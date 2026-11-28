"""实时视频叠加服务 v2：8GB 档完整实时管线 + 3D 骨架流式输出。

架构：
1. 标定：增量复用（drift < 80px 且 age < 5s 时跳过 RANSAC）
2. 检测：RTMO-s 人体框与 17 点关键点；RF-DETR-S 10 类职业比赛检测
3. 姿态：RTMO-s 每帧运行，RF-DETR 球员框通过 IoU 复用关键点
4. 时序平滑：运动预测补足遮挡帧，骨长约束校准
5. 3D 重建：bsk.perceive.pose_3d.Pose3DLifter（单应分解相机 + 骨长射线解，米制）
6. 渲染：烧录到 JPEG，前端只负责显示

性能目标：
- 1080p @ 30 FPS：~33ms/frame 预算
- 标定：< 10ms（复用 H）或 < 100ms（首次/重标定）
- 检测+姿态：RTMO-s + RF-DETR-S，共享每帧检测结果
- 时序平滑+3D：< 5ms
- 渲染：< 5ms

Profile（档位）：
- 4GB 档（BSK_PROFILE=4g，或无 CUDA 时的 auto 判定）：只出 2D，不计算任何 3D，
  请求里的 enable_3d 被忽略并在 status_3d 里给出 "profile_4g" 原因。
- 8GB 档（BSK_PROFILE=8g 或 auto 判定到 >=5.5GB 显存）：请求显式 enable_3d=1
  （或 BSK_RT_3D=1）时才计算，默认不付 3D 开销。

3D 安全闸（H 不可信宁可不输出）：
- calib 为空 / is_plausible 失败 / 重投影代价 > BSK_3D_MAX_ERR_PX /
  标定年龄 > BSK_3D_MAX_AGE_S / 单应分解不出相机 / 相机高度不在 2~60m
  → players_3d=[]，status_3d 写明原因，前端显示"标定不可用/3D 暂停"。
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from bsk import calibrate, court, render, schema
from bsk.perceive import shots as shotmod
from bsk.perceive.court_map import CourtMapState, apply_court_coords, filter_confident_off_court
from bsk.perceive.player_gate import (
    conservative_court_polygon, ground_point, player_quality,
    rfdetr_player_gate, safe_keypoints,
)
from bsk.perceive.pose_filter import PoseFilter, PoseFilterConfig
from bsk.roi_tracking import get_roi_tracker, apply_court_roi_from_calibration


# ── 模型懒加载 ───────────────────────────────────────────────

_models_lock = threading.Lock()
_models: dict = {}
_inference_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bsk-infer")
_POSE_EVERY = max(1, int(os.environ.get("BSK_POSE_EVERY", "2")))
_pose_cache: dict[tuple[str, int], tuple[int, list]] = {}
_people_cache: dict[str, tuple[int, list]] = {}

# 骨架时序状态层（抗抖动 + 遮挡维持）：默认开；BSK_POSE_FILTER=0 可关掉做 A/B 排障
_POSE_FILTER_ON = os.environ.get("BSK_POSE_FILTER", "1") != "0"
_pose_filters: dict[tuple[str, str], PoseFilter] = {}


def _get_ins():
    with _models_lock:
        if "ins" not in _models:
            from bsk.analyze import RtmdetIns
            _models["ins"] = RtmdetIns()
        return _models["ins"]


def _get_rtmo():
    """RTMO-s 姿态模型（ONNX/TensorRT），框+关键点质量与批量分析一致。"""
    with _models_lock:
        if "rtmo" not in _models:
            from bsk.perceive.persons import Rtmo
            _models["rtmo"] = Rtmo()
        return _models["rtmo"]


def _get_tracker(vid: str):
    """每个视频一个跟踪器：全局共用一个会导致多视频串 ID。"""
    with _models_lock:
        key = f"tracker:{vid}"
        if key not in _models:
            from bsk.analyze import Tracker
            _models[key] = Tracker(min_hits=2)
        return _models[key]


def _get_smoother():
    """已废弃：3D 时序平滑改由 bsk.perceive.pose_3d.Pose3DLifter 内部（深度域 EMA）承担。

    bsk/render_3d.py 的 PlayerSkeleton3D/TemporalSmoother 是早期的 bbox 比例高度法，
    保留给离线脚本，实时链路不再使用；这里只留一个显式报错，避免旧代码悄悄回流。
    """
    raise RuntimeError("TemporalSmoother 已从实时链路移除，请使用 Pose3DLifter（见 ensure_3d）")


# ── 档位（4GB / 8GB）────────────────────────────────────────
# 4GB 档必须保持"只出点"：不加载、不计算任何姿态 3D。
def _detect_profile() -> str:
    raw = os.environ.get("BSK_PROFILE", "auto").strip().lower()
    if raw in ("4g", "4gb", "low", "4"):
        return "4g"
    if raw in ("8g", "8gb", "high", "8"):
        return "8g"
    # auto：有可用 CUDA 且显存 >= 5.5GB 视作 8GB 档，其余按 4GB 档（保守）
    try:
        import torch

        if torch.cuda.is_available():
            _free, total = torch.cuda.mem_get_info()
            return "8g" if float(total) >= 5.5e9 else "4g"
    except Exception:
        pass
    return "4g"


_PROFILE = _detect_profile()


def rt_profile() -> str:
    """当前运行档位："4g" / "8g"。"""
    return _PROFILE


def profile_allows_3d() -> bool:
    return _PROFILE == "8g"


# ── 3D 安全闸参数 ───────────────────────────────────────────
# 单应不可信（拟合代价大 / 太旧 / 分解不出相机）时一律不输出 3D。
_3D_MAX_ERR_PX = float(os.environ.get("BSK_3D_MAX_ERR_PX", "35"))
_3D_MAX_AGE_S = float(os.environ.get("BSK_3D_MAX_AGE_S", "2.5"))
# BSK_RT_3D=1：8GB 档下即使请求没带 enable_3d 也算 3D（脚本/离线验收用）
_RT_3D_DEFAULT = os.environ.get("BSK_RT_3D", "0") == "1"

# 固定机位 H 覆盖（验收/回放用）：.npy = court→pixel（项目里 H_inv 的约定），
# 配 BSK_H_OVERRIDE_W/H 指明该 H 的参考分辨率；.json 见 temp/data/.tmp_H_demo.json 格式。
_H_OVERRIDE_PATH = os.environ.get("BSK_H_OVERRIDE", "").strip()
_H_OVERRIDE_CACHE: dict = {}


def _h_override():
    """读取固定物 H（只做一次）。返回 (H_court_to_pixel, (W,H), court_model) 或 None。"""
    if not _H_OVERRIDE_PATH:
        return None
    if _H_OVERRIDE_CACHE.get("path") == _H_OVERRIDE_PATH:
        return _H_OVERRIDE_CACHE.get("value")
    value = None
    try:
        from pathlib import Path

        p = Path(_H_OVERRIDE_PATH)
        if p.suffix.lower() == ".json":
            import json

            d = json.loads(p.read_text(encoding="utf-8"))
            H = np.asarray(d.get("H_court_to_pixel", d.get("H_inv")), dtype=np.float64)
            size = (int(d.get("width", 960)), int(d.get("height", 540)))
            model = str(d.get("court_model", "FIBA"))
        else:
            H = np.asarray(np.load(p), dtype=np.float64)
            size = (int(os.environ.get("BSK_H_OVERRIDE_W", "960")),
                    int(os.environ.get("BSK_H_OVERRIDE_H", "540")))
            model = os.environ.get("BSK_H_OVERRIDE_MODEL", "FIBA")
        if H.shape != (3, 3) or not np.all(np.isfinite(H)):
            raise ValueError("H 必须是有限 3x3")
        value = (H, size, model)
    except Exception as exc:
        print(f"[3D] BSK_H_OVERRIDE 读取失败，忽略: {type(exc).__name__}: {exc}")
        value = None
    _H_OVERRIDE_CACHE.update(path=_H_OVERRIDE_PATH, value=value)
    return value


def _override_calibration(shape: tuple) -> Optional[calibrate.Calibration]:
    """把固定机位 H 折算到当前解码分辨率，返回标定对象（source=override）。"""
    ov = _h_override()
    if ov is None:
        return None
    H, (rw, rh), model = ov
    sx, sy = shape[1] / max(rw, 1), shape[0] / max(rh, 1)
    S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    Hs = S @ H
    try:
        Hpix = np.linalg.inv(Hs)
    except np.linalg.LinAlgError:
        return None
    cal = calibrate.Calibration(
        H=Hpix.tolist(), H_inv=Hs.tolist(), reproj_error_px=0.0,
        keypoints_used=4, court_model=model, timestamp=time.time(),
    )
    cal.source = "override"      # 标记：免掉"陈旧"判定，但仍要过 plausible/相机分解
    return cal


# ── 3D 提升（Pose3DLifter）──────────────────────────────────
_lifters: dict[str, object] = {}
_lift_lock = threading.Lock()


def _get_lifter(vid: str, image_size: tuple[int, int], court_obj: court.Court):
    """每个视频一份提升器：深度/身高 EMA 状态按 track_id 跨帧延续。"""
    with _models_lock:
        lifter = _lifters.get(vid)
        if lifter is None:
            from bsk.perceive.pose_lift_net import HybridPose3DLifter

            lifter = HybridPose3DLifter(court=court_obj, image_size=image_size)
            _lifters[vid] = lifter
        else:
            if tuple(getattr(lifter, "image_size", ())) != tuple(image_size):
                lifter.set_image_size(image_size)
            lifter.court = court_obj
        return lifter


def _h_gate(cal: Optional[calibrate.Calibration], shape: tuple) -> dict:
    """单应可信度闸门。ok=False 时调用方不得输出任何 3D。"""
    w, h = int(shape[1]), int(shape[0])
    st: dict = {"ok": False, "calib_ok": False, "reason": "no_calib"}
    if cal is None:
        return st
    src = getattr(cal, "source", "line_calib")
    st["source"] = src
    try:
        H = np.asarray(cal.H_inv, dtype=np.float64)
        plausible = bool(cal.is_plausible(w, h))
        st["reproj_px"] = round(float(cal.reproj_error_px), 2)
        st["age_s"] = round(max(0.0, time.time() - float(getattr(cal, "timestamp", 0.0))), 2)
    except Exception as exc:
        st["reason"] = "calib_invalid"
        st["detail"] = f"{type(exc).__name__}: {exc}"
        return st
    if not plausible:
        st["reason"] = "implausible"          # 角点塌缩/超出画面：3D 没有意义
        return st
    if not np.all(np.isfinite(H)):
        st["reason"] = "h_nan"
        return st
    if src != "override":
        if st["reproj_px"] > _3D_MAX_ERR_PX:
            st["reason"] = "high_reproj"
            return st
        if st["age_s"] > _3D_MAX_AGE_S:
            st["reason"] = "stale"
            return st
    from bsk.perceive.pose_3d import camera_from_homography

    cam = camera_from_homography(H, (w, h))
    if cam is None:
        st["reason"] = "camera_failed"        # 单应退化，分解不出相机
        return st
    st["cam_height_m"] = round(float(cam.height), 2)
    st["f_px"] = round(float(cam.f), 1)
    st["f_estimated"] = bool(cam.f_estimated)
    if not (2.0 <= float(cam.height) <= 60.0):
        st["reason"] = "camera_implausible"   # 相机贴地/在天上：米制尺度不可信
        return st
    st.update(ok=True, calib_ok=True, reason="ok")
    return st


def _court_map_gate(cal: Optional[calibrate.Calibration], shape: tuple) -> dict:
    """球员脚点反投影闸门。只要求 H 可用于地平面，不要求相机可分解成 3D。"""
    w, h = int(shape[1]), int(shape[0])
    st: dict = {"ok": False, "calib_ok": False, "reason": "no_calib"}
    if cal is None:
        return st
    src = getattr(cal, "source", "line_calib")
    try:
        H = np.asarray(cal.H, dtype=np.float64)
        plausible = bool(cal.is_plausible(w, h))
        st["reproj_px"] = round(float(cal.reproj_error_px), 2)
        st["age_s"] = round(max(0.0, time.time() - float(getattr(cal, "timestamp", 0.0))), 2)
        st["source"] = src
    except Exception as exc:
        st["reason"] = "calib_invalid"
        st["detail"] = f"{type(exc).__name__}: {exc}"
        return st
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        st["reason"] = "h_invalid"
        return st
    if not plausible:
        st["reason"] = "implausible"
        return st
    if src != "override":
        if st["reproj_px"] > _3D_MAX_ERR_PX:
            st["reason"] = "high_reproj"
            return st
        if st["age_s"] > _3D_MAX_AGE_S:
            st["reason"] = "stale"
            return st
    st.update(ok=True, calib_ok=True, reason="ok")
    return st


@dataclass(slots=True)
class Player3D:
    """单个球员的 3D 结果（世界坐标，米）。visible=低置信度点，occluded=含推测点。"""
    track_id: int
    kpts: np.ndarray                 # (17,3)
    conf: np.ndarray                 # (17,)
    visible: np.ndarray              # (17,) bool
    stature: float
    root: np.ndarray                 # 骨盆
    ground: np.ndarray               # 双足中点（地面锚）
    reproj_err_px: float = 0.0
    occluded: bool = False
    height_px: float = 0.0
    bbox_px: tuple | None = None

    def to_dict(self, digits: int = 3) -> dict:
        return {
            "track_id": int(self.track_id),
            "kpts": np.round(self.kpts, digits).tolist(),
            "visible": [int(v) for v in self.visible],
            "conf": np.round(self.conf, 2).tolist(),
            "stature": round(float(self.stature), 2),
            "root": np.round(self.root, digits).tolist(),
            "ground": np.round(self.ground, digits).tolist(),
            "reproj_err_px": round(float(self.reproj_err_px), 2),
            "occluded": bool(self.occluded),
            "height_px": round(float(self.height_px), 1),
            "bbox_px": None if self.bbox_px is None else [round(float(v), 1) for v in self.bbox_px],
        }


def _lift_players_2d(poses: list[schema.Pose2D], objects: list[schema.Tracked],
                     img_shape: tuple, frame: int, status: dict) -> tuple[list[Player3D], dict]:
    """标定不可用时把识别到的 2D 骨架投影到虚拟球场，供 3D 并排预览。"""
    h, w = img_shape[:2]
    court_obj = court.FIBA
    boxes = {o.track_id: o.bbox_px for o in objects
             if o.category == "player" and o.bbox_px}
    out: list[Player3D] = []
    for pose in poses:
        bbox = boxes.get(pose.track_id)
        if bbox is None:
            continue
        k2 = np.asarray(pose.kpts, dtype=np.float64)
        if k2.shape != (17, 3):
            continue
        x0, _y0, x1, y1 = map(float, bbox)
        foot_x, foot_y = (x0 + x1) * 0.5, y1
        scale = max(y1 - _y0, 1.0)
        ground = np.array([(foot_x / w - 0.5) * court_obj.length,
                           (0.5 - foot_y / h) * court_obj.width, 0.0])
        k3 = np.empty((17, 3), dtype=np.float64)
        k3[:, 0] = ground[0] + (k2[:, 0] - foot_x) / scale
        k3[:, 1] = ground[1] + (k2[:, 1] - foot_y) / scale
        k3[:, 2] = np.maximum(0.0, (foot_y - k2[:, 1]) / scale * 1.8)
        conf = k2[:, 2]
        visible = conf >= 0.25
        if not visible.any():
            continue
        root = (k3[11] + k3[12]) * 0.5
        out.append(Player3D(
            track_id=int(pose.track_id), kpts=k3, conf=conf, visible=visible,
            stature=1.8, root=root, ground=ground, reproj_err_px=0.0,
            occluded=bool(getattr(pose, "occluded", False)),
            height_px=float(getattr(pose, "height_px", scale)), bbox_px=bbox,
        ))
    status = {**status, "ok": bool(out), "reason": "2d_fallback" if out else "lift_empty",
              "calib_ok": False, "calib_reason": status.get("reason"),
              "ms": 0.0, "n": len(out)}
    return out, status


def _lift_players(vid: str, poses: list[schema.Pose2D], objects: list[schema.Tracked],
                  cal: Optional[calibrate.Calibration], img_shape: tuple,
                  frame: int) -> tuple[list[Player3D], dict]:
    """把同一 FrameResult 的 2D 姿态提升到 3D；不重新检测/不预取未来帧。"""
    st = _h_gate(cal, img_shape)
    if not st["ok"]:
        return _lift_players_2d(poses, objects, img_shape, frame, st)
    if not poses:
        st.update(ok=True, reason="no_poses", ms=0.0, n=0)
        return [], st
    H = np.asarray(cal.H_inv, dtype=np.float64)
    model = getattr(cal, "court_model", "FIBA")
    court_obj = court.COURTS.get(model, court.FIBA)
    lifter = _get_lifter(vid, (int(img_shape[1]), int(img_shape[0])), court_obj)
    boxes = {o.track_id: o.bbox_px for o in objects
             if o.category == "player" and o.bbox_px}
    t0 = time.time()
    out: list[Player3D] = []
    for p in poses:
        k = np.asarray(p.kpts, dtype=np.float64)
        res = lifter.lift_result(k, boxes.get(p.track_id), H, p.track_id, frame)
        if res is None or not bool(res.visible.any()):
            continue
        if not res.on_court:          # 离球场 4m 以外（观众/替补席）不输出
            continue
        out.append(Player3D(
            track_id=int(p.track_id), kpts=res.kpts3d, conf=np.asarray(res.conf, dtype=np.float64),
            visible=np.asarray(res.visible, dtype=bool), stature=float(res.stature),
            root=np.asarray(res.root, dtype=np.float64), ground=np.asarray(res.ground, dtype=np.float64),
            reproj_err_px=float(res.reproj_err_px), occluded=bool(getattr(p, "occluded", False)),
            height_px=float(getattr(p, "height_px", 0.0)), bbox_px=boxes.get(p.track_id),
        ))
    st["ms"] = round((time.time() - t0) * 1000.0, 2)
    st["n"] = len(out)
    if hasattr(lifter, "status"):
        st["net"] = lifter.status()
    if not out:
        st["reason"] = "lift_empty"
    return out, st


def _cache_key(vid: str, n: int, backend: str, img_shape: tuple[int, int]) -> tuple:
    return (vid, n, backend, int(img_shape[1]), int(img_shape[0]))


def ensure_3d(vid: str, n: int, backend: str = "accurate",
              img_shape: tuple[int, int] | None = None) -> Optional[FrameResult]:
    """用缓存里的帧结果补算 3D（不重跑检测/姿态）。返回同一 FrameResult。"""
    r = get_cached(vid, n, backend, img_shape)
    if r is None:
        return None
    # reason=disabled 表示这一帧是按纯 2D 分析的，可以补算；其余已算过的原因不重试
    if r.status_3d.get("computed") and r.status_3d.get("reason") != "disabled":
        return r
    if not profile_allows_3d():
        r.status_3d = {"ok": False, "calib_ok": False, "reason": "profile_4g",
                       "computed": True, "enabled": True, "ms": 0.0}
        return r
    with _lift_lock:
        if r.status_3d.get("computed") and r.status_3d.get("reason") != "disabled":
            return r
        players, st = _lift_players(vid, r.poses, r.objects, r.calib, r.img_shape, r.frame)
        st["computed"] = True
        st["enabled"] = True
        r.players_3d = players
        r.status_3d = st
    return r


def frame_3d_payload(vid: str, n: int, backend: str = "accurate",
                     court_model: Optional[str] = None, limit: int = 24,
                     img_shape: tuple[int, int] | None = None) -> dict:
    """`/rt_3d` 的载荷：只读缓存 + 必要时用缓存结果补算 3D（绝不重跑检测）。

    增量友好：前端按帧号轮询，没算好时返回 status=no_frame，重试即可。
    """
    r = get_cached(vid, n, backend, img_shape)
    model = court_model or (getattr(r.calib, "court_model", None) if r is not None else None) \
        or "FIBA"
    base = {
        "frame": int(n),
        "court_model": model,
        "court": court.court_dims(model),
        "profile": _PROFILE,
        "calib_ok": False,
        "status": "no_frame",
        "players": [],
    }
    if r is None or r.img_shape == (0, 0):
        return base
    if profile_allows_3d():
        if not r.status_3d.get("ok"):
            r = ensure_3d(vid, n, backend) or r
    elif not r.status_3d.get("ok"):
        r.status_3d = {"ok": False, "calib_ok": False, "reason": "profile_4g",
                       "computed": True, "enabled": True, "ms": 0.0}
    st = r.status_3d or {}
    st2d = r.status_court2d or {}
    base.update(
        court_model=model,
        calib_ok=bool(st.get("calib_ok")),
        status=str(st.get("reason", "unknown")),
        detail={k: st.get(k) for k in
                ("source", "reproj_px", "age_s", "cam_height_m", "f_px", "ms", "n") if k in st},
        players=[p.to_dict() for p in r.players_3d[:limit]],
        court2d={k: st2d.get(k) for k in
                 ("ok", "reason", "n", "observed", "predicted", "rejected",
                  "outlier_rejected", "filtered_off_court", "reason_counts", "reproj_px", "age_s", "ms") if k in st2d},
        court_players=[
            {"track_id": int(o.track_id), "x": o.x, "y": o.y,
             "conf": round(float(o.conf), 2), "visible": bool(o.visible)}
            for o in r.objects
            if o.category == "player" and o.x is not None and o.y is not None
        ][:limit],
        poses_2d=len(r.poses),
    )
    return base


def _get_pose_filter(vid: str, backend: str) -> PoseFilter:
    """每个 (视频, 后端) 一份滤波状态：共用会把两个后端的 track 混在一起。"""
    with _models_lock:
        key = (vid, backend)
        pf = _pose_filters.get(key)
        if pf is None:
            # 候选已在跟踪前通过球员质量/语义门控；单帧有效球员也要立即出骨架，
            # 隔帧推理的显示帧由 PoseFilter 承担短时维持。
            pf = _pose_filters[key] = PoseFilter(PoseFilterConfig(min_confirm=1))
        return pf


# ── 球场标定（线分割 → 单应）────────────────────────────────
# 标定失败/不可信时返回 None：宁可不剔除观众，也绝不误删球员。
_CALIB_ON = os.environ.get("BSK_COURT_CALIB", "1") != "0"
_CALIB_EVERY = max(1, int(os.environ.get("BSK_CALIB_EVERY", "10")))
_CALIB_WEIGHTS = os.environ.get("BSK_CALIB_WEIGHTS", "checkpoints/calib_seg_mix2/best.pt")
_calib_trackers: dict = {}


def _get_calib_tracker(vid: str):
    """每个视频一份标定时序状态：每 K 帧拟合一次 → 近 5 次角点中值 → 失败沿用 ≤2s。"""
    with _models_lock:
        tr = _calib_trackers.get(vid)
        if tr is None:
            from bsk.line_calib import CalibrationTracker, LineCalibrator

            calib = LineCalibrator(weights=_CALIB_WEIGHTS, label=vid)
            tr = _calib_trackers[vid] = CalibrationTracker(
                calib, every=_CALIB_EVERY, window=5, stale_s=2.0)
        return tr


def _discard_calib_tracker(vid: str, roi_tracker) -> None:
    """丢弃旧线标定并停止后台 worker，同时清除依赖旧 H 的 ROI 状态。"""
    with _models_lock:
        old = _calib_trackers.pop(vid, None)
    if old is not None:
        stop = getattr(old, "stop", None)
        if callable(stop):
            stop()
    roi_tracker.court_mask = None
    roi_tracker.prev_gray = None
    roi_tracker.stable_count = 0


# ── FIBA 16 点标定（可选 ONNX；auto 在权重就绪时启用）──────────────
# 与球衣共享会话注册表和线程本地 CLAHE；轨迹/镜头状态按视频和后端隔离。
_COURT_KP_MODE = os.environ.get("BSK_COURT_KP", "auto").strip().lower()
_COURT_KP_ON = _COURT_KP_MODE not in ("0", "off", "false")
_COURT_KP_FORCE_CLASSICAL = _COURT_KP_MODE == "1"
_COURT_KP_EVERY = max(1, int(os.environ.get("BSK_COURT_KP_EVERY", "3")))
_COURT_KP_WEIGHTS = os.environ.get("BSK_COURT_KP_WEIGHTS", "data/models/court_keypoints16.onnx")
_court_kp_trackers: dict[tuple[str, str], object] = {}
_court_kp_warned = False


def _get_court_keypoint_tracker(vid: str, backend: str):
    """auto 模式没有权重时返回 None；显式 BSK_COURT_KP=1 可测试经典边线兜底。"""
    global _court_kp_warned
    if not _COURT_KP_ON:
        return None
    with _models_lock:
        key = (vid, backend)
        tr = _court_kp_trackers.get(key)
        if tr is None:
            from bsk.perceive.court_keypoints import CourtKeypointTracker, KeypointConfig

            tr = CourtKeypointTracker(KeypointConfig(
                weights=_COURT_KP_WEIGHTS, every_n=_COURT_KP_EVERY))
            _court_kp_trackers[key] = tr
            if not tr.model.ready and not _court_kp_warned:
                _court_kp_warned = True
                print(f"[court-kp] 16 点模型未启用：{tr.model.error}"
                      "（继续使用现有线标定；显式经典兜底：BSK_COURT_KP=1）")
        if not tr.model.ready and not _COURT_KP_FORCE_CLASSICAL:
            return None
        return tr


def court_keypoints_config() -> dict:
    """只读状态摘要，不创建模型或触发任何检测。"""
    with _models_lock:
        weights_ready = any(bool(getattr(getattr(tr, "model", None), "ready", False))
                            for tr in _court_kp_trackers.values())
        tracker_count = len(_court_kp_trackers)
    return {
        "mode": _COURT_KP_MODE,
        "weights": _COURT_KP_WEIGHTS,
        "enabled": _COURT_KP_ON,
        "active": _COURT_KP_ON and (_COURT_KP_FORCE_CLASSICAL or weights_ready),
        "trackers": tracker_count,
    }


_court_map_states: dict[tuple[str, str], CourtMapState] = {}


def _get_court_map_state(vid: str, backend: str) -> CourtMapState:
    """每个视频/兼容后端一份世界坐标状态，避免不同请求互相污染。"""
    with _models_lock:
        key = (vid, backend)
        state = _court_map_states.get(key)
        if state is None:
            state = _court_map_states[key] = CourtMapState(fps=30.0)
        return state


# ── 球衣号码识别（协作任务，需求见 docs/jersey_number_detection.md）──
# 默认开；BSK_JERSEY=0 整体关掉做 A/B 排障。号码框来自 RF-DETR，
# 每 2 帧采样，连续 3 次一致后发布。
_JERSEY_ON = os.environ.get("BSK_JERSEY", "1") != "0"
_JERSEY_WEIGHTS = Path(os.environ.get(
    "BSK_JERSEY_WEIGHTS", "data/models/jersey_pro_resnet18_aug.onnx"))
_jersey: dict[tuple[str, str], object] = {}
_jersey_warned = False
_rfdetr = None
_rfdetr_warned = False


def _get_rfdetr():
    """10 类检测器单例；球员、号码和球框共享同一次推理。"""
    global _rfdetr, _rfdetr_warned
    with _models_lock:
        if _rfdetr is None:
            from bsk.perceive.rfdetr import RfDetrDetector

            _rfdetr = RfDetrDetector(conf_thr=0.4)
        if not _rfdetr.ready and not _rfdetr_warned:
            _rfdetr_warned = True
            print(f"[rfdetr] 10 类检测器不可用：{_rfdetr.error}")
        return _rfdetr if _rfdetr.ready else None


def _rfdetr_people(pose_people: list, detections: list,
                   shape: tuple[int, int] | None = None) -> list:
    """RF-DETR 球员/裁判框 + RTMO 关键点，并用 RTMO 补齐漏检框。"""
    from bsk.perceive.persons import Person

    person_classes = {"player", "player-in-possession", "player-jump-shot",
                      "player-layup-dunk", "player-shot-block", "referee"}
    candidates = [d for d in detections
                  if d.class_name in person_classes and d.conf >= 0.30]
    if not candidates:
        return [person for person in pose_people if person.score >= 0.55]
    boxes = np.asarray([[d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0],
                         d.bbox[3] - d.bbox[1]] for d in candidates], np.float32)
    scores = np.asarray([d.conf for d in candidates], np.float32)
    keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.30, 0.5)
    out = []
    matched_pose: set[int] = set()
    for index in np.asarray(keep).reshape(-1):
        det = candidates[int(index)]
        best_iou, best_index, best = 0.0, -1, None
        for pose_index, person in enumerate(pose_people):
            a, b = det.bbox, person.bbox
            ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
            ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
            area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
            iou = inter / (area_a + area_b - inter)
            if iou > best_iou:
                best_iou, best_index, best = iou, pose_index, person
        if best_iou >= 0.10:
            matched_pose.add(best_index)
        kpts = best.kpts if best is not None and best_iou >= 0.25 else None
        if kpts is not None:
            kpts = kpts.copy()
            shift = np.array([
                (det.bbox[0] + det.bbox[2] - best.bbox[0] - best.bbox[2]) * 0.5,
                (det.bbox[1] + det.bbox[3] - best.bbox[1] - best.bbox[3]) * 0.5,
            ])
            kpts[:, :2] += shift
        category = "referee" if det.class_name == "referee" \
            or (best is not None and best.category == "referee") else "player"
        out.append(Person(bbox=det.bbox, score=det.conf, kpts=kpts, category=category))
    unmatched = [person for index, person in enumerate(pose_people) if index not in matched_pose]
    if shape is not None and len(out) >= 3:
        feet = np.asarray([[(p.bbox[0] + p.bbox[2]) * 0.5, p.bbox[3]] for p in out], np.float32)
        hull = cv2.convexHull(feet)
        margin = 0.12 * max(shape)
        out.extend(person for person in unmatched
                   if person.score >= 0.45 and cv2.pointPolygonTest(
                       hull, ((person.bbox[0] + person.bbox[2]) * 0.5, person.bbox[3]), True) >= -margin)
    else:
        out.extend(person for person in unmatched if person.score >= 0.55)
    return out


def _get_jersey(vid: str, backend: str):
    """每个 (视频, 后端) 一份号码状态；权重/运行时不可用时返回 None（链路整体退化）。"""
    global _jersey_warned
    if not _JERSEY_ON:
        return None
    with _models_lock:
        key = (vid, backend)
        rec = _jersey.get(key)
        if rec is None:
            from bsk.perceive.jersey import JerseyRecognizer
            rec = JerseyRecognizer(weights=_JERSEY_WEIGHTS)
            if not rec.ready and not _jersey_warned:
                _jersey_warned = True
                print(f"[jersey] 号码识别未启用：{rec.unavailable_reason}"
                      "（训练导出：python -m bsk.train_jersey；关闭：BSK_JERSEY=0）")
            _jersey[key] = rec
        return rec if rec.ready else None


def get_jersey_info(vid: str, backend: str) -> dict:
    with _models_lock:
        rec = _jersey.get((vid, backend))
    if rec is None:
        return {"published": 0, "crops": 0, "tracks": 0, "numbers": {}}
    numbers = rec.published_numbers(vid)
    return {"published": len(numbers), "numbers": numbers,
            "crops": rec._n_crop, "tracks": len(rec._states)}


# ─ 实时状态缓存 ─────────────────────────────────────────────

@dataclass
class FrameResult:
    """单帧实时分析结果。"""
    frame: int
    pts_us: int
    objects: list[schema.Tracked]
    poses: list[schema.Pose2D]
    segments: list[schema.Segment2D]
    calib: Optional[calibrate.Calibration]
    prev_rgb: Optional[np.ndarray]  # 用于运动检测
    players_3d: list[Player3D] = field(default_factory=list)  # 3D 骨架（米制世界坐标）
    img_shape: tuple[int, int] = (0, 0)  # (h, w) 检测所基于的分辨率
    court_cut: int = 0  # 被球场范围过滤掉的人数（观众/替补席）
    # 2D 俯视坐标状态：ok/reason/n/rejected/court_model/reproj_px/age_s
    status_court2d: dict = field(default_factory=dict)
    # 3D 状态：ok/reason/calib_ok/reproj_px/age_s/cam_height_m/ms/n/computed/enabled
    status_3d: dict = field(default_factory=dict)
    # FIBA 16 点在线配准；None 表示该帧未启用，失效结果仍保留 reason/H=None。
    court_keypoints: object | None = None

    def has_3d(self) -> bool:
        return bool(self.status_3d.get("ok")) and bool(self.players_3d)


class RealtimeCache:
    """线程安全的帧结果缓存，按 (视频, 帧号, 后端) 隔离。"""

    def __init__(self, max_size: int = 60):
        self._cache: OrderedDict[tuple, FrameResult] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def put(self, key: tuple, result: FrameResult) -> None:
        with self._lock:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def get(self, key: tuple) -> Optional[FrameResult]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def get_prev_rgb(self, vid: str, n: int) -> Optional[np.ndarray]:
        with self._lock:
            for fn in range(n - 1, max(0, n - 10), -1):
                for (v, f, *_rest), r in self._cache.items():
                    if v == vid and f == fn and r.prev_rgb is not None:
                        return r.prev_rgb
            return None


_cache = RealtimeCache(max_size=60)


# 注意：不做「提前分析后续帧」的预取。后续会接直播流，直播只有当前帧，
# 提前消费未来帧的架构在直播下无法成立；实时性靠单帧延迟与丢帧策略保证。

def engine_info() -> dict:
    """已加载的推理引擎信息（面板排障用）。"""
    with _models_lock:
        rtmo = _models.get("rtmo")
        rfdetr = _rfdetr
    return {
        "rtmo": getattr(rtmo, "ep", None) if rtmo is not None else None,
        "rfdetr": getattr(rfdetr, "ep", None) if rfdetr is not None else None,
        "rfdetr_model": str(rfdetr.path) if rfdetr is not None else None,
    }


def get_cached(vid: str, n: int, backend: str = "accurate",
               img_shape: tuple[int, int] | None = None) -> Optional[FrameResult]:
    """取已分析好的帧结果（只读缓存，不触发计算）。"""
    if img_shape is not None:
        return _cache.get(_cache_key(vid, n, backend, img_shape))
    with _cache._lock:
        for (v, f, b, _w, _h), result in reversed(_cache._cache.items()):
            if v == vid and f == n and b == backend:
                return result
    return None


def _get_court_bounds(H_world_to_pixel: np.ndarray, img_shape: tuple) -> Optional[tuple]:
    """Get court boundary in pixel coordinates for ROI filtering.

    Args:
        H_world_to_pixel: 标定的 H_inv（世界→像素），不是 pixel→court

    Returns:
        (x_min, y_min, x_max, y_max) or None if calibration unavailable
    """
    if H_world_to_pixel is None:
        return None

    from bsk.court import project_court_polygon
    poly = project_court_polygon(H_world_to_pixel)

    x_min = float(poly[:, 0].min()) - 50  # Margin for safety
    y_min = float(poly[:, 1].min()) - 50
    x_max = float(poly[:, 0].max()) + 50
    y_max = float(poly[:, 1].max()) + 50

    return (x_min, y_min, x_max, y_max)


def _court_polygon(H_world_to_pixel: np.ndarray, court_model: str = "FIBA") -> Optional[np.ndarray]:
    """球场外框多边形（像素）。比外接矩形更贴合：矩形会把角落里的人群一起收进来。"""
    if H_world_to_pixel is None:
        return None
    from bsk.court import COURTS, project_court_polygon

    poly = project_court_polygon(H_world_to_pixel, COURTS.get(court_model, court.FIBA))
    if not np.all(np.isfinite(poly)):
        return None
    return poly


def _inside_court(poly: np.ndarray, x: float, y: float, margin: float) -> bool:
    """点到球场多边形的有符号距离测试；margin 允许脚踩出边线一点。"""
    return cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(x), float(y)), True) >= -margin


def _point_in_rect(x: float, y: float, rect: tuple) -> bool:
    """Check if point is inside rectangle."""
    x_min, y_min, x_max, y_max = rect
    return x_min <= x <= x_max and y_min <= y <= y_max


def _fix_keypoint_symmetry(kpts: np.ndarray, bbox: tuple) -> np.ndarray:
    """Fix left-right keypoint asymmetry using temporal/mirror augmentation.

    COCO keypoint order:
    0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear,
    5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow,
    9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip,
    13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle

    Left-right pairs: (1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)
    """
    if kpts.shape[0] != 17:
        return kpts

    # Mirror pairs (left_idx, right_idx)
    mirror_pairs = [
        (1, 2),   # eyes
        (3, 4),   # ears
        (5, 6),   # shoulders
        (7, 8),   # elbows
        (9, 10),  # wrists
        (11, 12), # hips
        (13, 14), # knees
        (15, 16), # ankles
    ]

    bbox_height = bbox[3] - bbox[1]
    bbox_center_x = (bbox[0] + bbox[2]) / 2

    for left_idx, right_idx in mirror_pairs:
        left_conf = kpts[left_idx, 2]
        right_conf = kpts[right_idx, 2]

        # If one side is much better than the other, use symmetry to estimate
        if left_conf < 0.3 and right_conf > 0.7:
            # Right visible, left occluded → mirror right to left
            dx = kpts[right_idx, 0] - bbox_center_x
            kpts[left_idx, 0] = bbox_center_x - dx
            kpts[left_idx, 1] = kpts[right_idx, 1]
            kpts[left_idx, 2] = right_conf * 0.8  # Reduced confidence for estimated

        elif right_conf < 0.3 and left_conf > 0.7:
            # Left visible, right occluded → mirror left to right
            dx = bbox_center_x - kpts[left_idx, 0]
            kpts[right_idx, 0] = bbox_center_x + dx
            kpts[right_idx, 1] = kpts[left_idx, 1]
            kpts[right_idx, 2] = left_conf * 0.8

    return kpts


# ─ 核心实时管线 ─────────────────────────────────────────────

def analyze_frame_realtime(
    vid: str,
    n: int,
    rgb: np.ndarray,
    pts_us: int,
    run_seg: bool = False,
    enable_3d: bool = False,
    enable_calib: Optional[bool] = None,
    run_ball: bool = True,
    backend: str = "accurate",
) -> FrameResult:
    """实时分析单帧。

    Args:
        vid: 视频 ID
        n: 帧号
        rgb: RGB 图像 [H, W, 3]
        pts_us: 时间戳（微秒）
        run_seg: 是否运行分割（慢，默认关）
        enable_3d: 是否启用 3D 骨架重建（8GB 档）
        enable_calib: 是否做球场标定（None = 读 BSK_COURT_CALIB，默认开）

    Returns:
        FrameResult: 包含检测、姿态、分割、标定、3D 骨架结果
    """
    if enable_calib is None:
        enable_calib = _CALIB_ON
    t0 = time.time()

    # 1. 球场标定：优先 FIBA 16 点，失效时回退现有异步线标定。
    #    16 点硬切时会清空线标定历史和球场 ROI，绝不让旧机位 H 继续用于 3D/剔除。
    cal_tracker = None
    cal = None
    court_kp = None
    roi_tracker = get_roi_tracker(vid)
    if enable_calib and not _H_OVERRIDE_PATH:
        kp_tracker = _get_court_keypoint_tracker(vid, backend)
        if kp_tracker is not None:
            try:
                court_kp = kp_tracker.update(n, rgb)
            except Exception as exc:
                # 新模块不得拖垮已有实时链路；失败时 keypoints 状态仍明确标记失效。
                from bsk.perceive.court_keypoints import empty_result

                court_kp = empty_result(n, time.time(), n, "error", "none",
                                         f"runtime_error:{type(exc).__name__}", "none", {})
                # 异常发生在 update 内部时，旧 H 可能仍留在 tracker 中；必须同步失效，
                # 否则下一帧隔帧复用会把异常前的机位重新发布出来。
                kp_tracker.invalidate("runtime_error", frame=n)
                _discard_calib_tracker(vid, roi_tracker)
                print(f"[court-kp] 帧 {n} 跳过：{type(exc).__name__}: {exc}")
            hard_cut = court_kp.transition == "hard_cut" or court_kp.mode == "hard_cut"
            if hard_cut:
                # 不修改 line_calib.py：丢弃整个跟踪器即可使旧历史和后台槽位失效。
                _discard_calib_tracker(vid, roi_tracker)
            if court_kp.valid:
                cal = court_kp.to_calibration()
            elif not hard_cut:
                cal_tracker = _get_calib_tracker(vid)
                cal = cal_tracker.update(n, rgb)
        else:
            cal_tracker = _get_calib_tracker(vid)
            cal = cal_tracker.update(n, rgb)
    # 固定机位覆盖（验收/回放）：完全跳过线分割拟合，避免与实时标定互相扰动
    _cal_ov = _override_calibration(rgb.shape)
    if _cal_ov is not None:
        cal = _cal_ov
        cal_tracker = None
        court_kp = None
    t_mark = time.time()
    t_calib = t_mark - t0

    # Cache H_inv and INVERT it for pixel->court projection
    # cal.H_inv is court->pixel, we need pixel->court for 3D lifting
    H_inv_court_to_pixel = np.asarray(cal.H_inv, dtype=np.float64) if cal else None
    H_inv = None
    if H_inv_court_to_pixel is not None:
        try:
            H_inv = np.linalg.inv(H_inv_court_to_pixel)
        except np.linalg.LinAlgError as e:
            print(f"[WARN] Frame {n}: Failed to invert homography: {e}")

    # ROI-based tracking optimization: check if we can skip detection
    # roi_tracker 已在标定阶段创建，硬切时同步清除了旧球场掩码。

    # Apply court ROI mask from calibration (use court->pixel matrix)
    # 掩码目前只用于运动比例统计，没必要每帧重算（1080p 填充+膨胀约几 ms）
    if H_inv_court_to_pixel is not None and n % 30 == 0:
        apply_court_roi_from_calibration(roi_tracker, H_inv_court_to_pixel, rgb.shape)

    # 2. RF-DETR 每帧运行；RTMO 关键点默认每 2 帧更新，时序层负责补帧。
    pose_model = _get_rtmo()
    detector = _get_rfdetr()
    detect_future = _inference_pool.submit(detector.predict, rgb, conf=0.3) \
        if detector else None
    pose_cache_key = (vid, id(pose_model))
    # 有 RF-DETR 当前帧语义门控时可隔帧跑 RTMO；无检测器时逐帧确认候选，
    # 否则无法区分“暂时没模型输出”和“这一帧真的没人”。
    pose_every = _POSE_EVERY if detector is not None else 1
    cached_pose = _pose_cache.get(pose_cache_key)
    if cached_pose is None or n < cached_pose[0] or n - cached_pose[0] >= pose_every:
        pose_people = pose_model(rgb, thr=0.3)
        _pose_cache[pose_cache_key] = (n, pose_people)
    else:
        # 隔帧只复用“模型不必重跑”的调度，不把旧检测冒充当前帧观测。
        # Tracker/PoseFilter 在当前帧做遮挡预测，才能正确老化并判断是否离场。
        pose_people = []
    rfdetr_dets = detect_future.result() if detect_future else []
    tracker = _get_tracker(vid)
    use_player_gate = os.environ.get("BSK_PLAYER_DETECTOR", "rfdetr").strip().lower() == "rfdetr"
    if use_player_gate and detector is not None:
        # 只用当前帧检测语义放行；陈旧轨迹不能为陌生低分候选背书。
        live_boxes = getattr(tracker, "_live", {})
        last_frames = getattr(tracker, "_last_frame", {})
        max_lost = getattr(tracker, "max_lost", 16)
        known_boxes = [
            box for tid, box in live_boxes.items()
            if 0 <= n - last_frames.get(tid, n) <= max_lost
        ]
        ppl = rfdetr_player_gate(pose_people, rfdetr_dets, known_boxes=known_boxes)
    else:
        ppl = pose_people
    court_poly = _court_polygon(H_inv_court_to_pixel, cal.court_model) if cal is not None else None
    if court_poly is not None:
        court_poly = conservative_court_polygon(court_poly, rgb.shape[:2])
    court_margin = 0.0
    if court_poly is not None:
        age = (time.time() - cal.timestamp) if cal_tracker is None else cal_tracker.age_s
        court_margin = max(8.0, rgb.shape[1] * 0.025) + min(max(age, 0.0), 2.0) * 0.015 * rgb.shape[1]
    court_cut = 0
    filtered_people = []
    for person in ppl:
        # 身份分配之前过滤：场外人员/半身假阳性不能消耗 track_id。
        if not player_quality(person, rgb.shape[:2], backend):
            continue
        if court_poly is not None:
            fx, fy = ground_point(person.bbox, person.kpts)
            if not _inside_court(court_poly, fx, fy, court_margin):
                court_cut += 1
                continue
        filtered_people.append(person)
    ppl = filtered_people
    _people_cache[vid] = (n, ppl)
    boxes = [p.bbox for p in ppl]
    ids = tracker.assign(boxes, frame_idx=n)
    confirmed_people = [p for p, tid in zip(ppl, ids) if tid >= 0]

    # ROI 只接收已经获得身份的有效球员，避免场外候选污染运动统计。
    roi_tracker.update_detections(confirmed_people)

    t_after_detect = time.time()
    t_detect = t_after_detect - t_mark

    # 3. 球检测复用 RF-DETR 当前帧结果，不额外推理。
    ball_filter = getattr(detector, "filter_balls", None) if detector else None
    ball_dets = ball_filter(rgb, rfdetr_dets, max_det=1) if run_ball and ball_filter else []

    # 单球固定 ID，不消耗球员跟踪器的 ID 序列。
    ball_objects = []
    for det in ball_dets:
        ball_objects.append(schema.Tracked(
            track_id=0, category="ball", frame=n,
            bbox_px=det.bbox, conf=det.conf, visible=True,
        ))

    t_after_ball = time.time()
    t_ball = t_after_ball - t_after_detect

    # 4. 构建 Tracked 对象和 Pose2D（过滤假阳性）
    objects = []
    poses = []

    # Get court polygon for ROI filtering (needs world->pixel matrix).
    # 用真实多边形而不是外接矩形：矩形四个角会把替补席/观众一起圈进来。
    # 写进 Pose2D 的来源标记：+pf 表示过了时序状态层
    pose_model = ("rtmo_s+rfdetr_s"
                  if os.environ.get("BSK_PLAYER_DETECTOR", "rfdetr").strip().lower() == "rfdetr"
                  else "rtmo_s") + \
                 ("+pf" if _POSE_FILTER_ON else "_symmetric")

    for p, tid in zip(ppl, ids):
        if tid < 0:
            continue
        # Fix left-right asymmetry: mirror keypoints if needed (lazy copy)
        # 走时序状态层时不在这里补：逐帧镜像会把"这一帧左臂可信、下一帧右臂可信"
        # 的来回切换直接写进坐标，表现为左右肢抽搐；改由 PoseFilter 带迟滞地补。
        kpts_fixed = safe_keypoints(p.kpts) if p.kpts is not None else None
        if kpts_fixed is not None and not _POSE_FILTER_ON:
            # Check if symmetry fix is actually needed before copying
            needs_fix = False
            for left_idx, right_idx in [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]:
                left_conf = kpts_fixed[left_idx, 2]
                right_conf = kpts_fixed[right_idx, 2]
                if (left_conf < 0.3 and right_conf > 0.7) or (right_conf < 0.3 and left_conf > 0.7):
                    needs_fix = True
                    break

            if needs_fix:
                kpts_fixed = kpts_fixed.copy()
                kpts_fixed = _fix_keypoint_symmetry(kpts_fixed, p.bbox)

        objects.append(schema.Tracked(
            track_id=tid, category=p.category, frame=n,
            bbox_px=p.bbox, conf=p.score, visible=True
        ))
        if kpts_fixed is not None and p.category == "player":
            poses.append(schema.Pose2D(
                video_id=vid, frame=n, track_id=tid,
                kpts=kpts_fixed.tolist(), model=pose_model,
                height_px=float(p.bbox[3] - p.bbox[1])
            ))
    t_after_pose = time.time()

    # 4b. 时序状态层：在线滤波 + 关节级遮挡预测 + 人体级短时维持 + 骨长约束
    t_filter = 0.0
    if _POSE_FILTER_ON:
        t_f0 = time.time()
        pose_filter = _get_pose_filter(vid, backend)
        # 注入的筛选器也沿用同一策略；上游已拒绝未确认/无效候选。
        pose_filter.cfg.min_confirm = 1
        boxes_by_id = {o.track_id: o.bbox_px for o in objects if o.category == "player" and o.bbox_px}
        poses, occluded_boxes = pose_filter.apply(n, poses, boxes_by_id)
        pose_by_id = {p.track_id: p for p in poses}
        evicted = []
        for tid, box in occluded_boxes.items():
            if court_poly is not None:
                predicted_pose = pose_by_id.get(tid)
                kp = predicted_pose.kpts if predicted_pose is not None else None
                fx, fy = ground_point(box, kp)
                if not _inside_court(court_poly, fx, fy, court_margin):
                    # 遮挡预测已经外推出有效区域，不能继续补骨架或让状态复活。
                    evicted.append(tid)
                    court_cut += 1
                    continue
            # 整人框丢失但状态未超龄：补一个 visible=False 的框，ID 与骨架都不跳
            objects.append(schema.Tracked(
                track_id=tid, category="player", frame=n, bbox_px=box,
                conf=0.2, visible=False,
            ))
        if evicted:
            poses = [p for p in poses if p.track_id not in evicted]
            pose_filter.forget(evicted)
        t_filter = time.time() - t_f0
    t_pose = time.time() - t_after_ball

    # 4c. 球衣号码：RFDETR 每帧都产出 number 框，采样节拍由轨迹状态控制。
    t_jersey = 0.0
    jersey_rec = _get_jersey(vid, backend)
    if jersey_rec is not None:
        t_j0 = time.time()
        try:
            number_boxes = [d.bbox for d in rfdetr_dets
                            if d.class_name == "number" and d.conf >= 0.4]
            jersey_rec.recognize(vid, n, rgb, objects, poses,
                                 number_boxes=number_boxes)
        except Exception as exc:  # 号码是附加信息，坏了不能拖垮实时链路
            print(f"[jersey] 帧 {n} 跳过：{type(exc).__name__}: {exc}")
        t_jersey = time.time() - t_j0

    # 合并人员和球对象
    objects.extend(ball_objects)

    # 4d. 线标定反投影：把球员脚点映射到球场平面（4GB 档也输出）
    status_court2d = _court_map_gate(cal, rgb.shape[:2])
    court_state = _get_court_map_state(vid, backend)
    status_court2d.update(apply_court_coords(
        objects, poses, cal, state=court_state, frame=n,
        image_shape=rgb.shape[:2], calibration_status=status_court2d,
    ))
    objects, off_court = filter_confident_off_court(
        objects, status_court2d, getattr(cal, "court_model", "FIBA"))
    if off_court:
        dropped_ids = {o.track_id for o in off_court}
        poses = [p for p in poses if p.track_id not in dropped_ids]
        court_cut += len(off_court)
        status_court2d["filtered_off_court"] = len(off_court)

    # 5. 3D 骨架提升（8GB 档且显式要求；4GB 档完全不算）
    #    提升输入就是本帧最终的 poses/objects，保证 3D 与 2D 来自同一份检测结果。
    players_3d = []
    t_smooth = 0.0
    status_3d: dict = {"enabled": bool(enable_3d), "computed": True, "ok": False,
                       "reason": "disabled" if not enable_3d else "", "calib_ok": False}
    if enable_3d:
        if not profile_allows_3d():
            status_3d = {"enabled": False, "computed": True, "ok": False,
                         "calib_ok": False, "reason": "profile_4g",
                         "profile": _PROFILE, "ms": 0.0, "n": 0}
        else:
            t_3d0 = time.time()
            with _lift_lock:
                players_3d, status_3d = _lift_players(
                    vid, poses, objects, cal, rgb.shape[:2], n)
            status_3d.update(enabled=True, computed=True, profile=_PROFILE)
            t_smooth = time.time() - t_3d0
    else:
        status_3d["profile"] = _PROFILE

    # 6. 分割（可选，很慢）
    segments = []
    if run_seg:
        ins = _get_ins()
        masked = {}
        for d in ins(rgb, thr=0.3):
            masked[d.bbox] = d
        for p, tid in zip(ppl, ids):
            m = None  # TODO: implement mask matching
            if m is not None and m.polygon:
                segments.append(schema.Segment2D(
                    video_id=vid, frame=n, track_id=tid, category="player",
                    polygon=m.polygon, area_px=m.area_px, model="rtmdet_ins_s"
                ))
    t_seg = 0.0

    # 6. 缓存结果（包含 3D 骨架用于流式传输）
    # Optimize: Store downsampled grayscale for motion detection (16x memory reduction)
    h, w = rgb.shape[:2]
    small_gray = cv2.cvtColor(
        cv2.resize(rgb, (w // 4, h // 4), interpolation=cv2.INTER_AREA),
        cv2.COLOR_RGB2GRAY
    )

    result = FrameResult(
        frame=n, pts_us=pts_us, objects=objects, poses=poses,
        segments=segments, calib=cal, prev_rgb=small_gray,  # Small grayscale instead of full RGB
        players_3d=players_3d, img_shape=(h, w), court_cut=court_cut,
        status_court2d=status_court2d,
        status_3d=status_3d, court_keypoints=court_kp,
    )
    _cache.put(_cache_key(vid, n, backend, rgb.shape[:2]), result)

    total_ms = (time.time() - t0) * 1000
    if total_ms > 50:  # 警告：超过预算
        print(f"[WARN] Frame {n} took {total_ms:.1f}ms "
              f"(calib={t_calib*1000:.1f}, detect={t_detect*1000:.1f}, ball={t_ball*1000:.1f}, "
              f"pose={t_pose*1000:.1f}, filter={t_filter*1000:.1f}, "
              f"jersey={t_jersey*1000:.1f}, "
              f"smooth={t_smooth*1000:.1f}, seg={t_seg*1000:.1f})")

    return result


def render_frame_realtime(
    vid: str,
    n: int,
    rgb: np.ndarray,
    pts_us: int,
    layers: render.Layers,
    width: int,
    quality: int = 72,
    enable_3d: bool = False,
    enable_calib: Optional[bool] = None,
    run_ball: bool = True,
    backend: str = "accurate",
    court_model: str = "FIBA",
) -> bytes:
    """实时渲染单帧并返回 JPEG。"""
    # 获取或计算分析结果（缓存基于检测时的分辨率，尺寸不一致必须重算）
    cached = _cache.get(_cache_key(vid, n, backend, rgb.shape[:2]))
    if cached is None or cached.img_shape != rgb.shape[:2]:
        cached = analyze_frame_realtime(
            vid, n, rgb, pts_us, enable_3d=enable_3d, enable_calib=enable_calib,
            run_ball=run_ball, backend=backend,
        )
    elif enable_3d and not cached.status_3d.get("computed"):
        # 这一帧先前按纯 2D 分析过：用同一份缓存结果补算 3D（不重跑检测/姿态）
        cached = ensure_3d(vid, n, backend) or cached

    # 获取前一帧用于运动检测（已经是1/4分辨率灰度图）
    prev_rgb = _cache.get_prev_rgb(vid, n)

    # 准备渲染参数
    h, w = rgb.shape[:2]

    # 标定信息
    H_inv = np.asarray(cached.calib.H_inv, dtype=np.float64) if cached.calib else None

    # Use court model from request parameter or calibration
    court_obj = court.COURTS.get(court_model, court.FIBA)
    if cached.calib is not None:
        # 画出的球场线要和标定用的球场模型一致，否则线会整体缩一圈
        court_obj = court.COURTS.get(cached.calib.court_model, court_obj)

    # 运动检测（复用缓存的小尺寸灰度图，避免重复resize）
    motion = None
    if layers.motion and prev_rgb is not None:
        # Compute current frame's small grayscale to match prev size
        h_small, w_small = prev_rgb.shape
        g1_small = cv2.cvtColor(
            cv2.resize(rgb, (w_small, h_small), interpolation=cv2.INTER_AREA),
            cv2.COLOR_RGB2GRAY
        )
        d = cv2.absdiff(g1_small, prev_rgb)
        m = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        motion = cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0

        # Upsample motion mask back to full res if needed
        h_full, w_full = rgb.shape[:2]
        if motion.shape != (h_full, w_full):
            motion = cv2.resize(motion.astype(np.uint8) * 255, (w_full, h_full),
                               interpolation=cv2.INTER_NEAREST).astype(bool)

    # 地板分割（轻量版）：木色与青/蓝色漆区两类地板都覆盖
    floor = None
    if layers.floor:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        wood = cv2.inRange(hsv, np.array([8, 35, 80]), np.array([32, 170, 255]))
        paint = cv2.inRange(hsv, np.array([70, 60, 40]), np.array([110, 210, 220]))
        floor = (wood | paint) > 0

    # 构建 notes
    notes = []
    if cached.calib:
        src = getattr(cached.calib, "source", "line_calib")
        age = max(0.0, time.time() - cached.calib.timestamp)
        tail = " (stale)" if age > 0.35 else ""
        if src == "override":
            notes.append(f"calib: override {cached.calib.court_model}")
        else:
            notes.append(f"calib: ok err={cached.calib.reproj_error_px:.1f}px "
                         f"{cached.calib.court_model}{tail}")
    else:
        notes.append("calib: FAILED")
    if cached.court_cut:
        notes.append(f"court: -{cached.court_cut} off-court")

    st3 = cached.status_3d or {}
    if enable_3d:
        if st3.get("ok"):
            notes.append(f"3D: {len(cached.players_3d)} players {st3.get('ms', 0):.1f}ms")
        else:
            notes.append(f"3D: off ({st3.get('reason', 'unknown')})")

    notes.append(f"rt: {len(cached.objects)} obj, {len(cached.poses)} pose")

    # 渲染
    rendered = render.render_frame(
        rgb,
        frame=n,
        pts_us=pts_us,
        nframes=99999,  # 实时模式不知道总帧数
        shot_id=None,
        court_visible=None,
        objects=cached.objects,
        poses=cached.poses,
        segments=cached.segments,
        floor=floor,
        motion=motion,
        ball_history=[],
        layers=layers,
        sources={"boxes": "realtime", "pose": "realtime", "seg": "realtime" if cached.segments else "none",
                 "ball": "none", "camera": "realtime"},
        notes=notes,
        calib_H_inv=H_inv,
        court_model=court_obj,
        in_place=True,  # rgb 由本函数独占，省一次整幅拷贝
    )

    return render.to_jpeg(rendered, quality)


# ── 清理旧缓存 ─────────────────────────────────────────────

def cleanup_old_cache(max_age_sec: float = 300.0) -> None:
    """定期清理超过 max_age_sec 的缓存。"""
    now = time.time()
    with _cache._lock:
        to_remove = [k for k, r in _cache._cache.items()
                     if now - r.pts_us / 1e6 > max_age_sec]
        for n in to_remove:
            del _cache._cache[n]


# 启动后台清理线程
_cleaner_thread = threading.Thread(
    target=lambda: [cleanup_old_cache(), time.sleep(60)] * 99999,
    daemon=True
)
_cleaner_thread.start()


# 启动后台模型预热线程（消除首帧延迟）
def _warmup_models():
    """Pre-load and warm up all models at startup."""
    import time
    print("[warmup] Loading models...")
    t0 = time.time()

    try:
        t1 = time.time()
        _get_ins()
        print(f"[warmup] RTMDet-Ins ready ({time.time()-t1:.1f}s)")
    except Exception as e:
        print(f"[warn] Failed to warmup ins: {e}")

    try:
        detector = _get_rfdetr()
        if detector is not None:
            detector.predict(np.zeros((512, 512, 3), np.uint8))
        print("[warmup] RF-DETR 10-class detector ready")
    except Exception as e:
        print(f"[warmup] Failed to warmup RF-DETR: {e}")

    try:
        m = _get_rtmo()
        print(f"[warmup] RTMO ready (engine={getattr(m, 'ep', None)})")
    except Exception as e:
        print(f"[warn] Failed to warmup rtmo: {e}")

    print(f"[warmup] All models loaded in {time.time()-t0:.1f}s")


_warmup_thread: Optional[threading.Thread] = None


def start_model_warmup() -> None:
    """服务启动后再后台预热，避免导入模块阻塞 HTTP。"""
    global _warmup_thread
    if _warmup_thread is None or not _warmup_thread.is_alive():
        _warmup_thread = threading.Thread(target=_warmup_models, daemon=True)
        _warmup_thread.start()


__all__ = ["analyze_frame_realtime", "render_frame_realtime",
           "get_cached", "ensure_3d", "frame_3d_payload", "engine_info",
           "RealtimeCache", "FrameResult", "Player3D",
           "rt_profile", "profile_allows_3d"]
