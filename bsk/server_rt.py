"""实时视频叠加服务 v2：8GB 档完整实时管线 + 3D 骨架流式输出。

架构：
1. 标定：增量复用（drift < 80px 且 age < 5s 时跳过 RANSAC）
2. 检测：YOLOv8n-pose，每帧运行（~15ms on GPU）
3. 姿态：YOLOv8n-pose 自带，每帧运行
4. 时序平滑：运动预测补足遮挡帧，骨长约束校准
5. 3D 重建：bsk.perceive.pose_3d.Pose3DLifter（单应分解相机 + 骨长射线解，米制）
6. 渲染：烧录到 JPEG，前端只负责显示

性能目标：
- 1080p @ 30 FPS：~33ms/frame 预算
- 标定：< 10ms（复用 H）或 < 100ms（首次/重标定）
- 检测+姿态：< 20ms（YOLOv8n-pose）
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
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from bsk import calibrate, court, render, schema
from bsk.perceive import shots as shotmod
from bsk.perceive.court_map import CourtMapState, apply_court_coords
from bsk.perceive.pose_filter import PoseFilter
from bsk.roi_tracking import get_roi_tracker, apply_court_roi_from_calibration


# ── 模型懒加载 ───────────────────────────────────────────────

_models_lock = threading.Lock()
_models: dict = {}

# 骨架时序状态层（抗抖动 + 遮挡维持）：默认开；BSK_POSE_FILTER=0 可关掉做 A/B 排障
_POSE_FILTER_ON = os.environ.get("BSK_POSE_FILTER", "1") != "0"
_pose_filters: dict[tuple[str, str], PoseFilter] = {}


def _get_fast_pose():
    """获取快速姿态模型：YOLOv8n-pose (~15ms on GPU).
    
    首次启动会下载模型到 data/models/yolov8n-pose.pt（约 6MB）
    """
    with _models_lock:
        if "fast_pose" not in _models:
            from bsk.perceive.fast_pose import YoloV8Pose
            from pathlib import Path
            cache_dir = Path("data/models")
            cache_dir.mkdir(parents=True, exist_ok=True)
            _models["fast_pose"] = YoloV8Pose(device="cuda", cache_dir=cache_dir)
        return _models["fast_pose"]


def _get_ins():
    with _models_lock:
        if "ins" not in _models:
            from bsk.analyze import RtmdetIns
            _models["ins"] = RtmdetIns()
        return _models["ins"]


def _get_rtmo():
    """准确档姿态模型：RTMO-s（ONNX），框+关键点质量与批量分析一致。"""
    with _models_lock:
        if "rtmo" not in _models:
            from bsk.perceive.persons import Rtmo
            _models["rtmo"] = Rtmo()
        return _models["rtmo"]


def _get_ball():
    with _models_lock:
        if "ball" not in _models:
            from bsk.perceive.ball import get_ball_detector
            _models["ball"] = get_ball_detector(device="cuda")
        return _models["ball"]


def _get_tracker(vid: str):
    """每个视频一个跟踪器：全局共用一个会导致多视频串 ID。"""
    with _models_lock:
        key = f"tracker:{vid}"
        if key not in _models:
            from bsk.analyze import Tracker
            _models[key] = Tracker()
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


def _lift_players(vid: str, poses: list[schema.Pose2D], objects: list[schema.Tracked],
                  cal: Optional[calibrate.Calibration], img_shape: tuple,
                  frame: int) -> tuple[list[Player3D], dict]:
    """把同一 FrameResult 的 2D 姿态提升到 3D；不重新检测/不预取未来帧。"""
    st = _h_gate(cal, img_shape)
    if not st["ok"]:
        st["ms"] = 0.0
        st["n"] = 0
        return [], st
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


def ensure_3d(vid: str, n: int, backend: str = "fast") -> Optional[FrameResult]:
    """用缓存里的帧结果补算 3D（不重跑检测/姿态）。返回同一 FrameResult。"""
    r = _cache.get((vid, n, backend))
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


def frame_3d_payload(vid: str, n: int, backend: str = "fast",
                     court_model: Optional[str] = None, limit: int = 24) -> dict:
    """`/rt_3d` 的载荷：只读缓存 + 必要时用缓存结果补算 3D（绝不重跑检测）。

    增量友好：前端按帧号轮询，没算好时返回 status=no_frame，重试即可。
    """
    r = get_cached(vid, n, backend)
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
                  "outlier_rejected", "reason_counts", "reproj_px", "age_s", "ms") if k in st2d},
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
            pf = _pose_filters[key] = PoseFilter()
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


_court_map_states: dict[tuple[str, str], CourtMapState] = {}


def _get_court_map_state(vid: str, backend: str) -> CourtMapState:
    """每个视频/后端一份世界坐标状态，避免快速档与准确档互相污染。"""
    with _models_lock:
        key = (vid, backend)
        state = _court_map_states.get(key)
        if state is None:
            state = _court_map_states[key] = CourtMapState(fps=30.0)
        return state


# ── 球衣号码识别（协作任务，需求见 docs/jersey_number_detection.md）──
# 默认开；BSK_JERSEY=0 整体关掉做 A/B 排障。只对准确档生效：快速档的框抖得厉害，
# 躯干裁片经常偏出号码区，融合层会被大量空白票拖住。
_JERSEY_ON = os.environ.get("BSK_JERSEY", "1") != "0"
_jersey: dict[tuple[str, str], object] = {}
_jersey_warned = False


def _get_jersey(vid: str, backend: str):
    """每个 (视频, 后端) 一份号码状态；权重/运行时不可用时返回 None（链路整体退化）。"""
    global _jersey_warned
    if not _JERSEY_ON or backend != "accurate":
        return None
    with _models_lock:
        key = (vid, backend)
        rec = _jersey.get(key)
        if rec is None:
            from bsk.perceive.jersey import JerseyRecognizer
            rec = JerseyRecognizer()
            if not rec.ready and not _jersey_warned:
                _jersey_warned = True
                print(f"[jersey] 号码识别未启用：{rec.unavailable_reason}"
                      "（训练导出：python -m bsk.train_jersey；关闭：BSK_JERSEY=0）")
            _jersey[key] = rec
        return rec if rec.ready else None


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
    return {"rtmo": getattr(rtmo, "ep", None) if rtmo is not None else None}


def get_cached(vid: str, n: int, backend: str = "fast") -> Optional[FrameResult]:
    """取已分析好的帧结果（只读缓存，不触发计算）。"""
    return _cache.get((vid, n, backend))


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
    run_ball: bool = False,
    backend: str = "fast",
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

    # 1. 球场标定：线分割 → 单应。每 K 帧拟合一次（后台线程，不阻塞帧路径），
    #    取不到/不可信时 cal=None，后面不做任何剔除（安全兜底）。
    # 注意：变量名用 cal_tracker，避免与下面的人员跟踪器 tracker 撞名（撞了会
    # 把 tracker.age_s 错算成"人员 tracker 的年龄"）。
    cal_tracker = None
    cal = None
    if enable_calib and not _H_OVERRIDE_PATH:
        cal_tracker = _get_calib_tracker(vid)
        cal = cal_tracker.update(n, rgb)
    # 固定机位覆盖（验收/回放）：完全跳过线分割拟合，避免与实时标定互相扰动
    _cal_ov = _override_calibration(rgb.shape)
    if _cal_ov is not None:
        cal = _cal_ov
        cal_tracker = None
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
    roi_tracker = get_roi_tracker(vid)
    
    # Apply court ROI mask from calibration (use court->pixel matrix)
    # 掩码目前只用于运动比例统计，没必要每帧重算（1080p 填充+膨胀约几 ms）
    if H_inv_court_to_pixel is not None and n % 30 == 0:
        apply_court_roi_from_calibration(roi_tracker, H_inv_court_to_pixel, rgb.shape)
    
    # 2. 姿态检测 + 跟踪
    # fast: YOLOv8n-pose（~50ms，召回与关键点精度一般）
    # accurate: RTMO-s（与批量分析同一模型，框和骨架质量一致）
    if backend == "accurate":
        ppl = _get_rtmo()(rgb, thr=0.3)
        det_thr = 0.3
        min_h_frac, min_h_abs = 0.03, 36
        min_w_frac, min_w_abs = 0.015, 18
        min_kpts = 2
        ar_lo, ar_hi = 0.6, 5.0
    else:
        ppl = _get_fast_pose()(rgb, thr=0.45)
        det_thr = 0.45
        min_h_frac, min_h_abs = 0.05, 60
        min_w_frac, min_w_abs = 0.025, 30
        min_kpts = 4
        ar_lo, ar_hi = 0.8, 4.5
    boxes = [p.bbox for p in ppl]
    
    tracker = _get_tracker(vid)
    ids = tracker.assign(boxes)
    
    # Always update ROI tracker with fresh detections
    roi_tracker.update_detections(ppl)

    t_after_detect = time.time()
    t_detect = t_after_detect - t_mark

    # 3. 球检测（模型训练完成前默认关闭；实时档 130ms 太贵且不准）
    ball_dets = _get_ball()(rgb, thr=0.3) if run_ball else []
    
    # 为球分配独立 track_id（从人员 tracker 之后继续编号）
    ball_objects = []
    for det in ball_dets:
        tid = tracker._next  # 使用下一个可用 ID
        tracker._next += 1
        x0, y0, x1, y1 = det.bbox
        ball_objects.append(schema.Tracked(
            track_id=tid,
            category="ball",
            frame=n,
            bbox_px=det.bbox,
            conf=det.conf,
            visible=True,
        ))
    
    t_after_ball = time.time()
    t_ball = t_after_ball - t_after_detect
    
    # 4. 构建 Tracked 对象和 Pose2D（过滤假阳性）
    objects = []
    poses = []
    
    # Get court polygon for ROI filtering (needs world->pixel matrix).
    # 用真实多边形而不是外接矩形：矩形四个角会把替补席/观众一起圈进来。
    court_poly = _court_polygon(H_inv_court_to_pixel, cal.court_model) if cal is not None else None
    court_margin = 0.0
    if court_poly is not None:
        # 标定陈旧时 H 会随镜头漂移：margin 随时间放大，宁可少剔也不误删
        age = (time.time() - cal.timestamp) if cal_tracker is None else cal_tracker.age_s
        court_margin = max(8.0, rgb.shape[1] * 0.025) + min(max(age, 0.0), 2.0) * 0.015 * rgb.shape[1]
    # 写进 Pose2D 的来源标记：+pf 表示过了时序状态层
    pose_model = ("rtmo_s" if backend == "accurate" else "yolov8n_pose") + \
                 ("+pf" if _POSE_FILTER_ON else "_symmetric")
    court_cut = 0
    
    for p, tid in zip(ppl, ids):
        # Filter 1: Confidence threshold
        if p.score < det_thr:
            continue
        
        # Filter 2: Bounding box aspect ratio (standing person ~1:2 to 1:3)
        bw, bh = p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1]
        aspect_ratio = bh / max(bw, 1)
        # Seated spectators are typically wide (AR < 1.0), standing players are tall (AR > 1.0)
        if aspect_ratio < ar_lo or aspect_ratio > ar_hi:
            continue
        
        # Filter 3: Adaptive minimum size (resolution-aware)
        min_height = max(min_h_abs, rgb.shape[0] * min_h_frac)
        min_width = max(min_w_abs, rgb.shape[1] * min_w_frac)
        if bh < min_height or bw < min_width:
            continue
        
        # Filter 4: Court ROI (if calibration available) - PRIMARY crowd filter
        # Use bbox bottom-center as the ground contact point; spectators/bench
        # sit outside the court polygon so their feet fail this test.
        if court_poly is not None:
            fx, fy = (p.bbox[0] + p.bbox[2]) / 2, p.bbox[3]
            if not _inside_court(court_poly, fx, fy, court_margin):
                court_cut += 1
                continue  # Outside court = spectator/bench
        
        # Filter 5: Keypoint quality check (spectators often have poor keypoints due to occlusion/seating)
        if p.kpts is not None:
            # Count high-confidence keypoints
            high_conf_kpts = np.sum(p.kpts[:, 2] > 0.5)
            # Standing players/referees typically have >6 good keypoints
            # Seated spectators often have <4 due to unusual poses
            if high_conf_kpts < min_kpts:
                continue  # Likely seated spectator with poor pose
        
        # Fix left-right asymmetry: mirror keypoints if needed (lazy copy)
        # 走时序状态层时不在这里补：逐帧镜像会把"这一帧左臂可信、下一帧右臂可信"
        # 的来回切换直接写进坐标，表现为左右肢抽搐；改由 PoseFilter 带迟滞地补。
        kpts_fixed = p.kpts
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
            track_id=tid, category="player", frame=n,
            bbox_px=p.bbox, conf=p.score, visible=True
        ))
        if kpts_fixed is not None:
            poses.append(schema.Pose2D(
                video_id=vid, frame=n, track_id=tid,
                kpts=kpts_fixed.tolist(), model=pose_model,
                height_px=float(p.bbox[3] - p.bbox[1])
            ))
    t_after_pose = time.time()

    # 4b. 时序状态层：在线滤波 + 关节级遮挡预测 + 人体级短时维持 + 骨长约束
    t_filter = 0.0
    if _POSE_FILTER_ON and poses:
        t_f0 = time.time()
        boxes_by_id = {o.track_id: o.bbox_px for o in objects if o.category == "player" and o.bbox_px}
        poses, occluded_boxes = _get_pose_filter(vid, backend).apply(n, poses, boxes_by_id)
        for tid, box in occluded_boxes.items():
            # 整人框丢失但状态未超龄：补一个 visible=False 的框，ID 与骨架都不跳
            objects.append(schema.Tracked(
                track_id=tid, category="player", frame=n, bbox_px=box,
                conf=0.2, visible=False,
            ))
        t_filter = time.time() - t_f0
    t_pose = time.time() - t_after_ball

    # 4c. 球衣号码：poses 构建之后调用（bbox 与骨架都已就位），就地写 Tracked.jersey
    #     采样与融合都在 JerseyRecognizer 内部完成，这里只负责计时与异常兜底
    t_jersey = 0.0
    jersey_rec = _get_jersey(vid, backend)
    if jersey_rec is not None:
        t_j0 = time.time()
        try:
            jersey_rec.recognize(vid, n, rgb, objects, poses)
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
        status_3d=status_3d,
    )
    _cache.put((vid, n, backend), result)
    
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
    run_ball: bool = False,
    backend: str = "fast",
    court_model: str = "FIBA",
) -> bytes:
    """实时渲染单帧并返回 JPEG。"""
    # 获取或计算分析结果（缓存基于检测时的分辨率，尺寸不一致必须重算）
    cached = _cache.get((vid, n, backend))
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
        _get_fast_pose()
        print(f"[warmup] YOLOv8 ready ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"[warn] Failed to warmup fast_pose: {e}")
    
    try:
        t1 = time.time()
        _get_ins()
        print(f"[warmup] RTMDet-Ins ready ({time.time()-t1:.1f}s)")
    except Exception as e:
        print(f"[warn] Failed to warmup ins: {e}")
    
    try:
        _get_ball()
        print(f"[warmup] Ball detector ready")
    except Exception as e:
        print(f"[warn] Failed to warmup ball: {e}")

    try:
        m = _get_rtmo()
        print(f"[warmup] RTMO ready (engine={getattr(m, 'ep', None)})")
    except Exception as e:
        print(f"[warn] Failed to warmup rtmo: {e}")
    
    print(f"[warmup] All models loaded in {time.time()-t0:.1f}s")

_warmup_thread = threading.Thread(target=_warmup_models, daemon=True)
_warmup_thread.start()


__all__ = ["analyze_frame_realtime", "render_frame_realtime",
           "get_cached", "ensure_3d", "frame_3d_payload", "engine_info",
           "RealtimeCache", "FrameResult", "Player3D",
           "rt_profile", "profile_allows_3d"]
