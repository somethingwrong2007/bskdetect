"""可选增强：几何根节点 + 自训因果 TCN 形状（2D→3D 混合提升器）。

定位
----
`pose_3d.Pose3DLifter` 的**旁路增强**，不是替代品：米制锚（脚点/骨盆/身高）仍由几何法
给，学习法只替换**骨盆以上**的关节相对形状（头/肩/肘/腕/鼻眼耳）。髋/膝/踝保持几何解，
所以"踝不劣化"这条验收天然成立。对外接口与 `Pose3DLifter.lift_result` 完全一致。

为什么这样分工
--------------
- 几何法在**横向+竖直**方向几乎无歧义（脚点落地 + 身高 EMA），误差厘米级；
  弱在**沿光轴的前后深度**（骨长射线解对 2D 噪声极敏感）；
- 学习法（TCN）从"射线斜率 + 可见性掩码 + 角尺度"回归相机系根相对米制 3D，
  前后深度明显更准（闭环 −32%），但整体平移/尺度不可靠；
- 组合：`X = p_w + (root_geo − root_learned)`，即"用几何法的根，套学习法的形状"。
  研究口径见 `docs/research/pose_2d_to_3d.md` §4.3 / §9。

因果性（严格）
--------------
每个 track 维护长度 `K_WIN=16` 的**过去帧**特征队列；推理时把队列（不足 16 帧则重复
队首＝冷启动）喂给因果 TCN，只取窗口最后一帧的输出。网络本身是因果膨胀卷积（只做
左侧 padding），窗口里也**只有当前帧与历史帧**——没有任何预取未来帧的路径。

开关与回滚
----------
默认**关闭**（`use_net=False`）。打开方式：
    lifter = HybridPose3DLifter(..., use_net=True)        # 构造参数
    BSK_POSE3D_NET=1  （环境变量，构造时读取）
    lifter.enable_net(True)                               # 运行期切换（A/B）
模型缺失 / ORT 加载失败 / 连续异常 → 自动回退纯几何（不抛异常、不影响实时链路），
`status()` 可查原因。

依赖
----
运行时只需要 onnxruntime（项目已有）+ numpy。torch **不在**运行时依赖里，只用于
导出 ONNX（见 `docs/research/pose_2d_to_3d_bench.py` 的导出逻辑；本模块不依赖 docs/）。
"""

from __future__ import annotations

import math
import os
from collections import deque
from pathlib import Path

import numpy as np

from bsk.court import FIBA, Court
from bsk.perceive.pose_3d import CourtCamera, LiftResult, Pose3DLifter, VIS_CONF

K_WIN = 16                     # 因果窗口帧数（与训练一致，勿改）
N_IN = 17 * 2 + 17 + 1         # 射线斜率 34 + 掩码 17 + 角尺度 1
N_OUT = 17 * 3
CONF_TH = 0.15                 # 与训练/渲染层一致的可见阈值

# 骨盆以上（要被学习形状替换的关节）：鼻、眼、耳、肩、肘、腕
UPPER_IDX = tuple(range(0, 11))

ENV_ENABLE = "BSK_POSE3D_NET"          # "1"/"true"/"on" → 打开
ENV_ONNX = "BSK_POSE3D_NET_ONNX"       # 覆盖模型路径（A/B 用）
DEFAULT_ONNX = "pose_lift_tcn_occ.onnx"  # 推荐变体 B（遮挡增强）

_MAX_FALLBACK_FAILS = 3        # 连续异常达到该次数 → 永久回退几何


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_onnx_path() -> Path:
    p = os.environ.get(ENV_ONNX, "").strip()
    return Path(p) if p else repo_root() / "data" / "models" / DEFAULT_ONNX


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


# ======================================================================
#  学习形状：ONNX 会话 + 逐 track 因果窗口
# ======================================================================

class LearnedShape:
    """ONNX TCN：COCO17 像素 + 相机 → 相机系、根相对、米制 (17,3)（因果窗口）。"""

    def __init__(self, onnx_path: str | Path | None = None, k_win: int = K_WIN,
                 providers: tuple[str, ...] = ("CPUExecutionProvider",),
                 num_threads: int = 1):
        import onnxruntime as ort          # 延迟导入：未开启时零开销/零依赖风险

        self.path = Path(onnx_path) if onnx_path is not None else default_onnx_path()
        if not self.path.exists():
            raise FileNotFoundError(f"ONNX 模型不存在：{self.path}")
        self.k_win = int(k_win)
        so = ort.SessionOptions()
        so.log_severity_level = 3                      # 只报错误，日志不刷屏
        so.intra_op_num_threads = int(num_threads)     # 1 线程：小模型 + 不与实时链路抢核
        self.sess = ort.InferenceSession(str(self.path), so, providers=list(providers))
        self.provider = self.sess.get_providers()[0]
        self._in_name = self.sess.get_inputs()[0].name
        self._hist: dict[int, deque] = {}
        self.calls = 0

    # ── 特征（必须与 docs/research/pose_2d_to_3d_bench.py: featurize 逐项一致）──
    @staticmethod
    def featurize(uv: np.ndarray, conf: np.ndarray, cam: CourtCamera) -> np.ndarray:
        """(17,2) 像素 + (17,) 置信 → (52,)；缺失关节射线置 0、掩码 0。

        与 bench 的唯一差别是"输入已保证有限"（NaN 在 predict 里先清洗），
        公式（含 `max(scale_px, 4.0)` 与 `median(rad[m>0])`）完全一致。
        """
        x = (uv[:, 0] - cam.cx) / cam.f
        y = (uv[:, 1] - cam.cy) / cam.f
        m = (conf >= CONF_TH).astype(np.float64)
        m[~np.isfinite(uv).all(1)] = 0.0
        pel = 0.5 * (uv[11] + uv[12])
        rad = np.linalg.norm(uv - pel, axis=1)
        scale_px = float(np.median(rad[m > 0])) if (m > 0).any() else 50.0
        scale_px = max(scale_px, 4.0)
        return np.concatenate([x * m, y * m, m, [math.log(scale_px / cam.f)]])

    # ── 窗口管理 ────────────────────────────────────────────
    def forget(self, tid: int) -> None:
        self._hist.pop(tid, None)

    def reset(self) -> None:
        self._hist.clear()

    def window_len(self, tid: int) -> int:
        return len(self._hist.get(tid, ()))

    # ── 推理 ────────────────────────────────────────────────
    def predict(self, uv: np.ndarray, conf: np.ndarray, cam: CourtCamera,
                tid: int) -> np.ndarray:
        """→ (17,3) 相机系、根相对、米。只使用当前帧与历史帧。"""
        uv = np.asarray(uv, dtype=np.float64)[:, :2]
        conf = np.asarray(conf, dtype=np.float64).reshape(-1)
        # 缺失点的像素值填 0：featurize 里它们会被掩码乘 0（NaN*0 仍是 NaN，必须先清洗）
        bad = ~np.isfinite(uv).all(1)
        if bad.any():
            uv = uv.copy()
            uv[bad] = 0.0

        feat = self.featurize(uv, conf, cam).astype(np.float32)
        hist = self._hist.get(tid)
        if hist is None:
            hist = deque(maxlen=self.k_win)
            self._hist[tid] = hist
        hist.append(feat)
        win = list(hist)
        if len(win) < self.k_win:                     # 冷启动：重复第一帧（= 几何法首帧无历史）
            win = [win[0]] * (self.k_win - len(win)) + win
        x = np.asarray(win, dtype=np.float32)[None]   # (1,K,52)
        out = self.sess.run(None, {self._in_name: x})[0]
        self.calls += 1
        return np.asarray(out, dtype=np.float64)[0, -1].reshape(17, 3)


# ======================================================================
#  混合提升器
# ======================================================================

class HybridPose3DLifter(Pose3DLifter):
    """几何法给米制锚，学习法给骨盆以上形状；默认关闭，可 A/B、可回滚。"""

    def __init__(self, court: Court = FIBA, image_size: tuple[int, int] = (1920, 1080),
                 *, use_net: bool | None = None,
                 onnx_path: str | Path | None = None,
                 gate_missing: bool = False,
                 net: LearnedShape | None = None,
                 **geo_kwargs):
        super().__init__(court=court, image_size=image_size, **geo_kwargs)
        self.use_net = _env_flag(ENV_ENABLE) if use_net is None else bool(use_net)
        self.gate_missing = bool(gate_missing)   # 缺失关节回退几何：默认关（研究实测反而更差）
        self._onnx_path = Path(onnx_path) if onnx_path is not None else default_onnx_path()
        self.net: LearnedShape | None = net
        self._net_error: str | None = None
        self._net_fails = 0
        self._net_frame: dict[int, int] = {}
        self._net_used = 0
        self._net_skipped = 0
        if self.use_net and self.net is None:
            self._ensure_net()                   # 尽早暴露"模型缺失"，但不抛异常

    # ── 开关 / 诊断 ─────────────────────────────────────────
    def _ensure_net(self) -> LearnedShape | None:
        if self.net is not None:
            return self.net
        if self._net_error is not None:
            return None
        try:
            self.net = LearnedShape(self._onnx_path)
            print(f"[pose3d] 学习形状已启用（B）：{self._onnx_path.name} "
                  f"EP={self.net.provider} K={self.net.k_win}")
        except Exception as exc:                 # 缺模型/ORT 异常/坏文件 → 回退几何
            self._net_error = f"{type(exc).__name__}: {exc}"
            print(f"[pose3d] 学习形状不可用，回退纯几何：{self._net_error}")
            return None
        return self.net

    def enable_net(self, enabled: bool) -> bool:
        """运行期开关（A/B 与回滚）。返回"现在是否真的会用学习形状"。"""
        self.use_net = bool(enabled)
        if not self.use_net:
            return False
        return self._ensure_net() is not None

    def reload_net(self) -> bool:
        """清掉失败记录并重试加载（模型是后来才放上去的情形，不必重启进程）。"""
        self.net = None
        self._net_error = None
        self._net_fails = 0
        self._net_frame.clear()
        return self._ensure_net() is not None

    def status(self) -> dict:
        return {"enabled": bool(self.use_net), "ready": self.net is not None,
                "onnx": str(self._onnx_path), "provider": getattr(self.net, "provider", None),
                "error": self._net_error, "used": self._net_used,
                "skipped": self._net_skipped, "gate_missing": self.gate_missing}

    def reset(self) -> None:
        super().reset()
        if self.net is not None:
            self.net.reset()
        self._net_frame.clear()

    # ── 主路径 ──────────────────────────────────────────────
    def lift_result(self, kpts_px, bbox_px, H_court_to_pixel, track_id: int = -1,
                    frame: int = 0) -> LiftResult | None:
        if not self.use_net:
            # 关闭时是纯旁路：与 Pose3DLifter 位级相同（含其既有的异常/None 语义）
            return super().lift_result(kpts_px, bbox_px, H_court_to_pixel, track_id, frame)
        try:
            geo = super().lift_result(kpts_px, bbox_px, H_court_to_pixel, track_id, frame)
        except Exception as exc:      # 非法 H（NaN/Inf）等：与"不可解"同等待遇，不抛给实时链路
            self._net_skipped += 1
            self._net_error = f"geo: {type(exc).__name__}: {exc}"
            return None
        if geo is None:
            return None
        net = self._ensure_net()
        if net is None:
            return geo
        try:
            cam = self.camera(H_court_to_pixel) if H_court_to_pixel is not None else None
            if cam is None or not np.isfinite(cam.f) or cam.f <= 1e-6:
                self._net_skipped += 1
                return geo
            k = np.asarray(kpts_px, dtype=np.float64)
            if k.ndim != 2 or k.shape[0] != 17 or k.shape[1] < 2:
                self._net_skipped += 1
                return geo
            uv = k[:, :2]
            conf = k[:, 2].copy() if k.shape[1] > 2 else np.ones(17)

            self._sync_hist(track_id, frame)
            rel = net.predict(uv, conf, cam, track_id)
            if not np.isfinite(rel).all():
                self._net_skipped += 1
                return geo

            p_w = rel @ cam.R                                   # 相机系 → 世界轴（方向）
            shift = geo.root - 0.5 * (p_w[11] + p_w[12])        # 只借几何的根节点（米制锚）
            upper = p_w[:11] + shift
            if self.gate_missing:                               # 默认关；A/B 用
                miss = np.asarray(conf[:11] < VIS_CONF)
                upper[miss] = geo.kpts3d[:11][miss]

            xyz = geo.kpts3d.copy()                             # 髋/膝/踝保持几何解
            xyz[:11] = upper
            if not np.isfinite(xyz).all():
                self._net_skipped += 1
                return geo

            err = np.linalg.norm(cam.project(xyz)[0] - uv, axis=1)
            ok = np.isfinite(err)
            rp = float(np.median(err[ok])) if ok.any() else float("nan")
            geo.kpts3d = xyz
            geo.reproj_err_px = rp
            self._net_fails = 0
            self._net_used += 1
            return geo
        except Exception as exc:      # 绝不让学习分支拖垮实时：连续失败即永久回退
            self._net_fails += 1
            self._net_error = f"{type(exc).__name__}: {exc}"
            if self._net_fails >= _MAX_FALLBACK_FAILS:
                self.net = None
                print(f"[pose3d] 学习形状连续 {self._net_fails} 次失败，永久回退纯几何："
                      f"{self._net_error}")
            return geo

    # ── 窗口生命周期（与几何法的 track 状态同源）────────────
    def _sync_hist(self, tid: int, frame: int) -> None:
        last = self._net_frame.get(tid)
        if last is None or frame < last or frame - last > self.max_track_age:
            if self.net is not None:
                self.net.forget(tid)          # 新 track / 断档太久 → 冷启动
        self._net_frame[tid] = int(frame)
        if len(self._net_frame) > 128:
            dead = [k for k, f in self._net_frame.items() if frame - f > self.max_track_age]
            for k in dead:
                self._net_frame.pop(k, None)
                if self.net is not None:
                    self.net.forget(k)


__all__ = ["LearnedShape", "HybridPose3DLifter", "K_WIN", "N_IN", "N_OUT",
           "ENV_ENABLE", "ENV_ONNX", "default_onnx_path"]