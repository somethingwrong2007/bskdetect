"""FIBA 球场 16 点在线检测、镜头状态迁移与单应配准。

坐标契约与项目现有 Calibration 完全一致：
* 输入/检测点：原图像素 ``(u, v)``；
* 世界点：FIBA 米制球场，原点在中心，x 沿长边，y 沿短边；
* ``H``：像素 → 世界；``H_inv``：世界 → 像素。

16 个主标签采用全场对称的 4 角 + 2 中线端点 + 4 罚球线端点 + 4 三分底角拐点 +
2 三分弧顶。限制区角点不重复占标签，作为 8 个辅助几何点参与边线配准。这样标签总数
固定为 16，同时保留需求中的限制区几何。若采用其他训练标签顺序，必须在 ONNX sidecar
中逐项声明 ``keypoint_ids``，顺序不一致直接拒绝加载。

检测优先使用可选的 16 点 ONNX 模型；权重缺失时使用 CLAHE + Shi-Tomasi 特征点 +
鲁棒直线拟合，并由拟合线求交点，再在已有可信机位或点位足够时做有界 RANSAC 匹配。
没有语义模型且没有历史机位时，局部边线/特写无法唯一确定全场坐标，此时明确返回
配准失效而不是猜一个 H。

所有状态只使用当前帧和过去帧；不预取未来帧。硬切会清空历史并强制重新推理/RANSAC，
摇移使用光流位移辅助点位跟踪并平滑单应。低于置信门限或支撑不足时 H/H_inv 均为 None。
"""

from __future__ import annotations

import itertools
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from bsk.calibrate import Calibration
from bsk.court import FIBA

N_KEYPOINTS = 16
DEFAULT_WEIGHTS = Path(os.environ.get("BSK_COURT_KP_WEIGHTS", "data/models/court_keypoints16.onnx"))


def _build_specs() -> tuple[tuple[str, tuple[float, float]], ...]:
    hl, hw = FIBA.half_length, FIBA.half_width
    ft_x = hl - FIBA.free_throw_dist
    corner_y = hw - FIBA.three_pt_corner_dist
    rim_x, _ = FIBA.rim_pos("A")
    apex_x = rim_x - FIBA.three_pt_radius
    return (
        ("corner_A_near", (-hl, -hw)),
        ("corner_B_near", (-hl, hw)),
        ("corner_A_far", (hl, hw)),
        ("corner_B_far", (hl, -hw)),
        ("mid_near", (0.0, -hw)),
        ("mid_far", (0.0, hw)),
        ("ft_A_near", (ft_x, -FIBA.free_throw_width / 2)),
        ("ft_A_far", (ft_x, FIBA.free_throw_width / 2)),
        ("ft_B_near", (-ft_x, -FIBA.free_throw_width / 2)),
        ("ft_B_far", (-ft_x, FIBA.free_throw_width / 2)),
        ("three_corner_A_near", (hl, -corner_y)),
        ("three_corner_A_far", (hl, corner_y)),
        ("three_corner_B_near", (-hl, -corner_y)),
        ("three_corner_B_far", (-hl, corner_y)),
        ("three_apex_A", (apex_x, 0.0)),
        ("three_apex_B", (-apex_x, 0.0)),
    )


KEYPOINT_SPECS = _build_specs()
KEYPOINT_IDS = tuple(name for name, _ in KEYPOINT_SPECS)
WORLD_POINTS = np.asarray([pos for _, pos in KEYPOINT_SPECS], np.float64)
KEYPOINT_INDEX = {name: i for i, name in enumerate(KEYPOINT_IDS)}

# 限制区矩形角点：两侧各 4 个，仅用于边线辅助，不进入 16 点模型标签。
_paint_hw = FIBA.paint_width / 2
_ft_x = FIBA.half_length - FIBA.free_throw_dist
AUX_RESTRICTED_POINTS = np.asarray([
    (-FIBA.half_length, -_paint_hw), (-FIBA.half_length, _paint_hw),
    (-_ft_x, -_paint_hw), (-_ft_x, _paint_hw),
    (FIBA.half_length, -_paint_hw), (FIBA.half_length, _paint_hw),
    (_ft_x, -_paint_hw), (_ft_x, _paint_hw),
], np.float64)
REFERENCE_POINTS = np.vstack([WORLD_POINTS, AUX_RESTRICTED_POINTS])


def _court_segments() -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    """主线段 + 限制区 + 三分线，供局部边线支撑和位移修正。"""
    hl, hw = FIBA.half_length, FIBA.half_width
    fx = hl - FIBA.free_throw_dist
    rim_a, _ = FIBA.rim_pos("A")
    rim_b, _ = FIBA.rim_pos("B")
    cy = hw - FIBA.three_pt_corner_dist
    dx = math.sqrt(max(FIBA.three_pt_radius ** 2 - cy ** 2, 0.0))
    segs = [
        ((-hl, -hw), (hl, -hw)), ((-hl, hw), (hl, hw)),
        ((-hl, -hw), (-hl, hw)), ((hl, -hw), (hl, hw)),
        ((0.0, -hw), (0.0, hw)),
        ((-fx, -FIBA.free_throw_width / 2), (-fx, FIBA.free_throw_width / 2)),
        ((fx, -FIBA.free_throw_width / 2), (fx, FIBA.free_throw_width / 2)),
        ((-hl, -_paint_hw), (-fx, -_paint_hw)),
        ((-hl, _paint_hw), (-fx, _paint_hw)),
        ((hl, -_paint_hw), (fx, -_paint_hw)),
        ((hl, _paint_hw), (fx, _paint_hw)),
        ((-hl, -cy), (rim_b + dx, -cy)), ((-hl, cy), (rim_b + dx, cy)),
        ((hl, -cy), (rim_a - dx, -cy)), ((hl, cy), (rim_a - dx, cy)),
    ]
    # 两段三分弧按 1m 左右采样；只使用当前帧的像素线段，不读取任何后续帧。
    for rim_x, sign in ((rim_a, -1), (rim_b, 1)):
        limit = math.asin(cy / FIBA.three_pt_radius)
        angles = np.linspace(-limit, limit, 25)
        pts = [(rim_x + sign * FIBA.three_pt_radius * math.cos(a),
                FIBA.three_pt_radius * math.sin(a)) for a in angles]
        segs.extend(zip(pts[:-1], pts[1:]))
    return tuple(segs)


COURT_SEGMENTS = _court_segments()


@dataclass(frozen=True)
class KeypointConfig:
    weights: str | Path = DEFAULT_WEIGHTS
    device: str = "auto"
    every_n: int = 3
    retry_every: int = 3
    min_conf: float = 0.35
    min_publish_conf: float = 0.55
    min_matches: int = 4
    ransac_px: float = 14.0
    max_reproj_px: float = 22.0
    stale_s: float = 0.8
    cut_mad: float = 0.30
    cut_hist: float = 0.55
    max_detections: int = 48
    classical_hypotheses: int = 48
    line_gate_frac: float = 0.045
    min_line_families: int = 2
    input_hw: tuple[int, int] = (640, 640)


@dataclass
class DetectedKeypoint:
    name: str
    pixel: tuple[float, float]
    confidence: float
    world: tuple[float, float] | None = None
    inlier: bool = False


@dataclass
class LineObservation:
    points: list[DetectedKeypoint] = field(default_factory=list)
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)
    intersection_keypoints: list[DetectedKeypoint] = field(default_factory=list)


@dataclass
class CourtKeypointResult:
    frame: int
    timestamp: float
    inference_frame: int
    mode: str
    transition: str
    valid: bool
    reason: str
    source: str
    confidence: float
    inliers: int
    reproj_error_px: float
    keypoints: list[DetectedKeypoint]
    H: np.ndarray | None = None
    H_inv: np.ndarray | None = None
    line_families: int = 0
    stats: dict = field(default_factory=dict)

    def to_calibration(self) -> Calibration | None:
        if not self.valid or self.H is None or self.H_inv is None:
            return None
        cal = Calibration(
            H=self.H.tolist(), H_inv=self.H_inv.tolist(),
            reproj_error_px=float(self.reproj_error_px),
            keypoints_used=int(self.inliers), court_model="FIBA",
            timestamp=float(self.timestamp),
        )
        cal.source = f"court_keypoints:{self.source}"
        return cal

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "inference_frame": self.inference_frame,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "transition": self.transition,
            "valid": self.valid,
            "reason": self.reason,
            "source": self.source,
            "confidence": round(float(self.confidence), 4),
            "inliers": int(self.inliers),
            "reproj_error_px": (round(float(self.reproj_error_px), 3)
                               if np.isfinite(self.reproj_error_px) else None),
            "line_families": int(self.line_families),
            # H 是像素→世界；失效时两个矩阵都为 null，避免旧 H 被误用。
            "H": self.H.tolist() if self.valid and self.H is not None else None,
            "H_inv": self.H_inv.tolist() if self.valid and self.H_inv is not None else None,
            "keypoints": [
                {
                    "id": kp.name,
                    "pixel": ([round(float(kp.pixel[0]), 3), round(float(kp.pixel[1]), 3)]
                              if np.all(np.isfinite(kp.pixel)) else None),
                    "world": list(kp.world) if kp.world is not None else None,
                    "confidence": round(float(kp.confidence), 4),
                    "visible": bool(kp.pixel is not None and kp.confidence > 0),
                    "inlier": bool(kp.inlier),
                }
                for kp in self.keypoints
            ],
            "stats": self.stats,
        }


def empty_result(
    frame: int, now: float, inference_frame: int, mode: str, transition: str,
    reason: str, source: str, stats: dict,
) -> CourtKeypointResult:
    return CourtKeypointResult(
        frame=frame, timestamp=now, inference_frame=inference_frame,
        mode=mode, transition=transition, valid=False, reason=reason,
        source=source, confidence=0.0, inliers=0, reproj_error_px=float("inf"),
        keypoints=[DetectedKeypoint(name, (float("nan"), float("nan")), 0.0, world)
                   for name, world in KEYPOINT_SPECS],
        stats=dict(stats),
    )


def _project(H_world_to_pixel: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]).T
    out = H_world_to_pixel @ hom
    good = np.abs(out[2]) > 1e-9
    out[:2] = np.divide(out[:2], out[2], out=np.zeros_like(out[:2]), where=good)
    out[:2, ~good] = 1e9
    return out[:2].T


def _nearest_on_segments(
    points: np.ndarray,
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回每个点的最近线段投影、距离和线段索引。"""
    if not segments:
        return (np.full_like(points, np.nan), np.full(len(points), np.inf),
                np.full(len(points), -1, np.int32))
    best = np.full_like(points, np.nan)
    best_d = np.full(len(points), np.inf)
    best_i = np.full(len(points), -1, np.int32)
    for i, (a, b) in enumerate(segments):
        a = np.asarray(a, np.float64)
        b = np.asarray(b, np.float64)
        ab = b - a
        denom = float(ab @ ab)
        if denom <= 1e-9:
            continue
        t = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
        proj = a + t[:, None] * ab
        dist = np.linalg.norm(points - proj, axis=1)
        take = dist < best_d
        best[take] = proj[take]
        best_d[take] = dist[take]
        best_i[take] = i
    return best, best_d, best_i


class CourtKeypointModel:
    """可选 16 点 ONNX 模型；与球衣模型共用进程级会话注册表。"""

    def __init__(self, cfg: KeypointConfig) -> None:
        self.cfg = cfg
        self.path = Path(cfg.weights)
        self.session = None
        self.ep = None
        self.error: str | None = None
        self.meta: dict = {}
        self.input_name = ""
        self._lock = threading.Lock()
        self._mean = np.zeros((3, 1, 1), np.float32)
        self._std = np.ones((3, 1, 1), np.float32)
        self._load()

    @property
    def ready(self) -> bool:
        return self.session is not None

    def _load(self) -> None:
        if not self.path.exists():
            self.error = f"关键点权重不存在: {self.path}"
            return
        try:
            side = self.path.with_suffix(".json")
            if side.exists():
                self.meta = json.loads(side.read_text(encoding="utf-8"))
            ids = self.meta.get("keypoint_ids")
            if ids is not None and tuple(ids) != KEYPOINT_IDS:
                raise ValueError("sidecar keypoint_ids 与本模块 16 点标签顺序不一致")
            ih, iw = self.meta.get("input_hw", self.cfg.input_hw)
            self.input_hw = (int(ih), int(iw))
            if min(self.input_hw) <= 0:
                raise ValueError("input_hw 必须为正")
            self.preprocess = self.meta.get("preprocess", "rgb_0_255")
            if self.preprocess not in ("rgb_0_255", "gray_clahe"):
                raise ValueError(f"不支持的预处理: {self.preprocess}")
            self.coordinate_space = self.meta.get("coordinate_space", "input_pixels")
            if self.coordinate_space not in ("input_pixels", "normalized"):
                raise ValueError(f"不支持的坐标空间: {self.coordinate_space}")
            mean = np.asarray(self.meta.get("mean", [0, 0, 0]), np.float32).reshape(3, 1, 1)
            std = np.asarray(self.meta.get("std", [1, 1, 1]), np.float32).reshape(3, 1, 1)
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
                raise ValueError("mean/std 必须有限且 std > 0")
            self._mean, self._std = mean, std
            from bsk.perceive.onnx_sessions import get_session

            self.session, self.ep = get_session(
                self.path, device=self.cfg.device, use_trt=False)
            self.input_name = self.session.get_inputs()[0].name
        except Exception as exc:
            self.session = None
            self.error = f"关键点模型加载失败: {type(exc).__name__}: {exc}"

    def predict(self, rgb: np.ndarray) -> list[DetectedKeypoint]:
        if not self.ready:
            return []
        h, w = rgb.shape[:2]
        ih, iw = self.input_hw
        if self.preprocess == "gray_clahe":
            # 直接复用球衣号码的线程本地 CLAHE，避免 OpenCV CLAHE 跨线程竞争。
            from bsk.perceive.jersey import gray_input

            gray = gray_input(rgb, (iw, ih))
            img = np.repeat(gray[:, :, None], 3, axis=2)
        else:
            img = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)
        blob = ((img.astype(np.float32).transpose(2, 0, 1) - self._mean) / self._std)[None]
        blob = np.ascontiguousarray(blob, np.float32)
        try:
            with self._lock:
                out = self.session.run(None, {self.input_name: blob})[0]
        except Exception as exc:
            self.error = f"关键点推理失败: {type(exc).__name__}: {exc}"
            return []
        arr = np.asarray(out, np.float32).squeeze()
        if arr.size == N_KEYPOINTS * 3:
            arr = arr.reshape(N_KEYPOINTS, 3)
        elif arr.shape != (N_KEYPOINTS, 3):
            self.error = f"关键点输出形状不支持: {arr.shape}"
            return []
        if not np.all(np.isfinite(arr)):
            return []
        if self.coordinate_space == "input_pixels":
            # 模型输入是直接 resize（无 letterbox），输入像素需折算回原图。
            sx, sy = w / float(iw), h / float(ih)
        else:
            sx, sy = float(w), float(h)
        result = []
        for (name, _world), (x, y, conf) in zip(KEYPOINT_SPECS, arr):
            u, v = float(x * sx), float(y * sy)
            if not (0.0 <= u < w and 0.0 <= v < h):
                continue
            if not (0.0 <= conf <= 1.0):
                continue
            if conf >= self.cfg.min_conf:
                result.append(DetectedKeypoint(name, (u, v), float(conf)))
        return result


def detect_classical(rgb: np.ndarray, cfg: KeypointConfig) -> LineObservation:
    """Shi-Tomasi 特征点提线；输出拟合线交点和原图坐标下的可见线段。

        这里不再使用 Canny/Hough 直接抽取连续边线。特征点检测器先在缩放图上
        做白线证据筛选和迭代 RANSAC，再将无限直线裁剪到图像范围。两条拟合线
        的交点继续作为无语义配准的候选关键点，保持旧调用方接口不变。
    """
    h, w = rgb.shape[:2]
    scale = min(1.0, 960.0 / max(h, w))
    pw, ph = max(160, int(round(w * scale))), max(90, int(round(h * scale)))
    small = cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_AREA)
    # 特征点模块统一使用 OpenCV 的 BGR 色彩契约；本项目其余感知链路使用 RGB。
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)

    from bsk.perceive.court_line_features import LineFeatureConfig, detect_line_features

    feature_cfg = LineFeatureConfig(
        max_corners=1200,
        quality_level=0.025,
        distance_threshold=float(np.clip(cfg.ransac_px * scale / 3.0, 3.0, 6.0)),
        min_distance=2,
        ransac_iterations=max(160, min(320, cfg.classical_hypotheses * 4)),
        min_line_points=5,
        min_cluster_points=5,
        min_line_length=80.0 * scale,  # 原图 80px 折算到检测分辨率
        min_line_support=0.30,
        white_neighborhood_size=3,
        min_point_evidence=0.25,
        min_span_frac=0.025,
        min_span_bins=2,
        max_lines=24,
        use_fixed_roi=True,
        line_angle_prior_deg=(0.0, 90.0),
        angle_prior_tolerance_deg=35.0,
        intersection_max_line_distance=12.0 * scale,
        intersection_cluster_distance=25.0 * scale,
        min_intersection_angle_deg=10.0,
    )
    detected = detect_line_features(bgr, feature_cfg)
    segments = []
    for line in detected.lines:
        # 求交使用无限直线；标定支撑仍限制在有角点实证的可见跨度，避免弧线
        # 或远端延长段降低单应验证质量。
        origin = np.asarray(line.p0, np.float64)
        direction = np.asarray(line.direction, np.float64)
        if line.points:
            along = [float((np.asarray([p.x, p.y]) - origin) @ direction) for p in line.points]
            margin = feature_cfg.distance_threshold * 3.0
            a = origin + (min(along) - margin) * direction
            b = origin + (max(along) + margin) * direction
        else:
            a, b = origin, np.asarray(line.p1, np.float64)
        segments.append(((a[0] / scale, a[1] / scale), (b[0] / scale, b[1] / scale)))
    # 与 DL 输出使用同一 DetectedKeypoint 结构；经典交点没有语义标签，
    # 由 _solve_unlabeled 做有界 FIBA 对应搜索，不能在此伪造 world 坐标。
    points = [
        DetectedKeypoint(
            f"intersection_{index}",
            (point.pixel[0] / scale, point.pixel[1] / scale),
            point.confidence,
        )
        for index, point in enumerate(detected.intersection_keypoints[:cfg.max_detections])
    ]
    return LineObservation(
        points=points, segments=segments, intersection_keypoints=list(points),
    )


class CourtKeypointTracker:
    """每个视频/分辨率一份状态；因果在线，硬切清空，摇移平滑。"""

    def __init__(self, cfg: KeypointConfig | None = None) -> None:
        self.cfg = cfg or KeypointConfig()
        self.model = CourtKeypointModel(self.cfg)
        self._lock = threading.RLock()
        self._prev_small: np.ndarray | None = None
        self._prev_orb: np.ndarray | None = None
        self._prev_hist: np.ndarray | None = None
        self._prev_luma: float | None = None
        self._prev_texture: float | None = None
        self._motion_ticks = 0
        self._orb = cv2.ORB_create(nfeatures=160, fastThreshold=12)
        self._phase_window = cv2.createHanningWindow((160, 90), cv2.CV_32F)
        self._H: np.ndarray | None = None  # pixel -> world
        self._H_inv: np.ndarray | None = None  # world -> pixel
        self._last_detections: list[DetectedKeypoint] = []
        self._last_result: CourtKeypointResult | None = None
        self._last_source = "none"
        self._last_infer = -10**9
        self._last_attempt = -10**9
        self._last_valid = 0.0
        self._shape: tuple[int, int] | None = None
        self.stats = {"inferences": 0, "hard_cuts": 0, "failures": 0, "line_aids": 0}

    def _reset_h(self, reason: str) -> None:
        self._H = self._H_inv = None
        self._last_detections = []
        self._last_source = "none"
        self._last_valid = 0.0
        if self._last_result is not None:
            self._last_result.reason = reason

    def invalidate(self, reason: str, frame: int | None = None) -> None:
        """Thread-safe invalidation without erasing a newer concurrent result."""
        with self._lock:
            if (frame is not None and self._last_result is not None
                    and self._last_result.frame >= frame):
                return
            self._reset_h(reason)

    def _motion(self, rgb: np.ndarray) -> tuple[str, np.ndarray, float]:
        # 一次下采样同时服务 MAD/直方图/相位相关，避免同一帧重复做全分辨率灰度化。
        gray_small = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (160, 90),
                                interpolation=cv2.INTER_AREA)
        small = gray_small.astype(np.float32)
        small = (small - small.mean()) / max(float(small.std()), 1.0)
        hist = cv2.calcHist([gray_small], [0], None, [32], [0, 256]).reshape(-1)
        hist = hist.astype(np.float32) / max(float(hist.sum()), 1.0)
        luma = float(gray_small.mean())
        texture = float(gray_small.std())
        flow = np.zeros(2, np.float64)
        scale_ratio = 1.0
        if self._prev_small is None:
            self._prev_small, self._prev_orb, self._prev_hist = small, gray_small, hist
            self._prev_luma, self._prev_texture = luma, texture
            return "initial", flow, scale_ratio
        mad = float(np.mean(np.abs(small - self._prev_small)))
        corr = float(cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL))
        luma_cut = (self._prev_luma is not None and self._prev_texture is not None
                    and max(texture, self._prev_texture) < 3.0
                    and abs(luma - self._prev_luma) > 40.0)
        self._motion_ticks += 1
        # 绝大多数硬切可由低分辨率直方图/归一化 MAD 直接判定，先走便宜路径。
        immediate_cut = luma_cut or corr < self.cfg.cut_hist or mad > 0.50
        if immediate_cut:
            mode = "hard_cut"
        elif mad > 0.035:
            # 摇移用相位相关估计全局位移；ORB 只周期性检查缩放/相干性，避免每帧
            # 特征匹配吃掉实时预算。周期性检查仍只依赖已经到达的当前帧。
            shift, response = cv2.phaseCorrelate(
                self._prev_orb.astype(np.float32) * self._phase_window,
                gray_small.astype(np.float32) * self._phase_window)
            flow = np.asarray(shift, np.float64)
            phase_coherent = response > 0.08
            matches = []
            scale_checked = False
            if self._motion_ticks % 4 == 0 or mad > self.cfg.cut_mad:
                scale_checked = True
                k0, d0 = self._orb.detectAndCompute(self._prev_orb, None)
                k1, d1 = self._orb.detectAndCompute(gray_small, None)
                if d0 is not None and d1 is not None and len(k0) >= 8 and len(k1) >= 8:
                    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d0, d1, k=2)
                    matches = [pair[0] for pair in pairs
                               if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
                coherent = 0.0
                if len(matches) >= 8:
                    p0 = np.float32([k0[m.queryIdx].pt for m in matches])
                    p1 = np.float32([k1[m.trainIdx].pt for m in matches])
                    vec = p1 - p0
                    orb_flow = np.median(vec, axis=0)
                    resid = np.median(np.linalg.norm(vec - orb_flow, axis=1))
                    coherent = 1.0 if resid < max(3.0, 0.35 * float(np.linalg.norm(orb_flow))) else 0.0
                    if coherent:
                        flow = orb_flow
                    scales = np.asarray([k1[m.trainIdx].size / max(k0[m.queryIdx].size, 1e-6)
                                         for m in matches])
                    scale_ratio = float(np.median(scales))
                if mad > self.cfg.cut_mad and (len(matches) < 8 or coherent < 0.5):
                    mode = "hard_cut"
                elif scale_checked and (scale_ratio > 1.16 or scale_ratio < 0.86):
                    mode = "zoom"
                elif coherent or phase_coherent:
                    mode = "pan" if float(np.linalg.norm(flow)) > 0.08 else "static"
                elif mad > 0.22:
                    mode = "hard_cut"
                else:
                    mode = "static"
            else:
                mode = "pan" if phase_coherent else "static"
        else:
            mode = "static"
        self._prev_small, self._prev_orb, self._prev_hist = small, gray_small, hist
        self._prev_luma, self._prev_texture = luma, texture
        # 小图 160×90 的位移折算回原图。
        return mode, flow * np.asarray([rgb.shape[1] / 160.0, rgb.shape[0] / 90.0]), scale_ratio

    @staticmethod
    def _plausible(H_inv: np.ndarray, shape: tuple[int, int]) -> bool:
        try:
            if not np.all(np.isfinite(H_inv)):
                return False
            corners = np.asarray([
                [-FIBA.half_length, -FIBA.half_width, 1],
                [FIBA.half_length, -FIBA.half_width, 1],
                [FIBA.half_length, FIBA.half_width, 1],
                [-FIBA.half_length, FIBA.half_width, 1],
            ], np.float64)
            if np.any(np.abs(corners @ H_inv[2]) < 1e-9):
                return False
            H = np.linalg.inv(H_inv)
            cal = Calibration(H=H.tolist(), H_inv=H_inv.tolist(), reproj_error_px=0.0,
                              keypoints_used=16, court_model="FIBA", timestamp=time.time())
            return bool(cal.is_plausible(shape[1], shape[0]))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return False

    @staticmethod
    def _stable_labeled_support(world: np.ndarray, pixel: np.ndarray,
                                H_inv: np.ndarray, cfg: KeypointConfig) -> bool:
        """Reject collinear/underdetermined fits that look perfect only in-sample."""
        if len(world) < 4 or not np.isfinite(world).all() or not np.isfinite(pixel).all():
            return False
        max_area = 0.0
        for first, second, third in itertools.combinations(range(len(world)), 3):
            a, b, c = world[[first, second, third]]
            area = abs(float((b[0] - a[0]) * (c[1] - a[1]) -
                             (b[1] - a[1]) * (c[0] - a[0]))) / 2.0
            max_area = max(max_area, area)
        if max_area < 40.0:
            return False
        hull = cv2.convexHull(pixel.astype(np.float32).reshape(-1, 1, 2))
        if float(cv2.contourArea(hull)) < 400.0:
            return False
        if len(world) >= 5:
            reference = _project(H_inv, np.asarray([
                [-FIBA.half_length, -FIBA.half_width], [FIBA.half_length, -FIBA.half_width],
                [FIBA.half_length, FIBA.half_width], [-FIBA.half_length, FIBA.half_width],
            ], np.float64))
            held_out_errors = []
            corner_drift = []
            for held_out in range(len(world)):
                train = np.arange(len(world)) != held_out
                candidate, _ = cv2.findHomography(world[train], pixel[train], 0)
                if candidate is None or not np.isfinite(candidate).all():
                    return False
                predicted = _project(candidate, world[held_out:held_out + 1])[0]
                held_out_errors.append(float(np.linalg.norm(predicted - pixel[held_out])))
                corner_drift.extend(np.linalg.norm(_project(candidate, np.asarray([
                    [-FIBA.half_length, -FIBA.half_width], [FIBA.half_length, -FIBA.half_width],
                    [FIBA.half_length, FIBA.half_width], [-FIBA.half_length, FIBA.half_width],
                ], np.float64)) - reference, axis=1).tolist())
            if (float(np.median(held_out_errors)) > max(18.0, cfg.ransac_px * 1.5) or
                    float(np.median(corner_drift)) > 35.0):
                return False
        return True

    def _solve_labeled(
        self, detections: list[DetectedKeypoint], shape: tuple[int, int], min_matches: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None, list[DetectedKeypoint], float]:
        named = []
        for d in detections:
            if d.name not in KEYPOINT_INDEX or d.world is None:
                continue
            named.append(d)
        if len(named) < max(4, min_matches):
            return None, None, named, float("inf")
        pixel = np.asarray([d.pixel for d in named], np.float64)
        world = np.asarray([d.world for d in named], np.float64)
        if np.linalg.matrix_rank(world - world.mean(axis=0)) < 2:
            return None, None, named, float("inf")
        try:
            H_inv, mask = cv2.findHomography(world, pixel, cv2.RANSAC, self.cfg.ransac_px, maxIters=500)
        except cv2.error:
            return None, None, named, float("inf")
        if H_inv is None or not np.all(np.isfinite(H_inv)):
            return None, None, named, float("inf")
        try:
            H = np.linalg.inv(H_inv)
        except np.linalg.LinAlgError:
            return None, None, named, float("inf")
        if not self._plausible(H_inv, shape):
            return None, None, named, float("inf")
        if mask is None:
            return None, None, named, float("inf")
        keep = mask.reshape(-1).astype(bool)
        if int(keep.sum()) < max(4, min_matches):
            return None, None, named, float("inf")
        # 内点重投影必须在像素空间，且不能只看 RANSAC 的 4 个最小样本。
        try:
            H_inv, _ = cv2.findHomography(world[keep], pixel[keep], 0)
            H = np.linalg.inv(H_inv)
        except (cv2.error, np.linalg.LinAlgError):
            return None, None, named, float("inf")
        proj = _project(H_inv, world)
        errors = np.linalg.norm(proj - pixel, axis=1)
        keep = errors <= self.cfg.ransac_px
        if int(keep.sum()) < max(4, min_matches):
            return None, None, named, float("inf")
        err = float(np.median(errors[keep]))
        if not np.isfinite(err) or err > self.cfg.max_reproj_px:
            return None, None, named, float("inf")
        if not self._stable_labeled_support(world[keep], pixel[keep], H_inv, self.cfg):
            return None, None, named, float("inf")
        out = []
        for d, is_in in zip(named, keep):
            out.append(DetectedKeypoint(d.name, d.pixel, d.confidence, d.world, bool(is_in)))
        return H, H_inv, out, err

    def _match_prior(
        self, detections: list[DetectedKeypoint], shape: tuple[int, int],
    ) -> list[DetectedKeypoint]:
        if self._H_inv is None or not detections:
            return []
        pred = _project(self._H_inv, WORLD_POINTS)
        gate = max(28.0, self.cfg.ransac_px * 2.5, math.hypot(*shape) * 0.025)
        pairs = []
        for d in detections:
            dist = np.linalg.norm(pred - np.asarray(d.pixel), axis=1)
            for idx in np.argsort(dist)[:2]:
                if dist[idx] <= gate:
                    pairs.append((float(dist[idx]), d, int(idx)))
        pairs.sort(key=lambda x: x[0])
        used_px: set[int] = set()
        used_world: set[int] = set()
        out = []
        for _dist, d, idx in pairs:
            if id(d) in used_px or idx in used_world:
                continue
            used_px.add(id(d)); used_world.add(idx)
            out.append(DetectedKeypoint(KEYPOINT_IDS[idx], d.pixel, d.confidence,
                                        KEYPOINT_SPECS[idx][1]))
        return out

    def _solve_unlabeled(
        self, detections: list[DetectedKeypoint], shape: tuple[int, int], frame: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None, list[DetectedKeypoint], float]:
        """无语义模型且无历史机位时的有界随机对应搜索；失败优先于误标定。"""
        if len(detections) < self.cfg.min_matches or len(WORLD_POINTS) < 4:
            return None, None, [], float("inf")
        # 可见范围端点只用于有历史机位的 prior 匹配；无语义、无历史时让它们
        # 参与投票会拉偏单应，因此这里只使用置信度较高的线线交点。
        detections = [d for d in detections if d.confidence >= 0.28]
        if len(detections) < self.cfg.min_matches:
            return None, None, [], float("inf")
        # 只使用当前帧和帧号派生种子；不读取未来帧，跨视频状态也互不影响。
        rng = np.random.default_rng((int(frame) + 7919) & 0xFFFFFFFF)
        pix = np.asarray([d.pixel for d in detections], np.float64)
        best_score = 0
        best_named: list[DetectedKeypoint] = []

        def consider(H_inv: np.ndarray) -> None:
            nonlocal best_score, best_named
            if H_inv is None or not np.all(np.isfinite(H_inv)) or not self._plausible(H_inv, shape):
                return
            pred = _project(H_inv, WORLD_POINTS)
            dist = np.linalg.norm(pred[:, None, :] - pix[None, :, :], axis=2)
            nearest = dist.argmin(axis=1)
            mind = dist[np.arange(len(pred)), nearest]
            matched_world = np.flatnonzero(mind <= self.cfg.ransac_px * 1.5)
            if len(matched_world) < self.cfg.min_matches:
                return
            if len(np.unique(nearest[matched_world])) != len(matched_world):
                return
            if len(matched_world) > best_score:
                best_score = len(matched_world)
                best_named = [DetectedKeypoint(
                    KEYPOINT_IDS[i], detections[nearest[i]].pixel,
                    detections[nearest[i]].confidence, KEYPOINT_SPECS[i][1],
                ) for i in matched_world]

        # 全场机位优先用交点凸包的四条极值边找外框角，只需 24 个排列；比盲采
        # 更快且能处理白线断续。极值点不是角点时会自然失败，再进入有界随机搜索。
        extreme = {
            int(np.argmin(pix[:, 0])), int(np.argmax(pix[:, 0])),
            int(np.argmin(pix[:, 1])), int(np.argmax(pix[:, 1])),
            int(np.argmin(pix[:, 0] + pix[:, 1])), int(np.argmax(pix[:, 0] + pix[:, 1])),
            int(np.argmin(pix[:, 0] - pix[:, 1])), int(np.argmax(pix[:, 0] - pix[:, 1])),
        }
        for corner_idx in itertools.combinations(sorted(extreme), 4):
            pix4 = pix[list(corner_idx)]
            for perm in itertools.permutations(range(4)):
                try:
                    H_inv, _ = cv2.findHomography(WORLD_POINTS[:4][list(perm)], pix4, 0)
                except cv2.error:
                    continue
                consider(H_inv)
            if best_score >= min(10, len(detections)):
                break

        n_pix = len(detections)
        for _ in range(self.cfg.classical_hypotheses):
            if best_score >= min(12, n_pix):
                break
            pi = rng.choice(n_pix, 4, replace=False)
            wi = rng.choice(len(WORLD_POINTS), 4, replace=False)
            pix4 = pix[pi]
            for perm in itertools.permutations(range(4)):
                world4 = WORLD_POINTS[wi[list(perm)]]
                try:
                    H_inv, _ = cv2.findHomography(world4, pix4, 0)
                except cv2.error:
                    continue
                consider(H_inv)
        if best_score < self.cfg.min_matches:
            return None, None, [], float("inf")
        return self._solve_labeled(best_named, shape, self.cfg.min_matches)

    def _line_support(
        self, H_inv: np.ndarray, observation: LineObservation, shape: tuple[int, int],
    ) -> tuple[int, float, np.ndarray, float]:
        if not observation.segments:
            return 0, float("inf"), np.zeros(2), float("inf")
        world_samples = []
        for a, b in COURT_SEGMENTS:
            a, b = np.asarray(a), np.asarray(b)
            length = float(np.linalg.norm(b - a))
            count = max(2, min(12, int(length / 0.8) + 1))
            world_samples.extend(np.linspace(a, b, count))
        world = np.asarray(world_samples, np.float64)
        pred = _project(H_inv, world)
        nearest, dist, seg_idx = _nearest_on_segments(pred, observation.segments)
        gate = max(26.0, math.hypot(*shape) * self.cfg.line_gate_frac)
        good = np.isfinite(dist) & (dist <= gate)
        families = 0
        if good.any():
            used = np.unique(seg_idx[good])
            angles = []
            for idx in used:
                a, b = observation.segments[int(idx)]
                angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0
                angles.append(angle)
            # 同一条白线被 Hough 切成多段时只算一个线族；两条近平行线不能
            # 互相作为配准支撑，避免局部特写/反射产生伪全场 H。
            clusters: list[float] = []
            for angle in sorted(angles):
                if not clusters or all(min(abs(angle - c), 180 - abs(angle - c)) > 15.0
                                       for c in clusters):
                    clusters.append(angle)
            families = len(clusters)
            shift = np.median(nearest[good] - pred[good], axis=0)
            quality = float(np.median(np.clip(1.0 - dist[good] / gate, 0.0, 1.0)))
            residual_px = float(np.median(dist[good]))
        else:
            shift, quality, residual_px = np.zeros(2), 0.0, float("inf")
        return families, quality, shift, residual_px

    def _line_aid(
        self, observation: LineObservation, shape: tuple[int, int], mode: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None, int, float]:
        """有历史机位时，用可见主线段支撑估计小幅平移；硬切/局部特写不猜 H。"""
        if self._H_inv is None or mode == "hard_cut":
            return None, None, 0, float("inf")
        families, quality, shift, _residual = self._line_support(
            self._H_inv, observation, shape)
        if families < self.cfg.min_line_families or quality < 0.25:
            return None, None, families, float("inf")
        if np.linalg.norm(shift) > math.hypot(*shape) * 0.12:
            return None, None, families, float("inf")
        T = np.asarray([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]], [0.0, 0.0, 1.0]])
        H_inv = T @ self._H_inv
        if not self._plausible(H_inv, shape):
            return None, None, families, float("inf")
        try:
            H = np.linalg.inv(H_inv)
            _families2, _quality2, _shift2, residual = self._line_support(
                H_inv, observation, shape)
            return H, H_inv, families, residual
        except np.linalg.LinAlgError:
            return None, None, families, float("inf")

    def _smooth(
        self, H: np.ndarray, H_inv: np.ndarray, mode: str, shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._H_inv is None or mode in ("hard_cut", "initial"):
            return H, H_inv
        raw = _project(H_inv, REFERENCE_POINTS)
        prev = _project(self._H_inv, REFERENCE_POINTS)
        drift = float(np.median(np.linalg.norm(raw - prev, axis=1))) / max(math.hypot(*shape), 1)
        alpha = {"static": 0.30, "pan": 0.58, "zoom": 0.42}.get(mode, 0.50)
        if drift > 0.06:
            alpha = max(alpha, 0.78)
        ema = alpha * raw + (1.0 - alpha) * prev
        try:
            smooth_inv, _ = cv2.findHomography(REFERENCE_POINTS, ema, 0)
            smooth = np.linalg.inv(smooth_inv)
        except (cv2.error, np.linalg.LinAlgError):
            return H, H_inv
        if not np.all(np.isfinite(smooth)) or not self._plausible(smooth_inv, shape):
            return H, H_inv
        return smooth, smooth_inv

    @staticmethod
    def _publication_confidence(detections: list[DetectedKeypoint]) -> float:
        """Runtime and offline evaluation must use the same inlier-support gate."""
        confs = [d.confidence for d in detections if d.inlier]
        ratio = len(confs) / max(len(detections), 1)
        mean_conf = float(np.mean(confs)) if confs else 0.0
        return float(np.clip(mean_conf * (0.65 + 0.35 * ratio), 0.0, 1.0))

    def _make_valid(
        self, n: int, now: float, mode: str, transition: str, source: str,
        H: np.ndarray, H_inv: np.ndarray, detections: list[DetectedKeypoint],
        err: float, observation: LineObservation, shape: tuple[int, int],
    ) -> CourtKeypointResult:
        H, H_inv = self._smooth(H, H_inv, mode, shape)
        families, line_quality, _shift, line_residual = self._line_support(
            H_inv, observation, shape)
        if source in ("line_aid", "keypoints+lines"):
            if families < self.cfg.min_line_families or line_quality < 0.25:
                self._reset_h("line_support_lost_after_smoothing")
                self.stats["failures"] += 1
                return empty_result(n, now, n, mode, transition,
                                    "line_support_lost_after_smoothing", source, self.stats)
            err = line_residual
        by_name = {d.name: d for d in detections if d.inlier}
        visible = []
        for name, world in KEYPOINT_SPECS:
            d = by_name.get(name)
            visible.append(d if d is not None else DetectedKeypoint(
                name, (float("nan"), float("nan")), 0.0, world))
        confs = [d.confidence for d in visible if d.inlier]
        confidence = self._publication_confidence(detections)
        if source == "line_aid" and not confs:
            confidence = float(np.clip(0.25 + 0.60 * line_quality, 0.0, 0.85))
        if source in ("line_aid", "keypoints+lines"):
            confidence *= 0.75 + 0.25 * line_quality
        if source in ("onnx", "classical+prior") and confidence < self.cfg.min_publish_conf:
            self._reset_h("low_confidence")
            self.stats["failures"] += 1
            return empty_result(n, now, n, mode, transition, "low_confidence",
                                source, self.stats)
        self._H, self._H_inv = H, H_inv
        self._last_detections = [d for d in visible if d.inlier]
        self._last_source = source
        self._last_valid = now
        self._last_result = CourtKeypointResult(
            frame=n, timestamp=now, inference_frame=n, mode=mode, transition=transition,
            valid=True, reason="ok", source=source, confidence=confidence,
            inliers=len(confs), reproj_error_px=float(err), keypoints=visible,
            H=H, H_inv=H_inv, line_families=families, stats=dict(self.stats),
        )
        return self._last_result

    def _carry(self, n: int, now: float, mode: str, transition: str,
               flow: np.ndarray, stats: dict) -> CourtKeypointResult:
        if self._last_result is None or self._H is None or self._H_inv is None:
            return empty_result(n, now, self._last_infer, mode, transition,
                                 "no_calibration", "none", stats)
        age = now - self._last_valid
        if age > self.cfg.stale_s:
            self._reset_h("stale")
            return empty_result(n, now, self._last_infer, mode, transition,
                                 "stale", "none", stats)
        confidence = self._last_result.confidence * math.exp(-max(age, 0.0))
        if (self._last_source in ("onnx", "classical+prior")
                and confidence < self.cfg.min_publish_conf):
            self._reset_h("low_confidence")
            return empty_result(n, now, self._last_infer, mode, transition,
                                 "low_confidence", "none", stats)
        cached = []
        if mode == "pan" and np.all(np.isfinite(flow)) and float(np.linalg.norm(flow)) > 0.05:
            T = np.asarray([[1.0, 0.0, flow[0]], [0.0, 1.0, flow[1]], [0.0, 0.0, 1.0]])
            candidate = T @ self._H_inv
            if self._plausible(candidate, self._shape):
                self._H_inv = candidate
                self._H = np.linalg.inv(candidate)
        for d in self._last_detections:
            p = np.asarray(d.pixel) + flow
            cached.append(DetectedKeypoint(d.name, (float(p[0]), float(p[1])),
                                           d.confidence, d.world, True))
        by_name = {d.name: d for d in cached}
        visible = [by_name.get(name, DetectedKeypoint(
            name, (float("nan"), float("nan")), 0.0, world)) for name, world in KEYPOINT_SPECS]
        # 缓存的是最近一次观测位置；每帧只叠加相对上一帧的 flow，避免隔帧时
        # 把位移反复加到同一批原始点或每帧重置回旧坐标。
        self._last_detections = cached
        self._last_result = CourtKeypointResult(
            frame=n, timestamp=now, inference_frame=self._last_infer,
            mode=mode, transition=transition, valid=True,
            reason="ok" if age < self.cfg.stale_s * 0.7 else "stale_soon",
            source="temporal", confidence=confidence,
            inliers=self._last_result.inliers, reproj_error_px=self._last_result.reproj_error_px,
            keypoints=visible, H=self._H, H_inv=self._H_inv,
            line_families=self._last_result.line_families, stats=dict(stats),
        )
        return self._last_result

    def update(self, frame: int, rgb: np.ndarray) -> CourtKeypointResult:
        """处理当前帧；隔帧只复用过去结果，硬切当帧强制重算。"""
        with self._lock:
            now = time.time()
            if rgb is None or rgb.size == 0 or rgb.ndim != 3 or rgb.shape[2] != 3:
                return empty_result(frame, now, self._last_infer, "invalid", "none",
                                     "invalid_image", "none", self.stats)
            shape = rgb.shape[:2]
            if self._shape is not None and self._shape != shape:
                self._reset_h("resolution_changed")
                self._prev_small = self._prev_orb = self._prev_hist = None
                self._prev_luma = self._prev_texture = None
                self._last_infer = self._last_attempt = -10**9
            self._shape = shape
            mode, flow, _scale = self._motion(rgb)
            transition = "hard_cut" if mode == "hard_cut" else mode
            if mode == "hard_cut":
                self.stats["hard_cuts"] += 1
                self._reset_h("hard_cut")
            due = (
                mode == "hard_cut"
                or frame - self._last_infer >= self.cfg.every_n
                or (self._H is None and frame - self._last_attempt >= self.cfg.retry_every)
            )
            if not due:
                return self._carry(frame, now, mode, transition, flow, self.stats)
            self._last_attempt = frame
            self._last_infer = frame
            self.stats["inferences"] += 1

            observation = LineObservation()
            detections: list[DetectedKeypoint] = []
            used_model = False
            if self.model.ready:
                detections = [d for d in self.model.predict(rgb)
                              if d.confidence >= self.cfg.min_conf
                              and np.all(np.isfinite(d.pixel))
                              and 0 <= d.pixel[0] < rgb.shape[1]
                              and 0 <= d.pixel[1] < rgb.shape[0]]
                used_model = bool(detections)
            need_lines = not used_model or len(detections) < self.cfg.min_matches
            if need_lines:
                observation = detect_classical(rgb, self.cfg)
                # DL 点位不足时优先使用几何交点完成标定，避免弱模型阻断经典链路。
                if not used_model or len(detections) < self.cfg.min_matches:
                    detections = observation.intersection_keypoints or observation.points
                    used_model = False

            source = "onnx" if used_model else "classical"
            named = [DetectedKeypoint(d.name, d.pixel, d.confidence,
                                      KEYPOINT_SPECS[KEYPOINT_INDEX[d.name]][1])
                     for d in detections if d.name in KEYPOINT_INDEX]
            if not named and not used_model and self._H_inv is not None:
                named = self._match_prior(observation.intersection_keypoints or observation.points, shape)
                if named:
                    source = "classical+prior"

            H = H_inv = None
            err = float("inf")
            matched: list[DetectedKeypoint] = named
            if len(named) >= 4:
                H, H_inv, matched, err = self._solve_labeled(
                    named, shape, 4 if len(named) < self.cfg.min_matches else self.cfg.min_matches)
            if H is None and used_model:
                # Having four detections does not guarantee a usable H. If the
                # semantic fit fails, keep the classical fallback alive instead
                # of returning low_confidence without checking the court lines.
                observation = detect_classical(rgb, self.cfg)
                line_detections = observation.intersection_keypoints or observation.points
                line_named = [DetectedKeypoint(d.name, d.pixel, d.confidence,
                                               KEYPOINT_SPECS[KEYPOINT_INDEX[d.name]][1])
                              for d in line_detections if d.name in KEYPOINT_INDEX]
                if len(line_named) >= 4:
                    H, H_inv, matched, err = self._solve_labeled(
                        line_named, shape, 4 if len(line_named) < self.cfg.min_matches
                        else self.cfg.min_matches)
                    if H is not None:
                        source = "classical"
                if H is None and len(line_named) >= 4:
                    # If lines alone are insufficient, let semantic points add
                    # support. _make_valid still requires independent line-family
                    # evidence for this fused source.
                    combined = {d.name: d for d in line_named}
                    combined.update({d.name: d for d in named})
                    fused = list(combined.values())
                    H, H_inv, matched, err = self._solve_labeled(
                        fused, shape, 4 if len(fused) < self.cfg.min_matches
                        else self.cfg.min_matches)
                    if H is not None:
                        source = "keypoints+lines"
            if (H is None and self._H_inv is None and
                    (observation.points or observation.intersection_keypoints)):
                H, H_inv, matched, err = self._solve_unlabeled(
                    observation.intersection_keypoints or observation.points, shape, frame)
                if H is not None:
                    source = "classical_ransac"

            if H is not None:
                if source == "classical_ransac":
                    families, quality, _, _ = self._line_support(H_inv, observation, shape)
                    if families < self.cfg.min_line_families or quality < 0.35:
                        H = H_inv = None
                if H is not None and len(matched) < self.cfg.min_matches:
                    families, quality, _, _ = self._line_support(H_inv, observation, shape)
                    if families < self.cfg.min_line_families or quality < 0.25:
                        H = H_inv = None
                    else:
                        source = "keypoints+lines"
                        self.stats["line_aids"] += 1
                if H is not None:
                    return self._make_valid(frame, now, mode, transition, source,
                                            H, H_inv, matched, err, observation, shape)

            if observation.segments:
                H, H_inv, families, residual_px = self._line_aid(observation, shape, mode)
                if H is not None:
                    self.stats["line_aids"] += 1
                    # 边线辅助不伪造关键点；只保留本轮真实检测到的内点。
                    matched = [d for d in matched if d.confidence >= self.cfg.min_conf]
                    result = self._make_valid(frame, now, mode, transition, "line_aid",
                                              H, H_inv, matched, residual_px, observation, shape)
                    result.reason = "line_aid"
                    return result

            self._reset_h("low_confidence_or_insufficient_support")
            self.stats["failures"] += 1
            detail = "model_unavailable" if not self.model.ready else "low_confidence"
            return empty_result(frame, now, frame, mode, transition, detail,
                                 source, self.stats)


def get_keypoint_tracker(vid: str, cfg: KeypointConfig | None = None) -> CourtKeypointTracker:
    """每个视频一个独立状态；调用方负责缓存生命周期。"""
    return CourtKeypointTracker(cfg)


__all__ = [
    "KEYPOINT_IDS", "KEYPOINT_SPECS", "WORLD_POINTS", "N_KEYPOINTS",
    "KeypointConfig", "DetectedKeypoint", "CourtKeypointResult",
    "CourtKeypointModel", "CourtKeypointTracker", "detect_classical", "empty_result",
]
