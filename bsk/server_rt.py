"""实时视频叠加服务：8GB 档完整实时管线。

架构：
1. 标定：增量复用（drift < 80px 且 age < 5s 时跳过 RANSAC）
2. 检测：YOLOv8n-pose，每帧运行（~15ms on GPU）
3. 姿态：YOLOv8n-pose 自带，每帧运行（无需隔帧插值）
4. 分割：可选，默认关闭
5. 渲染：烧录到 JPEG，前端只负责显示

性能目标：
- 1080p @ 30 FPS：~33ms/frame 预算
- 标定：< 10ms（复用 H）或 < 100ms（首次/重标定）
- 检测+姿态：< 20ms（YOLOv8n-pose）
- 渲染：< 5ms

模型选择：
- YOLOv8n-pose: ~15ms/frame (GPU), Apache-2.0 许可
- 替代方案：MediaPipe BlazePose (~5ms CPU), RTMO-s (~380ms GPU, 太慢)
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from bsk import calibrate, court, render, schema
from bsk.realtime_calibrate import calibrate_frame
from bsk.perceive import shots as shotmod


# ── 模型懒加载 ───────────────────────────────────────────────

_models_lock = threading.Lock()
_models: dict = {}


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


def _get_rtmo():
    """兼容旧接口，已废弃，改用 _get_fast_pose()."""
    return _get_fast_pose()


def _get_ins():
    with _models_lock:
        if "ins" not in _models:
            from bsk.analyze import RtmdetIns
            _models["ins"] = RtmdetIns()
        return _models["ins"]


def _get_tracker():
    with _models_lock:
        if "tracker" not in _models:
            from bsk.analyze import Tracker
            _models["tracker"] = Tracker()
        return _models["tracker"]


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


class RealtimeCache:
    """线程安全的帧结果缓存。"""
    
    def __init__(self, max_size: int = 60):
        self._cache: OrderedDict[int, FrameResult] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
    
    def put(self, n: int, result: FrameResult) -> None:
        with self._lock:
            self._cache[n] = result
            self._cache.move_to_end(n)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
    
    def get(self, n: int) -> Optional[FrameResult]:
        with self._lock:
            if n in self._cache:
                self._cache.move_to_end(n)
                return self._cache[n]
            return None
    
    def get_prev(self, n: int) -> Optional[np.ndarray]:
        with self._lock:
            for fn in range(n - 1, max(0, n - 10), -1):
                if fn in self._cache and self._cache[fn].prev_rgb is not None:
                    return self._cache[fn].prev_rgb
            return None


_cache = RealtimeCache(max_size=60)

# ── 姿态插值 ───────────────────────────────────────────────

_pose_history: deque[tuple[int, list[schema.Pose2D]]] = deque(maxlen=10)


def _interpolate_poses(n: int, current: list[schema.Pose2D]) -> list[schema.Pose2D]:
    """如果当前帧没跑姿态，用前后帧插值。"""
    if current:
        _pose_history.append((n, current))
        return current
    
    # 找前后最近的姿态帧
    before = None
    after = None
    for fn, poses in reversed(_pose_history):
        if fn < n and (before is None or fn > before[0]):
            before = (fn, poses)
    for fn, poses in _pose_history:
        if fn > n and (after is None or fn < after[0]):
            after = (fn, poses)
    
    if before and after:
        # 线性插值
        t = (n - before[0]) / (after[0] - before[0])
        return _lerp_poses(before[1], after[1], t)
    elif before:
        return before[1]
    elif after:
        return after[1]
    return []


def _lerp_poses(p1: list[schema.Pose2D], p2: list[schema.Pose2D], t: float) -> list[schema.Pose2D]:
    """在两个姿态列表间线性插值。"""
    by_id1 = {p.track_id: p for p in p1}
    by_id2 = {p.track_id: p for p in p2}
    result = []
    for tid in set(by_id1.keys()) | set(by_id2.keys()):
        if tid in by_id1 and tid in by_id2:
            kpts1 = np.array(by_id1[tid].kpts)
            kpts2 = np.array(by_id2[tid].kpts)
            kpts = kpts1 * (1 - t) + kpts2 * t
            result.append(schema.Pose2D(
                video_id="", frame=0, track_id=tid,
                kpts=kpts.tolist(), model="interpolated",
                height_px=(kpts1[:, 1].max() - kpts1[:, 1].min()) * (1 - t) + 
                          (kpts2[:, 1].max() - kpts2[:, 1].min()) * t
            ))
        elif tid in by_id1:
            result.append(by_id1[tid])
        else:
            result.append(by_id2[tid])
    return result


# ─ 核心实时管线 ─────────────────────────────────────────────

def analyze_frame_realtime(
    vid: str,
    n: int,
    rgb: np.ndarray,
    pts_us: int,
    run_seg: bool = False,
    run_pose_every: int = 2,
) -> FrameResult:
    """实时分析单帧。
    
    Args:
        vid: 视频 ID
        n: 帧号
        rgb: RGB 图像 [H, W, 3]
        pts_us: 时间戳（微秒）
        run_seg: 是否运行分割（慢，默认关）
        run_pose_every: 姿态运行频率（每 N 帧跑一次）
    
    Returns:
        FrameResult: 包含检测、姿态、分割、标定结果
    """
    t0 = time.time()
    
    # 1. 增量标定（最快路径：复用上一帧 H）
    cal = calibrate_frame(vid, rgb)
    t_calib = time.time() - t0
    
    # 2. 快速姿态检测 + 跟踪（YOLOv8n-pose，每帧运行）
    fast_pose = _get_fast_pose()
    ppl = fast_pose(rgb, thr=0.3)
    boxes = [p.bbox for p in ppl]
    
    tracker = _get_tracker()
    ids = tracker.assign(boxes)
    t_detect = time.time() - t0 - t_calib
    
    # 3. 构建 Tracked 对象和 Pose2D
    objects = []
    poses = []
    for p, tid in zip(ppl, ids):
        objects.append(schema.Tracked(
            track_id=tid, category="player", frame=n,
            bbox_px=p.bbox, conf=p.score, visible=True
        ))
        if p.kpts is not None:
            poses.append(schema.Pose2D(
                video_id=vid, frame=n, track_id=tid,
                kpts=np.asarray(p.kpts).tolist(), model="yolov8n_pose",
                height_px=float(p.bbox[3] - p.bbox[1])
            ))
    t_pose = time.time() - t0 - t_calib - t_detect
    
    # 5. 分割（可选，很慢）
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
    t_seg = time.time() - t0 - t_calib - t_detect - t_pose
    
    # 4. 分割（可选，很慢）
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
    t_seg = time.time() - t0 - t_calib - t_detect - t_pose
    
    # 5. 缓存结果
    result = FrameResult(
        frame=n, pts_us=pts_us, objects=objects, poses=poses,
        segments=segments, calib=cal, prev_rgb=rgb.copy()
    )
    _cache.put(n, result)
    
    total_ms = (time.time() - t0) * 1000
    if total_ms > 50:  # 警告：超过预算
        print(f"[WARN] Frame {n} took {total_ms:.1f}ms "
              f"(calib={t_calib*1000:.1f}, detect={t_detect*1000:.1f}, "
              f"pose={t_pose*1000:.1f}, seg={t_seg*1000:.1f})")
    
    return result


def render_frame_realtime(
    vid: str,
    n: int,
    rgb: np.ndarray,
    pts_us: int,
    layers: render.Layers,
    width: int,
    quality: int = 72,
) -> bytes:
    """实时渲染单帧并返回 JPEG。"""
    # 获取或计算分析结果
    cached = _cache.get(n)
    if cached is None:
        cached = analyze_frame_realtime(vid, n, rgb, pts_us)
    
    # 获取前一帧用于运动检测
    prev_rgb = _cache.get_prev(n)
    
    # 准备渲染参数
    h, w = rgb.shape[:2]
    scale = min(width / w, 1.0) if width else 1.0
    scaled_w = int(w * scale)
    scaled_h = int(h * scale)
    
    if scale < 1.0:
        rgb = cv2.resize(rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    
    # 标定信息
    H_inv = np.array(cached.calib.H_inv) if cached.calib else None
    court_obj = court.COURTS.get(cached.calib.court_model, court.FIBA) if cached.calib else None
    
    # 运动检测
    motion = None
    if layers.motion and prev_rgb is not None:
        g1 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        g0 = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
        d = cv2.absdiff(g1, g0)
        m = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        motion = cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0
    
    # 地板分割（轻量版）
    floor = None
    if layers.floor:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lower = np.array([20, 40, 60])
        upper = np.array([50, 120, 200])
        floor = cv2.inRange(hsv, lower, upper) > 0
    
    # 构建 notes
    notes = []
    if cached.calib:
        notes.append(f"calib: err={cached.calib.reproj_error_px:.1f}px n={cached.calib.keypoints_used}")
    else:
        notes.append("calib: FAILED")
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
    )
    
    return render.to_jpeg(rendered, quality)


# ── 清理旧缓存 ─────────────────────────────────────────────

def cleanup_old_cache(max_age_sec: float = 300.0) -> None:
    """定期清理超过 max_age_sec 的缓存。"""
    now = time.time()
    with _cache._lock:
        to_remove = [n for n, r in _cache._cache.items() 
                     if now - r.pts_us / 1e6 > max_age_sec]
        for n in to_remove:
            del _cache._cache[n]


# 启动后台清理线程
_cleaner_thread = threading.Thread(
    target=lambda: [cleanup_old_cache(), time.sleep(60)] * 99999,
    daemon=True
)
_cleaner_thread.start()
