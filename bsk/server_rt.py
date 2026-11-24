"""实时视频叠加服务 v2：8GB 档完整实时管线 + 3D 骨架流式输出。

架构：
1. 标定：增量复用（drift < 80px 且 age < 5s 时跳过 RANSAC）
2. 检测：YOLOv8n-pose，每帧运行（~15ms on GPU）
3. 姿态：YOLOv8n-pose 自带，每帧运行
4. 时序平滑：运动预测补足遮挡帧，骨长约束校准
5. 3D 重建：从单目 2D 提升到世界坐标 3D（米制）
6. 渲染：烧录到 JPEG，前端只负责显示

性能目标：
- 1080p @ 30 FPS：~33ms/frame 预算
- 标定：< 10ms（复用 H）或 < 100ms（首次/重标定）
- 检测+姿态：< 20ms（YOLOv8n-pose）
- 时序平滑+3D：< 5ms
- 渲染：< 5ms

Profile requirement:
- 4GB mode: outputs only 2D coordinates (no 3D rendering)
- 8GB mode: enables full 3D skeleton + streaming to frontend
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from bsk import calibrate, court, render, schema
from bsk.realtime_calibrate import calibrate_frame
from bsk.perceive import shots as shotmod
from bsk.render_3d import TemporalSmoother, PlayerSkeleton3D
from bsk.roi_tracking import get_roi_tracker, apply_court_roi_from_calibration


# ── 模型懒加载 ───────────────────────────────────────────────

_models_lock = threading.Lock()
_models: dict = {}
_temporal_smoother: Optional[TemporalSmoother] = None


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


def _get_tracker():
    with _models_lock:
        if "tracker" not in _models:
            from bsk.analyze import Tracker
            _models["tracker"] = Tracker()
        return _models["tracker"]


def _get_smoother():
    """Get temporal smoother for 3D skeletons (8GB mode only)."""
    global _temporal_smoother
    if _temporal_smoother is None:
        from bsk.court import FIBA
        _temporal_smoother = TemporalSmoother(court=FIBA)
    return _temporal_smoother


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
    players_3d: list[PlayerSkeleton3D] = field(default_factory=list)  # 3D skeletons for streaming
    img_shape: tuple[int, int] = (0, 0)  # (h, w) 检测所基于的分辨率


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


def _get_court_bounds(H_inv: np.ndarray, img_shape: tuple) -> Optional[tuple]:
    """Get court boundary in pixel coordinates for ROI filtering.
    
    Returns:
        (x_min, y_min, x_max, y_max) or None if calibration unavailable
    """
    if H_inv is None:
        return None
    
    # Court corners in world coords (FIBA: 28m x 15m)
    from bsk.court import FIBA
    corners_world = np.array([
        [-FIBA.half_length, -FIBA.half_width, 1],
        [FIBA.half_length, -FIBA.half_width, 1],
        [FIBA.half_length, FIBA.half_width, 1],
        [-FIBA.half_length, FIBA.half_width, 1],
    ], dtype=np.float64)
    
    # Project to pixel coords
    corners_pixel = (H_inv @ corners_world.T).T
    corners_pixel = corners_pixel[:, :2] / corners_pixel[:, 2:3]
    
    x_min = float(corners_pixel[:, 0].min()) - 50  # Margin for safety
    y_min = float(corners_pixel[:, 1].min()) - 50
    x_max = float(corners_pixel[:, 0].max()) + 50
    y_max = float(corners_pixel[:, 1].max()) + 50
    
    return (x_min, y_min, x_max, y_max)


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
    enable_3d: bool = True,  # 8GB mode flag
) -> FrameResult:
    """实时分析单帧。
    
    Args:
        vid: 视频 ID
        n: 帧号
        rgb: RGB 图像 [H, W, 3]
        pts_us: 时间戳（微秒）
        run_seg: 是否运行分割（慢，默认关）
        enable_3d: 是否启用 3D 骨架重建（8GB 档）
    
    Returns:
        FrameResult: 包含检测、姿态、分割、标定、3D 骨架结果
    """
    t0 = time.time()
    
    # 1. 增量标定（最快路径：复用上一帧 H）
    cal = calibrate_frame(vid, rgb)
    t_calib = time.time() - t0
    
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
    if H_inv_court_to_pixel is not None:
        apply_court_roi_from_calibration(roi_tracker, H_inv_court_to_pixel, rgb.shape)
    
    # 2. 快速姿态检测 + 跟踪（YOLOv8n-pose，每帧运行）
    # Note: Disabled ROI skip optimization due to tracking instability
    # Static scene detection was causing players to disappear when they stopped moving
    fast_pose = _get_fast_pose()
    ppl = fast_pose(rgb, thr=0.45)  # Optimized threshold for basketball (reduces false positives)
    boxes = [p.bbox for p in ppl]
    
    tracker = _get_tracker()
    ids = tracker.assign(boxes)
    
    # Always update ROI tracker with fresh detections
    roi_tracker.update_detections(ppl)
    
    t_detect = time.time() - t0 - t_calib
    
    # 3. 球检测（轻量级，~5ms）
    from bsk.perceive.ball import get_ball_detector
    ball_det = get_ball_detector(device="cuda")
    ball_dets = ball_det(rgb, thr=0.3)
    
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
    
    t_ball = time.time() - t0 - t_calib - t_detect
    
    # 4. 构建 Tracked 对象和 Pose2D（过滤假阳性）
    objects = []
    poses = []
    
    # Get court bounds for ROI filtering (reuse cached H_inv)
    court_bounds = _get_court_bounds(H_inv, rgb.shape) if H_inv is not None else None
    
    for p, tid in zip(ppl, ids):
        # Filter 1: Confidence threshold (optimized for basketball)
        if p.score < 0.45:
            continue
        
        # Filter 2: Bounding box aspect ratio (standing person ~1:2 to 1:3)
        bw, bh = p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1]
        aspect_ratio = bh / max(bw, 1)
        # Seated spectators are typically wide (AR < 1.0), standing players are tall (AR > 1.0)
        if aspect_ratio < 0.8 or aspect_ratio > 4.5:
            continue
        
        # Filter 3: Adaptive minimum size (resolution-aware)
        min_height = max(60, rgb.shape[0] * 0.05)  # 5% of frame height
        min_width = max(30, rgb.shape[1] * 0.025)  # 2.5% of frame width
        if bh < min_height or bw < min_width:
            continue
        
        # Filter 4: Court ROI (if calibration available) - PRIMARY crowd filter
        # This excludes bench area and audience seats
        if court_bounds is not None:
            cx, cy = (p.bbox[0] + p.bbox[2]) / 2, (p.bbox[1] + p.bbox[3]) / 2
            if not _point_in_rect(cx, cy, court_bounds):
                continue  # Outside court = spectator/bench
        
        # Filter 5: Keypoint quality check (spectators often have poor keypoints due to occlusion/seating)
        if p.kpts is not None:
            # Count high-confidence keypoints
            high_conf_kpts = np.sum(p.kpts[:, 2] > 0.5)
            # Standing players/referees typically have >6 good keypoints
            # Seated spectators often have <4 due to unusual poses
            if high_conf_kpts < 4:
                continue  # Likely seated spectator with poor pose
        
        # Fix left-right asymmetry: mirror keypoints if needed (lazy copy)
        kpts_fixed = p.kpts
        if kpts_fixed is not None:
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
                kpts=kpts_fixed.tolist(), model="yolov8n_pose_symmetric",
                height_px=float(p.bbox[3] - p.bbox[1])
            ))
    t_pose = time.time() - t0 - t_calib - t_detect - t_ball
    
    # 合并人员和球对象
    objects.extend(ball_objects)
    
    # 5. 3D 骨架重建与时序平滑（8GB 档）
    players_3d = []
    t_smooth = 0.0
    if enable_3d:
        # H_inv already inverted above (pixel->court)
        
        # Lift 2D to 3D
        pairs_3d = []
        for obj, pose in zip(objects, poses):
            if H_inv is not None:
                skel = PlayerSkeleton3D.from_2d(obj, pose, H_inv)
                if skel:
                    pairs_3d.append(skel)
        
        # Apply temporal smoothing and occlusion prediction
        smoother = _get_smoother()
        players_3d = smoother.smooth(pairs_3d, n)
        t_smooth = time.time() - t0 - t_calib - t_detect - t_ball - t_pose
    
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
    t_seg = time.time() - t0 - t_calib - t_detect - t_pose - t_smooth
    
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
        players_3d=players_3d, img_shape=(h, w)
    )
    _cache.put(n, result)
    
    total_ms = (time.time() - t0) * 1000
    if total_ms > 50:  # 警告：超过预算
        print(f"[WARN] Frame {n} took {total_ms:.1f}ms "
              f"(calib={t_calib*1000:.1f}, detect={t_detect*1000:.1f}, ball={t_ball*1000:.1f}, "
              f"pose={t_pose*1000:.1f}, smooth={t_smooth*1000:.1f}, seg={t_seg*1000:.1f})")
    
    return result


def render_frame_realtime(
    vid: str,
    n: int,
    rgb: np.ndarray,
    pts_us: int,
    layers: render.Layers,
    width: int,
    quality: int = 72,
    enable_3d: bool = True,
    court_model: str = "FIBA",
) -> bytes:
    """实时渲染单帧并返回 JPEG。"""
    # 获取或计算分析结果（缓存基于检测时的分辨率，尺寸不一致必须重算）
    cached = _cache.get(n)
    if cached is None or cached.img_shape != rgb.shape[:2]:
        cached = analyze_frame_realtime(vid, n, rgb, pts_us, enable_3d=enable_3d)
    
    # 获取前一帧用于运动检测（已经是1/4分辨率灰度图）
    prev_rgb = _cache.get_prev(n)
    
    # 准备渲染参数
    h, w = rgb.shape[:2]
    scale = min(width / w, 1.0) if width else 1.0
    scaled_w = int(w * scale)
    scaled_h = int(h * scale)
    
    if scale < 1.0:
        rgb_resized = cv2.resize(rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    else:
        rgb_resized = rgb
    
    # 标定信息
    H_inv = np.asarray(cached.calib.H_inv, dtype=np.float64) if cached.calib else None
    
    # Use court model from request parameter or calibration
    court_obj = court.COURTS.get(court_model, court.FIBA)
    
    # Update smoother with correct court dimensions
    smoother = _get_smoother()
    if smoother.court.model != court_model:
        from bsk.court import COURTS
        smoother.court = COURTS.get(court_model, court.FIBA)
        print(f"[3D] Smoother court updated to {court_model}")
    
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
        h_full, w_full = rgb_resized.shape[:2]
        if motion.shape != (h_full, w_full):
            motion = cv2.resize(motion.astype(np.uint8) * 255, (w_full, h_full), 
                               interpolation=cv2.INTER_NEAREST).astype(bool)
    
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
    
    if enable_3d and cached.players_3d:
        notes.append(f"3D: {len(cached.players_3d)} players")
    
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
        _get_tracker()
        print(f"[warmup] Tracker ready")
    except Exception as e:
        print(f"[warn] Failed to warmup tracker: {e}")
    
    print(f"[warmup] All models loaded in {time.time()-t0:.1f}s")

_warmup_thread = threading.Thread(target=_warmup_models, daemon=True)
_warmup_thread.start()


__all__ = ["analyze_frame_realtime", "render_frame_realtime", "RealtimeCache", "FrameResult"]
