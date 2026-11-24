"""实时增量标定：跟踪摄像机运动，只在场景剧变时重新标定。

摄像机在直播中会平移、旋转、变焦，单应矩阵 H 随时在变。每帧都跑完整 RANSAC
太慢（~200ms），所以用增量策略：
- 每帧检测球场交点
- 与上一帧交点匹配，如果位移小（摄像机慢速移动），直接复用 H
- 如果位移大或匹配失败，重新跑 RANSAC 标定
- 用 H 的连续性做平滑，避免跳变

这样完整标定只在场景切换、大幅变焦时触发，平时只做轻量跟踪。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from bsk.auto_calibrate import _detect_intersections, auto_calibrate
from bsk.calibrate import Calibration
from bsk.court import FIBA, Court


@dataclass
class CalibState:
    """标定状态：当前 H 矩阵 + 上一帧交点 + 时间戳。"""
    H: np.ndarray | None = None
    H_inv: np.ndarray | None = None
    intersections: list[tuple[float, float]] = field(default_factory=list)
    timestamp: float = 0.0
    reproj_error: float = 0.0
    inliers: int = 0
    stable_count: int = 0  # 连续稳定帧数
    img_shape: tuple[int, int] | None = None  # H 对应的图像尺寸 (h, w)


class IncrementalCalibrator:
    """增量标定器：每个视频一个实例，跟踪摄像机运动。"""

    def __init__(self, court: Court = FIBA, drift_thresh: float = 150.0, max_age: float = 10.0):
        """
        Args:
            court: 球场几何模型
            drift_thresh: 交点漂移阈值（像素），超过则重新标定（放宽到 150px 避免频繁重标定）
            max_age: 标定最大有效期（秒），超时强制重新标定（延长到 10s）
        """
        self.court = court
        self.drift_thresh = drift_thresh
        self.max_age = max_age
        self.state = CalibState()

    def update(self, image: np.ndarray, force_recalibrate: bool = False) -> Calibration | None:
        """处理一帧，返回当前标定。

        Args:
            image: RGB 或 BGR 图像
            force_recalibrate: 强制重新标定

        Returns:
            Calibration 或 None（首次标定失败时）
        """
        now = time.time()
        shape = (image.shape[0], image.shape[1])
        intersections = _detect_intersections(image)

        if not intersections:
            return None

        # 分辨率变了：旧 H 是另一套像素坐标，必须重标定且不能拿它做初值
        shape_changed = self.state.img_shape is not None and self.state.img_shape != shape

        # 首次或强制重标定
        if self.state.H is None or force_recalibrate or shape_changed:
            return self._full_calibrate(
                image, intersections, now,
                init_H=None if shape_changed else self.state.H,
            )

        # 标定过期
        if now - self.state.timestamp > self.max_age:
            return self._full_calibrate(image, intersections, now)

        # 检查交点漂移（只有显著移动才重新标定）
        drift = self._compute_drift(intersections)
        
        # 如果漂移很小且稳定帧数 > 3，直接复用 H（跳过 RANSAC）
        if drift < 20.0 and self.state.stable_count > 3:
            self.state.intersections = intersections
            self.state.stable_count += 1
            return self._make_calibration(now)
        
        # 漂移超过阈值，重新标定
        if drift > self.drift_thresh:
            print(f"[calib] Drift {drift:.1f}px > {self.drift_thresh}px, recalibrating")
            return self._full_calibrate(image, intersections, now)

        # 中等漂移，继续复用
        self.state.intersections = intersections
        self.state.stable_count += 1
        return self._make_calibration(now)

    def _full_calibrate(
        self, image: np.ndarray, intersections: list[tuple[float, float]], now: float,
        init_H: np.ndarray | None = None,
    ) -> Calibration | None:
        """完整 RANSAC 标定，如果有上一帧的 H 则用快速模式。"""
        cal = auto_calibrate(image, court=self.court, init_H=init_H)
        if cal is None:
            return None

        H = np.array(cal.H, dtype=np.float64)
        self.state = CalibState(
            H=H,
            H_inv=np.array(cal.H_inv, dtype=np.float64),
            intersections=intersections,
            timestamp=now,
            reproj_error=cal.reproj_error_px,
            inliers=cal.keypoints_used,
            stable_count=0,
            img_shape=(image.shape[0], image.shape[1]),
        )
        return cal

    def _compute_drift(self, new_intersections: list[tuple[float, float]]) -> float:
        """计算新旧交点的平均位移。"""
        if not self.state.intersections or not new_intersections:
            return float("inf")

        old_pts = np.array(self.state.intersections, dtype=np.float64)
        new_pts = np.array(new_intersections, dtype=np.float64)

        # 简单最近邻匹配
        drifts = []
        for new_pt in new_pts:
            dists = np.linalg.norm(old_pts - new_pt, axis=1)
            drifts.append(dists.min())

        return float(np.mean(drifts)) if drifts else float("inf")

    def _make_calibration(self, now: float) -> Calibration:
        """从当前状态构建 Calibration 对象。"""
        import json
        return Calibration(
            H=self.state.H.tolist(),
            H_inv=self.state.H_inv.tolist(),
            reproj_error_px=self.state.reproj_error,
            keypoints_used=self.state.inliers,
            court_model=self.court.model,
            timestamp=now,
        )


# 全局标定器缓存：每个视频一个
_calibrators: dict[str, IncrementalCalibrator] = {}


def get_calibrator(vid: str, court: Court = FIBA) -> IncrementalCalibrator:
    """获取或创建视频的增量标定器。"""
    if vid not in _calibrators:
        _calibrators[vid] = IncrementalCalibrator(court=court)
    return _calibrators[vid]


def calibrate_frame(vid: str, image: np.ndarray, force: bool = False) -> Calibration | None:
    """便捷接口：标定一帧。"""
    calibrator = get_calibrator(vid)
    return calibrator.update(image, force_recalibrate=force)


__all__ = ["IncrementalCalibrator", "CalibState", "get_calibrator", "calibrate_frame"]
