"""单应标定：像素坐标 <-> 球场米制坐标。

用球场关键点（角点、三分弧、罚球线等）的像素位置与米制位置做 DLT，
求出 3x3 单应矩阵 H，使得 [u, v, 1]^T ~ H @ [x, y, 1]^T。

标定结果存到 derived/<vid>/homography.json，包含：
- H: 3x3 列表（行优先）
- H_inv: 逆矩阵
- reproj_error_px: 重投影误差（像素）
- keypoints_used: 实际使用的关键点数量
- court_model: 球场型号
- timestamp: 标定时间
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bsk.court import Court, CourtKeypoint, all_court_keypoints


@dataclass
class Calibration:
    H: list[list[float]]
    H_inv: list[list[float]]
    reproj_error_px: float
    keypoints_used: int
    court_model: str
    timestamp: float

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "H": self.H,
            "H_inv": self.H_inv,
            "reproj_error_px": self.reproj_error_px,
            "keypoints_used": self.keypoints_used,
            "court_model": self.court_model,
            "timestamp": self.timestamp,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def is_plausible(self, img_w: int, img_h: int) -> bool:
        """球场外框投影是否落在图像内的合理范围。

        Hough 交点噪声大时 RANSAC 会收敛到退化解（所有角点塌缩到一起），
        这种 H 的重投影误差甚至很小。用它做观众剔除会把球员删光，必须先挡掉。
        """
        from bsk.court import COURTS, FIBA, project_court_polygon

        try:
            poly = project_court_polygon(
                np.array(self.H_inv, dtype=np.float64),
                COURTS.get(self.court_model, FIBA),
            )
        except (np.linalg.LinAlgError, ValueError):
            return False
        if poly is None or not np.all(np.isfinite(poly)):
            return False

        # 面积占比：正常转播视角球场占画面 5%~160%（放大构图会超出画面）
        x, y = poly[:, 0], poly[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        img_area = float(img_w * img_h)
        if not (0.05 * img_area <= area <= 1.6 * img_area):
            return False

        # 跨度约束，防止塌缩成一条线
        if (x.max() - x.min()) < 0.25 * img_w or (y.max() - y.min()) < 0.08 * img_h:
            return False

        # 球场是凸矩形，投影必须近似凸（自交/扭曲的四边形象限错误）
        import cv2 as _cv2
        contour = poly.reshape(-1, 1, 2).astype(np.float32)
        poly_area = abs(_cv2.contourArea(contour))
        hull_area = abs(_cv2.contourArea(_cv2.convexHull(contour)))
        if hull_area <= 1.0 or poly_area < 0.9 * hull_area:
            return False

        # 角点不能全在画面外
        if x.max() < 0 or x.min() > img_w or y.max() < 0 or y.min() > img_h:
            return False
        return True


def calibrate(
    pixel_points: list[tuple[float, float]],
    court_points: list[tuple[float, float]],
) -> Calibration:
    """DLT 单应标定。输入两组对应点，返回 Calibration。

    pixel_points: 像素坐标 (u, v)
    court_points: 球场米制坐标 (x, y)
    """
    if len(pixel_points) < 4:
        raise ValueError(f"至少需要 4 个对应点，实际 {len(pixel_points)}")

    src = np.array(pixel_points, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.array(court_points, dtype=np.float64).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("单应矩阵求解失败")

    H_inv = np.linalg.inv(H)

    inliers = int(mask.sum()) if mask is not None else len(pixel_points)
    reproj_error = _reproj_error(H, src, dst)

    return Calibration(
        H=H.tolist(),
        H_inv=H_inv.tolist(),
        reproj_error_px=float(reproj_error),
        keypoints_used=inliers,
        court_model="FIBA",
        timestamp=time.time(),
    )


def _reproj_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    src_h = np.hstack([src.reshape(-1, 2), np.ones((len(src), 1))]).T
    dst_h = np.hstack([dst.reshape(-1, 2), np.ones((len(dst), 1))]).T
    proj = H @ src_h
    proj = proj[:2] / proj[2]
    diff = proj - dst_h[:2]
    return float(np.sqrt((diff ** 2).sum(axis=0)).mean())


def pixel_to_court(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    pt = np.array([u, v, 1.0])
    proj = H @ pt
    return float(proj[0] / proj[2]), float(proj[1] / proj[2])


def court_to_pixel(H_inv: np.ndarray, x: float, y: float) -> tuple[float, float]:
    pt = np.array([x, y, 1.0])
    proj = H_inv @ pt
    return float(proj[0] / proj[2]), float(proj[1] / proj[2])


__all__ = ["Calibration", "calibrate", "pixel_to_court", "court_to_pixel"]
