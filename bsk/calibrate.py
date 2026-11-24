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
