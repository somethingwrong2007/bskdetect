"""自动标定：从单帧图像自动估计相机到球场的单应矩阵。

传统 CV 路线，不需要训练数据：
1. 边缘检测 + Hough 直线 → 像素空间交点
2. RANSAC：采样交点子集，枚举到球场关键点的匹配，估计单应
3. 用内点集重新估计 H

关键：像素坐标和米制坐标在不同空间，RANSAC 必须在像素空间度量重投影误差。
"""

from __future__ import annotations

import itertools
import cv2
import numpy as np

from bsk.calibrate import Calibration, _reproj_error
from bsk.court import FIBA, Court, all_court_keypoints


def auto_calibrate(
    image: np.ndarray,
    court: Court = FIBA,
    min_intersections: int = 6,
    ransac_px: float = 15.0,
    n_iter: int = 500,
    init_H: np.ndarray | None = None,
) -> Calibration | None:
    """从单帧图像自动标定。

    Args:
        image: RGB 或 BGR 图像（numpy array）
        court: 球场几何模型
        min_intersections: 最少需要的内点交点数
        ransac_px: RANSAC 内点阈值（像素）
        n_iter: RANSAC 迭代次数
        init_H: 初始 H 矩阵（快速模式：跳过全量 RANSAC，直接验证+精化）

    Returns:
        Calibration 或 None（失败时）
    """
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]

    pixel_pts = _detect_intersections(image)
    if len(pixel_pts) < 4:
        return None

    kps = all_court_keypoints(court)
    court_pts = np.array([kp.pos for kp in kps], dtype=np.float64)

    pix_arr = np.array(pixel_pts, dtype=np.float64)
    n_pix = len(pixel_pts)

    # 快速模式：用 init_H 直接评估，跳过 RANSAC 搜索
    if init_H is not None:
        score, inlier_mask = _count_inliers(init_H, court_pts, pix_arr, ransac_px * 2)
        if score >= min_intersections:
            return _refine_and_build(init_H, inlier_mask, pix_arr, court_pts, ransac_px, court, min_intersections)
        # init_H 不够好，退回全量 RANSAC
        n_iter = min(n_iter, 100)

    best_H = None
    best_inlier_mask = None
    best_score = 0

    n_court = len(court_pts)

    for _ in range(n_iter):
        # 采样 4 个像素交点
        idx_pix = np.random.choice(n_pix, 4, replace=False)
        sample_pix = pix_arr[idx_pix]

        # 采样 4 个球场点
        idx_court = np.random.choice(n_court, 4, replace=False)
        sample_court = court_pts[idx_court]

        # 枚举 4! = 24 种匹配
        best_H_local, best_score_local, best_inlier_local = _try_permutations(
            sample_pix, sample_court, court_pts, pix_arr, ransac_px
        )

        if best_score_local > best_score:
            best_score = best_score_local
            best_H = best_H_local
            best_inlier_mask = best_inlier_local

    if best_H is None or best_score < min_intersections:
        return None

    return _refine_and_build(best_H, best_inlier_mask, pix_arr, court_pts, ransac_px, court, min_intersections)


def _refine_and_build(
    H: np.ndarray,
    inlier_mask: np.ndarray,
    pix_arr: np.ndarray,
    court_pts: np.ndarray,
    ransac_px: float,
    court_obj,
    min_intersections: int,
) -> Calibration | None:
    """用内点精化 H 并构建 Calibration。"""
    inlier_pix = pix_arr[inlier_mask]
    matched_pix, matched_court = _find_court_correspondences(H, inlier_pix, court_pts, ransac_px * 2)

    if len(matched_pix) < min_intersections:
        return None

    src = np.array(matched_court, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.array(matched_pix, dtype=np.float64).reshape(-1, 1, 2)
    H_refined, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px)
    if H_refined is None:
        return None

    H_inv = np.linalg.inv(H_refined)
    inliers = int(mask.sum()) if mask is not None else len(matched_pix)
    reproj = _reproj_error(H_refined, src, dst)

    import time
    return Calibration(
        H=H_refined.tolist(),
        H_inv=H_inv.tolist(),
        reproj_error_px=float(reproj),
        keypoints_used=inliers,
        court_model=court_obj.model,
        timestamp=time.time(),
    )


def _try_permutations(
    sample_pix: np.ndarray,
    sample_court: np.ndarray,
    all_court: np.ndarray,
    all_pix: np.ndarray,
    thresh: float,
) -> tuple[np.ndarray | None, int, np.ndarray | None]:
    """对 4 个采样点尝试所有 24 种匹配，返回最佳 H 和内点数。"""
    best_H = None
    best_score = 0
    best_mask = None

    for perm in itertools.permutations(range(4)):
        # 按排列构建对应关系
        src = sample_court[np.array(perm)].reshape(-1, 1, 2).astype(np.float64)
        dst = sample_pix.reshape(-1, 1, 2).astype(np.float64)

        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, thresh)
        if H is None:
            continue

        # 在像素空间度量所有球场点的重投影误差
        score, mask = _count_inliers(H, all_court, all_pix, thresh)
        if score > best_score:
            best_score = score
            best_H = H
            best_mask = mask

    return best_H, best_score, best_mask


def _count_inliers(
    H: np.ndarray,
    court_pts: np.ndarray,
    pixel_pts: np.ndarray,
    thresh: float,
) -> tuple[int, np.ndarray]:
    """把球场点通过 H 投到像素空间，统计与像素交点的匹配数。"""
    # court -> pixel via H_inv... 不对，H 是从 court 到 pixel 的映射
    # 实际上 findHomography(src=court, dst=pixel) 返回的 H 满足: pixel ~ H @ court
    # 所以直接把 court_pts 投影即可

    court_h = np.hstack([court_pts, np.ones((len(court_pts), 1))]).T  # 3 x N
    proj = H @ court_h  # 3 x N
    # 避免除零：w 接近零的投影无效
    w = proj[2:]
    valid = np.abs(w) > 1e-8
    proj = np.where(valid, proj[:2] / np.where(valid, w, 1.0), 1e6)  # 无效点投到远处
    proj = proj.T  # N x 2

    # 对每个投影点找最近的像素交点
    inlier_mask = np.zeros(len(pixel_pts), dtype=bool)
    for i, pp in enumerate(pixel_pts):
        dists = np.linalg.norm(proj - pp, axis=1)
        if dists.min() < thresh:
            inlier_mask[i] = True

    return int(inlier_mask.sum()), inlier_mask


def _find_court_correspondences(
    H: np.ndarray,
    pixel_pts: np.ndarray,
    court_pts: np.ndarray,
    thresh: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """对每个像素点，通过 H_inv 找到对应的球场点。返回对齐的 (matched_pix, matched_court)。"""
    H_inv = np.linalg.inv(H)
    matched_pix = []
    matched_court = []
    for pp in pixel_pts:
        pt = np.array([pp[0], pp[1], 1.0])
        proj = H_inv @ pt
        proj = proj[:2] / proj[2]
        dists = np.linalg.norm(court_pts - proj, axis=1)
        if dists.min() < thresh:
            matched_pix.append((pp[0], pp[1]))
            matched_court.append((court_pts[dists.argmin()][0], court_pts[dists.argmin()][1]))
    return matched_pix, matched_court


def _detect_intersections(image: np.ndarray) -> list[tuple[float, float]]:
    """检测图像中的直线交点。"""
    if len(image.shape) == 3:
        # 判断是 BGR 还是 RGB：不影响灰度转换
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    h, w = gray.shape[:2]

    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=min(h, w) // 10,
        maxLineGap=15,
    )

    if lines is None or len(lines) < 2:
        return []

    segments = []
    for line in lines:
        # HoughLinesP 输出形状可能是 (N, 1, 4) 或 (N, 4)
        if line.ndim == 2:
            x1, y1, x2, y2 = line[0]
        else:
            x1, y1, x2, y2 = line
        segments.append(((float(x1), float(y1)), (float(x2), float(y2))))

    return _find_intersections(segments, max(15, min(h, w) // 30))


def _find_intersections(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    min_dist: float = 20.0,
) -> list[tuple[float, float]]:
    """求所有线段对的交点，去重。"""
    pts = []
    n = len(segments)
    for i in range(n):
        for j in range(i + 1, n):
            pt = _line_intersection(segments[i], segments[j])
            if pt is not None:
                if not any(_dist(pt, p) < min_dist for p in pts):
                    pts.append(pt)
    return pts


def _line_intersection(
    seg1: tuple[tuple[float, float], tuple[float, float]],
    seg2: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float] | None:
    """求两条线段所在直线的交点。"""
    (x1, y1), (x2, y2) = seg1
    (x3, y3), (x4, y4) = seg2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if not (-0.5 < t < 1.5 and -0.5 < u < 1.5):
        return None

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return (x, y)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


__all__ = ["auto_calibrate"]
