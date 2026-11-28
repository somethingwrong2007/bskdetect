"""基于 Shi-Tomasi 特征点的球场边线检测（纯 OpenCV，无深度学习）。

检测流程
--------
1. 默认根据白色球场线估计自动 ROI；也可传入显式 mask/多边形；
2. 把 ROI 直接传给 ``goodFeaturesToTrack``，保证 ROI 外不提取角点；
3. 角点邻域 HSV 白线证据、局部 Sobel 强梯度和可选球员 mask 剔除假角点；
4. 自动模式下按“方向角 + 法向距离”聚类全部有效特征点，少于最小簇点数的簇丢弃，
   每个点簇独立 RANSAC；
5. 用最小二乘重新拟合，以 MAD 迭代剔除离群点，再校验最小线长和可选角度先验；
6. 合并邻近重复线，返回最终通过校验的拟合线、有效内点和可视化图。

``target_line`` 是保留给单帧调试的可选分支：给出后关闭自动 ROI，并使用原来的
目标线优先 RANSAC 流程。调用方也可用 ``use_roi_auto=False`` 关闭自动 ROI。
所有计算只使用当前帧，不读取视频或未来帧。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

# 固定转播机位球场地面多边形，坐标为相对图像宽高的 [0,1] 比例。
# 可通过 fixed_roi_polygon / --roi-polygon 按实际机位覆盖；None 表示关闭固定 ROI。
DEFAULT_FIXED_COURT_POLYGON = (
    (0.10, 0.23), (0.90, 0.23), (0.97, 0.92), (0.03, 0.92),
)

if TYPE_CHECKING:
    from bsk.perceive.court_keypoints import DetectedKeypoint


@dataclass(frozen=True)
class LineFeatureConfig:
    """边线特征检测参数；三个核心调试项依次为角点数、质量阈值、距离阈值。"""

    max_corners: int = 700          # Shi-Tomasi 最多返回多少个角点
    quality_level: float = 0.025    # Shi-Tomasi 质量阈值，越小候选越多
    distance_threshold: float = 5.0 # RANSAC 点到直线的最大距离（原图像素）
    min_distance: int = 7           # 角点之间的最小像素间距
    block_size: int = 3             # Shi-Tomasi 邻域尺寸
    use_harris: bool = False
    harris_k: float = 0.04
    border_margin: int = 3

    use_roi_auto: bool = True         # 无显式 ROI 且无 target_line 时自动估计球场区域
    use_fixed_roi: bool = True         # 优先使用固定球场地面多边形，避免观众席角点
    fixed_roi_polygon: tuple[tuple[float, float], ...] | None = DEFAULT_FIXED_COURT_POLYGON
    auto_roi_dilate: float = 0.035     # 自动 ROI 外扩比例，按图像短边计算
    hard_gradient_filter: bool = True  # 自动模式只保留白线/场地强梯度角点
    gradient_neighborhood_size: int = 5  # 梯度局部最大值窗口，必须为奇数
    min_gradient_strength: float = 0.10  # 局部梯度强度下限，按图像高分位归一化
    use_floor_support: bool = True     # 有场地颜色证据时排除观众席和球员前景
    floor_neighborhood_size: int = 7   # 角点邻域场地颜色证据窗口，必须为奇数
    min_floor_support: float = 0.08    # 邻域内场地颜色像素比例下限
    floor_roi_dilate_frac: float = 0.020  # 固定 ROI 与场地 mask 的外扩比例

    use_clahe: bool = True
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    white_threshold: int = 145      # HSV 白线亮度阈值 V
    max_saturation: int = 90        # HSV 白线低饱和度阈值 S
    white_neighborhood_size: int = 5  # 角点邻域白线证据窗口，必须为奇数
    min_point_evidence: float = 0.35  # 角点邻域白线像素比例下限；target_line 可放宽暗线

    ransac_iterations: int = 300
    min_cluster_points: int = 5   # 聚类阶段的最小点数，小于该值直接丢弃整簇
    min_line_points: int = 8      # RANSAC/最终直线的最小内点数
    min_line_length: float = 80.0 # 最终可见拟合线段的最小长度（原图像素）
    min_line_support: float = 0.38      # 白线/梯度证据下限，抑制纯噪声随机共线
    min_span_frac: float = 0.055    # 内点跨度至少占图像对角线的比例
    min_span_bins: int = 3          # 内点至少覆盖 6 个纵向区间中的 N 个
    line_angle_prior_deg: tuple[float, ...] = (0.0, 90.0)  # 默认剔除明显斜线；可清空关闭
    angle_prior_tolerance_deg: float = 35.0       # 与水平/垂直轴的最大允许偏差
    max_lines: int = 10
    merge_angle_deg: float = 8.0
    merge_distance_factor: float = 3.0
    cluster_angle_deg: float = 2.0       # 自动模式角度聚类粒度
    cluster_distance_factor: float = 1.5 # 自动模式法向距离聚类半径 / distance_threshold
    intersection_max_line_distance: float = 12.0  # 交点到两条源直线的最大距离
    intersection_cluster_distance: float = 25.0   # 同一点交点的聚类半径
    min_intersection_angle_deg: float = 10.0       # 近平行线不求交，避免远端噪声
    seed: int = 20261127           # 固定种子使同一帧的调试结果可复现

    def __post_init__(self) -> None:
        if self.max_corners < 2:
            raise ValueError("max_corners 必须 >= 2")
        if not 0.0 < self.quality_level <= 1.0:
            raise ValueError("quality_level 必须在 (0, 1] 内")
        if self.distance_threshold <= 0:
            raise ValueError("distance_threshold 必须 > 0")
        if self.min_distance < 1 or self.block_size < 3 or self.border_margin < 0:
            raise ValueError("min_distance/block_size/border_margin 取值不合法")
        if self.auto_roi_dilate < 0:
            raise ValueError("auto_roi_dilate 必须 >= 0")
        if not 0.0 < self.min_gradient_strength <= 1.0:
            raise ValueError("min_gradient_strength 必须在 (0, 1] 内")
        for name, value in (("gradient_neighborhood_size", self.gradient_neighborhood_size),
                            ("white_neighborhood_size", self.white_neighborhood_size),
                            ("floor_neighborhood_size", self.floor_neighborhood_size)):
            if value < 1 or value > 31 or value % 2 == 0:
                raise ValueError(f"{name} 必须是 [1, 31] 内的奇数")
        if not 0.0 <= self.min_floor_support <= 1.0:
            raise ValueError("min_floor_support 必须在 [0, 1] 内")
        if not 0.0 <= self.floor_roi_dilate_frac <= 0.20:
            raise ValueError("floor_roi_dilate_frac 必须在 [0, 0.20] 内")
        if not 0.0 < self.cluster_angle_deg < 90.0:
            raise ValueError("cluster_angle_deg 必须在 (0, 90) 内")
        if not 0.0 < self.cluster_distance_factor:
            raise ValueError("cluster_distance_factor 必须 > 0")
        if self.merge_angle_deg < 0 or self.merge_distance_factor <= 0:
            raise ValueError("merge_angle_deg 必须 >= 0，merge_distance_factor 必须 > 0")
        if self.ransac_iterations < 1 or self.max_lines < 1:
            raise ValueError("ransac_iterations 和 max_lines 必须 >= 1")
        if not 0.0 <= self.min_line_support <= 1.0:
            raise ValueError("min_line_support 必须在 [0, 1] 内")
        if not 0.0 <= self.min_point_evidence <= 1.0:
            raise ValueError("min_point_evidence 必须在 [0, 1] 内")
        if not 1 <= self.min_span_bins <= 6:
            raise ValueError("min_span_bins 必须在 [1, 6] 内")
        if self.clahe_grid < 1 or self.clahe_clip <= 0:
            raise ValueError("clahe_grid 必须 >= 1，clahe_clip 必须 > 0")
        if not (0 <= self.white_threshold <= 255 and 0 <= self.max_saturation <= 255):
            raise ValueError("白线阈值必须在 [0, 255] 内")
        if self.min_cluster_points < 2:
            raise ValueError("min_cluster_points 必须 >= 2")
        if self.min_line_points < 2:
            raise ValueError("min_line_points 必须 >= 2")
        if not math.isfinite(self.min_line_length) or self.min_line_length <= 0:
            raise ValueError("min_line_length 必须 > 0")
        if not 0.0 <= self.angle_prior_tolerance_deg <= 90.0:
            raise ValueError("angle_prior_tolerance_deg 必须在 [0, 90] 内")
        if any(not math.isfinite(float(angle)) for angle in self.line_angle_prior_deg):
            raise ValueError("line_angle_prior_deg 必须是有限角度")
        if self.fixed_roi_polygon is not None:
            polygon = np.asarray(self.fixed_roi_polygon, np.float64)
            if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
                    or not np.all(np.isfinite(polygon))
                    or np.any(polygon < 0.0) or np.any(polygon > 1.0)):
                raise ValueError("fixed_roi_polygon 必须是 [0,1] 范围内的 Nx2 多边形")
        if not 0.0 < self.min_span_frac <= 1.0:
            raise ValueError("min_span_frac 必须在 (0, 1] 内")
        if not math.isfinite(self.intersection_max_line_distance) or self.intersection_max_line_distance < 0:
            raise ValueError("intersection_max_line_distance 必须 >= 0")
        if not math.isfinite(self.intersection_cluster_distance) or self.intersection_cluster_distance <= 0:
            raise ValueError("intersection_cluster_distance 必须 > 0")
        if not 0.0 <= self.min_intersection_angle_deg < 90.0:
            raise ValueError("min_intersection_angle_deg 必须在 [0, 90) 内")


@dataclass
class LineFeaturePoint:
    """原图像素坐标下的一个有效 Shi-Tomasi 角点。"""

    x: float
    y: float
    response: float
    evidence: float = 0.0
    gradient: float = 0.0
    line_id: int = -1
    distance_to_line: float = math.inf

    def to_dict(self) -> dict:
        return {
            "x": round(float(self.x), 3),
            "y": round(float(self.y), 3),
            "response": round(float(self.response), 6),
            "evidence": round(float(self.evidence), 4),
            "gradient": round(float(self.gradient), 4),
            "line_id": int(self.line_id),
            "distance_to_line": (round(float(self.distance_to_line), 4)
                                 if math.isfinite(self.distance_to_line) else None),
        }


@dataclass
class FittedCourtLine:
    """一条鲁棒拟合的直线；p0/p1 是裁剪到图像范围后的可视化端点。"""

    line_id: int
    p0: tuple[float, float]
    p1: tuple[float, float]
    direction: tuple[float, float]
    normal: tuple[float, float]
    points: list[LineFeaturePoint] = field(default_factory=list)
    rms_error: float = 0.0
    span: float = 0.0
    support: float = 0.0
    score: float = 0.0
    confidence: float = 0.0

    @property
    def inlier_count(self) -> int:
        return len(self.points)

    def to_dict(self) -> dict:
        return {
            "line_id": self.line_id,
            "p0": [round(float(v), 3) for v in self.p0],
            "p1": [round(float(v), 3) for v in self.p1],
            "direction": [round(float(v), 6) for v in self.direction],
            "normal": [round(float(v), 6) for v in self.normal],
            "inlier_count": self.inlier_count,
            "rms_error": round(float(self.rms_error), 4),
            "span": round(float(self.span), 3),
            "support": round(float(self.support), 4),
            "score": round(float(self.score), 4),
            "confidence": round(float(self.confidence), 4),
            "points": [p.to_dict() for p in self.points],
        }


@dataclass
class LineIntersection:
    """两条无限延长拟合直线的交点，坐标始终为原图像素。"""

    x: float
    y: float
    confidence: float
    line_ids: tuple[int, int]
    distance_first: float = 0.0
    distance_second: float = 0.0

    def to_dict(self) -> dict:
        return {
            "x": round(float(self.x), 3),
            "y": round(float(self.y), 3),
            "confidence": round(float(self.confidence), 4),
            "line_ids": [int(v) for v in self.line_ids],
            "distance_first": round(float(self.distance_first), 4),
            "distance_second": round(float(self.distance_second), 4),
        }


@dataclass
class LineFeatureResult:
    lines: list[FittedCourtLine] = field(default_factory=list)
    points: list[LineFeaturePoint] = field(default_factory=list)
    intersections: list[LineIntersection] = field(default_factory=list)
    intersection_keypoints: list[DetectedKeypoint] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    visualization: np.ndarray | None = None

    @property
    def inlier_points(self) -> list[LineFeaturePoint]:
        return [p for p in self.points if p.line_id >= 0]

    def to_dict(self) -> dict:
        return {
            "lines": [line.to_dict() for line in self.lines],
            "points": [p.to_dict() for p in self.points],
            "intersections": [item.to_dict() for item in self.intersections],
            "intersection_keypoints": [
                {
                    "name": kp.name,
                    "pixel": [round(float(kp.pixel[0]), 3), round(float(kp.pixel[1]), 3)],
                    "confidence": round(float(kp.confidence), 4),
                    "world": list(kp.world) if kp.world is not None else None,
                    "inlier": bool(kp.inlier),
                }
                for kp in self.intersection_keypoints
            ],
            "stats": self.stats,
        }


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("输入图像为空")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("输入图像必须是灰度、BGR 或 BGRA numpy 数组")
    if gray.dtype.kind == "f" and not np.all(np.isfinite(gray)):
        raise ValueError("输入图像包含 NaN 或 Inf")
    return gray


def _roi_mask(shape: tuple[int, int], roi, name: str = "roi") -> np.ndarray | None:
    """roi 支持二值掩码或 Nx2 多边形；None 表示全图。"""
    if roi is None:
        return None
    h, w = shape
    arr = np.asarray(roi)
    if arr.ndim == 2 and arr.shape == (h, w):
        return (arr > 0).astype(np.uint8)
    if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 3:
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.rint(arr).astype(np.int32)], 1)
        return mask
    raise ValueError(f"{name} 必须是 (h,w) 二值掩码或 Nx2 多边形")


def _dominant_floor_mask(hsv: np.ndarray, *, include_upper: bool = False) -> tuple[np.ndarray, int]:
    """从画面下半部学习主场地颜色，返回保守的场地像素 mask。

    只在下半部采样，避免观众席决定颜色模型；要求主色占有足够比例才启用。
    返回的是实际颜色证据而不是凸包，广告牌和看台不会仅因位于两条线之间就被填入。
    颜色不明显（如灰色合成图）时返回空 mask，由白线凸包分支兜底。
    """
    h, w = hsv.shape[:2]
    short = max(1, min(h, w))
    y0, y1 = int(h * 0.66), max(int(h * 0.66) + 1, int(h * 0.92))
    x0, x1 = int(w * 0.06), max(int(w * 0.06) + 1, int(w * 0.94))
    sample = hsv[y0:y1, x0:x1]
    hue, saturation, value = (sample[:, :, i] for i in range(3))
    eligible = (saturation >= 25) & (saturation <= 230) & (value >= 35) & (value <= 242)
    if int(eligible.sum()) < max(32, h * w // 200):
        return np.zeros((h, w), np.uint8), 0

    hist = np.bincount((hue[eligible] * 18 // 180).astype(np.int32), minlength=18)
    peak = int(np.argmax(hist))
    if float(hist[peak]) / float(eligible.sum()) < 0.08:
        return np.zeros((h, w), np.uint8), 0
    center_hue = peak * 10.0 + 5.0
    median_s = float(np.median(saturation[eligible]))
    median_v = float(np.median(value[eligible]))
    hue_delta = np.abs(hsv[:, :, 0].astype(np.float32) - center_hue)
    hue_delta = np.minimum(hue_delta, 180.0 - hue_delta)
    floor = (
        (hue_delta <= 13.0) &
        (hsv[:, :, 1] >= max(45.0, median_s - 80.0)) &
        (hsv[:, :, 1] <= min(220.0, median_s + 60.0)) &
        (hsv[:, :, 2] >= max(55.0, median_v - 65.0)) &
        (hsv[:, :, 2] <= min(235.0, median_v + 65.0))
    ).astype(np.uint8)
    # 自动 ROI 分支忽略上半部，避免灯光和记分牌；固定转播 ROI 需要保留
    # 与下半部场地同色且连通的远端场地，因此可显式包含上半部。
    if not include_upper:
        floor[:int(h * 0.38)] = 0
    kernel_size = max(3, short // 85)
    if kernel_size % 2 == 0:
        kernel_size += 1
    floor = cv2.morphologyEx(
        floor, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    )
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(floor, connectivity=8)
    selected = []
    for label in range(1, count):
        _x, y, _bw, bh, area = (int(v) for v in stats[label])
        if area < max(24, int(h * w * 0.003)):
            continue
        if centroids[label, 1] < h * 0.45 or y + bh < h * 0.70:
            continue
        selected.append(label)
    if not selected:
        return np.zeros((h, w), np.uint8), 0
    mask = np.isin(labels, selected).astype(np.uint8)
    if float(mask.mean()) < 0.025:
        return np.zeros((h, w), np.uint8), 0
    return mask, len(selected)


def _foreground_obstacle_mask(floor_mask: np.ndarray, court_mask: np.ndarray) -> np.ndarray:
    """提取场地颜色之外的块状前景，用于排除球员、广告和观众席。

    白色球场线很细，先做椭圆开运算消除；保留下来的连通域还必须同时具有
    足够面积和最小包围盒边长，避免木地板纹理碎块被误判为前景。
    """
    h, w = floor_mask.shape
    short = max(1, min(h, w))
    size = max(5, int(round(short / 60.0)))
    if size % 2 == 0:
        size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    foreground = ((floor_mask == 0) & (court_mask > 0)).astype(np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    obstacles = np.zeros((h, w), np.uint8)
    min_area = max(32, int(round(h * w * 0.0004)))
    min_side = max(4.0, short * 0.012)
    for label in range(1, count):
        _x, _y, bw, bh, area = (int(v) for v in stats[label])
        if area >= min_area and min(bw, bh) >= min_side:
            obstacles[_labels == label] = 1
    # 只轻微外扩，覆盖角点落在前景边界外 1~2px 的情况。
    if obstacles.any():
        obstacles = cv2.dilate(obstacles, np.ones((3, 3), np.uint8))
    return obstacles


def _auto_court_mask(
    white: np.ndarray, cfg: LineFeatureConfig, hsv: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """由细长白色球场线估计自动 ROI，返回 mask 和采用的连通域数量。

    有明确场地颜色时，用“实际场地像素 + 与场地相连的主白线簇”限制 ROI，避免
    看台白字和广告牌把凸包撑到全图。颜色不明显时才退回细长白线连通域凸包。
    所有估计只使用当前帧；自动结果不可信时可由调用方传入显式球场 mask。
    """
    h, w = white.shape
    short = max(1, min(h, w))
    kernel_size = max(3, min(31, short // 70))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(white.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    # 不做 3x3 开运算：缩放后的球场线可能只有 1px，开运算会把整条线删掉。
    # 小白点由后续连通域面积和长宽条件剔除。
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if count <= 1:
        return np.zeros((h, w), np.uint8), 0

    min_area = max(10, int(round(h * w * 0.000035)))
    candidates = []
    for label in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[label])
        bbox_area = max(1, bw * bh)
        fill_ratio = area / bbox_area
        longest = max(bw, bh)
        # 球场线/弧是细长结构；实心广告牌、看台服装块和大面积白 floor 不应参与凸包。
        if area < min_area or longest < max(12.0, short * 0.025):
            continue
        if fill_ratio > 0.48:
            continue
        elongation = longest / max(1.0, min(bw, bh))
        candidates.append((label, area * (1.0 + min(elongation, 12.0))))
    if not candidates:
        return np.zeros((h, w), np.uint8), 0

    candidates.sort(key=lambda item: -item[1])
    radius = max(2, int(round(short * cfg.auto_roi_dilate)))
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    floor_mask, floor_components = (
        _dominant_floor_mask(hsv) if hsv is not None else (np.zeros_like(white), 0)
    )
    if floor_components:
        best_label = candidates[0][0]
        floor_ys, floor_xs = np.nonzero(floor_mask)
        floor_top = int(floor_ys.min())
        line_core = (labels == best_label)
        # 上方只保留与主场地相近的白线走廊；下方由场地颜色 mask 约束，
        # 避免场边广告牌的白色边框把 ROI 拉到观众席。看台文字不能凭凸包进入。
        upper_limit = max(int(h * 0.28), floor_top - int(short * 0.22))
        allowed = np.zeros((h, w), bool)
        allowed[upper_limit:floor_top, :] = True
        allowed &= (
            (np.arange(w)[None, :] >= max(0, int(floor_xs.min()) - radius)) &
            (np.arange(w)[None, :] <= min(w - 1, int(floor_xs.max()) + radius))
        )
        line_core &= allowed
        if not line_core.any():
            return np.zeros((h, w), np.uint8), 0
        corridor_radius = max(2, int(round(short * 0.012)))
        floor_radius = max(2, int(round(short * min(cfg.auto_roi_dilate, 0.015))))
        corridor_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (corridor_radius * 2 + 1, corridor_radius * 2 + 1)
        )
        floor_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (floor_radius * 2 + 1, floor_radius * 2 + 1)
        )
        mask = cv2.dilate(floor_mask, floor_kernel) | cv2.dilate(
            line_core.astype(np.uint8), corridor_kernel
        )
        if 0.02 < float(mask.mean()) < 0.88:
            return mask, floor_components + 1

    best_score = candidates[0][1]
    selected = [label for label, score in candidates if score >= best_score * 0.055]
    line_pixels = np.isin(labels, selected)
    ys, xs = np.nonzero(line_pixels)
    hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    mask = cv2.dilate(mask, dilate_kernel)
    return mask, len(selected)


def _gradient_strength_map(gray: np.ndarray, mask: np.ndarray | None,
                          neighborhood_size: int = 5) -> np.ndarray:
    """Sobel 梯度幅值的局部最大值，按 ROI 内 95 分位归一化。"""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    local = cv2.dilate(magnitude, np.ones((neighborhood_size, neighborhood_size), np.uint8))
    values = local[mask > 0] if mask is not None and mask.any() else local.reshape(-1)
    scale = float(np.percentile(values, 95)) if len(values) else 0.0
    if not math.isfinite(scale) or scale < 1e-6:
        scale = max(float(local.max()), 1e-6)
    return np.clip(local / scale, 0.0, 1.0).astype(np.float32)


def _line_geometry(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """总最小二乘拟合，返回直线上一点、单位方向、单位法线。"""
    vx, vy, x0, y0 = cv2.fitLine(
        points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
    ).reshape(-1)
    direction = np.asarray([vx, vy], np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise ValueError("退化直线")
    direction /= norm
    normal = np.asarray([-direction[1], direction[0]], np.float64)
    return np.asarray([x0, y0], np.float64), direction, normal


def _distances(points: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return np.abs((points - origin) @ normal)


def _sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """双线性采样，避免直线证据只在整数像素处跳动。"""
    return cv2.remap(
        image, points[:, 0].astype(np.float32).reshape(-1, 1),
        points[:, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).reshape(-1)


def _box_mean(image: np.ndarray, points: np.ndarray, size: int) -> np.ndarray:
    """以每个角点为中心采样奇数窗口均值，用于邻域白线/梯度证据。"""
    if size <= 1:
        return _sample(image, points)
    h, w = image.shape
    radius = size // 2
    cx = np.rint(points[:, 0]).astype(np.int32)
    cy = np.rint(points[:, 1]).astype(np.int32)
    x0, x1 = np.clip(cx - radius, 0, w - 1), np.clip(cx + radius, 0, w - 1)
    y0, y1 = np.clip(cy - radius, 0, h - 1), np.clip(cy + radius, 0, h - 1)
    integral = cv2.integral(image.astype(np.float64, copy=False))
    total = (integral[y1 + 1, x1 + 1] - integral[y0, x1 + 1] -
             integral[y1 + 1, x0] + integral[y0, x0])
    area = np.maximum(1, (x1 - x0 + 1) * (y1 - y0 + 1))
    return total / area


def _clip_line(origin: np.ndarray, direction: np.ndarray,
               shape: tuple[int, int]) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """把无限直线裁剪到图像边框，输出稳定、有限的绘图端点。"""
    h, w = shape
    lo, hi = -math.inf, math.inf
    for value, delta, upper in ((origin[0], direction[0], w - 1.0),
                                (origin[1], direction[1], h - 1.0)):
        if abs(delta) < 1e-12:
            if value < 0 or value > upper:
                return None
            continue
        a, b = (0.0 - value) / delta, (upper - value) / delta
        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    if hi <= lo:
        return None
    a, b = origin + lo * direction, origin + hi * direction
    return ((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))


def _clip_line_to_mask(
    origin: np.ndarray, direction: np.ndarray,
    shape: tuple[int, int], mask: np.ndarray | None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """把直线裁剪到 ROI 内最长连续段，避免可视化外延穿过观众席。"""
    clipped = _clip_line(origin, direction, shape)
    if clipped is None:
        return None
    if mask is None or not mask.any():
        return clipped
    start, end = (np.asarray(point, np.float64) for point in clipped)
    length = float(np.linalg.norm(end - start))
    count = max(2, int(math.ceil(length)) + 1)
    samples = start + np.linspace(0.0, 1.0, count)[:, None] * (end - start)
    xi = np.clip(np.rint(samples[:, 0]).astype(int), 0, shape[1] - 1)
    yi = np.clip(np.rint(samples[:, 1]).astype(int), 0, shape[0] - 1)
    inside = mask[yi, xi] > 0
    if not inside.any():
        return None
    # 选择最长连续 True 区间；简单取首末 True 会跨过 ROI 中的看台/广告空洞。
    padded = np.concatenate([[False], inside, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    begin, finish = int(changes[0]), int(changes[-1])
    best_length = -1
    for a, b in zip(changes[::2], changes[1::2]):
        if int(b - a) > best_length:
            begin, finish, best_length = int(a), int(b), int(b - a)
    if best_length < 2:
        return None
    # 向连续区间内部收回一个采样点，保证输出端点仍严格位于 mask 内，
    # 同时保持端点在原拟合直线上（不做坐标取整）。
    return (
        (float(samples[begin + 1, 0]), float(samples[begin + 1, 1])),
        (float(samples[finish - 2, 0]), float(samples[finish - 2, 1])),
    )


def _refine_line(points: np.ndarray, cfg: LineFeatureConfig,
                 evidence: np.ndarray, gradient: np.ndarray) -> FittedCourtLine | None:
    """MAD 迭代剔除离群点，并计算跨度、图像证据和拟合质量。"""
    if len(points) < cfg.min_line_points:
        return None
    work = np.asarray(points, np.float64)
    origin = direction = normal = None
    keep = np.ones(len(work), bool)
    for _ in range(5):
        if int(keep.sum()) < cfg.min_line_points:
            return None
        try:
            origin, direction, normal = _line_geometry(work[keep])
        except (cv2.error, ValueError):
            return None
        dist = _distances(work, origin, normal)
        median = float(np.median(dist[keep]))
        mad = float(np.median(np.abs(dist[keep] - median)))
        # 点到直线的残差服从半正态分布，直接对“绝对残差”求 MAD 会把
        # 阈值收得过小。用中位数估计线宽，MAD 仅作为下限，二者取较大值。
        threshold = min(
            cfg.distance_threshold,
            max(2.0, 1.8 * median, 2.0 * 1.4826 * mad),
        )
        new_keep = dist <= threshold
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    if int(keep.sum()) < cfg.min_line_points:
        return None
    inliers = work[keep]
    try:
        origin, direction, normal = _line_geometry(inliers)
    except (cv2.error, ValueError):
        return None
    dist = _distances(inliers, origin, normal)
    along = (inliers - origin) @ direction
    span = float(np.percentile(along, 95) - np.percentile(along, 5))
    if not math.isfinite(span) or span < max(cfg.min_line_length, 1e-6):
        return None

    # 图像证据：白线颜色 + 梯度方向。两者都只用于候选打分，避免反光时硬过滤真实点。
    sample_n = 31
    sample_t = np.linspace(float(along.min()), float(along.max()), sample_n)
    samples = origin[None, :] + sample_t[:, None] * direction[None, :]
    white_score = float(np.mean(_sample(evidence, samples)))
    gx = _sample(gradient[:, :, 0], samples)
    gy = _sample(gradient[:, :, 1], samples)
    mag = np.hypot(gx, gy)
    alignment = np.divide(np.abs(gx * normal[0] + gy * normal[1]), mag,
                          out=np.zeros_like(mag), where=mag > 1e-6)
    gradient_score = float(np.mean(alignment)) if len(alignment) else 0.0
    support = float(np.clip(0.60 * white_score + 0.40 * gradient_score, 0.0, 1.0))
    clipped = _clip_line(origin, direction, evidence.shape)
    if clipped is None:
        return None
    rms = float(np.sqrt(np.mean(dist ** 2)))
    return FittedCourtLine(
        line_id=-1, p0=clipped[0], p1=clipped[1],
        direction=(float(direction[0]), float(direction[1])),
        normal=(float(normal[0]), float(normal[1])),
        rms_error=rms, span=span, support=support,
    )


def _angle_delta(first: tuple[float, float] | np.ndarray,
                 second: tuple[float, float] | np.ndarray) -> float:
    """返回两条无向直线的最小夹角，范围 [0, 90]。"""
    cosine = abs(float(np.dot(np.asarray(first), np.asarray(second))))
    return math.degrees(math.acos(float(np.clip(cosine, 0.0, 1.0))))


def _angle_prior_accepts(direction: tuple[float, float] | np.ndarray,
                         cfg: LineFeatureConfig) -> bool:
    """可选球场线角度白名单；透视画面未提供先验时不猜测角度。"""
    if not cfg.line_angle_prior_deg:
        return True
    angle = math.degrees(math.atan2(float(direction[1]), float(direction[0]))) % 180.0
    for prior in cfg.line_angle_prior_deg:
        prior = float(prior) % 180.0
        delta = abs(angle - prior)
        delta = min(delta, 180.0 - delta)
        if delta <= cfg.angle_prior_tolerance_deg:
            return True
    return False


def _validate_final_line(line: FittedCourtLine, cfg: LineFeatureConfig,
                         shape: tuple[int, int], mask: np.ndarray | None,
                         *, check_angle_prior: bool) -> str | None:
    """RANSAC 后校验并裁剪最终线段；返回拒绝原因，None 表示通过。"""
    if line.span < cfg.min_line_length:
        return "short_line"
    if check_angle_prior and not _angle_prior_accepts(line.direction, cfg):
        return "angle_prior"
    origin = np.asarray([
        line.p0[0] + (line.p1[0] - line.p0[0]) * 0.5,
        line.p0[1] + (line.p1[1] - line.p0[1]) * 0.5,
    ], np.float64)
    clipped = _clip_line_to_mask(origin, np.asarray(line.direction, np.float64),
                                 shape, mask)
    if clipped is None:
        return "roi_span"
    if float(np.linalg.norm(np.asarray(clipped[1]) - np.asarray(clipped[0]))) < cfg.min_line_length:
        return "short_line"
    line.p0, line.p1 = clipped
    return None


def _model_stats(points: np.ndarray, origin: np.ndarray, direction: np.ndarray,
                 normal: np.ndarray, cfg: LineFeatureConfig,
                 diagonal: float) -> tuple[bool, float, int]:
    dist = _distances(points, origin, normal)
    inliers = dist <= cfg.distance_threshold
    count = int(inliers.sum())
    if count < cfg.min_line_points:
        return False, 0.0, count
    along = (points[inliers] - origin) @ direction
    span = float(along.max() - along.min())
    if span < max(12.0, cfg.min_span_frac * diagonal):
        return False, 0.0, count
    # 防止一条线只由端点附近的一簇角点撑起。
    bins = np.clip(((along - along.min()) / max(span, 1e-6) * 6).astype(int), 0, 5)
    occupied = len(np.unique(bins))
    if occupied < cfg.min_span_bins:
        return False, 0.0, count
    score = count * (0.75 + 0.25 * span / diagonal) + 4.0 * occupied
    return True, float(score), count


def _hypothesis_support(points: np.ndarray, origin: np.ndarray,
                        direction: np.ndarray, normal: np.ndarray,
                        cfg: LineFeatureConfig, evidence: np.ndarray,
                        gradient: np.ndarray) -> float:
    """只给几何上可行的假设计算白线/梯度证据，避免对所有 RANSAC 假设采样。"""
    inliers = points[_distances(points, origin, normal) <= cfg.distance_threshold]
    if len(inliers) < cfg.min_line_points:
        return 0.0
    along = (inliers - origin) @ direction
    sample_t = np.linspace(float(along.min()), float(along.max()), 17)
    samples = origin[None, :] + sample_t[:, None] * direction[None, :]
    white_score = float(np.mean(_sample(evidence, samples)))
    gx = _sample(gradient[:, :, 0], samples)
    gy = _sample(gradient[:, :, 1], samples)
    mag = np.hypot(gx, gy)
    alignment = np.divide(np.abs(gx * normal[0] + gy * normal[1]), mag,
                          out=np.zeros_like(mag), where=mag > 1e-6)
    gradient_score = float(np.mean(alignment)) if len(alignment) else 0.0
    return float(np.clip(0.60 * white_score + 0.40 * gradient_score, 0.0, 1.0))


def _angle_distance_clusters(xy: np.ndarray, cfg: LineFeatureConfig,
                             diagonal: float, rng: np.random.Generator
                             ) -> tuple[list[np.ndarray], int]:
    """按直线方向角和法向距离聚类，返回各簇索引和因点数不足丢弃的簇数。

    点对投票同时约束角度与到直线的距离，比只按角度聚类更能排除平行看台线。
    点数较大时确定性采样点对，限制内存；默认 700 个角点使用全部点对。
    """
    n = len(xy)
    if n < cfg.min_cluster_points:
        return [], 0
    pair_count = n * (n - 1) // 2
    if pair_count <= 300_000:
        pi, pj = np.triu_indices(n, 1)
    else:
        pi = rng.integers(0, n, 300_000)
        pj = rng.integers(0, n, 300_000)
        valid = pi != pj
        pi, pj = pi[valid], pj[valid]

    delta = xy[pj] - xy[pi]
    length = np.linalg.norm(delta, axis=1)
    valid = length >= max(12.0, diagonal * 0.012)
    if not np.any(valid):
        return [], 0
    pi, pj, delta, length = pi[valid], pj[valid], delta[valid], length[valid]
    direction = delta / length[:, None]
    # 统一到 [0, pi)，避免同一条无向直线因端点顺序被拆成两个方向。
    flip = (direction[:, 1] < 0) | ((np.abs(direction[:, 1]) < 1e-12) &
                                   (direction[:, 0] < 0))
    direction[flip] *= -1.0
    angle = np.mod(np.arctan2(direction[:, 1], direction[:, 0]), np.pi)
    normal = np.column_stack([-direction[:, 1], direction[:, 0]])
    rho = np.einsum("ij,ij->i", xy[pi], normal)

    angle_step = max(0.5, cfg.cluster_angle_deg)
    distance_step = max(1.0, cfg.distance_threshold * cfg.cluster_distance_factor)
    angle_bin = np.floor(np.degrees(angle) / angle_step).astype(np.int32)
    rho_bin = np.floor(rho / distance_step).astype(np.int32)
    # 结构化 dtype 避免把角度和距离两个整数手工打包成可能冲突的 int64。
    keys = np.rec.fromarrays([angle_bin, rho_bin], names=["angle", "rho"])
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)
    order = order[counts[order] >= 2]

    radius = cfg.distance_threshold * cfg.cluster_distance_factor
    # 先在稀疏 (角度, 距离) 网格上做 NMS，再计算完整点簇。若先截断票数最高的
    # 几百个 bin，同一条长直线会占满名额，导致其它方向的边线根本无法进入聚类。
    peak_cells: list[int] = []
    angle_bin_limit = max(1, int(math.ceil(2.0 * cfg.cluster_angle_deg / angle_step)))
    rho_bin_limit = max(1, int(math.ceil(2.0 * radius / distance_step)))
    for cell in order:
        cell_angle = int(unique[cell]["angle"])
        cell_rho = int(unique[cell]["rho"])
        repeated = any(
            abs(cell_angle - int(unique[old]["angle"])) <= angle_bin_limit and
            abs(cell_rho - int(unique[old]["rho"])) <= rho_bin_limit
            for old in peak_cells
        )
        if repeated:
            continue
        peak_cells.append(int(cell))
        if len(peak_cells) >= max(64, cfg.max_lines * 6):
            break

    clusters: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    rejected_clusters = 0
    for cell in peak_cells:
        members = inverse == cell
        seed_angle = float(np.median(angle[members]))
        seed_rho = float(np.median(rho[members]))
        seed_direction = np.asarray([math.cos(seed_angle), math.sin(seed_angle)])
        seed_normal = np.asarray([-seed_direction[1], seed_direction[0]])
        seed_origin = seed_normal * seed_rho
        dist = _distances(xy, seed_origin, seed_normal)
        indices = np.flatnonzero(dist <= radius)
        if len(indices) < cfg.min_cluster_points:
            rejected_clusters += 1
            continue
        along = (xy[indices] - seed_origin) @ seed_direction
        if float(along.max() - along.min()) < max(12.0, cfg.min_span_frac * diagonal):
            rejected_clusters += 1
            continue
        repeated = False
        for _old_score, old_direction, old_normal, old_origin, _old_indices in clusters:
            cosine = abs(float(np.dot(old_direction, seed_direction)))
            angle_delta = math.degrees(math.acos(float(np.clip(cosine, 0.0, 1.0))))
            offset = abs(float(np.dot(old_origin - seed_origin, old_normal)))
            if (angle_delta <= cfg.cluster_angle_deg * 2.0 and
                    offset <= radius * 2.0):
                repeated = True
                break
        if repeated:
            continue
        clusters.append((float(len(indices)), seed_direction, seed_normal,
                         seed_origin, indices))

    clusters.sort(key=lambda item: -item[0])
    return [item[-1] for item in clusters[:max(32, cfg.max_lines * 4)]], rejected_clusters


def _fit_cluster_ransac(indices: np.ndarray, xy: np.ndarray, response: np.ndarray,
                        white: np.ndarray, gradient: np.ndarray,
                        cfg: LineFeatureConfig, diagonal: float,
                        seed: int) -> tuple[FittedCourtLine, np.ndarray] | None:
    """只在单个角度距离簇内执行 RANSAC，避免不同边线互相提供伪内点。"""
    cluster = xy[indices]
    if len(cluster) < cfg.min_line_points:
        return None
    rng = np.random.default_rng(seed)
    trials: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    try:
        trials.append(_line_geometry(cluster))
    except (cv2.error, ValueError):
        pass
    weights = response[indices] * (0.05 + 0.95 * _sample(white, cluster)) + 1e-6
    weights /= weights.sum()
    for _ in range(cfg.ransac_iterations):
        i, j = rng.choice(len(cluster), 2, replace=False, p=weights)
        if np.linalg.norm(cluster[i] - cluster[j]) < max(12.0, diagonal * 0.012):
            continue
        trials.append(_line_geometry(cluster[[i, j]]))

    candidates = []
    for origin, direction, normal in trials:
        ok, score, _count = _model_stats(cluster, origin, direction, normal, cfg, diagonal)
        if ok:
            candidates.append((score, origin, direction, normal))
    candidates.sort(key=lambda item: -item[0])
    for base_score, origin, direction, normal in candidates[:32]:
        support = _hypothesis_support(cluster, origin, direction, normal,
                                      cfg, white, gradient)
        dist = _distances(cluster, origin, normal)
        inliers = cluster[dist <= cfg.distance_threshold]
        line = _refine_line(inliers, cfg, white, gradient)
        if line is None or line.support < cfg.min_line_support:
            continue
        final_origin = np.asarray([
            line.p0[0] + (line.p1[0] - line.p0[0]) * 0.5,
            line.p0[1] + (line.p1[1] - line.p0[1]) * 0.5,
        ])
        selected = indices[_distances(cluster, final_origin, np.asarray(line.normal)) <=
                           cfg.distance_threshold]
        if len(selected) < cfg.min_line_points:
            continue
        line.score = float(base_score * (0.15 + 0.85 * support))
        line.confidence = float(np.clip(
            0.30 * min(1.0, len(selected) / 30.0) +
            0.35 * line.support +
            0.35 * min(1.0, line.span / max(diagonal * 0.22, 1e-6)), 0.0, 1.0
        ))
        return line, selected
    return None


def _merge_nearby_lines(raw_lines: list[FittedCourtLine], cfg: LineFeatureConfig,
                        white: np.ndarray, gradient: np.ndarray
                        ) -> tuple[list[FittedCourtLine], int]:
    """合并角度和法向距离都接近的重复线，并对其内点并集重新拟合。"""
    lines = sorted(raw_lines, key=lambda line: (-line.score, -line.inlier_count, line.rms_error))
    merged_count = 0
    changed = True
    while changed:
        changed = False
        output: list[FittedCourtLine] = []
        consumed: set[int] = set()
        for i, line in enumerate(lines):
            if i in consumed:
                continue
            group = [line]
            for j in range(i + 1, len(lines)):
                if j in consumed:
                    continue
                candidate = lines[j]
                angle = _angle_delta(line.direction, candidate.direction)
                # 用两条可见支撑段的双向中位残差衡量距离，避免两条近似平行线
                # 在远离图像交点处因斜率轻微差异而产生很大的法向偏移。
                first_xy = np.asarray([[point.x, point.y] for point in line.points], np.float64)
                second_xy = np.asarray([[point.x, point.y] for point in candidate.points], np.float64)
                if not len(first_xy) or not len(second_xy):
                    continue
                first_to_second = _distances(first_xy, np.asarray(candidate.p0),
                                             np.asarray(candidate.normal))
                second_to_first = _distances(second_xy, np.asarray(line.p0),
                                             np.asarray(line.normal))
                offset = max(float(np.median(first_to_second)), float(np.median(second_to_first)))
                if (angle <= cfg.merge_angle_deg and
                        offset <= cfg.distance_threshold * cfg.merge_distance_factor):
                    group.append(candidate)
                    consumed.add(j)
            if len(group) == 1:
                output.append(line)
                continue

            unique_points = {id(point): point for item in group for point in item.points}
            xy = np.asarray([[point.x, point.y] for point in unique_points.values()], np.float64)
            refined = _refine_line(xy, cfg, white, gradient)
            best = max(group, key=lambda item: (item.score, item.inlier_count))
            if refined is not None:
                origin = np.asarray([
                    refined.p0[0] + (refined.p1[0] - refined.p0[0]) * 0.5,
                    refined.p0[1] + (refined.p1[1] - refined.p0[1]) * 0.5,
                ])
                keep = _distances(xy, origin, np.asarray(refined.normal)) <= cfg.distance_threshold
                kept_points = [point for point, ok in zip(unique_points.values(), keep) if ok]
                if len(kept_points) >= cfg.min_line_points:
                    refined.points = kept_points
                    refined.score = max(item.score for item in group)
                    refined.confidence = max(item.confidence for item in group)
                    output.append(refined)
                else:
                    output.append(best)
            else:
                output.append(best)
            changed = True
            merged_count += 1
        lines = output
    return lines, merged_count


def _fit_clustered_lines(xy: np.ndarray, points: list[LineFeaturePoint],
                         response: np.ndarray, point_white: np.ndarray,
                         white: np.ndarray, gradient: np.ndarray,
                         cfg: LineFeatureConfig, diagonal: float
                         ) -> tuple[list[FittedCourtLine], int, int, int]:
    """自动模式：角度距离聚类后逐簇 RANSAC，并合并邻近重复线。"""
    rng = np.random.default_rng(cfg.seed)
    clusters, rejected_clusters = _angle_distance_clusters(xy, cfg, diagonal, rng)
    raw_lines: list[FittedCourtLine] = []
    for cluster_id, indices in enumerate(clusters):
        fitted = _fit_cluster_ransac(
            indices, xy, response, white, gradient, cfg, diagonal,
            cfg.seed + cluster_id * 1009,
        )
        if fitted is None:
            continue
        line, selected = fitted
        line.points = [points[int(index)] for index in selected]
        raw_lines.append(line)

    merged, merged_count = _merge_nearby_lines(raw_lines, cfg, white, gradient)
    return merged[:cfg.max_lines], len(clusters), rejected_clusters, merged_count


def _fit_target_lines(xy: np.ndarray, points: list[LineFeaturePoint],
                      response: np.ndarray, point_white: np.ndarray,
                      white: np.ndarray, gradient: np.ndarray,
                      cfg: LineFeatureConfig, diagonal: float,
                      target_line: tuple[float, float, float, float],
                      shape: tuple[int, int]) -> list[FittedCourtLine]:
    """保留原 target-line 调试分支：目标种子优先的多线族 RANSAC。"""
    h, w = shape
    rng = np.random.default_rng(cfg.seed)
    remaining = np.arange(len(points))
    available = xy[remaining]
    raw_lines: list[FittedCourtLine] = []

    # RANSAC 点对只依赖当前帧，预先计算一次；后续提取不同线族时复用，避免
    # 每条线都重新做几百次 fitLine。真实交点允许属于多条线，因此不删除角点。
    weights = response[remaining] * (0.05 + 0.95 * point_white[remaining]) + 1e-6
    weights /= weights.sum()
    base_trials: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _trial in range(cfg.ransac_iterations):
        i, j = rng.choice(len(available), 2, replace=False, p=weights)
        if np.linalg.norm(available[i] - available[j]) < max(12.0, diagonal * 0.015):
            continue
        base_trials.append(_line_geometry(available[[i, j]]))
    grid_x = np.clip((available[:, 0] / max(w, 1) * 8).astype(int), 0, 7)
    grid_y = np.clip((available[:, 1] / max(h, 1) * 6).astype(int), 0, 5)
    representatives = []
    for grid_row in range(6):
        for grid_col in range(8):
            idx = np.flatnonzero((grid_x == grid_col) & (grid_y == grid_row))
            if len(idx):
                quality = response[idx] * (0.1 + 0.9 * point_white[idx])
                representatives.append(int(idx[int(np.argmax(quality))]))
    for a in range(len(representatives)):
        for b in range(a + 1, len(representatives)):
            i, j = representatives[a], representatives[b]
            if np.linalg.norm(available[i] - available[j]) >= max(12.0, diagonal * 0.015):
                base_trials.append(_line_geometry(available[[i, j]]))

    x1, y1, x2, y2 = (float(v) for v in target_line)
    seed_pts = np.asarray([[x1, y1], [x2, y2]], np.float64)
    seeds = [_line_geometry(seed_pts)]
    for _ in range(cfg.max_lines):
        if len(remaining) < cfg.min_line_points:
            break
        candidates = []
        trials = [(0, seed) for seed in seeds] + [(1, trial) for trial in base_trials]
        seeds.clear()
        for priority, (origin, direction, normal) in trials:
            repeated = False
            for old in raw_lines:
                cosine = abs(float(np.dot(old.direction, direction)))
                angle = math.degrees(math.acos(float(np.clip(cosine, 0.0, 1.0))))
                offset = abs(float(np.dot(np.asarray(old.p0) - origin, old.normal)))
                if (angle <= cfg.merge_angle_deg and
                        offset <= cfg.distance_threshold * cfg.merge_distance_factor):
                    repeated = True
                    break
            if repeated:
                continue
            ok, score, _count = _model_stats(
                available, origin, direction, normal, cfg, diagonal
            )
            if ok:
                candidates.append((priority, score, origin, direction, normal))
        candidates.sort(key=lambda item: (item[0], -item[1]))
        ranked = []
        for priority, base, origin, direction, normal in candidates[:160]:
            support = _hypothesis_support(
                available, origin, direction, normal, cfg, white, gradient
            )
            ranked.append((priority, base * (0.15 + 0.85 * support),
                           origin, direction, normal))
        ranked.sort(key=lambda item: (item[0], -item[1]))
        chosen = None
        for _priority, score, origin, direction, normal in ranked[:24]:
            dist = _distances(available, origin, normal)
            inliers = available[dist <= cfg.distance_threshold]
            line = _refine_line(inliers, cfg, white, gradient)
            if line is not None and line.support >= cfg.min_line_support:
                line.score = score
                chosen = (line, score)
                break
        if chosen is None:
            break
        line, _best_score = chosen
        final_origin = np.asarray([
            line.p0[0] + (line.p1[0] - line.p0[0]) * 0.5,
            line.p0[1] + (line.p1[1] - line.p0[1]) * 0.5,
        ])
        final_dist = _distances(xy, final_origin, np.asarray(line.normal))
        selected = remaining[final_dist[remaining] <= cfg.distance_threshold]
        if len(selected) < cfg.min_line_points:
            break
        line.points = [points[int(idx)] for idx in selected]
        line.confidence = float(np.clip(
            0.30 * min(1.0, len(selected) / 30.0) +
            0.35 * line.support +
            0.35 * min(1.0, line.span / max(diagonal * 0.22, 1e-6)), 0.0, 1.0
        ))

        duplicate = None
        for old in raw_lines:
            angle = math.degrees(math.acos(float(np.clip(
                np.dot(old.direction, line.direction), -1.0, 1.0
            )))) % 180.0
            angle = min(angle, 180.0 - angle)
            offset = abs(float(np.dot(np.asarray(old.p0) - final_origin, old.normal)))
            if angle <= cfg.merge_angle_deg and offset <= cfg.distance_threshold * cfg.merge_distance_factor:
                duplicate = old
                break
        if duplicate is not None:
            merged = {id(p): p for p in duplicate.points + line.points}
            arr = np.asarray([[p.x, p.y] for p in merged.values()], np.float64)
            refined = _refine_line(arr, cfg, white, gradient)
            if refined is not None:
                refined.points = list(merged.values())
                refined.score = max(duplicate.score, line.score)
                refined.confidence = max(duplicate.confidence, line.confidence)
                raw_lines[raw_lines.index(duplicate)] = refined
        else:
            raw_lines.append(line)
    return raw_lines


def _line_intersections(
    lines: list[FittedCourtLine], mask: np.ndarray | None, cfg: LineFeatureConfig,
) -> tuple[list[LineIntersection], dict]:
    """两两求无限长直线交点，并按 ROI、源线残差和距离聚类过滤。"""
    candidates: list[LineIntersection] = []
    parallel_rejected = outside_rejected = distance_rejected = 0
    h, w = mask.shape if mask is not None else (0, 0)
    for i, first in enumerate(lines):
        origin_a = np.asarray(first.p0, np.float64)
        direction_a = np.asarray(first.direction, np.float64)
        normal_a = np.asarray(first.normal, np.float64)
        for j in range(i + 1, len(lines)):
            second = lines[j]
            origin_b = np.asarray(second.p0, np.float64)
            direction_b = np.asarray(second.direction, np.float64)
            normal_b = np.asarray(second.normal, np.float64)
            den = float(direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0])
            if abs(den) < 1e-12:
                parallel_rejected += 1
                continue
            angle = math.degrees(math.asin(min(1.0, abs(den))))
            if angle < cfg.min_intersection_angle_deg:
                parallel_rejected += 1
                continue
            delta = origin_b - origin_a
            t = float((delta[0] * direction_b[1] - delta[1] * direction_b[0]) / den)
            point = origin_a + t * direction_a
            if not np.all(np.isfinite(point)):
                distance_rejected += 1
                continue
            if mask is not None:
                xi = int(np.clip(round(float(point[0])), 0, w - 1))
                yi = int(np.clip(round(float(point[1])), 0, h - 1))
                if not (0 <= point[0] <= w - 1 and 0 <= point[1] <= h - 1
                        and mask[yi, xi] > 0):
                    outside_rejected += 1
                    continue
            distance_a = abs(float((point - origin_a) @ normal_a))
            distance_b = abs(float((point - origin_b) @ normal_b))
            if max(distance_a, distance_b) > cfg.intersection_max_line_distance:
                distance_rejected += 1
                continue
            confidence = float(np.clip(
                0.35 + 0.30 * min(first.confidence, second.confidence)
                + 0.10 * min(1.0, angle / 45.0), 0.0, 0.95,
            ))
            candidates.append(LineIntersection(
                float(point[0]), float(point[1]), confidence, (i, j),
                distance_a, distance_b,
            ))

    # 同一真实交点被多条相邻线重复求出时，保留最高置信度并合并来源线。
    candidates.sort(key=lambda item: (-item.confidence, item.x, item.y))
    clusters: list[dict] = []
    for candidate in candidates:
        best = None
        best_distance = cfg.intersection_cluster_distance
        for cluster in clusters:
            distance = math.hypot(candidate.x - cluster["x"], candidate.y - cluster["y"])
            if distance <= best_distance:
                best, best_distance = cluster, distance
        if best is None:
            clusters.append({
                "x": candidate.x, "y": candidate.y,
                "confidence": candidate.confidence,
                "line_ids": set(candidate.line_ids),
                "distance_first": candidate.distance_first,
                "distance_second": candidate.distance_second,
            })
        else:
            best["line_ids"].update(candidate.line_ids)
            best["confidence"] = max(best["confidence"], candidate.confidence)
    intersections = [
        LineIntersection(
            cluster["x"], cluster["y"], cluster["confidence"],
            tuple(sorted(cluster["line_ids"])[:2]),
            cluster["distance_first"], cluster["distance_second"],
        )
        for cluster in clusters
    ]
    intersections.sort(key=lambda item: (-item.confidence, item.y, item.x))
    return intersections, {
        "intersection_candidates": len(candidates),
        "intersections": len(intersections),
        "intersection_parallel_rejected": parallel_rejected,
        "intersection_outside_rejected": outside_rejected,
        "intersection_distance_rejected": distance_rejected,
    }


def detect_line_features(
    image: np.ndarray,
    cfg: LineFeatureConfig | None = None,
    *,
    roi=None,
    player_mask=None,
    target_line: tuple[float, float, float, float] | None = None,
) -> LineFeatureResult:
    """检测图像中的球场直线候选。

    Args:
        image: BGR/BGRA 彩色图或灰度图。
        cfg: 可调检测参数。
        roi: 可选 (h,w) 二值球场 mask 或 Nx2 多边形；只在 ROI 内提取角点。
        player_mask: 可选 (h,w) 二值球员区域 mask 或 Nx2 多边形；区域内的角点不参与聚类。
        target_line: 可选 ``(x1,y1,x2,y2)`` 原图坐标目标线。给出后关闭自动 ROI，
            并使用保留的单目标线 RANSAC 调试分支。
    """
    cfg = cfg or LineFeatureConfig()
    if target_line is not None:
        target = np.asarray(target_line, dtype=np.float64)
        if target.shape != (4,) or not np.all(np.isfinite(target)):
            raise ValueError("target_line 必须是有限的 (x1,y1,x2,y2)")
        if np.linalg.norm(target[2:] - target[:2]) <= 1e-6:
            raise ValueError("target_line 的两个端点不能重合")
        target_line = tuple(float(v) for v in target)
    gray = _as_gray(image)
    h, w = gray.shape
    if min(h, w) < 8:
        return LineFeatureResult(stats={"reason": "image_too_small"})

    hsv = None
    if image.ndim == 3 and image.shape[2] >= 3:
        hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
        white = ((hsv[:, :, 2] >= cfg.white_threshold) &
                 (hsv[:, :, 1] <= cfg.max_saturation)).astype(np.float32)
    else:
        white = (gray >= cfg.white_threshold).astype(np.float32)

    mask = _roi_mask((h, w), roi)
    player_regions = _roi_mask((h, w), player_mask, "player_mask")
    roi_source = "provided" if mask is not None else "none"
    roi_components = 0
    # 场地颜色只从画面下半部学习。它既用于收紧固定 ROI，也用于判断角点
    # 是否真的位于球场地面；观众席纹理和白色球衣通常缺少周边木质场地支撑。
    floor_mask, floor_components = (
        _dominant_floor_mask(hsv, include_upper=True)
        if hsv is not None and cfg.use_floor_support and target_line is None
        else (np.zeros((h, w), np.uint8), 0)
    )
    floor_roi_used = False
    if mask is None and cfg.use_fixed_roi and cfg.fixed_roi_polygon is not None and target_line is None:
        polygon = np.asarray(cfg.fixed_roi_polygon, np.float64) * np.asarray([w - 1, h - 1])
        mask = _roi_mask((h, w), polygon)
        roi_source = "fixed_polygon"
        if floor_components:
            radius = max(1, int(round(min(h, w) * cfg.floor_roi_dilate_frac)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
            floor_evidence = cv2.dilate(floor_mask, kernel)
            mask &= floor_evidence
            floor_roi_used = True
    if mask is None and cfg.use_roi_auto and target_line is None:
        mask, roi_components = _auto_court_mask(white, cfg, hsv)
        roi_source = "auto"
    if mask is not None and not mask.any():
        return LineFeatureResult(stats={
            "reason": "empty_roi", "roi_source": roi_source,
            "roi_components": roi_components, "floor_roi_used": floor_roi_used,
        })
    obstacle_regions = (
        _foreground_obstacle_mask(floor_mask, mask)
        if floor_components and mask is not None and cfg.use_floor_support and target_line is None
        else None
    )

    if cfg.use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid)
        )
        work_gray = clahe.apply(gray)
    else:
        work_gray = gray

    corners = cv2.goodFeaturesToTrack(
        work_gray, maxCorners=cfg.max_corners, qualityLevel=cfg.quality_level,
        minDistance=cfg.min_distance, blockSize=cfg.block_size,
        useHarrisDetector=cfg.use_harris, k=cfg.harris_k, mask=mask,
    )
    raw_corner_count = 0 if corners is None else len(corners)
    if corners is None or len(corners) < cfg.min_line_points:
        return LineFeatureResult(stats={
            "reason": "not_enough_corners", "corners": raw_corner_count,
            "raw_corners": raw_corner_count, "roi_source": roi_source,
            "roi_components": roi_components,
        })

    xy = corners.reshape(-1, 2).astype(np.float64)
    inside = (
        (xy[:, 0] >= cfg.border_margin) & (xy[:, 0] <= w - 1 - cfg.border_margin) &
        (xy[:, 1] >= cfg.border_margin) & (xy[:, 1] <= h - 1 - cfg.border_margin)
    )
    if mask is not None:
        inside &= mask[np.rint(xy[:, 1]).astype(int), np.rint(xy[:, 0]).astype(int)] > 0
    xy = xy[inside]
    if len(xy) < cfg.min_line_points:
        return LineFeatureResult(stats={
            "reason": "not_enough_valid_corners", "corners": len(xy),
            "raw_corners": raw_corner_count, "roi_source": roi_source,
            "roi_components": roi_components,
        })

    # Shi-Tomasi 响应与白线证据分别计算；response 用于排序，evidence 用于直线打分。
    eigen = cv2.cornerMinEigenVal(work_gray, blockSize=cfg.block_size)
    response = eigen[np.rint(xy[:, 1]).astype(int), np.rint(xy[:, 0]).astype(int)]
    response_max = max(float(response.max()), 1e-6)
    response = np.clip(response / response_max, 0.0, 1.0)
    point_white = _box_mean(white, xy, cfg.white_neighborhood_size)

    # 自动模式第一重过滤：只保留白色球场线像素附近的角点。target_line 调试
    # 分支保留旧的暗线放宽策略，避免破坏既有单线排障用法。
    evidence_keep = point_white >= cfg.min_point_evidence
    if target_line is not None:
        tx1, ty1, tx2, ty2 = (float(v) for v in target_line)
        target_pts = np.asarray([[tx1, ty1], [tx2, ty2]], np.float64)
        if np.isfinite(target_pts).all() and np.linalg.norm(target_pts[1] - target_pts[0]) > 1e-6:
            target_origin, _target_direction, target_normal = _line_geometry(target_pts)
            evidence_keep |= _distances(xy, target_origin, target_normal) <= cfg.distance_threshold * 2.0
    white_rejected = int((~evidence_keep).sum())
    xy = xy[evidence_keep]
    response = response[evidence_keep]
    point_white = point_white[evidence_keep]
    if len(xy) < cfg.min_line_points:
        return LineFeatureResult(stats={
            "reason": "not_enough_line_evidence_corners", "corners": len(xy),
            "raw_corners": raw_corner_count, "white_rejected": white_rejected,
            "roi_source": roi_source, "roi_components": roi_components,
        })

    gx = cv2.Sobel(work_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(work_gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.stack([gx, gy], axis=2)
    gradient_map = _gradient_strength_map(
        work_gray, mask, cfg.gradient_neighborhood_size
    )
    point_gradient = _sample(gradient_map, xy)
    gradient_rejected = 0
    if target_line is None and cfg.hard_gradient_filter:
        # 自动模式第二重过滤：纯色地面和缺少线/场地边界梯度的白块不进入聚类。
        gradient_keep = point_gradient >= cfg.min_gradient_strength
        gradient_rejected = int((~gradient_keep).sum())
        xy = xy[gradient_keep]
        response = response[gradient_keep]
        point_white = point_white[gradient_keep]
        point_gradient = point_gradient[gradient_keep]
        if len(xy) < cfg.min_line_points:
            return LineFeatureResult(stats={
                "reason": "not_enough_gradient_corners", "corners": len(xy),
                "raw_corners": raw_corner_count, "white_rejected": white_rejected,
                "gradient_rejected": gradient_rejected, "roi_source": roi_source,
                "roi_components": roi_components,
            })

    player_rejected = 0
    if player_regions is not None:
        xi = np.clip(np.rint(xy[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.rint(xy[:, 1]).astype(int), 0, h - 1)
        player_keep = player_regions[yi, xi] == 0
        player_rejected = int((~player_keep).sum())
        xy = xy[player_keep]
        response = response[player_keep]
        point_white = point_white[player_keep]
        point_gradient = point_gradient[player_keep]
        if len(xy) < cfg.min_line_points:
            return LineFeatureResult(stats={
                "reason": "not_enough_corners_after_player_filter", "corners": len(xy),
                "raw_corners": raw_corner_count, "white_rejected": white_rejected,
                "gradient_rejected": gradient_rejected, "player_rejected": player_rejected,
                "roi_source": roi_source, "roi_components": roi_components,
            })

    obstacle_rejected = 0
    if obstacle_regions is not None and obstacle_regions.any():
        xi = np.clip(np.rint(xy[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.rint(xy[:, 1]).astype(int), 0, h - 1)
        obstacle_keep = obstacle_regions[yi, xi] == 0
        obstacle_rejected = int((~obstacle_keep).sum())
        xy = xy[obstacle_keep]
        response = response[obstacle_keep]
        point_white = point_white[obstacle_keep]
        point_gradient = point_gradient[obstacle_keep]
        if len(xy) < cfg.min_line_points:
            return LineFeatureResult(stats={
                "reason": "not_enough_corners_after_obstacle_filter", "corners": len(xy),
                "raw_corners": raw_corner_count, "white_rejected": white_rejected,
                "gradient_rejected": gradient_rejected, "player_rejected": player_rejected,
                "obstacle_rejected": obstacle_rejected, "roi_source": roi_source,
                "roi_components": roi_components, "floor_roi_used": floor_roi_used,
            })

    floor_rejected = 0
    point_floor = np.zeros(len(xy), np.float32)
    if floor_components and cfg.use_floor_support and target_line is None:
        # 只检查角点周边是否有连续场地，不要求中心像素本身是木地板：
        # 真实白线穿过场地时中心是白色，但周围仍有足够地面颜色证据。
        point_floor = _box_mean(floor_mask, xy, cfg.floor_neighborhood_size).astype(np.float32)
        floor_keep = point_floor >= cfg.min_floor_support
        floor_rejected = int((~floor_keep).sum())
        xy = xy[floor_keep]
        response = response[floor_keep]
        point_white = point_white[floor_keep]
        point_gradient = point_gradient[floor_keep]
        point_floor = point_floor[floor_keep]
        if len(xy) < cfg.min_line_points:
            return LineFeatureResult(stats={
                "reason": "not_enough_floor_supported_corners", "corners": len(xy),
                "raw_corners": raw_corner_count, "white_rejected": white_rejected,
                "gradient_rejected": gradient_rejected, "player_rejected": player_rejected,
                "floor_rejected": floor_rejected, "roi_source": roi_source,
                "roi_components": roi_components, "floor_roi_used": floor_roi_used,
            })

    points = [
        LineFeaturePoint(float(p[0]), float(p[1]), float(r), float(e), float(g))
        for p, r, e, g in zip(xy, response, point_white, point_gradient)
    ]
    filtered_corner_count = len(points)
    diagonal = float(math.hypot(w, h))
    cluster_count = 0
    rejected_clusters = 0
    merged_line_count = 0
    if target_line is None:
        raw_lines, cluster_count, rejected_clusters, merged_line_count = _fit_clustered_lines(
            xy, points, response, point_white, white, gradient, cfg, diagonal
        )
    else:
        raw_lines = _fit_target_lines(
            xy, points, response, point_white, white, gradient, cfg, diagonal,
            target_line, (h, w)
        )

    # 目标线约束：方向差 <= 20°，法向偏移 <= 8 倍距离阈值。
    if target_line is not None and raw_lines:
        tx1, ty1, tx2, ty2 = (float(v) for v in target_line)
        target_dir = np.asarray([tx2 - tx1, ty2 - ty1], np.float64)
        target_dir /= max(float(np.linalg.norm(target_dir)), 1e-9)
        target_normal = np.asarray([-target_dir[1], target_dir[0]])
        target_origin = np.asarray([tx1, ty1])
        filtered = []
        for line in raw_lines:
            cosine = abs(float(np.dot(target_dir, line.direction)))
            angle = math.degrees(math.acos(float(np.clip(cosine, 0.0, 1.0))))
            offset = abs(float(np.dot(np.asarray(line.p0) - target_origin, target_normal)))
            if angle <= 20.0 and offset <= cfg.distance_threshold * 8.0:
                filtered.append(line)
        raw_lines = filtered

    # RANSAC 后统一校验：最小线长、可选角度先验、ROI 内连续可见长度。
    short_lines_rejected = angle_lines_rejected = roi_lines_rejected = 0
    validated_lines: list[FittedCourtLine] = []
    for line in raw_lines:
        reason = _validate_final_line(
            line, cfg, (h, w), mask, check_angle_prior=target_line is None
        )
        if reason == "short_line":
            short_lines_rejected += 1
        elif reason == "angle_prior":
            angle_lines_rejected += 1
        elif reason == "roi_span":
            roi_lines_rejected += 1
        if reason is None:
            validated_lines.append(line)

    validated_lines.sort(key=lambda line: (-line.score, -line.inlier_count, line.rms_error))
    lines = validated_lines[:cfg.max_lines]
    for line_id, line in enumerate(lines):
        line.line_id = line_id
        final_origin = np.asarray([
            line.p0[0] + (line.p1[0] - line.p0[0]) * 0.5,
            line.p0[1] + (line.p1[1] - line.p0[1]) * 0.5,
        ])
        distances = _distances(xy, final_origin, np.asarray(line.normal))
        for p in line.points:
            idx = int(np.argmin(np.hypot(xy[:, 0] - p.x, xy[:, 1] - p.y)))
            p.line_id = line_id
            p.distance_to_line = float(distances[idx])

    # 只发布最终通过全部筛选的直线内点；被 HSV、梯度、聚类、RANSAC、
    # 线长/角度/ROI 校验或 max_lines 淘汰的点不会出现在 JSON 或可视化中。
    inlier_ids = {id(point) for line in lines for point in line.points}
    points = [point for point in points if id(point) in inlier_ids]

    stats = {
        "mode": (
            "target_line" if target_line is not None
            else "full_image" if roi_source == "none"
            else "roi_fixed" if roi_source == "fixed_polygon"
            else "roi_auto"
        ),
        "roi_source": roi_source,
        "roi_components": roi_components,
        "raw_corners": raw_corner_count,
        "filtered_corners": filtered_corner_count,
        "white_rejected": white_rejected,
        "gradient_rejected": gradient_rejected,
        "player_rejected": player_rejected,
        "obstacle_rejected": obstacle_rejected,
        "floor_rejected": floor_rejected,
        "floor_roi_used": floor_roi_used,
        "min_floor_support": cfg.min_floor_support,
        "floor_neighborhood_size": cfg.floor_neighborhood_size,
        "clusters": cluster_count,
        "rejected_clusters": rejected_clusters,
        "accepted_clusters": len(lines),
        "merged_lines": merged_line_count,
        "short_lines_rejected": short_lines_rejected,
        "angle_lines_rejected": angle_lines_rejected,
        "roi_lines_rejected": roi_lines_rejected,
        "corners": len(points),
        "lines": len(lines),
        "inliers": sum(line.inlier_count for line in lines),
        "max_corners": cfg.max_corners,
        "quality_level": cfg.quality_level,
        "distance_threshold": cfg.distance_threshold,
        "min_distance": cfg.min_distance,
        "white_threshold": cfg.white_threshold,
        "max_saturation": cfg.max_saturation,
        "min_point_evidence": cfg.min_point_evidence,
        "white_neighborhood_size": cfg.white_neighborhood_size,
        "min_gradient_strength": cfg.min_gradient_strength,
        "gradient_neighborhood_size": cfg.gradient_neighborhood_size,
        "min_cluster_points": cfg.min_cluster_points,
        "min_line_length": cfg.min_line_length,
        "line_angle_prior_deg": list(cfg.line_angle_prior_deg),
        "angle_prior_tolerance_deg": cfg.angle_prior_tolerance_deg,
    }
    intersections, intersection_stats = _line_intersections(lines, mask, cfg)
    stats.update(intersection_stats)
    stats.update({
        "intersection_max_line_distance": cfg.intersection_max_line_distance,
        "intersection_cluster_distance": cfg.intersection_cluster_distance,
        "min_intersection_angle_deg": cfg.min_intersection_angle_deg,
        "fixed_roi": roi_source == "fixed_polygon",
    })
    intersection_keypoints = []
    if intersections:
        # 延迟导入避免模块初始化环；对象与 ONNX 输出使用同一 Dataclass。
        from bsk.perceive.court_keypoints import DetectedKeypoint

        intersection_keypoints = [
            DetectedKeypoint(f"intersection_{index}", (item.x, item.y), item.confidence)
            for index, item in enumerate(intersections)
        ]
    return LineFeatureResult(
        lines=lines, points=points, intersections=intersections,
        intersection_keypoints=intersection_keypoints, stats=stats,
    )


def visualize_line_features(image: np.ndarray, result: LineFeatureResult) -> np.ndarray:
    """特征点标红、拟合边线标蓝、无限线交点标绿；不修改输入图。"""
    if result.visualization is not None:
        return result.visualization
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # 只绘制最终直线的有效内点；聚类/RANSAC/后校验淘汰的点不进入可视化。
    for line in result.lines:
        a = (int(round(line.p0[0])), int(round(line.p0[1])))
        b = (int(round(line.p1[0])), int(round(line.p1[1])))
        cv2.line(out, a, b, (255, 0, 0), 2, cv2.LINE_AA)
        for p in line.points:
            cv2.circle(out, (int(round(p.x)), int(round(p.y))), 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(out, f"L{line.line_id}:{line.inlier_count}",
                    (min(max(a[0], 4), out.shape[1] - 90), min(max(a[1] - 8, 16), out.shape[0] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    for index, intersection in enumerate(result.intersections):
        center = (int(round(intersection.x)), int(round(intersection.y)))
        cv2.circle(out, center, 6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(out, f"P{index}", (center[0] + 8, center[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    result.visualization = out
    return out


def _parse_polygon(value: str) -> tuple[tuple[float, float], ...]:
    try:
        values = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("roi-polygon 格式必须是 x1,y1,x2,y2,...") from exc
    if len(values) < 6 or len(values) % 2:
        raise argparse.ArgumentTypeError("roi-polygon 至少需要 3 个 [0,1] 坐标点")
    polygon = tuple(zip(values[::2], values[1::2]))
    if any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for point in polygon for v in point):
        raise argparse.ArgumentTypeError("roi-polygon 坐标必须在 [0,1] 内")
    return polygon


def _parse_target(value: str | None) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    parts = [float(v.strip()) for v in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("target-line 格式必须是 x1,y1,x2,y2")
    return tuple(parts)  # type: ignore[return-value]


def _parse_angles(value: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    try:
        angles = tuple(float(v.strip()) for v in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("line-angles 格式必须是 0,45,90") from exc
    if any(not math.isfinite(angle) for angle in angles):
        raise argparse.ArgumentTypeError("line-angles 必须是有限角度")
    return angles


def main() -> None:
    parser = argparse.ArgumentParser(description="Shi-Tomasi 特征点球场边线检测")
    parser.add_argument("image", type=Path, help="输入图片路径")
    parser.add_argument("--output", type=Path, help="可视化输出路径，默认写入输入图同目录")
    parser.add_argument("--max-corners", type=int, default=700, help="角点数量上限")
    parser.add_argument("--quality-level", type=float, default=0.025, help="Shi-Tomasi 质量阈值")
    parser.add_argument("--distance-threshold", type=float, default=5.0,
                        help="点到拟合线的距离阈值（像素）")
    parser.add_argument("--min-distance", type=int, default=7, help="角点最小间距（像素）")
    parser.add_argument("--min-line-points", type=int, default=8, help="每条线最少有效角点")
    parser.add_argument("--min-cluster-points", type=int, default=5,
                        help="角度距离簇的最小点数，不足时整簇丢弃")
    parser.add_argument("--min-line-length", type=float, default=80.0,
                        help="最终拟合线段的最小长度（像素）")
    parser.add_argument("--white-threshold", type=int, default=145,
                        help="HSV 白线亮度阈值 V")
    parser.add_argument("--max-saturation", type=int, default=90,
                        help="HSV 白线最大饱和度 S")
    parser.add_argument("--white-neighborhood-size", type=int, default=5,
                        help="角点邻域 HSV 白线证据窗口（奇数）")
    parser.add_argument("--min-point-evidence", type=float, default=0.35,
                        help="角点邻域白线像素比例下限")
    parser.add_argument("--gradient-neighborhood-size", type=int, default=5,
                        help="局部 Sobel 梯度最大值窗口（奇数）")
    parser.add_argument("--min-gradient-strength", type=float, default=0.10,
                        help="角点局部梯度强度下限")
    parser.add_argument("--line-angles", type=_parse_angles, default=(0.0, 90.0),
                        help="球场线角度白名单，如 0,45,90,135；默认水平/垂直 ±35°")
    parser.add_argument("--angle-prior-tolerance", type=float, default=35.0,
                        help="角度白名单允许偏差（度）")
    parser.add_argument("--ransac-iterations", type=int, default=300, help="RANSAC 迭代次数")
    parser.add_argument("--max-lines", type=int, default=10, help="最多输出多少条候选线")
    parser.add_argument("--merge-angle", type=float, default=8.0,
                        help="邻近重复线合并的最大角度差（度）")
    parser.add_argument("--merge-distance-factor", type=float, default=3.0,
                        help="重复线法向距离阈值 / distance-threshold")
    parser.add_argument("--cluster-angle", type=float, default=2.0,
                        help="角度距离聚类的角度粒度（度）")
    parser.add_argument("--cluster-distance-factor", type=float, default=1.5,
                        help="聚类的法向距离半径 / distance-threshold")
    roi_group = parser.add_mutually_exclusive_group()
    roi_group.add_argument("--use-roi-auto", dest="use_roi_auto", action="store_true",
                           default=True, help="开启自动球场 ROI（默认）")
    roi_group.add_argument("--no-use-roi-auto", dest="use_roi_auto", action="store_false",
                           help="关闭自动 ROI，使用全图；target-line 调试模式也会关闭自动 ROI")
    roi_group.add_argument("--no-fixed-roi", dest="use_fixed_roi", action="store_false",
                           default=True, help="关闭默认固定多边形 ROI，改用自动 ROI")
    parser.add_argument("--roi-polygon", type=_parse_polygon,
                        default=DEFAULT_FIXED_COURT_POLYGON,
                        help="固定球场地面多边形，坐标为相对宽高的 x,y，如 0.10,0.23,...")
    parser.add_argument("--intersection-max-line-distance", type=float, default=12.0,
                        help="交点到源拟合线的最大距离（像素）")
    parser.add_argument("--intersection-cluster-distance", type=float, default=25.0,
                        help="交点聚类半径（像素）")
    parser.add_argument("--min-intersection-angle", type=float, default=10.0,
                        help="求交两条线的最小夹角（度）")
    parser.add_argument("--roi-mask", type=Path,
                        help="可选球场二值 mask 图片；优先于自动 ROI")
    parser.add_argument("--player-mask", type=Path,
                        help="可选球员区域二值 mask；区域内角点不参与拟合")
    parser.add_argument("--target-line", type=_parse_target,
                        help="可选目标线 x1,y1,x2,y2（原图像素），启用原调试分支")
    parser.add_argument("--json", action="store_true", help="打印全部角点和拟合线 JSON")
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"无法读取图片: {args.image}")
    cfg = LineFeatureConfig(
        max_corners=args.max_corners, quality_level=args.quality_level,
        distance_threshold=args.distance_threshold, min_distance=args.min_distance,
        min_line_points=args.min_line_points,
        min_cluster_points=args.min_cluster_points,
        min_line_length=args.min_line_length,
        white_threshold=args.white_threshold, max_saturation=args.max_saturation,
        white_neighborhood_size=args.white_neighborhood_size,
        min_point_evidence=args.min_point_evidence,
        gradient_neighborhood_size=args.gradient_neighborhood_size,
        min_gradient_strength=args.min_gradient_strength,
        line_angle_prior_deg=args.line_angles,
        angle_prior_tolerance_deg=args.angle_prior_tolerance,
        max_lines=args.max_lines, merge_angle_deg=args.merge_angle,
        merge_distance_factor=args.merge_distance_factor,
        cluster_angle_deg=args.cluster_angle,
        cluster_distance_factor=args.cluster_distance_factor,
        ransac_iterations=args.ransac_iterations, use_roi_auto=args.use_roi_auto,
        use_fixed_roi=args.use_fixed_roi,
        fixed_roi_polygon=args.roi_polygon,
        intersection_max_line_distance=args.intersection_max_line_distance,
        intersection_cluster_distance=args.intersection_cluster_distance,
        min_intersection_angle_deg=args.min_intersection_angle,
    )
    roi = None
    if args.roi_mask is not None:
        roi = cv2.imread(str(args.roi_mask), cv2.IMREAD_GRAYSCALE)
        if roi is None:
            raise SystemExit(f"无法读取 ROI mask: {args.roi_mask}")
        if roi.shape != image.shape[:2]:
            raise SystemExit("ROI mask 尺寸必须与输入图片一致")
    player_mask = None
    if args.player_mask is not None:
        player_mask = cv2.imread(str(args.player_mask), cv2.IMREAD_GRAYSCALE)
        if player_mask is None:
            raise SystemExit(f"无法读取球员 mask: {args.player_mask}")
        if player_mask.shape != image.shape[:2]:
            raise SystemExit("球员 mask 尺寸必须与输入图片一致")
    result = detect_line_features(
        image, cfg, roi=roi, player_mask=player_mask, target_line=args.target_line
    )
    output = args.output or args.image.with_name(f"{args.image.stem}_court_lines.jpg")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), visualize_line_features(image, result)):
        raise SystemExit(f"可视化写入失败: {output}")
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps({
            "image": str(args.image), "output": str(output),
            **{key: value for key, value in result.stats.items() if key != "lines"},
            "line_count": result.stats.get("lines", 0),
            "intersection_count": len(result.intersections),
            "intersection_keypoint_count": len(result.intersection_keypoints),
            "lines": [{"line_id": line.line_id, "inliers": line.inlier_count,
                       "rms_error": round(line.rms_error, 3),
                       "confidence": round(line.confidence, 3)} for line in result.lines],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
