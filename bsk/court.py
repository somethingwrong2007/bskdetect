"""球场几何唯一真源：FIBA / NBA / 3x3 尺寸、关键点、线段。

坐标系：米制，原点在球场中心，x 沿长边（A 队进攻方向为正），y 沿短边。
所有下游模块（标定、俯视投影、3D 展示）只引用这里，禁止硬编码尺寸。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CourtModel = Literal["FIBA", "NBA", "3x3"]


@dataclass(frozen=True)
class Court:
    """一份球场 = 外框 + 中线 + 两个半场的三分/罚球/合理冲撞区。

    尺寸单位米。坐标按"原点在中心、x 朝 A 队方向"给出，
    实际渲染时再按进攻方向翻转。
    """

    model: CourtModel
    length: float
    width: float
    three_pt_radius: float
    three_pt_corner_dist: float
    free_throw_dist: float
    free_throw_width: float
    restricted_radius: float
    rim_height: float = 3.05
    rim_offset_from_baseline: float = 1.575
    # 限制区（罚球区漆块）宽度：注意 free_throw_width 是罚球线长度，不等于漆块宽度，
    # 两者在 FIBA/NBA 分别是 3.60/4.90 与 3.66/4.88。3D 查看器画地面时必须用 paint_width。
    paint_width: float = 4.90
    center_circle_radius: float = 1.80
    backboard_from_baseline: float = 1.20
    backboard_width: float = 1.80
    backboard_bottom: float = 2.90
    backboard_height: float = 1.05
    rim_radius: float = 0.225
    # 线条宽：FIBA 5 cm，NBA 2 in (5.08 cm)。
    line_width: float = 0.05
    # FIBA 中线向两侧边线外各延长 0.15 m；NBA 不延长。
    center_line_extension: float = 0.0
    # FIBA 罚球圈在限制区内的一半画虚线；NBA 为整圆实线。
    free_throw_dashed_inside: bool = False

    @property
    def half_length(self) -> float:
        return self.length / 2

    @property
    def half_width(self) -> float:
        return self.width / 2

    def baseline_x(self, side: Literal["A", "B"]) -> float:
        return self.half_length if side == "A" else -self.half_length

    def rim_pos(self, side: Literal["A", "B"]) -> tuple[float, float]:
        x = self.baseline_x(side) - self.rim_offset_from_baseline * (1 if side == "A" else -1)
        return (x, 0.0)

    # ── 线中心线几何（官方尺寸量到线的内/外沿，渲染时统一换算到中心线）──
    @property
    def line_half(self) -> float:
        return self.line_width * 0.5

    @property
    def free_throw_center_dist(self) -> float:
        """罚球线中心线距端线内沿：官方 5.80 m 量到最外沿。"""
        return self.free_throw_dist - self.line_half

    @property
    def three_pt_center_radius(self) -> float:
        """三分弧中心线半径：官方 6.75/7.24 m 量到圆弧外沿。"""
        return self.three_pt_radius - self.line_half

    @property
    def three_pt_corner_center_dist(self) -> float:
        """底角三分线中心线距边线：官方 0.90/0.9144 m 量到线外沿。"""
        return self.three_pt_corner_dist + self.line_half

    @property
    def free_throw_circle_center_radius(self) -> float:
        return max(0.0, self.free_throw_width * 0.5 - self.line_half)

    @property
    def center_circle_center_radius(self) -> float:
        return self.center_circle_radius - self.line_half

    @property
    def restricted_center_radius(self) -> float:
        """合理冲撞区半径：FIBA 量到半圆内沿，NBA 4 ft 量到圆弧。"""
        return self.restricted_radius + self.line_half

    def dims(self) -> dict:
        """给前端 3D 查看器的全部地面几何（米）；前端不再自己抄一套尺寸。"""
        return {
            "model": self.model,
            "length": self.length,
            "width": self.width,
            "three_pt_radius": self.three_pt_radius,
            "three_pt_corner_dist": self.three_pt_corner_dist,
            "free_throw_dist": self.free_throw_dist,
            "free_throw_width": self.free_throw_width,
            "restricted_radius": self.restricted_radius,
            "paint_width": self.paint_width,
            "center_circle_radius": self.center_circle_radius,
            "rim_height": self.rim_height,
            "rim_offset_from_baseline": self.rim_offset_from_baseline,
            "backboard_from_baseline": self.backboard_from_baseline,
            "backboard_width": self.backboard_width,
            "backboard_bottom": self.backboard_bottom,
            "backboard_height": self.backboard_height,
            "rim_radius": self.rim_radius,
            "line_width": self.line_width,
            "center_line_extension": self.center_line_extension,
            "free_throw_dashed_inside": self.free_throw_dashed_inside,
        }


def court_dims(model: str = "FIBA") -> dict:
    """按型号名取地面几何（未知型号回落 FIBA）。"""
    return COURTS.get(model, FIBA).dims()


FIBA = Court(
    model="FIBA",
    length=28.0,
    width=15.0,
    three_pt_radius=6.75,          # 外沿半径
    three_pt_corner_dist=0.90,     # 底角线外沿距边线内沿
    free_throw_dist=5.80,          # 罚球线最外沿距端线内沿
    free_throw_width=3.60,         # 罚球线长度
    restricted_radius=1.30,        # 2022+ 无撞人半圆内沿半径（旧规 1.25）
    paint_width=4.90,              # 限制区内沿宽度
    center_circle_radius=1.80,     # 中圈外沿半径
    rim_offset_from_baseline=1.575,
    backboard_from_baseline=1.20,
    line_width=0.05,
    center_line_extension=0.15,
    free_throw_dashed_inside=True,
)

NBA = Court(
    model="NBA",
    length=28.65,                  # 94 ft
    width=15.24,                   # 50 ft
    three_pt_radius=7.239,         # 23 ft 9 in（外沿）
    three_pt_corner_dist=0.9144,   # 3 ft
    free_throw_dist=5.7912,        # 19 ft
    free_throw_width=3.6576,       # 12 ft
    restricted_radius=1.2192,      # 4 ft
    paint_width=4.8768,            # 16 ft
    center_circle_radius=1.8288,   # 6 ft
    rim_offset_from_baseline=1.6002,  # 5 ft 3 in
    backboard_from_baseline=1.2192,   # 4 ft
    line_width=0.0508,             # 2 in
    center_line_extension=0.0,
    free_throw_dashed_inside=False,
)

COURT_3X3 = Court(
    model="3x3",
    length=15.0,
    width=11.0,
    three_pt_radius=6.75,
    three_pt_corner_dist=0.90,
    free_throw_dist=5.80,
    free_throw_width=3.60,
    restricted_radius=1.30,
    paint_width=4.90,
    center_circle_radius=1.80,
    rim_offset_from_baseline=1.575,
    backboard_from_baseline=1.20,
    line_width=0.05,
    center_line_extension=0.0,
    free_throw_dashed_inside=True,
)

COURTS: dict[CourtModel, Court] = {"FIBA": FIBA, "NBA": NBA, "3x3": COURT_3X3}


@dataclass(frozen=True)
class CourtKeypoint:
    """用于单应标定的球场关键点：角点、三分顶点、罚球线端点等。

    name 是语义标签；pos 是米制坐标；visible 表示该点在典型转播机位下是否可见。
    """

    name: str
    pos: tuple[float, float]
    visible: bool = True


def court_keypoints(court: Court, side: Literal["A", "B"] = "A") -> list[CourtKeypoint]:
    """返回一侧半场的标定关键点（角点 + 三分弧采样 + 罚球区角点）。"""
    bl = court.baseline_x(side)
    hw = court.half_width
    kps: list[CourtKeypoint] = []

    kps.append(CourtKeypoint(f"{side}_baseline_left", (bl, -hw)))
    kps.append(CourtKeypoint(f"{side}_baseline_right", (bl, hw)))
    corner_y = hw - court.three_pt_corner_center_dist
    kps.append(CourtKeypoint(f"{side}_corner_left", (bl, -corner_y)))
    kps.append(CourtKeypoint(f"{side}_corner_right", (bl, corner_y)))

    rim_x, rim_y = court.rim_pos(side)
    kps.append(CourtKeypoint(f"{side}_rim", (rim_x, rim_y)))

    sign = 1 if side == "A" else -1
    ft_x = bl - court.free_throw_center_dist * sign
    ft_hw = court.paint_width / 2 + court.line_half
    kps.append(CourtKeypoint(f"{side}_ft_left", (ft_x, -ft_hw)))
    kps.append(CourtKeypoint(f"{side}_ft_right", (ft_x, ft_hw)))

    import math
    n_arc = 9
    R = court.three_pt_center_radius
    for i in range(n_arc + 1):
        theta = -math.pi / 2 + math.pi * i / n_arc
        px = rim_x - sign * R * math.cos(theta)
        py = R * math.sin(theta)
        if abs(py) <= corner_y:
            kps.append(CourtKeypoint(f"{side}_arc_{i}", (px, py)))

    return kps


def all_court_keypoints(court: Court) -> list[CourtKeypoint]:
    kps = []
    kps.append(CourtKeypoint("center", (0.0, 0.0)))
    kps.append(CourtKeypoint("mid_left", (0.0, -court.half_width)))
    kps.append(CourtKeypoint("mid_right", (0.0, court.half_width)))
    kps.extend(court_keypoints(court, "A"))
    kps.extend(court_keypoints(court, "B"))
    return kps


def project_court_polygon(H_world_to_pixel, court: Court = FIBA) -> "np.ndarray":
    """球场外框投影到像素坐标，返回 (4, 2) 多边形（用于 ROI 剔除）。

    H_world_to_pixel 即标定里的 H_inv（世界→像素）。观众席与替补席都在
    这个多边形之外，用它做范围剔除最稳。
    """
    import numpy as np

    hl, hw = court.half_length, court.half_width
    corners = np.array([
        [-hl, -hw, 1.0],
        [hl, -hw, 1.0],
        [hl, hw, 1.0],
        [-hl, hw, 1.0],
    ], dtype=np.float64)
    pts = (np.asarray(H_world_to_pixel, dtype=np.float64) @ corners.T).T
    return (pts[:, :2] / pts[:, 2:3]).astype(np.float32)


__all__ = ["Court", "CourtKeypoint", "CourtModel", "COURTS", "FIBA", "NBA", "COURT_3X3",
           "court_keypoints", "all_court_keypoints", "project_court_polygon", "court_dims"]
