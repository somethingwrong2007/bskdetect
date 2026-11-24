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


FIBA = Court(
    model="FIBA",
    length=28.0,
    width=15.0,
    three_pt_radius=6.75,
    three_pt_corner_dist=0.90,
    free_throw_dist=5.80,
    free_throw_width=3.60,
    restricted_radius=1.25,
)

NBA = Court(
    model="NBA",
    length=28.65,
    width=15.24,
    three_pt_radius=7.24,
    three_pt_corner_dist=0.91,
    free_throw_dist=5.79,
    free_throw_width=3.66,
    restricted_radius=1.22,
)

COURT_3X3 = Court(
    model="3x3",
    length=15.0,
    width=11.0,
    three_pt_radius=6.75,
    three_pt_corner_dist=0.90,
    free_throw_dist=5.80,
    free_throw_width=3.60,
    restricted_radius=1.25,
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
    kps.append(CourtKeypoint(f"{side}_corner_left", (bl, -(hw - court.three_pt_corner_dist))))
    kps.append(CourtKeypoint(f"{side}_corner_right", (bl, hw - court.three_pt_corner_dist)))

    rim_x, rim_y = court.rim_pos(side)
    kps.append(CourtKeypoint(f"{side}_rim", (rim_x, rim_y)))

    ft_x = bl - court.free_throw_dist * (1 if side == "A" else -1)
    ft_hw = court.free_throw_width / 2
    kps.append(CourtKeypoint(f"{side}_ft_left", (ft_x, -ft_hw)))
    kps.append(CourtKeypoint(f"{side}_ft_right", (ft_x, ft_hw)))

    import math
    n_arc = 9
    sign = 1 if side == "A" else -1
    for i in range(n_arc + 1):
        theta = -math.pi / 2 + math.pi * i / n_arc
        px = rim_x - sign * court.three_pt_radius * math.cos(theta)
        py = court.three_pt_radius * math.sin(theta)
        if abs(py) <= hw - court.three_pt_corner_dist:
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


__all__ = ["Court", "CourtKeypoint", "CourtModel", "COURTS", "FIBA", "NBA", "COURT_3X3",
           "court_keypoints", "all_court_keypoints"]
