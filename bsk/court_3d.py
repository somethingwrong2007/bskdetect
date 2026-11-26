"""True 3D basketball court model with perspective rendering.

Renders the FIBA court from arbitrary camera viewpoints using a software
rasterizer (numpy + PIL only, no GPU / no heavy 3D library required).
Generates both RGB images and 6-class segmentation masks for synthetic
training data.

Coordinate system (same as bsk.court):
    Origin at court centre.
    x  – along the length (team-A attacks in +x).
    y  – along the width.
    z  – up (floor at z = 0).

All dimensions come from ``bsk.court.Court``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw

from bsk.court import Court, FIBA

# ── Segmentation class indices ──────────────────────────────────────────
CL_BACKGROUND = 0
CL_BOUNDARY = 1
CL_THREE_POINT = 2
CL_PAINT = 3
CL_CENTER_LINE = 4
CL_CENTER_CIRCLE = 5
NUM_CLASSES = 6

CLASS_NAMES = [
    "background", "boundary", "three_point",
    "paint", "center_line", "center_circle",
]

# Colour used when *colourising* masks for visualisation (RGB).
CLASS_VIS_COLORS = [
    (0, 0, 0),          # background – black
    (255, 255, 255),     # boundary
    (255, 220, 60),      # three-point  – gold
    (80, 220, 80),       # paint        – green
    (80, 140, 255),      # center-line  – blue
    (220, 80, 220),      # center-circle – magenta
]

# 线条抬离地面少许，避免与地板共面；线宽由 Court.line_width 决定
# （FIBA 5 cm / NBA 2 in），不再使用统一加粗值。
LINE_H = 0.008
FLOOR_THICKNESS = 0.05


# ======================================================================
#  3-D court mesh
# ======================================================================

@dataclass
class _Poly3D:
    """One flat polygon lying on (or just above) the court floor.

    *vertices* – list of (x, y, z) tuples (world space).
    *class_id* – segmentation class index.
    *fill_only* – 只用于地板填色（例如罚球区大面积漆块），不参与线标签渲染。
    """
    vertices: list[tuple[float, float, float]]
    class_id: int
    fill_only: bool = False


@dataclass
class CourtMesh3D:
    """Complete 3-D mesh of a basketball court.

    ``polys`` is a flat list of :class:`_Poly3D` that the renderer
    rasterises.  The floor is included so that the court has a visible
    surface even where no line marking is present.
    """
    court: Court
    polys: list[_Poly3D] = field(default_factory=list)

    # ── public factory ──────────────────────────────────────────────
    @classmethod
    def build(cls, court: Court | None = None) -> "CourtMesh3D":
        obj = cls(court=court or FIBA)
        obj._build()
        return obj

    # ── geometry construction ───────────────────────────────────────
    def _add(self, verts, class_id, fill_only: bool = False):
        self.polys.append(_Poly3D(list(verts), class_id, fill_only))

    def _build(self):
        c = self.court
        hl, hw = c.half_length, c.half_width
        z0, z1 = 0.0, FLOOR_THICKNESS
        # 场地尺寸量到界线内沿；地板要覆盖到界线外沿才能把边线画在地板上。
        eh, ew = hl + c.line_width, hw + c.line_width

        # ---- floor slab (background) ----
        self._add([
            (-eh, -ew, z0), (eh, -ew, z0),
            (eh, ew, z0), (-eh, ew, z0),
        ], CL_BACKGROUND)

        # ---- per-half features ----
        for side in ("A", "B"):
            self._paint_area(side)
            self._three_point_line(side)
            self._free_throw_circle(side)
            self._restricted_area(side)

        # ---- boundary ----
        self._boundary()
        # ---- centre line & circle ----
        self._center_line()
        self._center_circle()

    # ---- boundary --------------------------------------------------
    def _boundary(self):
        c = self.court
        hl, hw = c.half_length, c.half_width
        z0, z1 = 0.0, LINE_H
        w = c.line_width
        # 场地内沿在 ±hl/±hw，界线中心线向外移半个线宽。
        bx, by = hl + w * 0.5, hw + w * 0.5
        self._add_line(-bx, -by, bx, -by, w, z0, z1, CL_BOUNDARY)
        self._add_line(-bx, by, bx, by, w, z0, z1, CL_BOUNDARY)
        self._add_line(-bx, -by, -bx, by, w, z0, z1, CL_BOUNDARY)
        self._add_line(bx, -by, bx, by, w, z0, z1, CL_BOUNDARY)

    # ---- three-point line ------------------------------------------
    def _three_point_line(self, side: Literal["A", "B"]):
        c = self.court
        rim_x, _ = c.rim_pos(side)
        sign = 1.0 if side == "A" else -1.0
        bl = c.baseline_x(side)          # 端线内沿
        hw = c.half_width
        # 官方尺寸量到线外沿，这里全部换算到中心线。
        R = c.three_pt_center_radius
        y_limit = hw - c.three_pt_corner_center_dist

        dx = math.sqrt(max(R * R - y_limit * y_limit, 0.0))
        arc_x = rim_x - sign * dx

        z0, z1 = 0.0, LINE_H
        w = c.line_width

        # 底角直线段：与圆弧中心线精确相接
        self._add_line(bl, y_limit, arc_x, y_limit, w, z0, z1, CL_THREE_POINT)
        self._add_line(bl, -y_limit, arc_x, -y_limit, w, z0, z1, CL_THREE_POINT)

        # 圆弧朝球场中心鼓出：x = rim_x - sign * R * cos(theta)
        n_arc = 80
        half = math.asin(min(1.0, y_limit / R))
        angles = np.linspace(-half, half, n_arc + 1)
        for i in range(n_arc):
            a0, a1 = angles[i], angles[i + 1]
            p0 = (rim_x - sign * R * math.cos(a0), R * math.sin(a0))
            p1 = (rim_x - sign * R * math.cos(a1), R * math.sin(a1))
            self._add_line(p0[0], p0[1], p1[0], p1[1], w, z0, z1, CL_THREE_POINT)

    # ---- paint area ------------------------------------------------
    def _paint_area(self, side: Literal["A", "B"]):
        c = self.court
        bl = c.baseline_x(side)          # 端线内沿
        sign = 1.0 if side == "A" else -1.0
        ft_x = bl - sign * c.free_throw_center_dist
        # 限制区宽度 4.90/4.88 m 量到两侧线内沿；线中心线再向外半个线宽。
        lane_inner = c.paint_width * 0.5
        lane_line = lane_inner + c.line_half
        z0, z1 = 0.0, LINE_H
        w = c.line_width
        # 漆块只用于地板填色；标签里只保留边框线，避免整块面积被当成线。
        self._add([
            (bl, -lane_inner, z0), (ft_x, -lane_inner, z0),
            (ft_x, lane_inner, z0), (bl, lane_inner, z0),
        ], CL_PAINT, fill_only=True)
        self._add_line(bl, -lane_line, ft_x, -lane_line, w, z0, z1, CL_PAINT)
        self._add_line(bl, lane_line, ft_x, lane_line, w, z0, z1, CL_PAINT)
        # 延长罚球线（限制区上沿）连到两侧线中心
        self._add_line(ft_x, -lane_line, ft_x, lane_line, w, z0, z1, CL_PAINT)

    # ---- free-throw circle (line ring) -----------------------------
    def _free_throw_circle(self, side: Literal["A", "B"]):
        c = self.court
        bl = c.baseline_x(side)
        sign = 1.0 if side == "A" else -1.0
        ft_x = bl - sign * c.free_throw_center_dist
        radius = c.free_throw_circle_center_radius
        z0, z1 = 0.0, LINE_H
        w = c.line_width
        if not c.free_throw_dashed_inside:
            self._circle_ring(ft_x, 0.0, radius, 80, w, z0, z1, CL_PAINT)
            return
        # FIBA：限制区外半圆实线，限制区内半圆虚线。
        if side == "A":
            solid = (math.pi / 2, 3 * math.pi / 2)
            dashed = (-math.pi / 2, math.pi / 2)
        else:
            solid = (-math.pi / 2, math.pi / 2)
            dashed = (math.pi / 2, 3 * math.pi / 2)
        self._solid_arc(ft_x, 0.0, radius, solid[0], solid[1], 48, w, z0, z1, CL_PAINT)
        self._dashed_arc(ft_x, 0.0, radius, dashed[0], dashed[1], w, 0.38, 0.28,
                         z0, z1, CL_PAINT)

    # ---- restricted area (no-charge semi-circle) --------------------
    def _restricted_area(self, side: Literal["A", "B"]):
        """合理冲撞区：篮下小半圆 + 两条垂直端线的短线（到篮板前沿）。"""
        c = self.court
        rim_x, _ = c.rim_pos(side)       # 篮筐中心投影
        bl = c.baseline_x(side)
        sign = 1.0 if side == "A" else -1.0
        r = c.restricted_center_radius
        back_x = bl - sign * c.backboard_from_baseline
        z0, z1 = 0.0, LINE_H
        w = c.line_width

        # 半圆朝球场中心张开；两端点正好落在 y = ±r、x = rim_x。
        # A 侧基线在 +x，弧必须经过 π（朝 -x）；B 侧经过 0（朝 +x）。
        if side == "A":
            a0, a1 = math.pi / 2, 3 * math.pi / 2
        else:
            a0, a1 = -math.pi / 2, math.pi / 2
        self._solid_arc(rim_x, 0.0, r, a0, a1, 40, w, z0, z1, CL_PAINT)

        # 官方规则：两条短线垂直端线，长度 0.375 m，止于篮板前沿；
        # 不是一直画到端线。
        for sy in (-1.0, 1.0):
            self._add_line(back_x, sy * r, rim_x, sy * r, w, z0, z1, CL_PAINT)

    # ---- centre line -----------------------------------------------
    def _center_line(self):
        c = self.court
        hw = c.half_width
        ext = c.line_half + c.center_line_extension
        z0, z1 = 0.0, LINE_H
        self._add_line(0.0, -(hw + ext), 0.0, hw + ext, c.line_width, z0, z1, CL_CENTER_LINE)

    # ---- centre circle (line ring) ---------------------------------
    def _center_circle(self):
        radius = self.court.center_circle_center_radius
        z0, z1 = 0.0, LINE_H
        self._circle_ring(0.0, 0.0, radius, 80, self.court.line_width, z0, z1, CL_CENTER_CIRCLE)

    # ── geometric helpers ───────────────────────────────────────────
    @staticmethod
    def _line_seg_quad(x1, y1, x2, y2, w, z0, z1, class_id):
        """Return four vertices of a thin horizontal quad along a line."""
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return None
        # perpendicular in the xy-plane, scaled to half line-width
        px = -dy / length * w * 0.5
        py = dx / length * w * 0.5
        return [
            (x1 + px, y1 + py, z1), (x2 + px, y2 + py, z1),
            (x2 - px, y2 - py, z1), (x1 - px, y1 - py, z1),
        ]

    def _add_line(self, x1, y1, x2, y2, w, z0, z1, class_id):
        """Add a thin quad for a line segment to the mesh."""
        q = self._line_seg_quad(x1, y1, x2, y2, w, z0, z1, class_id)
        if q is not None:
            self._add(q, class_id)

    def _circle_ring(self, cx, cy, r, n, w, z0, z1, class_id):
        """Tessellate a circle ring into short quads."""
        self._solid_arc(cx, cy, r, 0.0, 2.0 * math.pi, n, w, z0, z1, class_id)

    def _solid_arc(self, cx, cy, r, a0, a1, n, w, z0, z1, class_id):
        n = max(2, int(n))
        prev = None
        for i in range(n + 1):
            a = a0 + (a1 - a0) * i / n
            p = (cx + r * math.cos(a), cy + r * math.sin(a))
            if prev is not None:
                q = self._line_seg_quad(prev[0], prev[1], p[0], p[1], w, z0, z1, class_id)
                if q is not None:
                    self._add(q, class_id)
            prev = p

    def _dashed_arc(self, cx, cy, r, a0, a1, w, dash, gap, z0, z1, class_id):
        """沿圆弧画虚线；dash/gap 单位为米。"""
        if r <= 1e-6:
            return
        direction = 1.0 if a1 >= a0 else -1.0
        arc_len = abs(a1 - a0) * r
        pos = 0.0
        while pos < arc_len - 1e-6:
            end = min(pos + dash, arc_len)
            sa = a0 + direction * pos / r
            ea = a0 + direction * end / r
            self._solid_arc(cx, cy, r, sa, ea, max(3, int(dash / 0.03)),
                            w, z0, z1, class_id)
            pos += dash + gap


# ======================================================================
#  3-D renderer  (software rasteriser, numpy + PIL)
# ======================================================================

class Court3DRenderer:
    """Renders a :class:`CourtMesh3D` from an arbitrary camera pose.

    Camera model
    ------------
    *pos*      – (x, y, z) camera centre in world space.
    *look_at*  – (x, y, z) point the camera looks at.
    *up*       – world-up direction, default ``(0, 0, 1)``.
    *focal_length* – focal length in **pixels**.
    *image_size*   – ``(H, W)`` of the output image.
    """

    # Floor colour presets (RGB) — different arena floor styles
    FLOOR_STYLES = {
        "maple": np.array([220, 185, 140], dtype=np.float64),      # light maple (NBA standard)
        "oak": np.array([185, 155, 110], dtype=np.float64),        # warm oak
        "walnut": np.array([140, 110, 75], dtype=np.float64),      # dark walnut
        "cherry": np.array([160, 100, 70], dtype=np.float64),      # reddish cherry
        "ash": np.array([200, 190, 160], dtype=np.float64),        # pale ash
        "bamboo": np.array([210, 195, 130], dtype=np.float64),     # yellowish bamboo
    }

    _BG_RGB = np.array([35, 35, 45], dtype=np.float64)
    _LINE_RGB = np.array([245, 245, 245], dtype=np.float64)

    def __init__(self, court_mesh: CourtMesh3D | None = None, floor_style: str = "oak"):
        self.mesh = court_mesh or CourtMesh3D.build()
        self.floor_style = floor_style
        self._FLOOR_RGB = self.FLOOR_STYLES.get(floor_style, self.FLOOR_STYLES["oak"])
        # Paint area is slightly darker than floor
        self._PAINT_RGB = self._FLOOR_RGB * 0.85
        self._PAINT2_RGB = self._FLOOR_RGB * 0.80

    # ── camera basis ────────────────────────────────────────────────
    @staticmethod
    def _view_basis(pos, look_at, up):
        fwd = np.array(look_at, dtype=np.float64) - np.array(pos, dtype=np.float64)
        fwd /= np.linalg.norm(fwd) + 1e-12
        right = np.cross(fwd, np.array(up, dtype=np.float64))
        rn = np.linalg.norm(right)
        if rn < 1e-8:
            # degenerate – pick an arbitrary right
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= rn
        true_up = np.cross(right, fwd)
        return right, true_up, fwd

    # ── perspective projection ──────────────────────────────────────
    def _project(self, pts_w, pos, R, U, F, H, W, fl):
        """Project Nx3 world points → 2-D pixel coords + depth.

        Returns (px_col, px_row, depth, visible_mask).
        """
        pts = np.asarray(pts_w, dtype=np.float64)          # (N,3)
        rel = pts - pos                                     # (N,3)
        cz = rel @ F                                        # depth in cam frame
        cx = rel @ R
        cy = rel @ U
        vis = cz > 0.1                                      # in front of camera
        safe_z = np.where(vis, cz, 1.0)
        col = cx / safe_z * fl + W * 0.5
        row = -cy / safe_z * fl + H * 0.5
        return col, row, cz, vis

    # ── polygon rasterisation (PIL-based, painter's algorithm) ──────
    def _render_polys(self, cam, H, W, class_filter=None, colour_map=None,
                      bg_rgb=None, skip_fill_only=False):
        """Core render loop using PIL ImageDraw with painter's algorithm."""
        pos = np.array(cam["pos"], dtype=np.float64)
        look_at = np.array(cam["look_at"], dtype=np.float64)
        up = np.array(cam.get("up", (0, 0, 1)), dtype=np.float64)
        fl = cam.get("focal_length", 1000.0)

        R, U, F = self._view_basis(pos, look_at, up)

        # Background
        bg = tuple(int(v) for v in (bg_rgb or self._BG_RGB))
        img = Image.new("RGB", (W, H), bg)
        draw = ImageDraw.Draw(img)

        # Collect and sort polygons by average depth (far first)
        render_list = []
        for poly in self.mesh.polys:
            if skip_fill_only and poly.fill_only:
                continue
            if class_filter is not None and poly.class_id not in class_filter:
                continue
            if colour_map is not None and poly.class_id not in colour_map:
                continue
            colour = (colour_map or {}).get(poly.class_id)
            if colour is None:
                colour = self._colour_for_class(poly.class_id)

            verts_w = np.array(poly.vertices, dtype=np.float64)
            cols, rows, deps, vis = self._project(verts_w, pos, R, U, F, H, W, fl)
            if vis.sum() < 3:
                continue

            avg_depth = float(deps[vis].mean())
            pts_2d = list(zip(cols[vis].tolist(), rows[vis].tolist()))
            render_list.append((avg_depth, pts_2d, colour))

        # Painter's algorithm: far polygons first
        render_list.sort(key=lambda x: -x[0])

        for _depth, pts, colour in render_list:
            int_pts = [(int(round(x)), int(round(y))) for x, y in pts]
            if len(int_pts) >= 3:
                draw.polygon(int_pts, fill=tuple(int(c) for c in colour))

        return np.array(img, dtype=np.float64), None

    def _colour_for_class(self, class_id):
        if class_id == CL_BACKGROUND:
            return self._FLOOR_RGB
        if class_id == CL_PAINT:
            return self._PAINT_RGB
        if class_id == CL_CENTER_CIRCLE:
            return self._PAINT2_RGB
        return self._LINE_RGB

    # ── public API ──────────────────────────────────────────────────
    def render_rgb(self, cam: dict) -> np.ndarray:
        """Return ``[H, W, 3]`` uint8 RGB image."""
        H, W = cam["image_size"]
        buf, _ = self._render_polys(cam, H, W)
        return np.clip(buf, 0, 255).astype(np.uint8)

    def render_mask(self, cam: dict, class_id: int) -> np.ndarray:
        """Return ``[H, W]`` uint8 binary mask for *class_id*."""
        H, W = cam["image_size"]
        cmap = {class_id: (255.0, 255.0, 255.0)}
        # Use black background so non-zero = class pixel
        buf, _ = self._render_polys(cam, H, W,
                                    class_filter={class_id},
                                    colour_map=cmap,
                                    bg_rgb=(0, 0, 0))
        return buf[:, :, 0].astype(np.uint8)

    def render_all_masks(self, cam: dict) -> np.ndarray:
        """Return ``[NUM_CLASSES, H, W]`` uint8 binary masks."""
        H, W = cam["image_size"]
        masks = np.zeros((NUM_CLASSES, H, W), dtype=np.uint8)
        for cid in range(1, NUM_CLASSES):
            masks[cid] = self.render_mask(cam, cid)
        return masks

    def render_sample(self, cam: dict):
        """Return ``(rgb, masks)`` – one RGB image + [6,H,W] masks."""
        return self.render_rgb(cam), self.render_all_masks(cam)

    def render_label(self, cam: dict) -> np.ndarray:
        """单次渲染直接输出 [H,W] 类别图（0..5）。

        render_all_masks 每类渲一遍要走 5 次光栅化；训练数据生成时按类别
        上色一次画完再解码回来，速度快约 5 倍。
        """
        H, W = cam["image_size"]
        palette = {cid: (float(cid * 51), 0.0, 0.0) for cid in range(NUM_CLASSES)}
        buf, _ = self._render_polys(
            cam, H, W,
            class_filter=set(range(1, NUM_CLASSES)),  # 不画地板：画家算法会用它盖掉线
            colour_map=palette,
            bg_rgb=(0, 0, 0),
            skip_fill_only=True,  # 漆块只在 RGB 里填色，标签里用边框线
        )
        return np.rint(buf[:, :, 0] / 51.0).astype(np.uint8)

    def render_floor_only(self, cam: dict, with_paint: bool = True) -> np.ndarray:
        """只渲染地板（不含任何线），用作多颜色地板的背景。

        with_paint=True 时把罚球区漆块也画上（用 ``_PAINT_RGB``），
        这样能生成"木地板 + 彩色漆区"的真实球场配色。
        """
        H, W = cam["image_size"]
        ids = {CL_BACKGROUND, CL_PAINT} if with_paint else {CL_BACKGROUND}
        buf, _ = self._render_polys(cam, H, W, class_filter=ids)
        return np.clip(buf, 0, 255).astype(np.uint8)

    def render_floor_mask(self, cam: dict, with_paint: bool = True) -> np.ndarray:
        """地板（含漆区）的覆盖掩码 [H, W]，255=地板，0=其它。

        用于把渲染地板贴到真实背景上，遮挡住真实画面里不一致的场地线。
        """
        H, W = cam["image_size"]
        white = (255.0, 255.0, 255.0)
        cmap = {CL_BACKGROUND: white, CL_PAINT: white}
        buf, _ = self._render_polys(
            cam, H, W, class_filter=set(cmap), colour_map=cmap, bg_rgb=(0, 0, 0),
        )
        return buf[:, :, 0].astype(np.uint8)

    def render_rgb_line_only(self, cam: dict) -> np.ndarray:
        """只渲染线（不含地板），返回 RGB 线层，便于软件叠合。"""
        H, W = cam["image_size"]
        palette = {cid: self._colour_for_class(cid) for cid in range(1, NUM_CLASSES)}
        buf, _ = self._render_polys(
            cam, H, W, class_filter=set(range(1, NUM_CLASSES)),
            colour_map=palette, bg_rgb=(0, 0, 0),
        )
        return np.clip(buf, 0, 255).astype(np.uint8)


# ======================================================================
#  Random broadcast camera
# ======================================================================

def random_broadcast_camera(
    image_size: tuple[int, int] = (384, 384),
    rng: np.random.Generator | None = None,
) -> dict:
    """Generate a random but realistic broadcast-camera pose.

    The camera orbits at 8–20 m height, 12–30 m from court centre,
    always looking roughly at the court.
    """
    if rng is None:
        rng = np.random.default_rng()

    height = rng.uniform(8.0, 20.0)
    azimuth = rng.uniform(0, 2 * math.pi)
    distance = rng.uniform(12.0, 30.0)

    cx = distance * math.cos(azimuth)
    cy = distance * math.sin(azimuth)
    pos = (cx, cy, height)

    # look at a random point near court centre
    lx = rng.uniform(-5.0, 5.0)
    ly = rng.uniform(-3.0, 3.0)
    look_at = (lx, ly, 0.0)

    return {
        "pos": pos,
        "look_at": look_at,
        "up": (0.0, 0.0, 1.0),
        "focal_length": float(rng.uniform(700, 1400)),
        "image_size": image_size,
    }


# ======================================================================
#  End-to-end sample generation
# ======================================================================

def generate_3d_sample(
    renderer: Court3DRenderer,
    output_size: int = 384,
    rng: np.random.Generator | None = None,
    random_floor: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate one synthetic training sample.

    Returns ``(image, masks, camera_params)`` where
    *image* is ``[H, W, 3]`` uint8 RGB,
    *masks* is ``[6, H, W]`` uint8.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Random floor style for diversity
    if random_floor:
        style = rng.choice(list(Court3DRenderer.FLOOR_STYLES.keys()))
        renderer.floor_style = style
        renderer._FLOOR_RGB = renderer.FLOOR_STYLES[style]
        renderer._PAINT_RGB = renderer._FLOOR_RGB * 0.85
        renderer._PAINT2_RGB = renderer._FLOOR_RGB * 0.80

    cam = random_broadcast_camera((output_size, output_size), rng)
    rgb, masks = renderer.render_sample(cam)

    # ── add realistic noise / post-processing to RGB only ──────────
    img = rgb.astype(np.float64)

    # Gaussian noise
    noise = rng.normal(0, 6, img.shape)
    img = img + noise

    # Brightness jitter
    brightness = rng.uniform(0.85, 1.15)
    img = img * brightness

    # Simple vignette
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx_ = H / 2.0, W / 2.0
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx_) ** 2)
    max_d = math.hypot(cy, cx_)
    vignette = 1.0 - 0.25 * (dist / max_d)
    img = img * vignette[:, :, np.newaxis]

    img = np.clip(img, 0, 255).astype(np.uint8)

    # Apply floor pattern (striped, checkerboard, etc.)
    if rng.random() < 0.3:
        img = _apply_floor_pattern(img, masks, rng)

    return img, masks, cam


def _apply_floor_pattern(img: np.ndarray, masks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply floor patterns (stripes, checkerboard) to background areas."""
    H, W = img.shape[:2]
    bg_mask = masks[0] == 0  # background = floor area

    if not bg_mask.any():
        return img

    pattern_type = rng.choice(["stripes", "checkerboard", "planks"])

    if pattern_type == "stripes":
        # Horizontal or vertical stripes
        stripe_w = rng.integers(20, 60)
        stripe_h = rng.integers(20, 60)
        if rng.random() < 0.5:
            # Horizontal stripes
            pattern = np.zeros((H, W), dtype=np.float64)
            for y in range(0, H, stripe_h * 2):
                pattern[y:y + stripe_h, :] = 1.0
        else:
            # Vertical stripes
            pattern = np.zeros((H, W), dtype=np.float64)
            for x in range(0, W, stripe_w * 2):
                pattern[:, x:x + stripe_w] = 1.0
        # Apply subtle variation
        variation = (pattern - 0.5) * 30
        img[bg_mask] = np.clip(img[bg_mask].astype(np.float64) + variation[bg_mask, np.newaxis], 0, 255).astype(np.uint8)

    elif pattern_type == "checkerboard":
        # Checkerboard pattern
        cell_size = rng.integers(30, 80)
        pattern = np.zeros((H, W), dtype=np.float64)
        for y in range(0, H, cell_size):
            for x in range(0, W, cell_size):
                if ((y // cell_size) + (x // cell_size)) % 2 == 0:
                    pattern[y:y + cell_size, x:x + cell_size] = 1.0
        variation = (pattern - 0.5) * 25
        img[bg_mask] = np.clip(img[bg_mask].astype(np.float64) + variation[bg_mask, np.newaxis], 0, 255).astype(np.uint8)

    elif pattern_type == "planks":
        # Wood plank pattern (parallel lines)
        plank_w = rng.integers(40, 100)
        pattern = np.zeros((H, W), dtype=np.float64)
        for x in range(0, W, plank_w):
            pattern[:, x:x + 3] = 1.0  # plank gaps
        variation = (pattern - 0.3) * 20
        img[bg_mask] = np.clip(img[bg_mask].astype(np.float64) + variation[bg_mask, np.newaxis], 0, 255).astype(np.uint8)

    return img


# ======================================================================
#  Visualisation helpers
# ======================================================================

def colorize_masks(masks: np.ndarray) -> np.ndarray:
    """Convert ``[NUM_CLASSES, H, W]`` masks to ``[H, W, 3]`` RGB image."""
    _, H, W = masks.shape
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    for cid in range(1, NUM_CLASSES):
        color = CLASS_VIS_COLORS[cid]
        vis[masks[cid] > 0] = color
    return vis


# ======================================================================
#  Deterministic test cameras (for the __main__ block)
# ======================================================================

def _test_cameras(image_size=(384, 384)):
    """Three fixed broadcast-style cameras for visual testing."""
    return [
        {   # high side-line angle
            "pos": (18.0, 14.0, 14.0),
            "look_at": (0.0, 0.0, 0.0),
            "up": (0, 0, 1),
            "focal_length": 900.0,
            "image_size": image_size,
        },
        {   # low corner angle
            "pos": (-16.0, -10.0, 8.0),
            "look_at": (2.0, 1.0, 0.0),
            "up": (0, 0, 1),
            "focal_length": 1100.0,
            "image_size": image_size,
        },
        {   # nearly top-down
            "pos": (2.0, 0.5, 22.0),
            "look_at": (0.0, 0.0, 0.0),
            "up": (0, 1, 0),
            "focal_length": 600.0,
            "image_size": image_size,
        },
    ]


# ======================================================================
#  Main – generate test images
# ======================================================================

if __name__ == "__main__":
    print("Building FIBA court 3-D mesh …")
    mesh = CourtMesh3D.build(FIBA)
    print(f"  {len(mesh.polys)} polygons")

    renderer = Court3DRenderer(mesh)
    out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    cams = _test_cameras()
    for idx, cam in enumerate(cams):
        print(f"\n── Sample {idx} ──")
        print(f"  pos      = {cam['pos']}")
        print(f"  look_at  = {cam['look_at']}")
        print(f"  up       = {cam['up']}")
        print(f"  focal    = {cam['focal_length']:.0f} px")

        rgb = renderer.render_rgb(cam)
        masks = renderer.render_all_masks(cam)

        # save RGB
        rgb_path = os.path.join(out_dir, f"test_3d_{idx}.png")
        Image.fromarray(rgb).save(rgb_path)
        print(f"  RGB  → {rgb_path}  {rgb.shape}")

        # save colourised mask
        mask_vis = colorize_masks(masks)
        mask_path = os.path.join(out_dir, f"test_3d_{idx}_mask.png")
        Image.fromarray(mask_vis).save(mask_path)
        print(f"  mask → {mask_path}  {mask_vis.shape}")

        # per-class pixel counts
        for cid in range(NUM_CLASSES):
            cnt = int(np.count_nonzero(masks[cid]))
            print(f"    [{cid}] {CLASS_NAMES[cid]:15s}: {cnt:>8,} px")

    # Also generate one sample with random broadcast camera + noise
    print("\n── Random broadcast sample ──")
    rng = np.random.default_rng(42)
    img, masks, cam = generate_3d_sample(renderer, 384, rng)
    print(f"  pos      = {cam['pos']}")
    print(f"  look_at  = {cam['look_at']}")
    print(f"  focal    = {cam['focal_length']:.0f} px")
    p = os.path.join(out_dir, "test_3d_random.png")
    Image.fromarray(img).save(p)
    print(f"  saved → {p}")
    mv = colorize_masks(masks)
    Image.fromarray(mv).save(os.path.join(out_dir, "test_3d_random_mask.png"))

    print("\nDone.")
