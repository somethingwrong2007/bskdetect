"""Pixel-perfect 2D top-down court template renderer for segmentation training data.

Renders RGB images and 6-class binary masks from the FIBA/NBA/3x3 Court model.
All dimensions come from bsk.court — nothing is hardcoded here.
"""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np

from bsk.court import Court, FIBA


class CourtTemplate:
    """Renders a top-down view of a basketball court at a given resolution."""

    def __init__(self, court: Court | None = None):
        self.court = court if court is not None else FIBA

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _canvas_size(self, ppm: int) -> tuple[int, int]:
        """Return (width, height) in pixels including 2-meter margin."""
        margin = 2.0
        w = int(math.ceil((self.court.length + 2 * margin) * ppm))
        h = int(math.ceil((self.court.width + 2 * margin) * ppm))
        return w, h

    def _margin_px(self, ppm: int) -> float:
        return 2.0 * ppm

    def _to_px(self, x_m: float, y_m: float, ppm: int) -> tuple[int, int]:
        """Convert court coords (origin=center, x=length, y=width) to pixel coords.

        In the image: col increases rightward (same as x), row increases downward.
        We flip y so that positive y is upward on screen (row = margin - y*ppm).
        """
        margin = self._margin_px(ppm)
        col = int(round(x_m * ppm + margin + self.court.half_length * ppm))
        row = int(round(-y_m * ppm + margin + self.court.half_width * ppm))
        return col, row

    def _to_px_arr(self, pts_m: list[tuple[float, float]], ppm: int) -> np.ndarray:
        """Convert list of (x,y) meter points to (N,1,2) int32 array for cv2."""
        pts = np.array([[self._to_px(x, y, ppm)] for x, y in pts_m], dtype=np.int32)
        return pts

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def _draw_line(self, mask: np.ndarray, p1: tuple[float, float],
                   p2: tuple[float, float], ppm: int, thickness: int = 3) -> None:
        c1, r1 = self._to_px(p1[0], p1[1], ppm)
        c2, r2 = self._to_px(p2[0], p2[1], ppm)
        cv2.line(mask, (c1, r1), (c2, r2), 255, thickness)

    def _draw_polyline(self, mask: np.ndarray, pts: list[tuple[float, float]],
                       ppm: int, closed: bool = False, thickness: int = 3) -> None:
        if len(pts) < 2:
            return
        arr = self._to_px_arr(pts, ppm)
        cv2.polylines(mask, [arr], closed, 255, thickness)

    def _draw_circle(self, mask: np.ndarray, center: tuple[float, float],
                     radius_m: float, ppm: int, thickness: int = 3) -> None:
        cx, cy = self._to_px(center[0], center[1], ppm)
        r_px = int(round(radius_m * ppm))
        cv2.circle(mask, (cx, cy), r_px, 255, thickness)

    def _draw_arc(self, mask: np.ndarray, center: tuple[float, float],
                  radius_m: float, angle_start: float, angle_end: float,
                  ppm: int, thickness: int = 3, num_segments: int = 120) -> None:
        """Draw a circular arc from angle_start to angle_end (radians)."""
        pts = []
        # Handle wraparound: if going from positive to negative, add 2*pi to end
        if angle_start > angle_end:
            angle_end += 2 * math.pi
        for i in range(num_segments + 1):
            t = angle_start + (angle_end - angle_start) * i / num_segments
            px = center[0] + radius_m * math.cos(t)
            py = center[1] + radius_m * math.sin(t)
            pts.append(px)
            pts.append(py)
        pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        # Convert to pixel coords
        margin = self._margin_px(ppm)
        ox = margin + self.court.half_length * ppm
        oy = margin + self.court.half_width * ppm
        px_pts = pts_arr.copy()
        px_pts[:, 0, 0] = px_pts[:, 0, 0] * ppm + ox
        px_pts[:, 0, 1] = -px_pts[:, 0, 1] * ppm + oy
        px_pts = px_pts.astype(np.int32)
        cv2.polylines(mask, [px_pts], False, 255, thickness)

    # ------------------------------------------------------------------
    # Three-point line geometry
    # ------------------------------------------------------------------

    def _three_pt_arc_angles(self, side: Literal["A", "B"]) -> tuple[float, float]:
        """三分弧在 ``(cx + R cos t, cy + R sin t)`` 参数化下的角度范围。

        球心在篮筐投影点；弧朝球场中心鼓出。A 侧经过 π，B 侧经过 0。
        """
        c = self.court
        y_limit = c.half_width - c.three_pt_corner_center_dist
        half = math.asin(min(1.0, y_limit / c.three_pt_center_radius))
        if side == "A":
            return math.pi - half, math.pi + half
        return -half, half

    def _draw_three_point_line(self, mask: np.ndarray, side: Literal["A", "B"],
                               ppm: int, thickness: int = 3) -> None:
        c = self.court
        bl = c.baseline_x(side)
        sign = 1.0 if side == "A" else -1.0
        R = c.three_pt_center_radius
        y_limit = c.half_width - c.three_pt_corner_center_dist
        rim_x, rim_y = c.rim_pos(side)

        dx = math.sqrt(max(R * R - y_limit * y_limit, 0.0))
        arc_x = rim_x - sign * dx

        self._draw_line(mask, (bl, y_limit), (arc_x, y_limit), ppm, thickness)
        self._draw_line(mask, (bl, -y_limit), (arc_x, -y_limit), ppm, thickness)

        a0, a1 = self._three_pt_arc_angles(side)
        self._draw_arc(mask, (rim_x, rim_y), R, a0, a1, ppm, thickness,
                       num_segments=200)

    def _draw_dashed_arc(self, mask: np.ndarray, center: tuple[float, float],
                         radius_m: float, angle_start: float, angle_end: float,
                         ppm: int, thickness: int = 3, dash: float = 0.38,
                         gap: float = 0.28) -> None:
        total = abs(angle_end - angle_start) * radius_m
        direction = 1.0 if angle_end >= angle_start else -1.0
        pos = 0.0
        while pos < total:
            end = min(pos + dash, total)
            s = angle_start + direction * pos / radius_m
            e = angle_start + direction * end / radius_m
            self._draw_arc(mask, center, radius_m, s, e, ppm, thickness,
                           num_segments=max(3, int(dash / 0.02)))
            pos += dash + gap

    # ------------------------------------------------------------------
    # Free throw area
    # ------------------------------------------------------------------

    def _draw_paint_area(self, mask: np.ndarray, side: Literal["A", "B"],
                         ppm: int, thickness: int = 3) -> None:
        c = self.court
        bl = c.baseline_x(side)
        sign = 1.0 if side == "A" else -1.0
        ft_hw = c.paint_width / 2.0 + c.line_half

        # 限制区矩形：边线中心线 + 延长罚球线
        ft_line_x = bl - sign * c.free_throw_center_dist
        corners = [
            (bl, -ft_hw),
            (ft_line_x, -ft_hw),
            (ft_line_x, ft_hw),
            (bl, ft_hw),
        ]
        self._draw_polyline(mask, corners, ppm, closed=True, thickness=thickness)

        ft_r = c.free_throw_circle_center_radius
        if c.free_throw_dashed_inside:
            if sign > 0:
                self._draw_arc(mask, (ft_line_x, 0.0), ft_r,
                               math.pi / 2, math.pi * 1.5, ppm, thickness)
                self._draw_dashed_arc(mask, (ft_line_x, 0.0), ft_r,
                                      -math.pi / 2, math.pi / 2, ppm, thickness)
            else:
                self._draw_arc(mask, (ft_line_x, 0.0), ft_r,
                               -math.pi / 2, math.pi / 2, ppm, thickness)
                self._draw_dashed_arc(mask, (ft_line_x, 0.0), ft_r,
                                      math.pi / 2, math.pi * 1.5, ppm, thickness)
        else:
            self._draw_circle(mask, (ft_line_x, 0.0), ft_r, ppm, thickness)

    # ------------------------------------------------------------------
    # Center circle
    # ------------------------------------------------------------------

    def _draw_center_circle(self, mask: np.ndarray, ppm: int, thickness: int = 3) -> None:
        self._draw_circle(mask, (0.0, 0.0),
                          self.court.center_circle_center_radius, ppm, thickness)

    # ------------------------------------------------------------------
    # Boundary
    # ------------------------------------------------------------------

    def _draw_boundary(self, mask: np.ndarray, ppm: int, thickness: int = 3) -> None:
        c = self.court
        hl = c.half_length
        hw = c.half_width
        offset = self.court.line_half
        corners = [
            (-hl - offset, -hw - offset),
            (hl + offset, -hw - offset),
            (hl + offset, hw + offset),
            (-hl - offset, hw + offset),
        ]
        self._draw_polyline(mask, corners, ppm, closed=True, thickness=thickness)

    # ------------------------------------------------------------------
    # Center line
    # ------------------------------------------------------------------

    def _draw_center_line(self, mask: np.ndarray, ppm: int, thickness: int = 3) -> None:
        c = self.court
        extent = c.half_width + c.line_half + c.center_line_extension
        self._draw_line(mask, (0.0, -extent), (0.0, extent), ppm, thickness)

    def _draw_restricted_area(self, mask: np.ndarray, side: Literal["A", "B"],
                              ppm: int, thickness: int = 3) -> None:
        c = self.court
        sign = 1.0 if side == "A" else -1.0
        rim_x, _ = c.rim_pos(side)
        back_x = sign * c.half_length - sign * c.backboard_from_baseline
        radius = c.restricted_center_radius
        if sign > 0:
            self._draw_arc(mask, (rim_x, 0.0), radius, math.pi / 2, math.pi * 1.5, ppm, thickness)
        else:
            self._draw_arc(mask, (rim_x, 0.0), radius, -math.pi / 2, math.pi / 2, ppm, thickness)
        self._draw_line(mask, (back_x, -radius), (rim_x, -radius), ppm, thickness)
        self._draw_line(mask, (back_x, radius), (rim_x, radius), ppm, thickness)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_rgb(self, pixels_per_meter: int = 50) -> np.ndarray:
        """Return [H, W, 3] BGR image with white lines on dark background."""
        ppm = pixels_per_meter
        cw, ch = self._canvas_size(ppm)
        thickness = max(2, ppm // 16)

        # Draw all lines on a single combined mask, then apply to RGB
        combined = np.zeros((ch, cw), dtype=np.uint8)

        self._draw_boundary(combined, ppm, thickness)
        self._draw_center_line(combined, ppm, thickness)
        self._draw_center_circle(combined, ppm, thickness)
        for side in ("A", "B"):
            self._draw_three_point_line(combined, side, ppm, thickness)
            self._draw_paint_area(combined, side, ppm, thickness)
            self._draw_restricted_area(combined, side, ppm, thickness)

        # Dark green background with white lines
        img = np.full((ch, cw, 3), 30, dtype=np.uint8)
        img[:, :, 1] = 50
        # Where mask is nonzero, set to white
        white_pixels = combined > 0
        img[white_pixels] = [255, 255, 255]

        return img

    def render_mask(self, pixels_per_meter: int = 50) -> np.ndarray:
        """Return [6, H, W] uint8 array with binary masks for each class.

        Channels:
          0: background (all zeros)
          1: boundary (sidelines + baselines)
          2: three_point (three-point line including straight segments and arc)
          3: paint (free throw lane rectangle + free throw circle)
          4: center_line (midcourt line)
          5: center_circle (center circle)
        """
        ppm = pixels_per_meter
        cw, ch = self._canvas_size(ppm)
        thickness = max(3, ppm // 16)

        # Create single-channel masks
        m_boundary = np.zeros((ch, cw), dtype=np.uint8)
        m_three_pt = np.zeros((ch, cw), dtype=np.uint8)
        m_paint = np.zeros((ch, cw), dtype=np.uint8)
        m_center_line = np.zeros((ch, cw), dtype=np.uint8)
        m_center_circle = np.zeros((ch, cw), dtype=np.uint8)

        # Boundary
        self._draw_boundary(m_boundary, ppm, thickness)

        # Center line
        self._draw_center_line(m_center_line, ppm, thickness)

        # Center circle
        self._draw_center_circle(m_center_circle, ppm, thickness)

        # Both sides
        for side in ("A", "B"):
            self._draw_three_point_line(m_three_pt, side, ppm, thickness)
            self._draw_paint_area(m_paint, side, ppm, thickness)
            self._draw_restricted_area(m_paint, side, ppm, thickness)

        # Stack: channel 0 = background (all zeros)
        bg = np.zeros((ch, cw), dtype=np.uint8)
        result = np.stack([bg, m_boundary, m_three_pt, m_paint, m_center_line, m_center_circle], axis=0)
        return result

    def render_combined(self, pixels_per_meter: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Return (rgb, mask) together."""
        return self.render_rgb(pixels_per_meter), self.render_mask(pixels_per_meter)


# ------------------------------------------------------------------
# Colorized mask for visualization
# ------------------------------------------------------------------

_CLASS_COLORS = [
    (0, 0, 0),        # 0: background - black
    (255, 255, 255),   # 1: boundary - white
    (0, 255, 255),     # 2: three_point - yellow
    (0, 255, 0),       # 3: paint - green
    (255, 0, 0),       # 4: center_line - blue
    (255, 0, 255),     # 5: center_circle - magenta
]

_CLASS_NAMES = [
    "background", "boundary", "three_point", "paint", "center_line", "center_circle"
]


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert [6, H, W] mask to [H, W, 3] BGR colorized image."""
    _, h, w = mask.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for ch in range(1, 6):  # skip background
        color = _CLASS_COLORS[ch]
        img[mask[ch] > 0] = color
    return img


# ------------------------------------------------------------------
# Main: generate test images
# ------------------------------------------------------------------

if __name__ == "__main__":
    import os

    tmpl = CourtTemplate()
    ppm = 50

    rgb, mask = tmpl.render_combined(ppm)

    # Save RGB
    out_dir = os.path.dirname(os.path.abspath(__file__))
    # Go to project root
    proj_dir = os.path.dirname(os.path.dirname(out_dir))
    rgb_path = os.path.join(proj_dir, "test_template_rgb.png")
    cv2.imwrite(rgb_path, rgb)
    print(f"Saved RGB image to {rgb_path}  shape={rgb.shape}")

    # Save colorized mask
    color_mask = colorize_mask(mask)
    mask_path = os.path.join(proj_dir, "test_template_mask.png")
    cv2.imwrite(mask_path, color_mask)
    print(f"Saved colorized mask to {mask_path}  shape={color_mask.shape}")

    # Print mask info
    print(f"\nMask shape: {mask.shape}  (channels, H, W)")
    print(f"Per-class pixel counts:")
    for i, name in enumerate(_CLASS_NAMES):
        count = int(np.count_nonzero(mask[i]))
        print(f"  [{i}] {name:15s}: {count:>10,} pixels")
