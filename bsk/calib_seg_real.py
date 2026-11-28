"""纯合成球场线数据生成：只使用 FIBA/NBA 3D 球场模型。

关键约束：
1. 相机、地板、线、标签必须来自同一套 3D 相机与球场几何，绝不把渲染结果
   贴到真实转播帧上（那会产生两套透视，线必然错位）。
2. 地板颜色/木纹随机，线固定为白色系，避免模型把黄线/彩线当成不同类别。
3. 以边线主机位、罚球、底线、角位、高位为主，补充非常规机位与变焦。
4. 用程序化看台、木纹、灯光、真实位置替补席遮挡和 JPEG 噪声补足真实感。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageDraw

from bsk.court import FIBA, NBA, Court
from bsk.court_3d import (
    CL_PAINT,
    NUM_CLASSES,
    Court3DRenderer,
    CourtMesh3D,
)
from bsk.team_palettes import TEAM_PALETTES, hex_rgb

try:
    _SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
except Exception:
    _SRGB_PROFILE = None

# 线固定为一个颜色，只允许全局灯光把它整体压暗/提亮
LINE_RGB = np.array([242.0, 242.0, 236.0])

WOOD_COLORS = [
    np.array([220, 185, 140], np.float32),   # maple
    np.array([185, 155, 110], np.float32),   # oak
    np.array([140, 110, 75], np.float32),    # walnut
    np.array([160, 100, 70], np.float32),    # cherry
    np.array([200, 190, 160], np.float32),   # ash
    np.array([210, 195, 130], np.float32),   # bamboo
]

SOLID_COLORS = [
    np.array([38, 74, 150], np.float32),
    np.array([28, 102, 66], np.float32),
    np.array([140, 46, 46], np.float32),
    np.array([58, 64, 74], np.float32),
    np.array([188, 146, 52], np.float32),
    np.array([52, 132, 145], np.float32),
    np.array([96, 102, 112], np.float32),
]

PAINT_COLORS = [
    np.array([38, 74, 150], np.float32),
    np.array([28, 102, 66], np.float32),
    np.array([140, 46, 46], np.float32),
    np.array([58, 64, 74], np.float32),
    np.array([188, 146, 52], np.float32),
    np.array([52, 132, 145], np.float32),
]



# ── 相机 ────────────────────────────────────────────────────────

def sample_camera(size: tuple[int, int], court: Court, rng: np.random.Generator,
                  mode: str | None = None) -> dict:
    """常用转播机位为主，少量低位、斜向和贴边非常规机位。"""
    h, w = size
    hl, hw = court.half_length, court.half_width
    profiles = ["broadcast", "opposite", "free_throw", "courtside", "baseline", "corner_high", "half", "high", "close",
                "midcourt", "unusual_low", "unusual_oblique", "unusual_tight"]
    probabilities = [0.17, 0.07, 0.07, 0.08, 0.08, 0.07, 0.09, 0.04, 0.03, 0.24, 0.02, 0.02, 0.02]
    mode = mode or str(rng.choice(profiles, p=probabilities))
    side = float(rng.choice([-1.0, 1.0]))
    unusual = mode.startswith("unusual")

    if mode in ("broadcast", "opposite"):
        poi = (rng.uniform(-3.5, 3.5), rng.uniform(-0.8, 0.8), 0.0)
        pos = (poi[0] + rng.uniform(-3.0, 3.0), side * (hw + rng.uniform(6.0, 11.0)), rng.uniform(8.0, 16.0))
        fl = rng.uniform(850.0, 1550.0)
    elif mode == "free_throw":
        bl = side * hl
        ft_x = bl - side * court.free_throw_center_dist
        poi = (bl - side * court.rim_offset_from_baseline, 0.0, 0.8)
        pos = (ft_x - side * rng.uniform(3.0, 8.0), side * rng.uniform(-5.0, 5.0), rng.uniform(2.8, 6.5))
        fl = rng.uniform(720.0, 1350.0)
    elif mode == "midcourt":
        # 中线与边线交界上方的主转播机位，沿边线横向随机约 ±4.5 m，视线偏向一侧端线。
        target_side = float(rng.choice([-1.0, 1.0]))
        pos = (rng.uniform(-4.5, 4.5), side * (hw + rng.uniform(5.5, 11.0)), rng.uniform(5.0, 9.0))
        poi = (target_side * rng.uniform(0.40, 0.72) * hl + rng.uniform(-0.05, 0.05) * hl,
               rng.uniform(-0.08, 0.08) * hw, rng.uniform(0.6, 1.4))
        fl = rng.uniform(700.0, 1350.0)
    elif mode == "courtside":
        poi = (rng.uniform(-3.5, 3.5), 0.0, 0.0)
        pos = (poi[0] + rng.uniform(-3.0, 3.0), side * (hw + rng.uniform(5.0, 10.0)), rng.uniform(2.2, 4.2))
        fl = rng.uniform(850.0, 1450.0)
    elif mode == "baseline":
        poi = (side * rng.uniform(0.18 * hl, 0.48 * hl), rng.uniform(-2.0, 2.0), 0.0)
        pos = (side * (hl + rng.uniform(3.0, 7.0)), rng.uniform(-5.0, 5.0), rng.uniform(3.0, 7.5))
        fl = rng.uniform(700.0, 1350.0)
    elif mode == "corner_high":
        poi = (rng.uniform(-0.2 * hl, 0.2 * hl), rng.uniform(-0.2 * hw, 0.2 * hw), 0.0)
        pos = (side * (hl + rng.uniform(5.0, 10.0)), -side * (hw + rng.uniform(5.0, 10.0)), rng.uniform(10.0, 20.0))
        fl = rng.uniform(700.0, 1250.0)
    elif mode == "half":
        poi = (side * rng.uniform(0.30 * hl, 0.58 * hl), rng.uniform(-0.35 * hw, 0.35 * hw), 0.0)
        pos = (side * (hl + rng.uniform(3.0, 8.0)), side * (hw + rng.uniform(3.0, 8.0)), rng.uniform(5.0, 12.0))
        fl = rng.uniform(950.0, 1800.0)
    elif mode == "high":
        poi = (rng.uniform(-0.2 * hl, 0.2 * hl), rng.uniform(-0.2 * hw, 0.2 * hw), 0.0)
        pos = (poi[0] + rng.uniform(-8.0, 8.0), side * (hw + rng.uniform(6.0, 12.0)), rng.uniform(18.0, 26.0))
        fl = rng.uniform(650.0, 1150.0)
    elif mode == "close":
        poi = (side * rng.uniform(0.52 * hl, 0.82 * hl), rng.uniform(-0.35 * hw, 0.35 * hw), 0.0)
        pos = (side * (hl + rng.uniform(2.0, 6.0)), side * (hw + rng.uniform(3.0, 8.0)), rng.uniform(3.0, 8.0))
        fl = rng.uniform(1350.0, 2300.0)
    elif mode == "unusual_low":
        poi = (-side * rng.uniform(0.25 * hl, 0.65 * hl), 0.0, 0.0)
        pos = (side * rng.uniform(0.35 * hl, 0.85 * hl), side * (hw + rng.uniform(0.8, 5.0)), rng.uniform(1.8, 4.5))
        fl = rng.uniform(600.0, 1150.0)
    elif mode == "unusual_oblique":
        poi = (side * rng.uniform(0.35 * hl, 0.78 * hl), rng.uniform(-0.25 * hw, 0.25 * hw), 0.0)
        pos = (rng.uniform(-(hl + 8.0), hl + 8.0), side * (hw + rng.uniform(4.0, 10.0)), rng.uniform(4.0, 19.0))
        fl = rng.uniform(500.0, 1100.0)
    else:
        poi = (-side * rng.uniform(0.35 * hl, 0.75 * hl), -side * rng.uniform(0.15 * hw, 0.45 * hw), 0.0)
        pos = (side * rng.uniform(0.72 * hl, 1.05 * hl), side * rng.uniform(0.70 * hw, 1.08 * hw), rng.uniform(2.0, 6.0))
        fl = rng.uniform(1550.0, 2600.0)

    look = (poi[0] + rng.uniform(-1.0, 1.0), poi[1] + rng.uniform(-0.8, 0.8), poi[2])
    roll_limit = 8.0 if unusual else 1.8
    roll = math.radians(float(rng.uniform(-roll_limit, roll_limit)))
    up = (math.sin(roll), 0.0, math.cos(roll))
    return {
        "pos": tuple(float(v) for v in pos),
        "look_at": tuple(float(v) for v in look),
        "up": up,
        "focal_length": float(fl),
        "image_size": (h, w),
        "mode": mode,
    }


def project_court_quad(renderer: Court3DRenderer, cam: dict, court: Court) -> np.ndarray:
    pos = np.array(cam["pos"], dtype=np.float64)
    look = np.array(cam["look_at"], dtype=np.float64)
    up = np.array(cam.get("up", (0, 0, 1)), dtype=np.float64)
    R, U, F = renderer._view_basis(pos, look, up)
    h, w = cam["image_size"]
    hl, hw = court.half_length, court.half_width
    corners = np.array([[-hl, -hw, 0.0], [hl, -hw, 0.0], [hl, hw, 0.0], [-hl, hw, 0.0]])
    col, row, _cz, _vis = renderer._project(
        corners, pos, R, U, F, h, w, cam.get("focal_length", 1000.0))
    return np.column_stack([col, row])


def world_to_image_homography(renderer: Court3DRenderer, cam: dict) -> np.ndarray:
    """球场米制坐标到图像像素的精确单应，用于厘米级误差验收。"""
    h, w = cam["image_size"]
    pos = np.array(cam["pos"], dtype=np.float64)
    look = np.array(cam["look_at"], dtype=np.float64)
    up = np.array(cam.get("up", (0, 0, 1)), dtype=np.float64)
    R, U, F = renderer._view_basis(pos, look, up)
    fl = float(cam["focal_length"])
    rp, up_, fp = float(R @ pos), float(U @ pos), float(F @ pos)
    return np.array([
        [fl * R[0] + 0.5 * w * F[0], fl * R[1] + 0.5 * w * F[1], -fl * rp - 0.5 * w * fp],
        [-fl * U[0] + 0.5 * h * F[0], -fl * U[1] + 0.5 * h * F[1], fl * up_ - 0.5 * h * fp],
        [F[0], F[1], -fp],
    ], dtype=np.float64)


def _quad_ok(quad: np.ndarray, size: tuple[int, int]) -> bool:
    h, w = size
    if not np.all(np.isfinite(quad)):
        return False
    cx, cy = float(quad[:, 0].mean()), float(quad[:, 1].mean())
    if not (0.02 * w < cx < 0.98 * w and 0.05 * h < cy < 0.99 * h):
        return False
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    return area >= 0.05 * w * h


# ── 地板纹理 ────────────────────────────────────────────────────

def _floor_palette(rng: np.random.Generator,
                   team: dict | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """返回 (地板色, 漆区色, 纹理类型)；球队配色优先用于漆区。"""
    if rng.random() < 0.72:
        kind = "wood"
        base = WOOD_COLORS[int(rng.integers(0, len(WOOD_COLORS)))].copy()
        base *= rng.uniform(0.86, 1.10)
    else:
        kind = "solid"
        base = SOLID_COLORS[int(rng.integers(0, len(SOLID_COLORS)))].copy()
        base *= rng.uniform(0.82, 1.12)

    if team is not None:
        paint = np.array(hex_rgb(team["primary"]), np.float32) * rng.uniform(0.84, 1.06)
    elif rng.random() < 0.65:
        paint = PAINT_COLORS[int(rng.integers(0, len(PAINT_COLORS)))].copy()
        paint *= rng.uniform(0.82, 1.10)
    else:
        paint = base * rng.uniform(0.42, 0.72)
    return np.clip(base, 12, 245), np.clip(paint, 10, 235), kind


def _floor_homography(renderer: Court3DRenderer, cam: dict, court: Court,
                      apron: float) -> tuple[np.ndarray, int, int, float]:
    """球场平面纹理坐标 → 图像像素的单应，以及纹理画布尺寸。

    纹理画布覆盖球场外扩 apron 米，行 0 对应 y_max（图像上方），
    这样 warpPerspective 后木纹方向与真实地面一致。
    """
    h, w = cam["image_size"]
    pos = np.array(cam["pos"], dtype=np.float64)
    look = np.array(cam["look_at"], dtype=np.float64)
    up = np.array(cam.get("up", (0, 0, 1)), dtype=np.float64)
    R, U, F = renderer._view_basis(pos, look, up)
    fl = float(cam.get("focal_length", 1000.0))

    rp = float(R @ pos)
    up_ = float(U @ pos)
    fp = float(F @ pos)
    Hw = np.array([
        [fl * R[0] + 0.5 * w * F[0], fl * R[1] + 0.5 * w * F[1], -fl * rp - 0.5 * w * fp],
        [-fl * U[0] + 0.5 * h * F[0], -fl * U[1] + 0.5 * h * F[1], fl * up_ - 0.5 * h * fp],
        [F[0], F[1], -fp],
    ], dtype=np.float64)

    hl, hw = court.half_length, court.half_width
    x_min, x_max = -hl - apron, hl + apron
    y_min, y_max = -hw - apron, hw + apron
    extent = max(x_max - x_min, y_max - y_min)
    res = max(44.0, min(80.0, 2800.0 / max(extent, 1e-6)))
    tw = max(64, int(round((x_max - x_min) * res)))
    th = max(64, int(round((y_max - y_min) * res)))
    T = np.array([
        [1.0 / res, 0.0, x_min],
        [0.0, -1.0 / res, y_max],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return Hw @ T, tw, th, res


def _grain_texture(court: Court, apron: float, res: float, kind: str,
                   rng: np.random.Generator) -> np.ndarray:
    """球场平面上的亮度乘子（木纹/接缝/细噪声），供透视 warp 使用。"""
    hl, hw = court.half_length, court.half_width
    tw = max(64, int(round((2 * hl + 2 * apron) * res)))
    th = max(64, int(round((2 * hw + 2 * apron) * res)))

    small = rng.normal(0.0, 1.0, (max(2, th // 28), max(2, tw // 28))).astype(np.float32)
    field = cv2.resize(small, (tw, th), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), 2.0)
    field /= max(float(field.std()), 1e-3)
    grain = 1.0 + 0.035 * field

    if kind == "wood":
        yy = np.arange(th, dtype=np.float32)[:, None] / max(res, 1e-6)
        xx = np.arange(tw, dtype=np.float32)[None, :] / max(res, 1e-6)
        if rng.random() < 0.5:
            along, across = xx, yy
        else:
            along, across = yy, xx

        board_w = float(rng.uniform(0.10, 0.18))
        seam = (np.mod(across, board_w) < 0.007).astype(np.float32)
        seam = cv2.GaussianBlur(seam, (3, 3), 0.6)
        grain *= 1.0 - 0.13 * seam

        streak = np.sin(2.0 * math.pi * (across * 7.0 + 0.25 * field))
        grain *= 1.0 + 0.018 * streak
        grain *= 1.0 + 0.012 * np.sin(2.0 * math.pi * along * 0.8)

    grain = cv2.GaussianBlur(grain.astype(np.float32), (0, 0), 0.6)
    return np.clip(grain, 0.72, 1.22)


def _arena_surround(size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """程序化体育馆看台：暗顶、模糊观众团块、低对比噪点。"""
    h, w = size
    top = float(rng.uniform(12, 32))
    bottom = float(rng.uniform(34, 72))
    tint = np.array([rng.uniform(0.85, 1.05), rng.uniform(0.85, 1.05),
                     rng.uniform(0.90, 1.15)], np.float32)
    grad = np.linspace(top, bottom, h, dtype=np.float32)[:, None, None]
    img = grad * tint[None, None, :] * np.ones((1, w, 1), np.float32)

    crowd = rng.random((max(2, h // 8), max(2, w // 8), 3)).astype(np.float32)
    crowd = cv2.resize(crowd, (w, h), interpolation=cv2.INTER_CUBIC)
    crowd = cv2.GaussianBlur(crowd, (0, 0), 2.2)
    falloff = np.clip((0.72 - np.arange(h, dtype=np.float32)[:, None] / max(h, 1)) * 2.0, 0, 1)
    crowd = (crowd - 0.5) * 38.0 * falloff[..., None]
    img += crowd

    lights = rng.normal(0.0, 1.0, (max(2, h // 24), max(2, w // 24))).astype(np.float32)
    lights = cv2.resize(lights, (w, h), interpolation=cv2.INTER_CUBIC)
    img += 6.0 * lights[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def _composite(base: np.ndarray, layer: np.ndarray, mask: np.ndarray,
               soft: float = 0.9) -> np.ndarray:
    a = mask.astype(np.float32) / 255.0
    if soft > 0:
        a = cv2.GaussianBlur(a, (0, 0), soft)
    a = a[..., None]
    return np.clip(layer.astype(np.float32) * a + base.astype(np.float32) * (1.0 - a),
                   0, 255).astype(np.uint8)


def _render_floor(renderer: Court3DRenderer, cam: dict, court: Court,
                  size: tuple[int, int], rng: np.random.Generator,
                  team: dict | None = None):
    """渲染与相机严格一致的地板 + 漆区 + 透视木纹，返回 (rgb, mask)。"""
    base_rgb, paint_rgb, kind = _floor_palette(rng, team)
    renderer._FLOOR_RGB = base_rgb.astype(np.float64)
    renderer._PAINT_RGB = paint_rgb.astype(np.float64)
    renderer._PAINT2_RGB = (np.array(hex_rgb(team["secondary"]), np.float64)
                            if team is not None else paint_rgb.astype(np.float64))

    floor = renderer.render_floor_only(cam, with_paint=True).astype(np.float32)
    mask = renderer.render_floor_mask(cam, with_paint=True)
    paint = renderer.render_mask(cam, CL_PAINT).astype(np.float32) / 255.0

    apron = float(rng.uniform(1.2, 3.6))
    H, _tw, _th, res = _floor_homography(renderer, cam, court, apron)
    grain = _grain_texture(court, apron, res, kind, rng)
    grain_w = cv2.warpPerspective(
        grain, H, (size[1], size[0]), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    # 漆区是实色涂层，木纹只轻微透出
    eff = 1.0 + (grain_w - 1.0) * (1.0 - 0.78 * paint)
    floor *= eff[..., None]
    floor = np.clip(floor, 0, 255)

    if team is not None:
        apron_base = (np.array(hex_rgb(team["secondary"]), np.float32) * 0.42
                      + np.array([24, 29, 37], np.float32))
    else:
        apron_base = base_rgb * float(rng.uniform(0.62, 0.86))
    apron_rgb = apron_base * float(rng.uniform(0.88, 1.08)) * grain_w[..., None]
    apron_rgb = np.clip(apron_rgb, 0, 255)
    return floor, mask, apron_rgb, H, apron


def _apron_mask(renderer: Court3DRenderer, cam: dict, court: Court,
                apron: float, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    hl, hw = court.half_length + apron, court.half_width + apron
    corners = np.array([[-hl, -hw, 0.0], [hl, -hw, 0.0], [hl, hw, 0.0], [-hl, hw, 0.0]])
    pos = np.array(cam["pos"], dtype=np.float64)
    look = np.array(cam["look_at"], dtype=np.float64)
    up = np.array(cam.get("up", (0, 0, 1)), dtype=np.float64)
    R, U, F = renderer._view_basis(pos, look, up)
    col, row, _z, _vis = renderer._project(
        corners, pos, R, U, F, h, w, cam.get("focal_length", 1000.0))
    quad = np.column_stack([col, row])
    if not np.all(np.isfinite(quad)):
        return np.zeros((h, w), np.uint8)
    quad = np.clip(quad, -1e5, 1e5).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [quad], 255)
    return mask


# ── 标签 ────────────────────────────────────────────────────────

def _label_from_raw(raw: np.ndarray) -> np.ndarray:
    """渲染类别图 → 线标签。

    漆区大面积填充已在 CourtMesh3D.render_label 中跳过，raw 里只剩
    边线/三分线/罚球区框线/罚球圈/合理冲撞区等真实线条，直接使用即可。
    """
    return raw.copy()


def _add_thickness(label: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() >= 0.55:
        return label
    thick = label.copy()
    kernel = np.ones((3, 3), np.uint8)
    for cid in (1, 2, 3, 4, 5):
        m = (label == cid).astype(np.uint8)
        if m.any():
            thick[cv2.dilate(m, kernel, iterations=1) > 0] = cid
    return thick


def _thin_label(label: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """按概率腐蚀 1px，让 960 宽下的线接近真实转播的 2~4px。"""
    if rng.random() >= 0.40:
        return label
    out = np.zeros_like(label)
    for cid in (1, 2, 3, 4, 5):
        m = (label == cid).astype(np.uint8)
        if not m.any():
            continue
        e = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
        out[e > 0] = cid
        if int(e.sum()) < max(20, int(0.25 * m.sum())):
            out[m > 0] = cid
    return out


def _choose_camera(renderer: Court3DRenderer, court: Court, size: tuple[int, int],
                   rng: np.random.Generator, tries: int = 12):
    def _accept(raw: np.ndarray) -> tuple[bool, float]:
        lab = _label_from_raw(raw)
        frac = float((lab > 0).mean())
        n_cls = int(sum((lab == c).sum() > 80 for c in range(1, NUM_CLASSES)))
        return (0.006 <= frac <= 0.35 and n_cls >= 2), frac

    best = None
    for i in range(tries + 8):
        forced = None if i < tries else str(rng.choice(["wide", "wide", "half"]))
        cam = sample_camera(size, court, rng, mode=forced)
        quad = project_court_quad(renderer, cam, court)
        if not _quad_ok(quad, size):
            continue
        raw = renderer.render_label(cam)
        ok, frac = _accept(raw)
        if ok:
            return cam, raw
        if best is None or frac > best[0]:
            best = (frac, cam, raw)
    if best is not None:
        return best[1], best[2]
    cam = sample_camera(size, court, rng, mode="wide")
    return cam, renderer.render_label(cam)


# ── 遮挡 ────────────────────────────────────────────────────────

def _crowd_layer(size: tuple[int, int], rng: np.random.Generator,
                 edge: str) -> tuple[np.ndarray, np.ndarray]:
    """一排观众剪影（黑/深色头肩）+ 护栏/广告板 + 偶尔的器材黑块。

    这是转播里最普遍的遮挡：第一排观众沿画面底边/侧边压住球场边缘。
    不做写实小人，只用低对比剪影，避免把伪影喂给分割模型。
    """
    h, w = size
    horizontal = edge in ("bottom", "top")
    along = float(w if horizontal else h)
    depth = float(h if horizontal else w)
    unit = float(min(h, w))

    layer = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w), np.float32)

    def _put(m: np.ndarray, color) -> None:
        nonlocal layer, alpha
        a = m[..., None]
        layer = layer * (1.0 - a) + np.asarray(color, np.float32) * a
        alpha = np.maximum(alpha, m)

    def _xy(along_px: float, inward_px: float):
        if edge == "bottom":
            return int(round(along_px)), int(round(h - inward_px))
        if edge == "left":
            return int(round(inward_px)), int(round(along_px))
        return int(round(w - inward_px)), int(round(along_px))

    def _rect(a0, a1, d0, d1, mask=None):
        pts = [_xy(a0, d0), _xy(a1, d0), _xy(a1, d1), _xy(a0, d1)]
        m = np.zeros((h, w), np.float32) if mask is None else mask
        cv2.fillPoly(m, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA)
        return m

    # 护栏/广告板：压在最外侧的深色带，顶沿偶尔有反光条
    band = max(3.0, float(rng.uniform(0.028, 0.052)) * depth)
    barrier = np.array([rng.uniform(13, 30), rng.uniform(13, 30), rng.uniform(15, 36)], np.float32)
    _put(_rect(-2, along + 2, 0, band), barrier)
    if rng.random() < 0.60:
        trim = np.clip(barrier + rng.uniform(26, 62), 0, 255)
        _put(_rect(-2, along + 2, band * 0.52, band * 0.68), trim)

    # 观众 1~2 排；先画远排，再画近排。人像保持直立：头在肩上方。
    n_rows = int(rng.integers(1, 3))
    hr_near = float(rng.uniform(0.018, 0.029) * unit)
    for r in reversed(range(n_rows)):
        hr = hr_near * (1.0 - 0.16 * r)
        row_in = band + hr * (0.95 + 2.20 * r) + rng.uniform(-0.2, 0.3) * hr
        step = hr * float(rng.uniform(1.7, 2.4))
        a = -step * float(rng.uniform(0.0, 1.0))
        while a < along + step:
            if rng.random() > 0.10:
                hx = a + float(rng.uniform(-0.30, 0.30)) * step
                rr = hr * float(rng.uniform(0.84, 1.12))

                def _dark(base_lo, base_hi):
                    v = rng.uniform(base_lo, base_hi)
                    return np.clip(np.array([v + rng.uniform(-4, 9),
                                             v + rng.uniform(-4, 6),
                                             v + rng.uniform(-3, 11)], np.float32), 8, 210)

                head_col = _dark(80, 155) if rng.random() < 0.16 else _dark(24, 72)
                cloth = _dark(72, 128) if rng.random() < 0.14 else _dark(18, 62)

                sh = np.zeros((h, w), np.float32)
                if horizontal:
                    # 头朝场内，肩朝画面边缘（观众面向球场）
                    sx, sy = _xy(hx, max(band * 0.4, row_in - rr * 0.65))
                    hxx, hyy = _xy(hx, row_in + rr * 0.30)
                    ax = (1.95 * rr, 1.05 * rr)
                else:
                    # 侧边观众：人像在图像里仍保持竖直
                    sx, sy = _xy(hx + rr * 1.15, row_in)
                    hxx, hyy = _xy(hx, row_in)
                    ax = (1.95 * rr, 1.05 * rr)
                cv2.ellipse(sh, (sx, sy),
                            (max(2, int(round(ax[0]))), max(2, int(round(ax[1])))),
                            0, 0, 360, 1.0, -1, cv2.LINE_AA)
                _put(sh, cloth)
                hd = np.zeros((h, w), np.float32)
                cv2.circle(hd, (hxx, hyy), max(2, int(round(rr))), 1.0, -1, cv2.LINE_AA)
                _put(hd, head_col)
            a += step * float(rng.uniform(0.78, 1.18))

    # 偶尔加一个器材/摄影机黑块，压在角落
    if rng.random() < 0.30:
        bw_ = float(rng.uniform(0.06, 0.13)) * (w if horizontal else h)
        bd_ = float(rng.uniform(0.05, 0.10)) * depth
        a0 = float(rng.choice([0.0, along - bw_]))
        block = np.array([rng.uniform(8, 22), rng.uniform(8, 22), rng.uniform(10, 26)], np.float32)
        _put(_rect(a0, a0 + bw_, 0, bd_), block)
        if rng.random() < 0.5:
            m = np.zeros((h, w), np.float32)
            _rect(a0 + bw_ * 0.15, a0 + bw_ * 0.85, bd_ * 0.78, bd_ * 0.86, mask=m)
            _put(m, np.clip(block + 24, 0, 255))

    sigma = max(0.5, unit * 0.0016)
    layer = cv2.GaussianBlur(layer, (0, 0), sigma)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)
    return np.clip(layer, 0, 255), np.clip(alpha, 0, 1)


def _bench_layer(renderer: Court3DRenderer, cam: dict, court: Court,
                 rng: np.random.Generator, team: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """按球场真实位置投影替补席、广告板和坐姿剪影，可大片压住近侧边线。"""
    h, w = cam["image_size"]
    pos = np.array(cam["pos"], dtype=np.float64)
    look = np.array(cam["look_at"], dtype=np.float64)
    up = np.array(cam.get("up", (0, 0, 1)), dtype=np.float64)
    R, U, F = renderer._view_basis(pos, look, up)
    side = 1.0 if pos[1] >= 0 else -1.0
    if abs(pos[1]) < court.half_width + 1.0:
        side = float(rng.choice([-1.0, 1.0]))
    layer = Image.new("RGB", (w, h))
    alpha = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(layer)
    mask = ImageDraw.Draw(alpha)

    def project(points):
        pts = np.asarray(points, dtype=np.float64)
        col, row, _depth, visible = renderer._project(pts, pos, R, U, F, h, w, cam["focal_length"])
        return [(int(round(x)), int(round(y))) for x, y in zip(col, row)], bool(visible.all())

    def polygon(points, color):
        pixels, visible = project(points)
        if visible and len(pixels) >= 3:
            draw.polygon(pixels, fill=tuple(int(v) for v in color))
            mask.polygon(pixels, fill=255)

    primary = hex_rgb(team["primary"]) if team else (38, 74, 150)
    secondary = hex_rgb(team["secondary"]) if team else (188, 146, 52)
    board_y = side * (court.half_width + 4.15)
    polygon([(-4.5, board_y, 0.05), (4.5, board_y, 0.05),
             (4.5, board_y, 1.12), (-4.5, board_y, 1.12)], secondary)
    polygon([(-4.5, board_y - side * 0.02, 0.84), (4.5, board_y - side * 0.02, 0.84),
             (4.5, board_y - side * 0.02, 0.92), (-4.5, board_y - side * 0.02, 0.92)], (228, 231, 232))
    for x in np.linspace(-4.1, 4.1, 9):
        polygon([(x, board_y - side * 0.03, 0.12), (x + 0.48, board_y - side * 0.03, 0.12),
                 (x + 0.48, board_y - side * 0.03, 0.70), (x, board_y - side * 0.03, 0.70)], primary)

    bench_y = side * (court.half_width + 3.35)
    polygon([(-3.6, bench_y + side * 0.28, 0.42), (3.6, bench_y + side * 0.28, 0.42),
             (3.6, bench_y + side * 0.28, 1.02), (-3.6, bench_y + side * 0.28, 1.02)], primary)
    for i in range(10):
        x = -3.24 + i * 0.72
        polygon([(x - 0.31, bench_y - side * 0.28, 0.42), (x + 0.31, bench_y - side * 0.28, 0.42),
                 (x + 0.31, bench_y + side * 0.24, 0.42), (x - 0.31, bench_y + side * 0.24, 0.42)], primary)
        polygon([(x - 0.27, bench_y + side * 0.20, 0.48), (x + 0.27, bench_y + side * 0.20, 0.48),
                 (x + 0.27, bench_y + side * 0.20, 0.98), (x - 0.27, bench_y + side * 0.20, 0.98)], secondary)

    for i in range(8):
        x = -2.88 + i * 0.82 + float(rng.uniform(-0.08, 0.08))
        cloth = tuple(int(v) for v in np.clip(np.array(primary) * float(rng.uniform(0.28, 0.55)), 12, 180))
        skin = (int(rng.uniform(92, 176)), int(rng.uniform(62, 132)), int(rng.uniform(42, 102)))
        polygon([(x - 0.22, bench_y - side * 0.06, 0.62), (x + 0.22, bench_y - side * 0.06, 0.62),
                 (x + 0.18, bench_y - side * 0.06, 1.10), (x - 0.18, bench_y - side * 0.06, 1.10)], cloth)
        center, visible = project([(x, bench_y - side * 0.06, 1.27)])
        offset, _ = project([(x + 0.12, bench_y - side * 0.06, 1.27)])
        if visible:
            cx, cy = center[0]
            radius = max(2, int(math.hypot(offset[0][0] - cx, offset[0][1] - cy)))
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=skin)
            mask.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)

    return np.asarray(layer, dtype=np.float32), np.asarray(alpha, dtype=np.float32) / 255.0


def _synthetic_occluders(img: np.ndarray, size: tuple[int, int],
                         rng: np.random.Generator,
                         renderer: Court3DRenderer | None = None,
                         cam: dict | None = None,
                         court: Court | None = None,
                         team: dict | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """随机选择真实位置替补席、第一排观众或护栏遮挡。"""
    occl = np.zeros(img.shape[:2], np.float32)
    choice = float(rng.random())
    kind = "none"
    if choice < 0.38 and renderer is not None and cam is not None and court is not None:
        kind = "bench"
        layer, alpha = _bench_layer(renderer, cam, court, rng, team)
        if float(alpha.mean()) < 0.002:
            kind = "crowd"
            edge = str(rng.choice(["bottom", "left", "right"], p=[0.64, 0.18, 0.18]))
            layer, alpha = _crowd_layer(size, rng, edge)
    elif choice < 0.88:
        kind = "crowd"
        edge = str(rng.choice(["bottom", "left", "right"], p=[0.64, 0.18, 0.18]))
        layer, alpha = _crowd_layer(size, rng, edge)
    else:
        return img, occl, kind
    a = alpha[..., None]
    img[:] = np.clip(layer * a + img.astype(np.float32) * (1.0 - a), 0, 255).astype(img.dtype)
    return img, alpha, kind


# ── 合成 ────────────────────────────────────────────────────────

def _draw_lines(img: np.ndarray, label: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    alpha = (label > 0).astype(np.float32) * float(rng.uniform(0.82, 0.97))
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0.65)
    shade = cv2.GaussianBlur(
        rng.normal(1.0, 0.025, (max(2, img.shape[0] // 48), max(2, img.shape[1] // 48)))
        .astype(np.float32), (0, 0), 1.2)
    shade = cv2.resize(shade, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    color = LINE_RGB[None, None, :] * np.clip(shade, 0.90, 1.05)[..., None]
    return np.clip(img.astype(np.float32) * (1 - alpha[..., None])
                   + color * alpha[..., None], 0, 255).astype(np.uint8)


def _lighting(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    lo = rng.normal(0.0, 1.0, (max(2, h // 90), max(2, w // 90))).astype(np.float32)
    lo = cv2.resize(lo, (w, h), interpolation=cv2.INTER_CUBIC)
    light = 1.0 + 0.085 * lo

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w * 0.5) / max(w * 0.5, 1)
    ny = (yy - h * 0.5) / max(h * 0.5, 1)
    vignette = 1.0 - 0.12 * np.clip((nx * nx + ny * ny) * 0.6, 0, 1)
    out = img.astype(np.float32) * light[..., None] * vignette[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """轻微曝光/白平衡/噪声，不做整图换色，保证线始终是同一颜色。"""
    out = img.astype(np.float32)
    out *= float(rng.uniform(0.90, 1.08))
    wb = 1.0 + rng.normal(0.0, 0.018, 3).astype(np.float32)
    out *= wb[None, None, :]
    gamma = float(rng.uniform(0.96, 1.04))
    out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, gamma)
    out += rng.normal(0, float(rng.uniform(0.5, 2.8)), out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)

    if rng.random() < 0.25:
        out = cv2.GaussianBlur(out, (3, 3), 0.4)
    if rng.random() < 0.6:
        q = int(rng.integers(78, 95))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out


def compose_sample(
    renderers: dict[str, Court3DRenderer],
    court: Court,
    size: tuple[int, int],
    rng: np.random.Generator,
    team: dict | None = None,
    return_meta: bool = False,
):
    """纯 3D 合成一张球场线样本，返回 (RGB, 类别图)。"""
    h, w = size
    renderer = renderers[court.model]
    cam, raw = _choose_camera(renderer, court, size, rng)

    floor, floor_mask, apron_rgb, _H, apron = _render_floor(
        renderer, cam, court, size, rng, team)
    surround = _arena_surround(size, rng)
    apron_mask = _apron_mask(renderer, cam, court, apron, size)

    img = _composite(surround, apron_rgb, apron_mask, soft=1.0)
    img = _composite(img, floor, floor_mask, soft=0.8)

    label = _label_from_raw(raw)
    label = _add_thickness(label, rng)
    label = _thin_label(label, rng)

    if rng.random() < 0.12:
        bs = 16
        small = rng.random((h // bs + 1, w // bs + 1)) < float(rng.uniform(0.02, 0.07))
        small = cv2.resize(small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        small = cv2.dilate(small, np.ones((3, 3), np.uint8))
        label[small > 0] = 0

    img, occl, occlusion = _synthetic_occluders(img, size, rng, renderer, cam, court, team)
    label[occl > 0.55] = 0

    img = _draw_lines(img, label, rng)

    img = _lighting(img, rng)
    img = _photometric(img, rng)
    if return_meta:
        return img, label, {
            "camera_mode": cam["mode"],
            "camera_pos": list(cam["pos"]),
            "look_at": list(cam["look_at"]),
            "focal_length": float(cam["focal_length"]),
            "H_world_to_px": world_to_image_homography(renderer, cam).tolist(),
            "occlusion": occlusion,
            "occluded_fraction": float((occl > 0.55).mean()),
        }
    return img, label


# ── 数据集 ──────────────────────────────────────────────────────

LABEL_VIS_COLORS = np.array([
    [0, 0, 0],          # background
    [255, 255, 255],    # boundary
    [255, 220, 60],     # three_point
    [80, 220, 80],      # paint
    [80, 140, 255],     # center_line
    [220, 80, 220],     # center_circle
], np.uint8)


def _save_label_png(path: Path, label: np.ndarray) -> None:
    """索引调色板 PNG：肉眼可见类别颜色，PIL 读取仍得到 0..5 索引。"""
    palette = LABEL_VIS_COLORS.reshape(-1).tolist()
    palette += [0] * (768 - len(palette))
    im = Image.fromarray(label, mode="P")
    im.putpalette(palette)
    kw = {"icc_profile": _SRGB_PROFILE} if _SRGB_PROFILE else {}
    im.save(path, **kw)


def build_mixed_dataset(
    out_dir: str | Path,
    n_samples: int = 6000,
    size: tuple[int, int] = (360, 640),
    courts: list[Court] | None = None,
    seed: int = 0,
) -> Path:
    courts = courts or [FIBA, NBA]
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    renderers = {c.model: Court3DRenderer(CourtMesh3D.build(c)) for c in courts}
    rng = np.random.default_rng(seed)

    meta = []
    for i in range(n_samples):
        court = courts[i % len(courts)]
        team_pool = [t for t in TEAM_PALETTES if t["model"] == court.model] or TEAM_PALETTES
        team = team_pool[(i // len(courts)) % len(team_pool)]
        img, label, sample_meta = compose_sample(renderers, court, size, rng, team, return_meta=True)
        img_kw = {"quality": 92}
        if _SRGB_PROFILE:
            img_kw["icc_profile"] = _SRGB_PROFILE
        Image.fromarray(img).save(out / "images" / f"{i:06d}.jpg", **img_kw)
        # 调色板 PNG 直接打开就是彩色类别图；PIL 读取仍返回 0..5 索引。
        _save_label_png(out / "masks" / f"{i:06d}.png", label)
        meta.append({"idx": i, "court_model": court.model, "team": team["name"], "image_size": [size[0], size[1]], **sample_meta})
        if (i + 1) % 200 == 0:
            print(f"  generated {i + 1}/{n_samples}", flush=True)

    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"dataset saved to {out} ({n_samples} samples, size={size[1]}x{size[0]})")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="纯 3D 合成球场线数据")
    ap.add_argument("--out", default="data/seg_synth")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", default="360x640", help="HxW")
    ap.add_argument("--courts", default="FIBA,NBA")
    args = ap.parse_args()

    hh, ww = (int(v) for v in args.size.lower().split("x"))
    court_map = {"FIBA": FIBA, "NBA": NBA}
    selected = [court_map[c.strip().upper()] for c in args.courts.split(",")
                if c.strip().upper() in court_map]
    build_mixed_dataset(args.out, args.n, (hh, ww), selected or [FIBA, NBA], seed=args.seed)


__all__ = [
    "sample_camera", "compose_sample", "build_mixed_dataset",
]