"""真实域混合数据生成：3D 球场线渲染叠加到真实/合成背景上。

目标：给快速实时标定用的球场线分割网络造训练数据，覆盖多机位、多地板颜色、
遮挡等真实转播情况。

要点：
1. 机位分档：全景 / 半场 / 近景 / 底线 / 高位，先选画面里的关注点（POI），
   再在球场外侧绕 POI 摆相机，保证画面里有线；渲染后按线像素比例再筛一遍，
   避免大量"空地板"样本；
2. 背景：85% 用真实转播画面当"体育馆背景"，把渲染的多颜色地板（木色/彩色
   漆区）贴到它的场地位置上——既保留真实看台纹理，又让场地几何与标签严格
   一致（真实画面里不一致的线被地板盖住，剩下露出来的白结构标 ignore）；
   15% 用合成看台渐变 + 渲染地板；
3. 线标签：网格里的线本身就是 ~10cm 宽的带状面，直接取其填充；只有罚球区
   的大面积漆块转成外框线（否则漆区会变成整块区域标签）；
4. 遮挡：真实画面贴片 + 着衣颜色块模拟人物/器材，约 35% 概率在前排边缘压
   一条波状观众遮挡带（挡住一条边线）；
5. 真实背景露出的白色结构标 ignore(255)，另留 8% 全背景负样本。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from bsk.court import FIBA, NBA, Court
from bsk.court_3d import (
    NUM_CLASSES,
    Court3DRenderer,
    CourtMesh3D,
)

IGNORE = 255
LINE_CLASSES = (1, 2, 4, 5)  # boundary, three_point, center_line, center_circle

# 漆区（lane）常见配色：深蓝/深绿/深红/近黑/金黄/青，None 表示地板色系加深
PAINT_STYLES = [
    np.array([38, 74, 150], dtype=np.float64),
    np.array([28, 102, 66], dtype=np.float64),
    np.array([140, 46, 46], dtype=np.float64),
    np.array([32, 32, 38], dtype=np.float64),
    np.array([188, 146, 52], dtype=np.float64),
    np.array([52, 132, 145], dtype=np.float64),
    None,
]


# ── 背景池 ──────────────────────────────────────────────────────

def extract_backgrounds(
    video_path: str | Path,
    out_dir: str | Path,
    n_frames: int = 240,
    size: tuple[int, int] = (960, 540),
) -> int:
    """从视频均匀抽帧存为 JPEG（原样保存，不做线清理）。

    背景里真实存在的球场线在合成阶段会被地板盖住或标成 ignore(255)，
    既保留真实纹理（域一致），又不把未标注线当成"负样本"。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        raise RuntimeError(f"无法读取视频帧数: {video_path}")
    idxs = np.linspace(0, total - 1, n_frames).astype(int)
    saved = 0
    for i, fidx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"bg_{i:04d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        saved += 1
    cap.release()
    return saved


def load_backgrounds(bg_dir: str | Path) -> list[np.ndarray]:
    files = sorted(Path(bg_dir).glob("*.jpg"))
    return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in files]


def _white_structures(rgb: np.ndarray) -> np.ndarray:
    """背景中的白色线状结构（真实球场线/广告牌白字等），训练时标为 ignore。"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[:, :, 2] > 150) & (hsv[:, :, 1] < 75)).astype(np.uint8)
    return cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)


# ── 相机 ────────────────────────────────────────────────────────

def sample_camera(size: tuple[int, int], court: Court, rng: np.random.Generator,
                  mode: str | None = None) -> dict:
    """按机位档位随机相机：全景/半场/近景/底线/高位，含变焦与轻微滚动。

    先取一个球场上的关注点 POI，相机在其外侧按距离摆放并看向 POI，
    这样无论哪个档位画面里都有线，不会出现"空地板"样本。
    """
    h, w = size
    hl, hw = court.half_length, court.half_width
    reach = max(hl, hw)

    if mode is None:
        mode = str(rng.choice(["wide", "half", "close", "corner", "high"],
                              p=[0.24, 0.30, 0.16, 0.18, 0.12]))
    if mode == "wide":
        poi = (rng.uniform(-0.2 * hl, 0.2 * hl), rng.uniform(-0.2 * hw, 0.2 * hw), 0.0)
        dist = rng.uniform(reach * 1.0, reach * 1.9)
        z = rng.uniform(8.0, 20.0)
        fl = rng.uniform(650, 1250)
        az_mid = None  # 全景机位绕场一周
    elif mode == "half":
        # 只拍半场：相机在该半场外侧，看向那一半
        side = float(rng.choice([-1.0, 1.0]))
        poi = (side * rng.uniform(0.30 * hl, 0.60 * hl), rng.uniform(-0.45 * hw, 0.45 * hw), 0.0)
        dist = rng.uniform(reach * 0.50, reach * 0.95)
        z = rng.uniform(4.5, 12.0)
        fl = rng.uniform(700, 1500)
        az_mid = math.atan2(poi[1], poi[0])
    elif mode == "close":
        side = float(rng.choice([-1.0, 1.0]))
        poi = (side * rng.uniform(0.55 * hl, 0.85 * hl), rng.uniform(-0.4 * hw, 0.4 * hw), 0.0)
        dist = rng.uniform(reach * 0.30, reach * 0.65)
        z = rng.uniform(3.0, 9.0)
        fl = rng.uniform(850, 1800)  # 长焦贴近
        az_mid = math.atan2(poi[1], poi[0])
    elif mode == "corner":
        sx = float(rng.choice([-1.0, 1.0]))
        sy = float(rng.choice([-1.0, 1.0]))
        poi = (sx * rng.uniform(0.35 * hl, 0.8 * hl), sy * rng.uniform(0.25 * hw, 0.75 * hw), 0.0)
        dist = rng.uniform(reach * 0.45, reach * 1.1)
        z = rng.uniform(5.0, 14.0)
        fl = rng.uniform(700, 1400)
        az_mid = math.atan2(poi[1], poi[0])
    else:  # high
        poi = (rng.uniform(-0.25 * hl, 0.25 * hl), rng.uniform(-0.25 * hw, 0.25 * hw), 0.0)
        dist = rng.uniform(reach * 0.9, reach * 1.5)
        z = rng.uniform(15.0, 24.0)
        fl = rng.uniform(600, 1100)
        az_mid = None

    if az_mid is None:
        azimuth = float(rng.uniform(0, 2 * math.pi))
    else:
        # 从球场中心指向 POI 的方向附近摆机（真实转播机位都在场外）
        azimuth = az_mid + float(rng.uniform(-0.9, 0.9))

    pos = (poi[0] + dist * math.cos(azimuth), poi[1] + dist * math.sin(azimuth), z)
    look = (poi[0] + rng.uniform(-1.0, 1.0), poi[1] + rng.uniform(-0.8, 0.8), poi[2])
    roll = math.radians(float(rng.uniform(-6.0, 6.0)))
    up = (math.sin(roll), 0.0, math.cos(roll))
    return {
        "pos": pos,
        "look_at": look,
        "up": up,
        "focal_length": float(fl),
        "image_size": (h, w),
        "mode": mode,
    }


def project_court_quad(renderer: Court3DRenderer, cam: dict, court: Court) -> np.ndarray:
    """球场外框四角的投影像素坐标 [4, 2]。"""
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


def _quad_ok(quad: np.ndarray, size: tuple[int, int], low: bool = False) -> bool:
    """外框四角是否构成"能看出球场"的画面（不退化、不太小、在画面内）。

    low=True 时要求球场落在画面下半部（直接画在真实转播画面上时，
    场地一定在下半部，避免线画到看台上）。
    """
    h, w = size
    if not np.all(np.isfinite(quad)):
        return False
    cx, cy = float(quad[:, 0].mean()), float(quad[:, 1].mean())
    if low:
        if not (0.12 * w < cx < 0.88 * w and 0.50 * h < cy < 0.98 * h):
            return False
        # 四个角都不能进画面上部（那里是看台/观众，线画上去会教模型在人群里乱找线）
        if float(quad[:, 1].min()) < 0.26 * h:
            return False
    elif not (0.05 * w < cx < 0.95 * w and 0.10 * h < cy < 0.98 * h):
        return False
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    min_area = 0.18 if low else 0.06
    if area < min_area * w * h:
        return False
    n_in = int(((quad[:, 0] >= 0) & (quad[:, 0] < w) & (quad[:, 1] >= 0) & (quad[:, 1] < h)).sum())
    # 特写机位四角都在画外也允许：面积判据已保证场地足够大
    return n_in >= 2 or area > 0.5 * w * h


# ── 背景 ────────────────────────────────────────────────────────

def _floor_texture(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """地板条纹/格纹纹理（模拟木地板拼条与漆区图案）。"""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    if rng.random() < 0.55:
        period = float(rng.integers(18, 70))
        stripes = (np.sin((xx + 0.35 * yy) / period * 2 * math.pi) > 0).astype(np.float32)
        factor = 1.0 - float(rng.uniform(0.04, 0.12)) * stripes
    else:
        period = int(rng.integers(40, 130))
        checker = ((xx // period + yy // period) % 2).astype(np.float32)
        factor = 1.0 - float(rng.uniform(0.03, 0.10)) * checker
    return np.clip(img.astype(np.float32) * factor[..., None], 0, 255)


def _apron_layer(renderer: Court3DRenderer, cam: dict, size: tuple[int, int],
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """球场外侧的"地板延展区"（apron）：真实球馆的地板比画线范围更大。

    没有这层时，合成地板的外边缘与边线完全重合，模型会学成
    "地板边缘 = 边线"（在真实画面里地板边缘在界外，预测就整体偏移）。
    """
    h, w = size
    court = renderer.mesh.court
    apron = float(rng.uniform(1.0, 4.0))
    hl, hw = court.half_length + apron, court.half_width + apron
    corners = np.array([[-hl, -hw, 0.0], [hl, -hw, 0.0], [hl, hw, 0.0], [-hl, hw, 0.0]])
    pos = np.array(cam["pos"], dtype=np.float64)
    look = np.array(cam["look_at"], dtype=np.float64)
    up = np.array(cam.get("up", (0, 0, 1)), dtype=np.float64)
    R, U, F = renderer._view_basis(pos, look, up)
    col, row, _z, _vis = renderer._project(
        corners, pos, R, U, F, h, w, cam.get("focal_length", 1000.0))
    poly = np.column_stack([col, row])
    if not np.all(np.isfinite(poly)):
        return np.zeros((h, w, 3), np.uint8), np.zeros((h, w), np.uint8)
    # 扩出去的地板配色要么同色稍暗，要么换成另一块木色（真实球场常见的界外配色）
    shade = renderer._FLOOR_RGB * float(rng.uniform(0.62, 1.08))
    layer = np.zeros((h, w, 3), np.uint8)
    cv2.fillPoly(layer, [poly.astype(np.int32)],
                 tuple(int(v) for v in np.clip(shade, 0, 255)))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return layer, mask


def _floor_frac(rgb: np.ndarray, quad: np.ndarray) -> float:
    """画面里四边形区域有多少是"地板色"（木色/青蓝漆区）。

    真实转播帧贴线前用它筛机位：线只能画在地板上，不能画在看台人群上。
    """
    mask = np.zeros(rgb.shape[:2], np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    n = int((mask > 0).sum())
    if n < 100:
        return 0.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    wood = cv2.inRange(hsv, np.array([8, 25, 70]), np.array([34, 200, 255]))
    cyan = cv2.inRange(hsv, np.array([75, 45, 40]), np.array([115, 230, 240]))
    floor = ((wood > 0) | (cyan > 0)) & (mask > 0)
    return float(floor.sum()) / float(n)


def _render_floor_layer(renderer: Court3DRenderer, cam: dict, rng: np.random.Generator):
    """多颜色地板层：随机木色 + 随机漆区色 + 纹理，返回 (RGB, 覆盖掩码)。"""
    style = str(rng.choice(list(Court3DRenderer.FLOOR_STYLES.keys())))
    tint = np.array([rng.uniform(0.80, 1.12) for _ in range(3)], dtype=np.float64)
    floor_rgb = np.clip(Court3DRenderer.FLOOR_STYLES[style] * tint, 25, 248)
    renderer._FLOOR_RGB = floor_rgb

    paint = PAINT_STYLES[int(rng.integers(0, len(PAINT_STYLES)))]
    if paint is None:
        renderer._PAINT_RGB = floor_rgb * float(rng.uniform(0.45, 0.80))
    else:
        renderer._PAINT_RGB = np.clip(paint * float(rng.uniform(0.85, 1.15)), 15, 235)
    renderer._PAINT2_RGB = renderer._PAINT_RGB

    floor = renderer.render_floor_only(cam, with_paint=True)
    mask = renderer.render_floor_mask(cam, with_paint=True)
    floor = _floor_texture(floor, rng).astype(np.uint8)
    return floor, mask


def _real_background(bg_pool: list[np.ndarray], size: tuple[int, int],
                     rng: np.random.Generator) -> np.ndarray:
    """从真实转播帧里随机裁一块缩放成目标尺寸。"""
    h, w = size
    src = bg_pool[int(rng.integers(0, len(bg_pool)))]
    scale = float(rng.uniform(0.9, 1.3))
    bh, bw = src.shape[:2]
    cw, ch = int(bw / scale), int(bh / scale)
    if cw >= bw or ch >= bh:
        return cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
    x0 = int(rng.integers(0, bw - cw + 1))
    y0 = int(rng.integers(0, bh - ch + 1))
    return cv2.resize(src[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_AREA)


def _synthetic_surround(size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """合成"体育馆看台"背景：上暗下亮的渐变 + 噪点 + 模糊的观众团块。"""
    h, w = size
    base = np.array([rng.uniform(14, 40), rng.uniform(14, 40), rng.uniform(20, 50)], dtype=np.float32)
    grad = np.linspace(1.40, 0.60, h, dtype=np.float32)[:, None, None]
    img = base[None, None, :] * grad * np.ones((1, w, 1), dtype=np.float32)
    # 稀疏模糊团块，弱化成"远处看台"而不是满天星点
    spec = (rng.random((h // 4 + 1, w // 4 + 1)) > 0.86).astype(np.float32)
    spec = cv2.resize(spec, (w, h), interpolation=cv2.INTER_LINEAR)
    spec = cv2.GaussianBlur(spec, (7, 7), 0)
    img += spec[..., None] * 22.0
    img += rng.normal(0, 4.0, (h, w, 3)).astype(np.float32)
    img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), 0)
    return img


def _paste_floor(bg: np.ndarray, floor: np.ndarray, mask: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    """把渲染地板按掩码软边贴到背景上。"""
    a = (mask > 0).astype(np.float32)
    a = cv2.GaussianBlur(a, (5, 5), float(rng.uniform(0.8, 1.6)))[..., None]
    out = floor.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def _floor_lum_texture(bg_pool: list[np.ndarray], size: tuple[int, int],
                       rng: np.random.Generator) -> np.ndarray:
    """从真实转播帧取"地板亮度纹理"（木纹/漆块/灯光起伏），归一化成乘子。

    取画面下半部（地板区域）的随机一块，去掉高亮白线（近似 inpaint）后模糊，
    除以均值得到 1.0 附近的亮度乘子，用来给渲染地板加上真实地板的质感，
    避免模型只认出"纯色地板"而看不真实木地板。
    """
    h, w = size
    src = bg_pool[int(rng.integers(0, len(bg_pool)))]
    sh, sw = src.shape[:2]
    y0 = int(rng.uniform(0.45, 0.70) * sh)
    crop = src[y0:sh, :]
    tex = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(tex, cv2.COLOR_RGB2GRAY)
    bright = cv2.dilate((gray > 165).astype(np.uint8), np.ones((3, 3), np.uint8))
    tex = cv2.inpaint(tex, bright, 3, cv2.INPAINT_TELEA)
    tex = cv2.GaussianBlur(tex, (7, 7), 0)
    lum = cv2.cvtColor(tex, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lum /= max(float(lum.mean()), 1.0)
    return np.clip(lum, 0.45, 1.7)


def _compose_background(
    renderer: Court3DRenderer,
    cam: dict,
    bg_pool: list[np.ndarray],
    size: tuple[int, int],
    rng: np.random.Generator,
    real_direct: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (背景图, ignore 掩码)。

    real_direct=True：直接拿真实转播画面当底（真实木地板/漆区/灯光/人群都在），
      线按球场几何画在上面，真实白线标 ignore。这是域一致性最强的样本。
    否则：合成多颜色地板（随机木色 + 彩色漆区 + 真实地板亮度纹理）贴到
      真实看台或合成看台上，覆盖不同地板配色。
    """
    if real_direct and bg_pool:
        bg = _real_background(bg_pool, size, rng)
        return bg, _white_structures(bg)

    floor, mask = _render_floor_layer(renderer, cam, rng)
    if bg_pool:
        # 真实地板亮度纹理（木纹/漆块/灯光），乘到纯色地板上
        lum = _floor_lum_texture(bg_pool, size, rng)
        floor = np.clip(floor.astype(np.float32) * lum[..., None], 0, 255).astype(np.uint8)

    apron, apron_mask = _apron_layer(renderer, cam, size, rng)
    if bg_pool:
        apron = np.clip(apron.astype(np.float32) * lum[..., None], 0, 255).astype(np.uint8)
    apron = _floor_texture(apron, rng).astype(np.uint8)

    if bg_pool and rng.random() < 0.85:
        src = _real_background(bg_pool, size, rng)
        src = cv2.GaussianBlur(src, (3, 3), 0)
        ignore = _white_structures(src)
        ignore[mask > 0] = 0
        # 先贴"地板延展区"，再贴球场地板：地板边缘就不会和边线重合
        bg = _paste_floor(src, apron, apron_mask, rng)
        bg = _paste_floor(bg, floor, mask, rng)
        ignore[cv2.dilate((mask > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 0
    else:
        bg = _paste_floor(_synthetic_surround(size, rng), apron, apron_mask, rng)
        bg = _paste_floor(bg, floor, mask, rng)
        ignore = np.zeros(size, np.uint8)
    return bg, ignore


# ── 标签与遮挡 ──────────────────────────────────────────────────

def _label_from_raw(raw: np.ndarray) -> np.ndarray:
    """渲染类别图 → 线标签。

    网格中的线（边线/三分/中线/中圈）本身就是带宽度的带状面：填充即线，
    不能再做轮廓提取，否则每条线会变成双线、中圈变双圈。
    罚球区既有大面积填充（漆区）又有细圆环（罚球圈）：只把大面积填充
    转成外框线，细圆环原样保留。
    """
    label = np.zeros_like(raw)
    m3 = (raw == 3).astype(np.uint8)
    if m3.any():
        dist = cv2.distanceTransform(m3, cv2.DIST_L2, 5)
        thick = ((m3 > 0) & (dist > 3.0)).astype(np.uint8)
        thin = ((m3 > 0) & (dist <= 3.0)).astype(np.uint8)
        edge = cv2.morphologyEx(thick, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        paint_line = np.maximum(thin, edge)
        label[paint_line > 0] = 3
    for cid in LINE_CLASSES:
        label[raw == cid] = cid
    return label


def _add_thickness(label: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() >= 0.5:
        return label
    thick = label.copy()
    kernel3 = np.ones((3, 3), np.uint8)
    for cid in (1, 2, 3, 4, 5):
        m = (label == cid).astype(np.uint8)
        if m.any():
            thick[cv2.dilate(m, kernel3, iterations=1) > 0] = cid
    return thick


def _floor_only_mask(rgb: np.ndarray) -> np.ndarray:
    """背景里"球场地板"的像素（木色/漆区色），不含白色。

    真实转播帧直接画线时必须用它做遮挡：球员、看台、广告牌上的线在真实画面
    里是被挡住的，训练样本里若把它们画出来，模型会学到"人身上也有线"，
    真实推理就会在球衣/看台边缘冒出假线（实测真实帧边界类大面积误检）。
    白色（真实球场线/球衣）也排除：它们本身就是 ignore 源。
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    wood = cv2.inRange(hsv, np.array([5, 25, 55]), np.array([36, 235, 255]))
    paint = cv2.inRange(hsv, np.array([60, 35, 25]), np.array([140, 255, 250]))
    floor = ((wood > 0) | (paint > 0)).astype(np.uint8)
    floor = cv2.morphologyEx(floor, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    floor = cv2.morphologyEx(floor, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    # 边缘容忍 3px：线宽本身会跨过地板色/白线边界
    return cv2.dilate(floor, np.ones((7, 7), np.uint8))


def _thin_label(label: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """按概率把线腐蚀细 1px。

    3D 网格画的线在近景有 6~12px 宽，真实转播 960 宽下地板线只有 2~4px；
    不细化的话模型只认"粗白条"，真实细线一律不出（实测真实帧召回≈0）。
    某一类被腐蚀光了就保留原样，避免样本里丢掉整整一类。
    只做 1 次腐蚀（2 次在 960 宽下会把边线打到 1px，召回反而掉）。
    """
    r = rng.random()
    if r >= 0.40:
        return label
    iters = 1
    out = np.zeros_like(label)
    for cid in (1, 2, 3, 4, 5):
        m = (label == cid).astype(np.uint8)
        if not m.any():
            continue
        e = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=iters)
        out[e > 0] = cid
        if int(e.sum()) < max(20, int(0.25 * m.sum())):
            out[m > 0] = cid
    return out


# 着衣配色（深色/灰/白/队服色），用于模拟人物遮挡
_CLOTH_RGB = [
    (20, 20, 24), (40, 40, 48), (70, 70, 75), (110, 110, 115),
    (200, 200, 205), (30, 50, 110), (120, 30, 30), (30, 90, 50),
    (150, 120, 40), (60, 30, 80),
]


def _blob_occluder(img: np.ndarray, label: np.ndarray, x: int, y: int, bw: int, bh: int,
                   rng: np.random.Generator) -> None:
    """贴一块"人/器材"色块：竖向渐变 + 噪点 + 软边，同步清空标签。"""
    h, w = label.shape
    x1, y1 = min(w, x + bw), min(h, y + bh)
    if x1 <= x or y1 <= y:
        return
    cw, ch = x1 - x, y1 - y
    base = np.array(_CLOTH_RGB[int(rng.integers(0, len(_CLOTH_RGB)))], dtype=np.float32)
    base *= float(rng.uniform(0.8, 1.15))
    grad = np.linspace(1.15, 0.75, ch, dtype=np.float32)[:, None, None]
    patch = np.clip(base[None, None, :] * grad * np.ones((1, cw, 1), np.float32), 0, 255)
    patch += rng.normal(0, 6, (ch, cw, 3)).astype(np.float32)
    patch = cv2.GaussianBlur(np.clip(patch, 0, 255).astype(np.uint8), (3, 3), 0)

    roi = img[y:y1, x:x1].astype(np.float32)
    m = np.ones((ch, cw, 1), np.float32)
    m = cv2.GaussianBlur(m, (5, 5), 1.0)[..., None]
    img[y:y1, x:x1] = np.clip(patch.astype(np.float32) * m + roi * (1 - m), 0, 255).astype(img.dtype)
    label[y:y1, x:x1] = 0


def _random_occluders(img: np.ndarray, label: np.ndarray, bg_pool, rng: np.random.Generator,
                      max_n: int = 5) -> None:
    """随机遮挡：优先贴真实画面纹理（含球员/观众），少量用着衣色块。"""
    h, w = label.shape
    src_pool = bg_pool if bg_pool else [img]
    budget = 0.38 * h * w  # 总遮挡面积上限，避免把线全盖掉
    used = 0.0
    for _ in range(int(rng.integers(0, max_n + 1))):
        bw = int(rng.uniform(0.03, 0.13) * w)
        bh = int(rng.uniform(0.10, 0.40) * h)
        if bw < 4 or bh < 4 or used + bw * bh > budget:
            continue
        x = int(rng.integers(0, max(1, w - bw)))
        y = int(rng.integers(0, max(1, h - bh)))
        used += bw * bh
        if rng.random() < 0.9:
            src = src_pool[int(rng.integers(0, len(src_pool)))]
            sx = int(rng.integers(0, max(1, src.shape[1] - bw)))
            sy = int(rng.integers(0, max(1, src.shape[0] - bh)))
            patch = cv2.resize(src[sy:sy + bh, sx:sx + bw], (bw, bh))
            img[y:y + bh, x:x + bw] = patch
            label[y:y + bh, x:x + bw] = 0
        else:
            _blob_occluder(img, label, x, y, bw, bh, rng)


def _crowd_band(img: np.ndarray, label: np.ndarray, bg_pool, rng: np.random.Generator) -> None:
    """在画面一侧压一条波状观众遮挡带（模拟前排观众挡住一条边线）。

    纹理取自真实转播帧的看台区域（上半部），竖直方向镜像平铺，避免拉伸失真。
    """
    if rng.random() > 0.35:
        return
    h, w = label.shape
    side = int(rng.integers(0, 4))  # 0 下 1 上 2 左 3 右
    depth = int(rng.uniform(0.08, 0.22) * (h if side in (0, 1) else w))
    wave_amp = float(rng.uniform(3, 12))
    wave_len = float(rng.uniform(0.08, 0.25)) * (w if side in (0, 1) else h)
    phase = float(rng.uniform(0, 2 * math.pi))

    mask = np.zeros((h, w), np.uint8)
    if side in (0, 1):
        base = (h - depth) if side == 0 else depth
        for x in range(w):
            off = wave_amp * math.sin(2 * math.pi * x / max(wave_len, 1) + phase)
            y = int(max(0, min(h, base + off)))
            if side == 0:
                mask[y:, x] = 1
            else:
                mask[:y, x] = 1
    else:
        base = (w - depth) if side == 2 else depth
        for y in range(h):
            off = wave_amp * math.sin(2 * math.pi * y / max(wave_len, 1) + phase)
            x = int(max(0, min(w, base + off)))
            if side == 2:
                mask[y, x:] = 1
            else:
                mask[y, :x] = 1

    if bg_pool:
        src = bg_pool[int(rng.integers(0, len(bg_pool)))]
        sh, sw = src.shape[:2]
        strip = src[:max(8, int(0.45 * sh))]          # 看台区域
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        bh, bw = y1 - y0, x1 - x0
        tile_h = max(6, bh // 3)
        tile = cv2.resize(strip, (bw, tile_h))
        rows = [tile, tile[::-1]]
        band = np.vstack(rows * (bh // (2 * tile_h) + 2))[:bh]
        canvas = np.zeros((h, w, 3), np.uint8)
        canvas[y0:y1, x0:x1] = band
    else:
        canvas = (rng.random((h, w, 3)) * 110 + 45).astype(np.uint8)
    m = mask > 0
    img[m] = canvas[m].astype(img.dtype)
    label[m] = 0


# ── 合成样本 ────────────────────────────────────────────────────

def _line_color(rng: np.random.Generator) -> np.ndarray:
    """球场线颜色：以白为主，少量偏黄的漆线。"""
    if rng.random() < 0.15:
        base = np.array([235.0, 220.0, 120.0])
    else:
        base = np.array([240.0, 240.0, 240.0])
    return np.clip(base * rng.uniform(0.82, 1.06), 0, 255)


def _photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float32)
    out = out * rng.uniform(0.82, 1.18) + rng.uniform(-18, 18)
    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + int(rng.integers(-10, 11))) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.8, 1.25), 0, 255)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    out += rng.normal(0, rng.uniform(1.0, 7.0), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    r = rng.random()
    if r < 0.35:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    elif r < 0.45:
        k = int(rng.choice([5, 7]))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        out = cv2.filter2D(out, -1, kernel)
    return out


def _choose_camera(renderer: Court3DRenderer, court: Court, size: tuple[int, int],
                   rng: np.random.Generator, tries: int = 10, low: bool = False):
    """挑一个"画面里有线"的机位，返回 (cam, 类别图)。

    判据：外框不退化、线像素占比 0.8%~35%、至少 2 类线有足够像素。
    随机采样失败时强制来一个全景机位，保证样本不至于几乎全空。
    low=True 时球场须落在画面下半部（真实转播画面直接画线时用）。
    """
    def _accept(raw: np.ndarray) -> tuple[bool, float]:
        lab = _label_from_raw(raw)
        frac = float((lab > 0).mean())
        n_cls = int(sum((lab == c).sum() > 80 for c in range(1, NUM_CLASSES)))
        return (0.008 <= frac <= 0.35 and n_cls >= 2), frac

    best = None
    for i in range(tries + 8):
        # 前 tries 次随机档位，之后强制全景/半场（必能看到大片场地）
        forced = None if i < tries else str(rng.choice(["wide", "wide", "half"]))
        cam = sample_camera(size, court, rng, mode=forced)
        quad = project_court_quad(renderer, cam, court)
        if not _quad_ok(quad, size, low=low):
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


def compose_sample(
    renderers: dict[str, Court3DRenderer],
    court: Court,
    bg_pool: list[np.ndarray],
    size: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """生成一张样本：背景（真实/合成看台 + 渲染地板）+ 3D 球场线 + 遮挡 + 光度增强。"""
    h, w = size
    renderer = renderers[court.model]
    # 70% 直接在真实转播画面上画线（真实地板/人群/灯光，域差最小）
    real_direct = bool(bg_pool) and rng.random() < 0.70
    if real_direct:
        cand = None
        for _ in range(4):
            cam_i, raw_i = _choose_camera(renderer, court, size, rng, low=True)
            bg_i = _real_background(bg_pool, size, rng)
            frac = _floor_frac(bg_i, project_court_quad(renderer, cam_i, court))
            if cand is None or frac > cand[0]:
                cand = (frac, cam_i, raw_i, bg_i)
            if frac >= 0.62:
                break
        _frac, cam, raw, bg = cand
        ignore = _white_structures(bg)
    else:
        cam, raw = _choose_camera(renderer, court, size, rng)
        bg, ignore = _compose_background(renderer, cam, bg_pool, size, rng, real_direct=False)

    # 8% 无线帧负样本：教模型不要从球衣/广告牌里"脑补"出线
    if rng.random() < 0.08:
        lab = np.zeros((h, w), np.uint8)
        lab[ignore > 0] = IGNORE
        return _photometric(bg, rng), lab

    label = _label_from_raw(raw)
    label = _add_thickness(label, rng)
    label = _thin_label(label, rng)

    # 真实背景直接画线：只在"露出地板"的像素上保留线标签。
    # 球员/看台/广告牌会挡住线，训练样本必须同样被挡住，否则模型会把
    # 球衣、观众席上的白色结构也当成球场线。
    if real_direct:
        fk = _floor_only_mask(bg)
        label[(label > 0) & (fk == 0)] = 0

    # 磨损：按块稀疏擦掉少量线段（注意不能逐像素随机，否则膨胀后覆盖全图）
    if rng.random() < 0.25:
        bs = 16
        small = (rng.random((h // bs + 1, w // bs + 1)) < rng.uniform(0.04, 0.14))
        small = cv2.resize(small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        small = cv2.dilate(small, np.ones((3, 3), np.uint8))
        label[small > 0] = 0

    # 背景露出的真实白线标 ignore，避免未标注线被当作负样本
    label[(ignore > 0) & (label == 0)] = IGNORE

    # 画线：软边缘 + 亮度沿长度起伏（真实木地板上白线并不"纯白刺眼"）
    line_alpha = (label > 0).astype(np.float32) * float(rng.uniform(0.40, 0.90))
    line_alpha = cv2.GaussianBlur(line_alpha, (3, 3), float(rng.uniform(0.3, 0.9)))
    color_layer = np.zeros_like(bg, np.float32)
    for cid in range(1, NUM_CLASSES):
        m = label == cid
        if m.any():
            color_layer[m] = _line_color(rng)
    shade = rng.normal(0.0, 0.12, ((h + 7) // 8, (w + 7) // 8)).astype(np.float32)
    shade = cv2.resize(shade, (w, h), interpolation=cv2.INTER_CUBIC)
    color_layer *= np.clip(1.0 + shade, 0.62, 1.15)[:, :, None]
    img = bg.astype(np.float32) * (1 - line_alpha[..., None]) + color_layer * line_alpha[..., None]

    _crowd_band(img, label, bg_pool, rng)
    _random_occluders(img, label, bg_pool, rng)

    # 灯光不均匀（球馆顶灯/镜头渐晕）：低频平滑乘子
    lo = rng.normal(0.0, 1.0, (6, 10)).astype(np.float32)
    lo = cv2.resize(lo, (w, h), interpolation=cv2.INTER_CUBIC)
    img = np.clip(img * (1.0 + 0.10 * lo)[..., None], 0, 255)

    img = _photometric(img, rng)

    if rng.random() < 0.5:
        q = int(rng.integers(45, 90))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            img = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img, label


# ── 数据集 ──────────────────────────────────────────────────────

def build_mixed_dataset(
    out_dir: str | Path,
    bg_dir: str | Path,
    n_samples: int = 6000,
    size: tuple[int, int] = (360, 640),
    courts: list[Court] | None = None,
    seed: int = 0,
) -> Path:
    courts = courts or [FIBA, NBA]
    bg_pool = load_backgrounds(bg_dir)
    if not bg_pool:
        print(f"[warn] 背景池为空（{bg_dir}），全部使用合成看台")

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    renderers = {c.model: Court3DRenderer(CourtMesh3D.build(c)) for c in courts}
    rng = np.random.default_rng(seed)

    meta = []
    for i in range(n_samples):
        court = courts[i % len(courts)]
        img, label = compose_sample(renderers, court, bg_pool, size, rng)
        cv2.imwrite(str(out / "images" / f"{i:06d}.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        cv2.imwrite(str(out / "masks" / f"{i:06d}.png"), label)
        meta.append({"idx": i, "court_model": court.model})
        if (i + 1) % 200 == 0:
            print(f"  generated {i + 1}/{n_samples}", flush=True)

    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"dataset saved to {out} ({n_samples} samples, size={size[1]}x{size[0]})")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="生成球场线分割混合数据")
    ap.add_argument("--out", default="data/seg_mix")
    ap.add_argument("--bg", default="data/seg_real_bg")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", default="360x640", help="HxW")
    args = ap.parse_args()
    hh, ww = (int(v) for v in args.size.lower().split("x"))
    build_mixed_dataset(args.out, args.bg, args.n, (hh, ww), seed=args.seed)


__all__ = [
    "extract_backgrounds", "load_backgrounds", "sample_camera",
    "compose_sample", "build_mixed_dataset",
]