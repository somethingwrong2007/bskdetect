"""WebGPU 3D 球馆合成器：球场 + 篮架 + 看台 + 灯光 + 篮球检测标签。

渲染器选型
----------
* 训练数据：pygfx 0.18 + wgpu 0.33。wgpu 是 WebGPU 原生实现，本机走
  D3D12/Vulkan，可无窗口离屏渲染，支持 PBR 材质、阴影贴图和实例化。
* 实时查看器：frontend 已带 Three.js 0.187 的 three.webgpu.js，
  可在浏览器里用 WebGPURenderer 打开同一套球场参数。

坐标约定
--------
bsk.court 使用 z-up（x 长边、y 短边、z 向上）；pygfx 使用 y-up。
``world_pos`` 统一做 (x, z, -y) 转换，保持右手系不变。

输出
----
images/*.jpg + labels/*.txt（YOLO: class cx cy w h）+ meta.json
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    import pygfx as gfx
    from pygfx.renderers.wgpu.engine import renderstate as _rsmod
except Exception as exc:  # pragma: no cover - 只在使用合成器时需要
    gfx = None
    _rsmod = None
    _GFX_IMPORT_ERROR = exc
else:
    _GFX_IMPORT_ERROR = None

from bsk.calib_seg_real import sample_camera
from bsk.court import FIBA, NBA, Court
from bsk.court_3d import CL_BACKGROUND, CL_PAINT, CourtMesh3D


def _require_gfx() -> None:
    if gfx is None:
        raise RuntimeError(f"pygfx/wgpu 未安装：{_GFX_IMPORT_ERROR}")


# pygfx 0.18 的阴影贴图固定 1024；改到 2048 后阴影边缘明显更实。
# 在第一次创建 RenderState 前替换 _setup_light_resources 即可。
def _install_shadow_resolution(size: int = 2048) -> None:
    if _rsmod is None:
        return
    cls = _rsmod.LightRenderState
    if getattr(cls, "_bsk_shadow_patched", False):
        return
    orig = cls._setup_light_resources

    def _setup(self):
        self.shadow_map_size = (int(size), int(size))
        orig(self)

    cls._setup_light_resources = _setup
    cls._bsk_shadow_patched = True


def to_scene(x: float, y: float, z: float) -> np.ndarray:
    """bsk.court (z-up) → pygfx (y-up)。"""
    return np.array([x, z, -y], np.float64)


def _rot_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], np.float64)


def _quad_geometry(polys, y_offset: float = 0.0):
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    indices: list[int] = []
    for poly in polys:
        base = len(positions)
        for x, y, z in poly.vertices:
            positions.append((x, z + y_offset, -y))
            normals.append((0.0, 1.0, 0.0))
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    if not positions:
        return None
    return gfx.Geometry(
        positions=np.array(positions, np.float32),
        indices=np.array(indices, np.uint32).reshape(-1, 3),
        normals=np.array(normals, np.float32),
    )


def _torus_geometry(radius: float, tube: float, n_u: int = 48, n_v: int = 12):
    verts, norms, idx = [], [], []
    for i in range(n_u):
        u = 2 * math.pi * i / n_u
        cu, su = math.cos(u), math.sin(u)
        for j in range(n_v):
            v = 2 * math.pi * j / n_v
            cv, sv = math.cos(v), math.sin(v)
            rr = radius + tube * cv
            verts.append((rr * cu, tube * sv, rr * su))
            norms.append((cv * cu, sv, cv * su))
    for i in range(n_u):
        i2 = (i + 1) % n_u
        for j in range(n_v):
            j2 = (j + 1) % n_v
            a, b = i * n_v + j, i2 * n_v + j
            c, d = i2 * n_v + j2, i * n_v + j2
            idx.extend([a, b, c, a, c, d])
    return gfx.Geometry(
        positions=np.array(verts, np.float32),
        indices=np.array(idx, np.uint32).reshape(-1, 3),
        normals=np.array(norms, np.float32),
    )


def _wood_texture(rng: np.random.Generator, base_rgb: np.ndarray, size=(1024, 512)):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    plank_w = float(rng.uniform(58, 92))
    seam = (np.mod(xx, plank_w) < 1.3).astype(np.float32)
    seam = cv2.GaussianBlur(seam, (0, 0), 0.6)
    grain = (np.sin(xx * 0.06 + 2.5 * np.sin(yy * 0.018)) +
             0.55 * np.sin(xx * 0.013 + 1.7) +
             0.35 * rng.normal(0, 1, (h, w)))
    grain = cv2.GaussianBlur(grain.astype(np.float32), (0, 0), 1.6)
    grain = 1.0 + 0.055 * grain
    board_tint = 1.0 + 0.06 * np.sin((xx // plank_w) * 1.7 + 0.4)
    field = grain * board_tint * (1.0 - 0.20 * seam)
    rgb = np.clip(base_rgb[None, None, :] * field[..., None], 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb, np.full((h, w, 1), 255, np.uint8)])
    return gfx.Texture(data=np.ascontiguousarray(rgba), dim=2, format="rgba8unorm")


def _environment_cube(size: int = 128):
    """程序化球馆 HDR 环境：暗墙/地板 + 顶部灯板，供 PBR 反射使用。"""
    faces = np.zeros((size, size, 6, 4), np.float16)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    for face in range(6):
        if face == 3:  # -Y：木地板方向
            base = np.array([0.30, 0.18, 0.095], np.float32)
        elif face == 2:  # +Y：天花板
            base = np.array([0.11, 0.12, 0.15], np.float32)
        else:
            base = np.array([0.075, 0.088, 0.12], np.float32)
        faces[..., face, :3] = base
        faces[..., face, 3] = 1.0

    def panel(face, x0, y0, x1, y1, rgb):
        m = (xx >= x0 * size) & (xx < x1 * size) & (yy >= y0 * size) & (yy < y1 * size)
        faces[..., face, :3][m] = np.array(rgb, np.float16)

    # 顶部大面积柔光板，先抬整体环境，再用灯带形成高光
    panel(2, 0.04, 0.04, 0.96, 0.96, (1.10, 1.06, 0.98))
    # 顶部灯板，HDR 值明显高于 1，形成真实高光
    for x0, x1 in ((0.08, 0.30), (0.40, 0.62), (0.70, 0.92)):
        panel(2, x0, 0.18, x1, 0.34, (7.5, 6.8, 5.8))
        panel(2, x0, 0.66, x1, 0.82, (7.5, 6.8, 5.8))
    # 侧墙灯带
    for face in (0, 1, 4, 5):
        panel(face, 0.10, 0.30, 0.90, 0.36, (1.8, 2.0, 2.4))
        panel(face, 0.10, 0.64, 0.90, 0.70, (1.2, 1.4, 1.8))
    tex = gfx.Texture(data=np.ascontiguousarray(faces), dim=2,
                      size=(size, size, 6), format="rgba16float",
                      colorspace="physical", generate_mipmaps=True)
    return gfx.TextureMap(tex, min_filter="linear", mag_filter="linear",
                          mipmap_filter="linear")


def _ball_textures(size=(512, 256)):
    """按球面经纬度生成 8 面板篮球接缝和皮革法线贴图。"""
    w, h = size
    u = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    v = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    phi = 2 * math.pi * u
    theta = math.pi * (1.0 - v)
    x = -np.cos(phi) * np.sin(theta)
    y = np.cos(theta)
    z = np.sin(phi) * np.sin(theta)
    # 三个大圆把球面分成 8 块：赤道、经线、45° 倾斜大圆。
    d0 = np.abs(y)
    d1 = np.abs(x)
    d2 = np.abs(x + z) / math.sqrt(2.0)
    seam = np.minimum(np.minimum(d0, d1), d2) < 0.016
    panel_id = ((y > 0).astype(np.uint8) |
                ((x > 0).astype(np.uint8) << 1) |
                ((x + z > 0).astype(np.uint8) << 2))
    shade = 0.92 + 0.035 * panel_id
    rng = np.random.default_rng(17)
    leather = rng.normal(0, 2.2, (h, w)).astype(np.float32)
    base = np.zeros((h, w, 3), np.float32)
    base[:] = (222, 101, 27)
    base *= shade[..., None]
    base += leather[..., None]
    base[seam] = (34, 22, 16)
    rgba = np.dstack([np.clip(base, 0, 255).astype(np.uint8),
                      np.full((h, w, 1), 255, np.uint8)])

    height = rng.normal(0, 1, (h, w)).astype(np.float32)
    height = cv2.GaussianBlur(height, (0, 0), 0.65)
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    normal = np.dstack([-gx * 0.55, -gy * 0.55, np.ones_like(gx)])
    normal /= np.linalg.norm(normal, axis=2, keepdims=True) + 1e-6
    normal_rgba = np.dstack([np.clip((normal * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8),
                             np.full((h, w, 1), 255, np.uint8)])
    albedo = gfx.Texture(data=np.ascontiguousarray(rgba), dim=2, format="rgba8unorm")
    normal_map = gfx.Texture(data=np.ascontiguousarray(normal_rgba), dim=2,
                             format="rgba8unorm", colorspace="physical")
    return albedo, normal_map


def _line_segments(points: Iterable[tuple[float, float, float]], color="#e8e8e0",
                   thickness: float = 1.2):
    pts = np.array(points, np.float32).reshape(-1, 3)
    geo = gfx.Geometry(positions=pts)
    return gfx.Line(geo, gfx.LineMaterial(color=color, thickness=thickness))


def _box(size, position, material, cast_shadow=True, receive_shadow=True):
    mesh = gfx.Mesh(gfx.box_geometry(*size), material)
    mesh.local.position = tuple(position)
    mesh.cast_shadow = cast_shadow
    mesh.receive_shadow = receive_shadow
    return mesh


@dataclass
class Arena:
    court: Court
    size: tuple[int, int]
    scene: object
    camera: object
    renderer: object
    target: object
    ball: object
    main_light: object
    fill_light: object
    spot_lights: list
    rng: np.random.Generator

    def set_camera(self, cam: dict) -> None:
        w, h = self.size[1], self.size[0]
        fl = float(cam.get("focal_length", 1000.0))
        fov = math.degrees(2.0 * math.atan((w + h) / (4.0 * fl)))
        self.camera.fov = max(10.0, min(100.0, fov))
        self.camera.aspect = w / h
        self.camera.local.position = to_scene(*cam["pos"])
        self.camera.world.reference_up = to_scene(*cam["up"])
        self.camera.look_at(to_scene(*cam["look_at"]))

    def set_ball(self, x: float, y: float, z: float, visible: bool = True) -> None:
        self.ball.visible = bool(visible)
        self.ball.local.position = to_scene(x, y, z)

    def set_lighting(self, rng: np.random.Generator) -> None:
        # 球馆有多盏顶灯，主光+补光+四盏面光共同软化阴影；避免单盏灯造成“假投影”。
        self.main_light.intensity = float(rng.uniform(1.18, 1.68))
        self.fill_light.intensity = float(rng.uniform(0.60, 0.92))
        for s in self.spot_lights:
            s.intensity = float(rng.uniform(3.7, 5.2))

    def project(self, p_world: np.ndarray) -> tuple[float, float, float]:
        w, h = self.size[1], self.size[0]
        pm = np.array(self.camera.projection_matrix)
        vm = np.array(self.camera.view_matrix)
        clip = pm @ (vm @ np.r_[to_scene(*p_world), 1.0])
        if clip[3] <= 1e-6:
            return float("nan"), float("nan"), float(clip[3])
        ndc = clip[:3] / clip[3]
        u = (ndc[0] * 0.5 + 0.5) * w
        v = (1.0 - (ndc[1] * 0.5 + 0.5)) * h
        return float(u), float(v), float(clip[3])

    def ball_bbox(self, p_world: np.ndarray, radius: float = 0.12):
        w, h = self.size[1], self.size[0]
        pm = np.array(self.camera.projection_matrix)
        u, v, depth = self.project(p_world)
        if not np.isfinite(u) or depth <= 0.2:
            return None
        r_px = abs(float(pm[0, 0])) * (w * 0.5) * radius / max(depth, 1e-6)
        if not np.isfinite(r_px) or r_px < 1.5:
            return None
        x0, x1 = u - r_px, u + r_px
        y0, y1 = v - r_px, v + r_px
        if x1 <= 1 or y1 <= 1 or x0 >= w - 1 or y0 >= h - 1:
            return None
        cx0, cy0 = max(0.0, x0), max(0.0, y0)
        cx1, cy1 = min(w - 1.0, x1), min(h - 1.0, y1)
        if cx1 - cx0 < 2.0 or cy1 - cy0 < 2.0:
            return None
        # 中心至少落在画面内，避免只有几像素露出
        if not (0.0 <= u <= w - 1 and 0.0 <= v <= h - 1):
            return None
        return cx0, cy0, cx1, cy1

    def visible_ball_mask(self):
        """从 WebGPU pick pass 读取真实可见球像素，自动排除被篮板/篮架/看台遮挡。"""
        pick = self.renderer._blender.get_texture("pick")
        if pick is None:
            return None
        h, w = self.size
        bpr = ((w * 8 + 255) // 256) * 256
        raw = self.renderer._device.queue.read_texture(
            {"texture": pick, "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": bpr, "rows_per_image": h},
            (w, h, 1),
        )
        rows = np.frombuffer(raw, np.uint8).reshape(h, bpr)[:, : w * 8]
        ids = rows.view(np.uint64).reshape(h, w) & ((1 << 20) - 1)
        return ids == self.ball.id

    @staticmethod
    def mask_bbox(mask):
        if mask is None:
            return None
        ys, xs = np.nonzero(mask)
        if len(xs) < 3:
            return None
        h, w = mask.shape
        x0, x1 = max(0.0, float(xs.min()) - 1.0), min(w - 1.0, float(xs.max()) + 1.0)
        y0, y1 = max(0.0, float(ys.min()) - 1.0), min(h - 1.0, float(ys.max()) + 1.0)
        if x1 - x0 < 2.0 or y1 - y0 < 2.0:
            return None
        return x0, y0, x1, y1

    def visible_ball_bbox(self):
        return self.mask_bbox(self.visible_ball_mask())

    def render(self) -> np.ndarray:
        self.renderer.render(self.scene, self.camera)
        img = self.renderer.snapshot()
        if img.shape[0] != self.size[0] or img.shape[1] != self.size[1]:
            img = cv2.resize(img, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
        rgb = img[..., :3]
        if np.issubdtype(rgb.dtype, np.integer):
            rgb = rgb.astype(np.float32) / 255.0
        return rgb.astype(np.float32, copy=False)


def _build_court_geometry(scene, court: Court, rng: np.random.Generator):
    mesh = CourtMesh3D.build(court)
    base_wood = np.array([rng.uniform(150, 195), rng.uniform(95, 135), rng.uniform(48, 82)], np.float32)
    floor = gfx.Mesh(
        gfx.plane_geometry(court.length + 2 * court.line_width + 3.6,
                           court.width + 2 * court.line_width + 3.6),
        gfx.MeshPhysicalMaterial(map=_wood_texture(rng, base_wood), roughness=0.60,
                                  metalness=0.0, clearcoat=0.07, clearcoat_roughness=0.52,
                                  env_map_intensity=0.20),
    )
    floor.local.euler = (-math.pi / 2, 0, 0)
    floor.receive_shadow = True
    scene.add(floor)

    paint_polys = [p for p in mesh.polys if p.class_id == CL_PAINT and p.fill_only]
    pg = _quad_geometry(paint_polys, 0.002)
    if pg is not None:
        paint_rgb = np.array([rng.uniform(28, 55), rng.uniform(48, 92), rng.uniform(115, 175)])
        paint = gfx.Mesh(pg, gfx.MeshStandardMaterial(
            color=tuple(float(v) / 255 for v in paint_rgb), roughness=0.68,
                             metalness=0.0, env_map_intensity=0.12))
        paint.receive_shadow = True
        scene.add(paint)

    line_polys = [p for p in mesh.polys if p.class_id != CL_BACKGROUND and not p.fill_only]
    lg = _quad_geometry(line_polys, 0.008)
    # 球场线固定为白色，不随灯光变暗/变色
    lines = gfx.Mesh(lg, gfx.MeshBasicMaterial(color="#f4f4ee"))
    scene.add(lines)


def _build_hoop(scene, court: Court, side: str):
    sign = 1.0 if side == "A" else -1.0
    bl = court.half_length * sign
    rim_x = bl - sign * court.rim_offset_from_baseline
    bb_x = bl - sign * court.backboard_from_baseline
    rim_y = court.rim_height
    board_center_y = rim_y + 0.375

    board_mat = gfx.MeshPhysicalMaterial(color="#dce8f2", roughness=0.08, metalness=0.0,
                                         opacity=0.38, clearcoat=0.85, clearcoat_roughness=0.06,
                                         ior=1.45, env_map_intensity=1.15)
    board = _box((0.045, 1.05, 1.80), (bb_x, board_center_y, 0.0), board_mat)
    scene.add(board)

    front_x = bb_x - sign * 0.026
    border = []
    def rect(y0, y1, z0, z1):
        border.extend([
            (front_x, y0, z0), (front_x, y1, z0),
            (front_x, y1, z0), (front_x, y1, z1),
            (front_x, y1, z1), (front_x, y0, z1),
            (front_x, y0, z1), (front_x, y0, z0),
        ])
    rect(rim_y - 0.20, rim_y + 0.85, -0.90, 0.90)
    rect(rim_y + 0.075, rim_y + 0.525, -0.295, 0.295)
    scene.add(_line_segments(border, color="#111418", thickness=2.2))

    rim = gfx.Mesh(
        _torus_geometry(court.rim_radius, 0.012),
        gfx.MeshPhysicalMaterial(color="#f05a1f", roughness=0.26, metalness=0.62,
                                  clearcoat=0.18, env_map_intensity=1.0),
    )
    rim.local.position = (rim_x, rim_y, 0.0)
    rim.cast_shadow = True
    scene.add(rim)

    segs = []
    n = 12
    top_y = rim_y - 0.012
    bot_y = rim_y - 0.45
    bot_r = court.rim_radius * 0.52
    for i in range(n):
        a = 2 * math.pi * i / n
        a2 = 2 * math.pi * (i + 1) / n
        p0 = (rim_x + court.rim_radius * math.cos(a), top_y, court.rim_radius * math.sin(a))
        p1 = (rim_x + bot_r * math.cos(a), bot_y, bot_r * math.sin(a))
        p2 = (rim_x + bot_r * math.cos(a2), bot_y, bot_r * math.sin(a2))
        segs.extend([p0, p1, p1, p2])
    for yy in (rim_y - 0.15, rim_y - 0.30, rim_y - 0.45):
        rr = court.rim_radius * (1.0 - (rim_y - yy) / 0.6 * 0.48)
        for i in range(n):
            a, a2 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
            segs.extend([
                (rim_x + rr * math.cos(a), yy, rr * math.sin(a)),
                (rim_x + rr * math.cos(a2), yy, rr * math.sin(a2)),
            ])
    scene.add(_line_segments(segs, color="#d8d8d0", thickness=1.1))

    # 底座必须完全在端线外；旧位置 0.65 m 会让 1.35 m 底座压住底线。
    pole_x = bl + sign * 2.20
    pole = _box((0.18, 3.55, 0.18), (pole_x, 1.775, 0.0),
                gfx.MeshStandardMaterial(color="#171d28", roughness=0.55, metalness=0.25))
    scene.add(pole)
    arm_len = abs(bb_x - pole_x) + 0.08
    arm = _box((arm_len, 0.11, 0.11), ((pole_x + bb_x) / 2, rim_y + 0.22, 0.0),
               gfx.MeshStandardMaterial(color="#171d28", roughness=0.55, metalness=0.25))
    scene.add(arm)
    pad_rgb = (0.55, 0.10, 0.10) if side == "A" else (0.08, 0.18, 0.55)
    pad = _box((0.85, 0.75, 0.85), (pole_x, 0.375, 0.0),
               gfx.MeshStandardMaterial(color=pad_rgb, roughness=0.85, metalness=0.0))
    scene.add(pad)
    base = _box((1.00, 0.10, 1.00), (pole_x, 0.05, 0.0),
                gfx.MeshStandardMaterial(color="#10141c", roughness=0.9, metalness=0.0))
    scene.add(base)


def _build_benches(scene, court: Court, rng: np.random.Generator):
    """两侧替补席：十席长凳、靠背、金属腿、水瓶和毛巾。"""
    hw = court.half_width
    frame_mat = gfx.MeshStandardMaterial(color="#252c36", roughness=0.45, metalness=0.55)
    bottle_mat = gfx.MeshStandardMaterial(color="#9fd6e8", roughness=0.25, metalness=0.0,
                                           opacity=0.75)
    towel_mat = gfx.MeshStandardMaterial(color="#e7e7e2", roughness=0.95, metalness=0.0)
    team_colors = [(0.10, 0.24, 0.62), (0.62, 0.10, 0.12)]
    length = 7.2
    n_seats = 10
    seat_w = length / n_seats
    for side in (-1.0, 1.0):
        z = side * (hw + 3.35)
        team = team_colors[0 if side > 0 else 1]
        cushion = gfx.MeshStandardMaterial(color=team, roughness=0.78, metalness=0.0)
        scene.add(_box((length, 0.10, 0.56), (0.0, 0.42, z), frame_mat))
        scene.add(_box((length, 0.56, 0.10), (0.0, 0.73, z + side * 0.28), frame_mat))
        for i in range(n_seats):
            x = -length / 2 + seat_w * (i + 0.5)
            scene.add(_box((seat_w - 0.05, 0.09, 0.52), (x, 0.49, z), cushion))
            scene.add(_box((seat_w - 0.05, 0.43, 0.09), (x, 0.75, z + side * 0.23), cushion))
        for i in range(7):
            x = -length / 2 + 0.18 + i * (length - 0.36) / 6
            for dz in (-0.20, 0.20):
                scene.add(_box((0.07, 0.42, 0.07), (x, 0.21, z + dz), frame_mat))
        for i in range(4):
            x = -length / 2 + 0.7 + i * 1.75
            bottle = gfx.Mesh(gfx.cylinder_geometry(0.045, 0.045, 0.20, radial_segments=10),
                              bottle_mat)
            bottle.local.position = (x, 0.60, z - side * 0.20)
            bottle.cast_shadow = True
            scene.add(bottle)
            towel = _box((0.32, 0.035, 0.22), (x + 0.35, 0.55, z), towel_mat)
            scene.add(towel)


def _build_stands(scene, court: Court, rng: np.random.Generator):
    hl, hw = court.half_length, court.half_width
    margin = 4.2
    rows = 6
    row_depth = 1.05
    row_h = 0.24
    stand_mat = gfx.MeshStandardMaterial(color="#2b3543", roughness=0.88, metalness=0.02)

    for r in range(rows):
        # 第一排贴近场地标高，后面每排只抬 24 cm；球馆看台不是八层大看台。
        top = 0.12 + r * row_h
        d = margin + (r + 0.5) * row_depth
        x_len = 2.0 * (hl + margin + (r + 1.0) * row_depth)
        z_len = 2.0 * (hw + margin + (r + 1.0) * row_depth)
        for s in (-1.0, 1.0):
            scene.add(_box((x_len, top, row_depth), (0.0, top * 0.5, s * (hw + d)), stand_mat))
            scene.add(_box((row_depth, top, z_len), (s * (hl + d), top * 0.5, 0.0), stand_mat))

    # 广告板/护栏
    bar_mat = gfx.MeshStandardMaterial(color="#0d1118", roughness=0.55, metalness=0.15,
                                       emissive="#163a68", emissive_intensity=0.35)
    for s in (-1.0, 1.0):
        scene.add(_box((2 * hl + 2.2, 0.85, 0.10), (0.0, 0.425, s * (hw + 2.55)), bar_mat))
        scene.add(_box((0.10, 0.85, 2 * hw + 2.2), (s * (hl + 2.55), 0.425, 0.0), bar_mat))

    # 观众：低多边形头肩剪影，实例化
    spots = []
    for r in range(rows):
        top = (r + 1) * row_h
        d = margin + (r + 0.5) * row_depth
        span_x = hl + margin + (r + 0.5) * row_depth - 0.3
        span_z = hw + margin + (r + 0.5) * row_depth - 0.3
        for s in (-1.0, 1.0):
            x = -span_x
            while x <= span_x:
                if rng.random() < 0.52:
                    spots.append((x + rng.uniform(-0.18, 0.18), top, s * (hw + d),
                                  rng.uniform(-0.4, 0.4)))
                x += rng.uniform(0.72, 0.92)
            z = -span_z
            while z <= span_z:
                if rng.random() < 0.52:
                    spots.append((s * (hl + d), top, z + rng.uniform(-0.18, 0.18),
                                  rng.uniform(-0.4, 0.4)))
                z += rng.uniform(0.72, 0.92)

    torso_colors = ["#1d2b40", "#2b384b", "#3a414d", "#452d2d", "#253a3a", "#41414f"]
    head_colors = ["#624634", "#866044", "#40352e", "#76543c"]
    torso_groups = {i: [] for i in range(len(torso_colors))}
    head_groups = {i: [] for i in range(len(head_colors))}
    for (x, y, z, ang) in spots:
        c = int(rng.integers(0, len(torso_colors)))
        torso_groups[c].append((x, y + 0.28, z, ang))
        head_groups[int(rng.integers(0, len(head_colors)))].append((x, y + 0.68, z, ang))

    for ci, items in torso_groups.items():
        if not items:
            continue
        im = gfx.InstancedMesh(gfx.box_geometry(0.36, 0.55, 0.26),
                               gfx.MeshStandardMaterial(color=torso_colors[ci], roughness=0.88),
                               len(items))
        im.cast_shadow = True
        im.receive_shadow = True
        for i, (x, y, z, ang) in enumerate(items):
            m = np.eye(4)
            m[:3, :3] = _rot_y(ang)
            m[:3, 3] = (x, y, z)
            im.set_matrix_at(i, m)
        scene.add(im)

    head_geo = gfx.sphere_geometry(0.115, 8, 6)
    for ci, items in head_groups.items():
        if not items:
            continue
        im = gfx.InstancedMesh(head_geo,
                               gfx.MeshStandardMaterial(color=head_colors[ci], roughness=0.9),
                               len(items))
        im.cast_shadow = True
        for i, (x, y, z, _ang) in enumerate(items):
            m = np.eye(4)
            m[:3, 3] = (x, y, z)
            im.set_matrix_at(i, m)
        scene.add(im)


def _build_lights(scene):
    # 真实球馆用分布式宽光束顶灯：照度高、重叠覆盖、暗区少。
    # 只让两盏中央灯投影，其余灯负责抬亮球场，避免多组硬阴影。
    scene.add(gfx.AmbientLight(color="#c3cfe2", intensity=0.56))
    main = gfx.DirectionalLight(color="#fff1df", intensity=1.42, cast_shadow=False)
    main.local.position = (0.0, 12.0, 0.0)
    scene.add(main)
    fill = gfx.DirectionalLight(color="#c3d4ff", intensity=0.74, cast_shadow=False)
    fill.local.position = (-9.0, 7.0, -6.0)
    scene.add(fill)
    spots = []
    positions = [(-10.0, -4.2), (-3.3, -4.2), (3.3, -4.2), (10.0, -4.2),
                 (-10.0, 4.2), (-3.3, 4.2), (3.3, 4.2), (10.0, 4.2)]
    for i, (x, z) in enumerate(positions):
        sp = gfx.SpotLight(color="#fff1dc", intensity=4.4, angle=0.92, penumbra=0.68,
                           distance=42.0, decay=1.0, cast_shadow=(i in (1, 6)))
        sp.local.position = (x, 11.0, z)
        sp.look_at(np.array([0.0, 0.0, 0.0]))
        sp.shadow.bias = -0.00025
        scene.add(sp)
        spots.append(sp)
    # 看台两侧的柔和补光，避免观众席黑成一片
    for z, color in ((-10.0, "#c7d8ff"), (10.0, "#ffe4c4")):
        pl = gfx.PointLight(color=color, intensity=7.0, distance=32.0, decay=1.0,
                            cast_shadow=False)
        pl.local.position = (0.0, 5.5, z)
        scene.add(pl)
    return main, fill, spots


def _camera_blocked_by_stands(cam: dict, court: Court, rows=6, row_depth=1.05,
                              row_h=0.24, margin=4.2) -> bool:
    x, y, z = cam["pos"]
    ex = max(0.0, abs(x) - (court.half_length + margin))
    ez = max(0.0, abs(y) - (court.half_width + margin))
    if ex <= 0 and ez <= 0:
        return False
    r = int(max(ex, ez) / row_depth)
    top = 0.12 + min(rows - 1, r) * row_h
    return y < top + 0.6


def build_arena(court: Court, size=(540, 960), seed: int = 0,
                shadow_map: int = 2048) -> Arena:
    _require_gfx()
    _install_shadow_resolution(shadow_map)
    rng = np.random.default_rng(seed)
    h, w = size
    target = gfx.Texture(size=(w, h, 1), dim=2, format="rgba8unorm")
    renderer = gfx.WgpuRenderer(target, pixel_ratio=1, ppaa="default", color_precision="float16")
    scene = gfx.Scene()
    scene.environment = _environment_cube()
    _build_court_geometry(scene, court, rng)
    _build_hoop(scene, court, "A")
    _build_hoop(scene, court, "B")
    _build_benches(scene, court, rng)
    _build_stands(scene, court, rng)
    main, fill, spots = _build_lights(scene)

    ball_albedo, ball_normal = _ball_textures()
    ball_mat = gfx.MeshStandardMaterial(map=ball_albedo, normal_map=ball_normal,
                                         normal_scale=(0.38, 0.38), color="#ffffff",
                                         roughness=0.64, metalness=0.0, env_map_intensity=0.32)
    ball_mat.pick_write = True
    ball = gfx.Mesh(gfx.sphere_geometry(0.12, 32, 16), ball_mat)
    ball.cast_shadow = True
    scene.add(ball)

    camera = gfx.PerspectiveCamera(50.0, w / h, depth_range=(0.1, 300.0))
    return Arena(court=court, size=size, scene=scene, camera=camera, renderer=renderer,
                 target=target, ball=ball, main_light=main, fill_light=fill,
                 spot_lights=spots, rng=rng)


def _sample_ball(court: Court, rng: np.random.Generator, camera_cam: dict) -> tuple[float, float, float]:
    """在球场/篮架附近采样一个篮球位置；多数样本贴近相机 POI，避免正样本大量出画。"""
    hl, hw = court.half_length, court.half_width
    mode = str(rng.choice(["arc", "air", "floor", "rim"], p=[0.40, 0.28, 0.19, 0.13]))
    side = 1.0 if rng.random() < 0.5 else -1.0
    rim_x = side * hl - side * court.rim_offset_from_baseline
    poi_x, poi_y, _ = camera_cam.get("look_at", (0.0, 0.0, 0.0))
    if mode == "arc":
        start_x = rim_x - side * rng.uniform(4.0, 9.5)
        start_y = rng.uniform(-5.0, 5.0)
        t = rng.uniform(0.25, 0.92)
        x = start_x + (rim_x - start_x) * t
        y = start_y * (1.0 - t) + rng.uniform(-0.4, 0.4) * t
        z = (1.9 + 2.8 * math.sin(math.pi * t)) * (1.0 - t) + (court.rim_height + 0.15) * t
    elif mode == "air":
        if rng.random() < 0.72:
            x, y = poi_x + rng.uniform(-4.0, 4.0), poi_y + rng.uniform(-3.0, 3.0)
        else:
            x, y = rng.uniform(-hl + 1.0, hl - 1.0), rng.uniform(-hw + 1.0, hw - 1.0)
        z = rng.uniform(1.2, 4.2)
    elif mode == "floor":
        if rng.random() < 0.72:
            x, y = poi_x + rng.uniform(-4.0, 4.0), poi_y + rng.uniform(-3.0, 3.0)
        else:
            x, y = rng.uniform(-hl + 1.0, hl - 1.0), rng.uniform(-hw + 1.0, hw - 1.0)
        z = rng.uniform(0.12, 1.1)
    else:
        x = rim_x + rng.uniform(-1.4, 1.4)
        y = rng.uniform(-1.2, 1.2)
        z = rng.uniform(2.6, 4.3)
    return float(np.clip(x, -hl + 0.4, hl - 0.4)), float(np.clip(y, -hw + 0.4, hw - 0.4)), float(z)


def _camera_ok(camera_cam: dict, court: Court) -> bool:
    if _camera_blocked_by_stands(camera_cam, court):
        return False
    x, y, z = camera_cam["pos"]
    if z < 3.2:
        return False
    # 太贴近板/篮架会穿模
    for side in (1.0, -1.0):
        bx = side * court.half_length - side * court.backboard_from_baseline
        if abs(x - bx) < 0.5 and abs(y) < 1.2 and z < 4.2:
            return False
    return True


def generate_ball_dataset(
    out_dir: str | Path,
    n_samples: int = 200,
    size: tuple[int, int] = (540, 960),
    seed: int = 0,
    courts: list[Court] | None = None,
    negative_rate: float = 0.06,
    motion_blur_rate: float = 0.35,
    preview: int = 8,
) -> Path:
    """生成球检测合成数据。labels 为 YOLO 单类（0=ball）。"""
    _require_gfx()
    courts = courts or [FIBA, NBA]
    out = Path(out_dir)
    for sub in ("images", "labels", "previews"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arenas = []
    for c in courts:
        arenas.append(build_arena(c, size, seed=int(rng.integers(0, 1_000_000))))
    meta = []
    t0 = time.time()
    for i in range(n_samples):
        arena = arenas[i % len(arenas)]
        court = arena.court
        # 选一个不穿看台的机位
        cam = None
        for _ in range(24):
            cand = sample_camera(size, court, rng)
            if _camera_ok(cand, court):
                cam = cand
                break
        if cam is None:
            cam = sample_camera(size, court, rng, mode="high")
        arena.set_camera(cam)
        arena.set_lighting(rng)

        is_neg = rng.random() < negative_rate
        bbox = None
        ball_pos = None
        ball_mask = None
        blurred = False
        if not is_neg:
            for _ in range(50):
                x, y, z = _sample_ball(court, rng, cam)
                arena.set_ball(x, y, z, visible=True)
                # 先用投影框快速剔除出画样本，再渲染；最终标签取 pick pass 的真实可见像素。
                if arena.ball_bbox((x, y, z)) is None:
                    continue
                img = arena.render()
                ball_mask = arena.visible_ball_mask()
                bbox = arena.mask_bbox(ball_mask)
                if bbox is not None:
                    ball_pos = (x, y, z)
                    break
            if bbox is not None and rng.random() < motion_blur_rate:
                # 用快门曝光内的 3 次子帧平均模拟运动模糊；静态球馆不变，只有球和球影移动。
                theta = float(rng.uniform(0, 2 * math.pi))
                speed = float(rng.uniform(4.0, 14.0))
                shutter = float(rng.uniform(1 / 800, 1 / 250))
                delta = np.array([speed * math.cos(theta) * shutter,
                                  speed * math.sin(theta) * shutter,
                                  rng.uniform(-3.0, 4.0) * shutter])
                dn = float(np.linalg.norm(delta))
                if dn > 0.15:
                    delta *= 0.15 / dn
                frames = [img]
                union = ball_mask.copy()
                p = np.array(ball_pos, float)
                for s in (-1.0, 1.0):
                    pp = p + s * delta * 0.5
                    pp[2] = max(0.12, pp[2])
                    arena.set_ball(*pp)
                    f = arena.render()
                    m = arena.visible_ball_mask()
                    frames.append(f)
                    if m is not None:
                        union |= m
                img = ((frames[0].astype(np.float32) +
                        frames[1].astype(np.float32) +
                        frames[2].astype(np.float32)) / 3.0)
                bbox = arena.mask_bbox(union)
                arena.set_ball(*ball_pos)
                blurred = True
            if bbox is None:
                is_neg = True
                arena.set_ball(0, 0, -2, visible=False)
                img = arena.render()
        else:
            arena.set_ball(0, 0, -2, visible=False)
            img = arena.render()

        # 后处理：曝光/白平衡/噪声/暗角/JPEG
        img = _postprocess(img, rng)
        cv2.imwrite(str(out / "images" / f"{i:06d}.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            w, h = size[1], size[0]
            cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
            bw, bh = (x1 - x0) / w, (y1 - y0) / h
            (out / "labels" / f"{i:06d}.txt").write_text(
                f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
        else:
            (out / "labels" / f"{i:06d}.txt").write_text("", encoding="utf-8")

        if i < preview:
            vis = img.copy()
            if bbox is not None:
                cv2.rectangle(vis, (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.imwrite(str(out / "previews" / f"{i:06d}.jpg"),
                        cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        meta.append({
            "idx": i,
            "court_model": court.model,
            "camera_mode": cam.get("mode"),
            "ball": None if ball_pos is None else [round(v, 4) for v in ball_pos],
            "bbox": None if bbox is None else [round(float(v), 2) for v in bbox],
            "motion_blur": bool(blurred),
            "negative": bool(bbox is None),
        })
        if (i + 1) % 20 == 0:
            dt = time.time() - t0
            print(f"  rendered {i + 1}/{n_samples}  ({dt / (i + 1):.2f}s/frame)", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"ball dataset saved to {out} ({n_samples} samples, size={size[1]}x{size[0]})")
    return out


def _tone_map(linear: np.ndarray, exposure: float = 1.0,
              wb: tuple[float, float, float] = (1.0, 1.0, 1.0),
              vignette: float = 0.14) -> np.ndarray:
    """线性 HDR → 曝光/白平衡/暗角 → ACES → sRGB uint8。"""
    x = np.maximum(linear.astype(np.float32), 0.0) * float(exposure)
    x *= np.asarray(wb, np.float32)[None, None, :]
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w * 0.5) / max(w * 0.5, 1)
    ny = (yy - h * 0.5) / max(h * 0.5, 1)
    x *= (1.0 - vignette * np.clip(nx * nx + ny * ny, 0, 1))[..., None]
    mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    srgb = np.where(mapped <= 0.0031308, 12.92 * mapped,
                    1.055 * np.power(np.maximum(mapped, 0), 1 / 2.4) - 0.055)
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)


def _postprocess(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    exposure = float(rng.uniform(1.18, 1.58))
    wb = tuple(float(v) for v in (1.0 + rng.normal(0.0, 0.025, 3)))
    out = _tone_map(img, exposure=exposure, wb=wb)
    out = np.clip(out.astype(np.float32) +
                  rng.normal(0, float(rng.uniform(0.5, 2.6)), out.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        q = int(rng.integers(80, 95))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out


def render_top_view(out_path: str | Path, court: Court = FIBA,
                    size: tuple[int, int] = (600, 1000), seed: int = 0) -> Path:
    """输出正投影顶视图，用于审计球场线方向、尺寸和篮架位置。"""
    _require_gfx()
    arena = build_arena(court, size, seed=seed)
    h, w = size
    view_w = court.length + 6.0
    view_h = view_w * h / w
    cam = gfx.OrthographicCamera(width=view_w, height=view_h, depth_range=(0.1, 120.0))
    cam.local.position = (0.0, 30.0, 0.0)
    cam.world.reference_up = np.array([0.0, 0.0, -1.0])
    cam.look_at(np.array([0.0, 0.0, 0.0]))
    arena.camera = cam
    arena.set_ball(0, 0, -2, visible=False)
    img = _tone_map(arena.render(), exposure=1.42, vignette=0.05)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="WebGPU 3D 球馆球检测合成数据")
    ap.add_argument("--out", default="temp/ball3d_probe")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--size", default="540x960")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--courts", default="FIBA,NBA")
    ap.add_argument("--top-view", action="store_true", help="只渲染一张顶视图到 --out")
    ap.add_argument("--motion-blur-rate", type=float, default=0.35)
    args = ap.parse_args()
    hh, ww = (int(v) for v in args.size.lower().split("x"))
    cmap = {"FIBA": FIBA, "NBA": NBA}
    cs = [cmap[x.strip().upper()] for x in args.courts.split(",") if x.strip().upper() in cmap]
    if args.top_view:
        target = Path(args.out).with_suffix(".jpg")
        render_top_view(target, (cs or [FIBA])[0], (hh, ww), args.seed)
        print(f"top view saved to {target}")
    else:
        generate_ball_dataset(args.out, args.n, (hh, ww), args.seed, cs or [FIBA],
                              motion_blur_rate=args.motion_blur_rate)