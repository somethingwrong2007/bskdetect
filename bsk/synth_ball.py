"""合成篮球检测训练数据：3D 球体投影 + 物理弹道 + 数据增强。

核心思路：
1. 用标准篮球直径 (0.246m) 构建 3D 球体
2. 通过摄像机 H_inv 投影到像素空间
3. 模拟真实弹道（重力、反弹、旋转）
4. 添加干扰：遮挡、光照、运动模糊、颜色抖动
5. 输出 YOLO 格式标注 (image, label)

性能目标：
- 单帧生成 < 50ms
- 支持批量生成 10k+ 样本
- 自动平衡正负样本比例
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# ── 物理参数 ───────────────────────────────────────────────

BALL_DIAMETER_M = 0.246  # FIBA 标准篮球直径
BALL_RADIUS_M = BALL_DIAMETER_M / 2
GRAVITY = 9.81  # m/s²
BOUNCE_DAMPING = 0.7  # 反弹能量损失
DRAG_COEFF = 0.02  # 空气阻力


@dataclass
class BallState:
    """球的物理状态（世界坐标，米制）。"""
    x: float = 0.0  # 沿球场长边
    y: float = 0.0  # 沿短边
    z: float = 1.0  # 垂直高度（地面为0）
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    angular_vel: float = 0.0  # 旋转速度（影响马格努斯效应）


@dataclass
class BallRenderConfig:
    """渲染配置。"""
    court_model: str = "FIBA"
    img_width: int = 1920
    img_height: int = 1080
    ball_color_base: tuple[int, int, int] = (255, 140, 0)  # 橙色
    enable_motion_blur: bool = True
    enable_occlusion: bool = True
    enable_lighting: bool = True
    occlusion_prob: float = 0.3
    blur_prob: float = 0.4


class BallPhysics:
    """篮球物理模拟器。"""

    def __init__(self, state: BallState | None = None):
        self.state = state or BallState()
        self.trail: list[BallState] = []  # 轨迹历史

    def step(self, dt: float = 1/30, floor_z: float = BALL_RADIUS_M) -> BallState:
        """推进一帧物理状态。

        Args:
            dt: 时间步长（秒），30FPS ≈ 0.033s
            floor_z: 地面高度（球触地时 z = 半径）

        Returns:
            更新后的球状态
        """
        s = self.state
        self.trail.append(BallState(**vars(s)))

        # 重力
        s.vz -= GRAVITY * dt

        # 空气阻力
        speed = math.sqrt(s.vx**2 + s.vy**2 + s.vz**2)
        drag = DRAG_COEFF * speed
        s.vx *= (1 - drag * dt)
        s.vy *= (1 - drag * dt)
        s.vz *= (1 - drag * dt)

        # 马格努斯效应（旋转导致侧向偏移）
        if s.angular_vel != 0:
            s.vx += s.angular_vel * s.vz * dt * 0.1
            s.vy += s.angular_vel * s.vx * dt * 0.05

        # 位置更新
        s.x += s.vx * dt
        s.y += s.vy * dt
        s.z += s.vz * dt

        # 地面碰撞
        if s.z <= floor_z:
            s.z = floor_z
            s.vz = -s.vz * BOUNCE_DAMPING
            # 地面摩擦
            s.vx *= 0.95
            s.vy *= 0.95
            # 旋转衰减
            s.angular_vel *= 0.9

        # 篮板/篮筐碰撞（简化：假设在 x=±14m 处）
        from bsk.court import FIBA
        if abs(s.x) > FIBA.half_length - 0.5:
            s.vx = -s.vx * 0.6

        self.state = s
        return s

    def reset(self, x: float = 0, y: float = 0, z: float = 2.0,
              vx: float = 5.0, vy: float = 2.0, vz: float = 0.0):
        """重置球状态（用于生成新轨迹）。"""
        self.state = BallState(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, angular_vel=random.uniform(-2, 2))
        self.trail.clear()


class BallRenderer:
    """篮球渲染器：3D → 2D 投影 + 视觉增强。"""

    def __init__(self, H_inv: np.ndarray, config: BallRenderConfig | None = None):
        """
        Args:
            H_inv: 世界坐标→像素坐标的单应矩阵 (3x3)
            config: 渲染配置
        """
        self.H_inv = H_inv
        self.config = config or BallRenderConfig()

    def project_sphere(self, state: BallState) -> tuple[int, int, int] | None:
        """将球心（3D 世界坐标）投影到像素空间，返回 (cx, cy, radius_px)。

        原理：
        1. 球心投影：(x, y, z) → (u, v) - 使用齐次坐标
        2. 半径投影：(x+r, y, z) → (u', v)，计算 pixel_radius = |u' - u|
        """
        # 球心投影（包含 z 坐标的完整 3D→2D 投影）
        # H_inv 是 3x3 矩阵，假设球场在 z=0 平面
        # 对于空中的球，需要透视修正
        pt_center = np.array([state.x, state.y, 1.0])
        p_center = self.H_inv @ pt_center
        if abs(p_center[2]) < 1e-8:
            return None
        cx = p_center[0] / p_center[2]
        cy = p_center[1] / p_center[2]

        # 高度引起的像素偏移修正（透视效果）
        # 球越高，在图像中位置会略微偏移
        height_offset = state.z * 0.5  # 简化的透视偏移系数
        cy -= height_offset

        # 半径投影（近似：用球边缘点）
        r_world = BALL_RADIUS_M
        pt_edge = np.array([state.x + r_world, state.y, 1.0])
        p_edge = self.H_inv @ pt_edge
        if abs(p_edge[2]) < 1e-8:
            return None
        ux = p_edge[0] / p_edge[2]
        radius_px = abs(ux - cx)

        # 高度透视修正：球越高，看起来越小（距离摄像机更远）
        perspective_scale = 1.0 / (1.0 + state.z * 0.15)
        radius_px *= perspective_scale

        return (int(cx), int(cy), max(int(radius_px), 3))

    def render_ball(self, img: np.ndarray, state: BallState,
                    prev_state: BallState | None = None) -> np.ndarray:
        """在图像上绘制篮球（带运动模糊和光照）。

        Args:
            img: RGB 图像 [H, W, 3]
            state: 当前球状态
            prev_state: 上一帧状态（用于运动模糊）

        Returns:
            绘制后的图像（原地修改）
        """
        proj = self.project_sphere(state)
        if proj is None:
            return img

        cx, cy, radius = proj
        h, w = img.shape[:2]

        # 边界检查
        if cx + radius < 0 or cx - radius > w or cy + radius < 0 or cy - radius > h:
            return img

        # 运动模糊：根据位移量决定模糊强度
        if self.config.enable_motion_blur and prev_state:
            prev_proj = self.project_sphere(prev_state)
            if prev_proj:
                dx = cx - prev_proj[0]
                dy = cy - prev_proj[1]
                motion_len = math.sqrt(dx**2 + dy**2)
                if motion_len > 5:  # 只有快速运动才加模糊
                    angle = math.degrees(math.atan2(dy, dx))
                    kernel_size = min(int(motion_len / 3), 15)
                    if kernel_size > 1:
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                        # 在球区域局部应用模糊
                        y0 = max(0, cy - radius - kernel_size)
                        y1 = min(h, cy + radius + kernel_size)
                        x0 = max(0, cx - radius - kernel_size)
                        x1 = min(w, cx + radius + kernel_size)
                        roi = img[y0:y1, x0:x1]
                        blurred = cv2.blur(roi, (kernel_size, kernel_size))
                        # 只混合球区域
                        mask = np.zeros((y1-y0, x1-x0), dtype=np.uint8)
                        cv2.circle(mask, (cx-x0, cy-y0), radius+kernel_size, 255, -1)
                        mask = mask.astype(np.float32) / 255.0
                        for c in range(3):
                            roi[:, :, c] = (blurred[:, :, c] * mask + roi[:, :, c] * (1 - mask)).astype(np.uint8)
                        img[y0:y1, x0:x1] = roi

        # 球体颜色（基础色 + 光照扰动）
        base_color = self.config.ball_color_base
        if self.config.enable_lighting:
            # 随机光照变化 ±20%
            factor = random.uniform(0.8, 1.2)
            color = tuple(min(255, int(c * factor)) for c in base_color)
        else:
            color = base_color

        # 绘制球体（渐变模拟3D效果）
        for r_offset in range(radius, 0, -2):
            shade = 1.0 - (radius - r_offset) / radius * 0.3  # 边缘暗化
            shaded_color = tuple(int(c * shade) for c in color)
            cv2.circle(img, (cx, cy), r_offset, shaded_color, -1, cv2.LINE_AA)

        # 高光（模拟反光）
        highlight_offset = int(radius * 0.3)
        cv2.circle(img, (cx - highlight_offset, cy - highlight_offset),
                   max(radius // 3, 2), (255, 255, 255), -1, cv2.LINE_AA)

        # 篮球纹路（简化：两条黑色弧线）
        cv2.ellipse(img, (cx, cy), (radius, radius//2), 0, 0, 180, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), (radius//2, radius), 90, 0, 180, (20, 20, 20), 2, cv2.LINE_AA)

        return img

    def add_random_occlusion(self, img: np.ndarray, state: BallState,
                             players: list[tuple[float, float, float, float]]) -> np.ndarray:
        """模拟球员遮挡：用半透明矩形覆盖球体。

        Args:
            img: RGB 图像
            state: 球状态
            players: 球员 bounding boxes [(x0, y0, x1, y1), ...]

        Returns:
            遮挡后的图像
        """
        if not self.config.enable_occlusion or random.random() > self.config.occlusion_prob:
            return img

        proj = self.project_sphere(state)
        if proj is None:
            return img

        cx, cy, radius = proj
        h, w = img.shape[:2]

        # 检查是否有球员框与球重叠
        for px0, py0, px1, py1 in players:
            # 计算球与球员框的 IoU
            bx0 = max(0, cx - radius)
            by0 = max(0, cy - radius)
            bx1 = min(w, cx + radius)
            by1 = min(h, cy + radius)

            ix0 = max(bx0, px0)
            iy0 = max(by0, py0)
            ix1 = min(bx1, px1)
            iy1 = min(by1, py1)

            if ix1 > ix0 and iy1 > iy0:  # 有重叠
                # 用球员颜色部分遮挡球
                overlap_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.rectangle(overlap_mask, (int(ix0), int(iy0)), (int(ix1), int(iy1)), 255, -1)
                # 球区域
                ball_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(ball_mask, (cx, cy), radius, 255, -1)
                # 交集
                combined = cv2.bitwise_and(overlap_mask, ball_mask)
                if combined.any():
                    # 用球员球衣颜色（灰色）覆盖
                    img[combined > 0] = (150, 150, 150)

        return img


def generate_trajectory(H_inv: np.ndarray, num_frames: int = 60,
                        config: BallRenderConfig | None = None) -> list[dict]:
    """生成一条完整的球轨迹并渲染到背景图上。

    Args:
        H_inv: 世界→像素单应矩阵
        num_frames: 轨迹长度（帧数）
        config: 渲染配置

    Returns:
        每帧数据列表：[{"img": np.ndarray, "bbox": [x0,y0,x1,y1], "visible": bool}, ...]
    """
    config = config or BallRenderConfig()
    physics = BallPhysics()
    renderer = BallRenderer(H_inv, config)

    # 随机初始状态
    from bsk.court import FIBA
    x0 = random.uniform(-FIBA.half_length/2, FIBA.half_length/2)
    y0 = random.uniform(-FIBA.half_width/2, FIBA.half_width/2)
    z0 = random.uniform(1.5, 3.0)
    vx0 = random.uniform(-8, 8)
    vy0 = random.uniform(-4, 4)
    vz0 = random.uniform(-2, 2)
    physics.reset(x=x0, y=y0, z=z0, vx=vx0, vy=vy0, vz=vz0)

    # 创建空白背景（实际使用时替换为真实视频帧）
    bg = np.zeros((config.img_height, config.img_width, 3), dtype=np.uint8)

    frames = []
    prev_state = None
    for _ in range(num_frames):
        state = physics.step(dt=1/30)
        proj = renderer.project_sphere(state)

        if proj:
            cx, cy, radius = proj
            x0 = max(0, cx - radius)
            y0 = max(0, cy - radius)
            x1 = min(config.img_width, cx + radius)
            y1 = min(config.img_height, cy + radius)
            visible = (x1 > x0 and y1 > y0 and radius >= 3)
        else:
            x0, y0, x1, y1 = 0, 0, 0, 0
            visible = False

        # 渲染球到背景
        img_copy = bg.copy()
        renderer.render_ball(img_copy, state, prev_state)

        frames.append({
            "img": img_copy,
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
            "visible": visible,
            "world_pos": (state.x, state.y, state.z),
        })
        prev_state = state

    return frames


def export_yolo_trajectory(output_dir: str | Path,
                           H_inv_list: list[np.ndarray] | None = None,
                           config_list: list[BallRenderConfig] | None = None,
                           num_trajectories: int = 100,
                           frames_per_traj: int = 60) -> Path:
    """批量生成 YOLO 格式训练数据集（支持多种摄像机视角）。

    目录结构：
    output_dir/
      ├── images/
      │   ├── train/
      │   │   ├── ball_000001.jpg
      │   │   └── ...
      │   └── val/
      ├── labels/
      │   ├── train/
      │   │   ├── ball_000001.txt  # YOLO: class cx cy w h (normalized)
      │   │   └── ...
      │   └── val/
      └── dataset.yaml

    Args:
        output_dir: 输出目录
        H_inv_list: 单应矩阵列表（None 则用单位矩阵）
        config_list: 渲染配置列表（None 则用默认配置）
        num_trajectories: 轨迹数量
        frames_per_traj: 每条轨迹帧数

    Returns:
        输出目录路径
    """
    import yaml

    out = Path(output_dir)
    img_train = out / "images" / "train"
    img_val = out / "images" / "val"
    lbl_train = out / "labels" / "train"
    lbl_val = out / "labels" / "val"

    for d in [img_train, img_val, lbl_train, lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    if H_inv_list is None:
        H_inv_list = [np.eye(3)] * num_trajectories
    if config_list is None:
        config_list = [BallRenderConfig()] * num_trajectories

    idx = 0
    for t in range(num_trajectories):
        H_inv = H_inv_list[t % len(H_inv_list)]
        config = config_list[t % len(config_list)]
        frames = generate_trajectory(H_inv, frames_per_traj, config)
        for i, frame_data in enumerate(frames):
            if not frame_data["visible"]:
                continue

            # 决定 train/val  split (80/20)
            is_val = (t % 5 == 0)
            img_dir = img_val if is_val else img_train
            lbl_dir = lbl_val if is_val else lbl_train

            img_name = f"ball_{idx:06d}.jpg"
            lbl_name = f"ball_{idx:06d}.txt"

            # 保存图像
            img_bgr = cv2.cvtColor(frame_data["img"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_dir / img_name), img_bgr)

            # 保存 YOLO 标注 (class cx cy w h, normalized)
            h, w = config.img_height, config.img_width
            x0, y0, x1, y1 = frame_data["bbox"]
            cx = (x0 + x1) / 2 / w
            cy = (y0 + y1) / 2 / h
            bw = (x1 - x0) / w
            bh = (y1 - y0) / h

            with open(lbl_dir / lbl_name, "w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            idx += 1

    # 生成 dataset.yaml
    yaml_content = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["ball"],
    }
    with open(out / "dataset.yaml", "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"[synth_ball] Generated {idx} samples in {out}")
    return out


def export_yolo_dataset(output_dir: str | Path, num_trajectories: int = 100,
                        frames_per_traj: int = 60,
                        H_inv: np.ndarray | None = None,
                        config: BallRenderConfig | None = None) -> Path:
    """简化接口：单视角批量生成（向后兼容）。"""
    H_inv_list = [H_inv] if H_inv is not None else None
    config_list = [config] if config is not None else None
    return export_yolo_trajectory(
        output_dir=output_dir,
        H_inv_list=H_inv_list,
        config_list=config_list,
        num_trajectories=num_trajectories,
        frames_per_traj=frames_per_traj,
    )
    """批量生成 YOLO 格式训练数据集。

    目录结构：
    output_dir/
      ├── images/
      │   ├── train/
      │   │   ├── ball_000001.jpg
      │   │   └── ...
      │   └── val/
      ├── labels/
      │   ├── train/
      │   │   ├── ball_000001.txt  # YOLO: class cx cy w h (normalized)
      │   │   └── ...
      │   └── val/
      └── dataset.yaml

    Args:
        output_dir: 输出目录
        num_trajectories: 轨迹数量
        frames_per_traj: 每条轨迹帧数
        H_inv: 单应矩阵（None 则用单位矩阵模拟俯视视角）
        config: 渲染配置

    Returns:
        输出目录路径
    """
    from pathlib import Path
    import yaml

    out = Path(output_dir)
    img_train = out / "images" / "train"
    img_val = out / "images" / "val"
    lbl_train = out / "labels" / "train"
    lbl_val = out / "labels" / "val"

    for d in [img_train, img_val, lbl_train, lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    H_inv = H_inv if H_inv is not None else np.eye(3)
    config = config or BallRenderConfig()

    idx = 0
    for t in range(num_trajectories):
        frames = generate_trajectory(H_inv, frames_per_traj, config)
        for i, frame_data in enumerate(frames):
            if not frame_data["visible"]:
                continue

            # 决定 train/val  split (80/20)
            is_val = (t % 5 == 0)
            img_dir = img_val if is_val else img_train
            lbl_dir = lbl_val if is_val else lbl_train

            img_name = f"ball_{idx:06d}.jpg"
            lbl_name = f"ball_{idx:06d}.txt"

            # 保存图像
            img_bgr = cv2.cvtColor(frame_data["img"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_dir / img_name), img_bgr)

            # 保存 YOLO 标注 (class cx cy w h, normalized)
            h, w = config.img_height, config.img_width
            x0, y0, x1, y1 = frame_data["bbox"]
            cx = (x0 + x1) / 2 / w
            cy = (y0 + y1) / 2 / h
            bw = (x1 - x0) / w
            bh = (y1 - y0) / h

            with open(lbl_dir / lbl_name, "w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            idx += 1

    # 生成 dataset.yaml
    yaml_content = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["ball"],
    }
    with open(out / "dataset.yaml", "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"[synth_ball] Generated {idx} samples in {out}")
    return out


if __name__ == "__main__":
    # 测试：生成少量样本验证
    from bsk.court import FIBA

    # 模拟一个俯视视角的单应矩阵（简化测试用）
    # 实际使用时应从真实标定文件加载
    scale = 100  # 1米 = 100像素
    H_inv = np.array([
        [scale, 0, 960],   # x → u
        [0, -scale, 540],  # y → v (翻转y轴)
        [0, 0, 1]
    ], dtype=np.float64)

    config = BallRenderConfig(
        img_width=1280,
        img_height=720,
        enable_motion_blur=True,
        enable_occlusion=True,
        enable_lighting=True,
    )

    # 生成 5 条轨迹测试
    frames = generate_trajectory(H_inv, num_frames=30, config=config)
    print(f"Generated {len(frames)} frames")

    # 保存第一帧查看效果
    if frames:
        test_img = frames[0]["img"]
        cv2.imwrite("test_ball_synth.jpg", cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
        print(f"Saved test image: test_ball_synth.jpg (bbox={frames[0]['bbox']})")

    # 批量生成数据集（小规模测试）
    export_yolo_dataset("data/ball_synth", num_trajectories=10, frames_per_traj=30,
                        H_inv=H_inv, config=config)
