"""合成训练数据生成器：通过随机透视变换生成球场图像。

思路：
1. 从标准球场模板（俯视图）开始
2. 随机采样相机参数（位置、朝向、焦距）
3. 应用透视变换生成训练图像
4. 同时变换关键点坐标作为标签

优点：
- 无需手动标注
- 可以生成大量多样化数据
- 关键点坐标精确已知
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Iterator

from bsk.court import FIBA, Court


def draw_court_template(court: Court = FIBA, pixels_per_meter: float = 50) -> np.ndarray:
    """绘制标准球场俯视图模板。

    Args:
        court: 球场几何参数
        pixels_per_meter: 像素/米 比例

    Returns:
        template: 球场图像 (H, W, 3)
        keypoints_m: 关键点的米制坐标 [K, 2]
    """
    # 计算图像尺寸（留出边1米边距）
    margin = 1.0 * pixels_per_meter
    W = int((court.length + 2) * pixels_per_meter)
    H = int((court.width + 2) * pixels_per_meter)

    # 创建白色背景
    template = np.ones((H, W, 3), dtype=np.uint8) * 255

    # 坐标转换：米制 -> 像素（原点在左上角，x 向右，y 向下）
    def m2px(x, y):
        px = int((x + court.half_length + 1) * pixels_per_meter)
        py = int((y + court.half_width + 1) * pixels_per_meter)
        return (px, py)

    # 绘制球场线（白色背景上画黑色线）
    color = (0, 0, 0)
    thick = 2

    # 外框
    p1 = m2px(-court.half_length, -court.half_width)
    p2 = m2px(court.half_length, -court.half_width)
    p3 = m2px(court.half_length, court.half_width)
    p4 = m2px(-court.half_length, court.half_width)
    cv2.polylines(template, [np.array([p1, p2, p3, p4])], True, color, thick)

    # 中线
    m1 = m2px(0, -court.half_width)
    m2 = m2px(0, court.half_width)
    cv2.line(template, m1, m2, color, thick)

    # 两个半场的三分线和罚球区
    for side_sign in (1, -1):
        bl = court.half_length * side_sign
        rim_x = bl - court.rim_offset_from_baseline * side_sign
        ft_x = bl - court.free_throw_dist * side_sign
        ft_hw = court.free_throw_width / 2

        # 罚球区
        fp1 = m2px(bl, -ft_hw)
        fp2 = m2px(ft_x, -ft_hw)
        fp3 = m2px(ft_x, ft_hw)
        fp4 = m2px(bl, ft_hw)
        cv2.polylines(template, [np.array([fp1, fp2, fp3, fp4])], True, color, thick)

        # 三分线（简化为折线）
        corner_y = court.half_width - court.three_pt_corner_dist
        tp1 = m2px(bl, -corner_y)
        tp2 = m2px(rim_x, -corner_y)
        tp3 = m2px(rim_x, corner_y)
        tp4 = m2px(bl, corner_y)
        cv2.polylines(template, [np.array([tp1, tp2, tp3, tp4])], False, color, thick)

    # 中圈
    center = m2px(0, 0)
    cv2.circle(template, center, int(1.80 * pixels_per_meter), color, thick)

    # 定义 8 个关键点（米制坐标）
    keypoints_m = np.array([
        # 4 个角点
        [-court.half_length, -court.half_width],
        [court.half_length, -court.half_width],
        [court.half_length, court.half_width],
        [-court.half_length, court.half_width],
        # 4 个罚球区外角
        [-court.half_length + court.free_throw_dist, -court.free_throw_width / 2],
        [-court.half_length + court.free_throw_dist, court.free_throw_width / 2],
        [court.half_length - court.free_throw_dist, -court.free_throw_width / 2],
        [court.half_length - court.free_throw_dist, court.free_throw_width / 2],
    ], dtype=np.float32)

    return template, keypoints_m


def random_camera_params() -> dict:
    """随机采样相机参数。"""
    return {
        # 相机位置（相对于球场中心，米）
        "x": np.random.uniform(-20, 20),
        "y": np.random.uniform(-15, 15),
        "z": np.random.uniform(8, 25),  # 高度
        # 相机朝向（欧拉角，弧度）
        "roll": np.random.uniform(-0.3, 0.3),
        "pitch": np.random.uniform(-1.2, -0.3),  # 向下看
        "yaw": np.random.uniform(-0.5, 0.5),
        # 焦距（像素）
        "f": np.random.uniform(500, 1500),
    }


def camera_params_to_homography(params: dict, image_size: tuple[int, int], court: Court = FIBA) -> np.ndarray:
    """将相机参数转换为单应矩阵（球场平面 -> 图像平面）。

    简化版本：假设球场在 Z=0 平面，计算 3x3 单应矩阵。
    """
    H_img, W_img = image_size

    # 相机内参
    K = np.array([
        [params["f"], 0, W_img / 2],
        [0, params["f"], H_img / 2],
        [0, 0, 1],
    ])

    # 相机外参（旋转 + 平移）
    # 简化：只用 pitch 和 yaw
    cp, sp = np.cos(params["pitch"]), np.sin(params["pitch"])
    cy, sy = np.cos(params["yaw"]), np.sin(params["yaw"])
    cr, sr = np.cos(params["roll"]), np.sin(params["roll"])

    # 旋转矩阵（ZYX 欧拉角）
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R = Rz @ Ry @ Rx

    # 平移向量
    t = np.array([params["x"], params["y"], params["z"]])

    # 构建 3x4 投影矩阵
    P = K @ np.hstack([R, t.reshape(3, 1)])

    # 提取单应矩阵（假设 Z=0 平面）
    H = P[:, [0, 1, 3]]  # 取 X, Y, t 列

    return H


def generate_synthetic_sample(
    template: np.ndarray,
    keypoints_m: np.ndarray,
    court: Court = FIBA,
    image_size: tuple[int, int] = (256, 256),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成一个合成训练样本。

    Args:
        template: 球场模板图像
        keypoints_m: 关键点米制坐标 [K, 2]
        court: 球场几何
        image_size: 输出图像尺寸

    Returns:
        image: 合成图像 [H, W, 3]
        keypoints_px: 关键点像素坐标 [K, 2]
        H: 单应矩阵 [3, 3]
    """
    H_img, W_img = image_size

    # 随机相机参数
    params = random_camera_params()
    H = camera_params_to_homography(params, image_size, court)

    # 变换模板图像
    image = cv2.warpPerspective(template, H, (W_img, H_img), flags=cv2.INTER_LINEAR)

    # 变换关键点
    keypoints_h = np.hstack([keypoints_m, np.ones((len(keypoints_m), 1))]).T  # [3, K]
    keypoints_proj = H @ keypoints_h  # [3, K]
    keypoints_proj = keypoints_proj[:2] / keypoints_proj[2:]  # [2, K]
    keypoints_px = keypoints_proj.T  # [K, 2]

    # 添加噪声和增强
    image = _augment_image(image)

    return image, keypoints_px, H


def _augment_image(image: np.ndarray) -> np.ndarray:
    """数据增强：颜色抖动、模糊、噪声。"""
    # 颜色抖动
    if np.random.rand() < 0.5:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] += np.random.uniform(-10, 10)  # 色相
        hsv[:, :, 1] *= np.random.uniform(0.8, 1.2)  # 饱和度
        hsv[:, :, 2] *= np.random.uniform(0.8, 1.2)  # 亮度
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 高斯模糊
    if np.random.rand() < 0.3:
        ksize = np.random.choice([3, 5])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    # 添加噪声
    if np.random.rand() < 0.3:
        noise = np.random.randn(*image.shape) * 10
        image = np.clip(image + noise, 0, 255).astype(np.uint8)

    return image


def generate_dataset(
    num_samples: int,
    output_dir: Path,
    court: Court = FIBA,
    image_size: tuple[int, int] = (256, 256),
) -> None:
    """批量生成合成训练数据集。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成球场模板
    template, keypoints_m = draw_court_template(court, pixels_per_meter=50)

    # 生成样本
    images = []
    keypoints_list = []
    homographies = []

    for i in range(num_samples):
        image, keypoints_px, H = generate_synthetic_sample(template, keypoints_m, court, image_size)
        images.append(image)
        keypoints_list.append(keypoints_px)
        homographies.append(H)

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_samples} samples")

    # 保存为 PyTorch 格式
    data = {
        "images": np.stack(images),  # [N, H, W, 3]
        "keypoints": np.stack(keypoints_list),  # [N, K, 2]
        "homographies": np.stack(homographies),  # [N, 3, 3]
    }

    np.save(output_dir / "synthetic_dataset.npy", data)
    print(f"Saved dataset to {output_dir / 'synthetic_dataset.npy'}")


if __name__ == "__main__":
    # 测试生成
    generate_dataset(1000, Path("data/synthetic"), num_samples=1000)
