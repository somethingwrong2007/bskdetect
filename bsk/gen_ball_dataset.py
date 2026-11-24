"""批量生成大规模球检测训练数据。

目标：10,000+ 高质量标注样本，覆盖各种场景。
"""

from __future__ import annotations

import numpy as np
from bsk.court import FIBA
from bsk.synth_ball import BallRenderConfig, export_yolo_trajectory


def generate_diverse_trajectories(output_dir: str = "data/ball_synth_large",
                                   num_trajectories: int = 200,
                                   frames_per_traj: int = 60):
    """生成多样化轨迹数据集。

    变化维度：
    1. 摄像机角度（不同 H_inv）
    2. 光照条件（白天/夜晚/室内）
    3. 球的颜色（标准橙色/磨损/反光）
    4. 运动速度（慢速传球/快速投篮）
    5. 遮挡程度（无遮挡/部分遮挡/严重遮挡）
    """
    from pathlib import Path
    import random

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    configs = []

    # 场景 1: 标准俯视（模拟主摄像机）
    for i in range(50):
        scale = random.uniform(80, 120)  # 像素/米
        offset_u = random.uniform(-100, 100)
        offset_v = random.uniform(-50, 50)
        H_inv = np.array([
            [scale, 0, 960 + offset_u],
            [0, -scale, 540 + offset_v],
            [0, 0, 1]
        ], dtype=np.float64)

        config = BallRenderConfig(
            img_width=1280,
            img_height=720,
            ball_color_base=(255, random.randint(120, 160), 0),  # 橙色变化
            enable_motion_blur=random.random() > 0.3,
            enable_occlusion=random.random() > 0.4,
            enable_lighting=True,
        )
        configs.append((H_inv, config))

    # 场景 2: 侧视角度（模拟边线摄像机）
    for i in range(50):
        # 透视变换：近大远小
        scale_x = random.uniform(60, 100)
        scale_y = random.uniform(40, 80)
        skew = random.uniform(-0.2, 0.2)
        H_inv = np.array([
            [scale_x, skew, 640],
            [0, -scale_y, 360],
            [0, 0, 1]
        ], dtype=np.float64)

        config = BallRenderConfig(
            img_width=1280,
            img_height=720,
            ball_color_base=(random.randint(200, 255), random.randint(100, 150), 0),
            enable_motion_blur=True,
            enable_occlusion=True,
            enable_lighting=True,
        )
        configs.append((H_inv, config))

    # 场景 3: 低角度（模拟底线摄像机）
    for i in range(50):
        scale = random.uniform(100, 150)
        H_inv = np.array([
            [scale, 0, 640],
            [0, -scale * 0.6, 600],  # y轴压缩（透视）
            [0, 0, 1]
        ], dtype=np.float64)

        config = BallRenderConfig(
            img_width=1280,
            img_height=720,
            ball_color_base=(255, 140, random.randint(0, 50)),  # 偏红
            enable_motion_blur=True,
            enable_occlusion=False,  # 底线视角遮挡少
            enable_lighting=True,
        )
        configs.append((H_inv, config))

    # 场景 4: 极端透视（模拟观众席手机拍摄）
    for i in range(50):
        scale = random.uniform(40, 80)
        H_inv = np.array([
            [scale, random.uniform(-0.3, 0.3), 640],
            [random.uniform(-0.1, 0.1), -scale * 0.5, 400],
            [0, 0, 1]
        ], dtype=np.float64)

        config = BallRenderConfig(
            img_width=1280,
            img_height=720,
            ball_color_base=(random.randint(180, 255), random.randint(80, 140), 0),
            enable_motion_blur=False,  # 手机视频通常模糊
            enable_occlusion=True,
            enable_lighting=False,  # 自动曝光
        )
        configs.append((H_inv, config))

    print(f"[gen_ball_dataset] Generated {len(configs)} diverse configurations")

    # 批量导出
    all_H = [c[0] for c in configs]
    all_cfg = [c[1] for c in configs]

    export_yolo_trajectory(
        output_dir=output_dir,
        H_inv_list=all_H,
        config_list=all_cfg,
        num_trajectories=num_trajectories,
        frames_per_traj=frames_per_traj,
    )


if __name__ == "__main__":
    generate_diverse_trajectories(
        output_dir="data/ball_synth_large",
        num_trajectories=200,
        frames_per_traj=60,
    )
    print("[gen_ball_dataset] Done! Dataset ready for training.")
