"""合成训练数据生成器：使用 2D 模板 + 3D 渲染生成球场线分割数据集。

数据流程：
1. 2D 模板（court_template.py）：像素级精确的俯视图 + 6 类掩码
2. 3D 渲染（court_3d.py）：从各种相机角度渲染透视图像
3. 数据增强：噪声、亮度、模糊等

支持 FIBA 和 NBA 两种球场。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from bsk.court import FIBA, NBA, Court
from bsk.court_3d import (
    CourtMesh3D,
    Court3DRenderer,
    generate_3d_sample,
    random_broadcast_camera,
    colorize_masks,
    NUM_CLASSES,
    CLASS_NAMES,
)
from bsk.calib_seg_net import INPUT_SIZE


def generate_dataset(
    num_samples: int,
    output_dir: str | Path,
    courts: list[Court] | None = None,
    image_size: int = INPUT_SIZE,
    seed: int | None = None,
) -> None:
    """批量生成合成训练数据集。

    Args:
        num_samples: 总样本数（FIBA/NBA 各半）
        output_dir: 输出目录
        courts: 球场列表，默认 [FIBA, NBA]
        image_size: 输出图像尺寸
        seed: 随机种子（可复现）
    """
    if courts is None:
        courts = [FIBA, NBA]

    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # 为每个球场创建渲染器
    renderers = {court.model: Court3DRenderer(CourtMesh3D.build(court)) for court in courts}

    # 每个球场的样本数
    samples_per_court = num_samples // len(courts)
    remainder = num_samples % len(courts)

    meta = []
    sample_idx = 0

    for court in courts:
        renderer = renderers[court.model]
        n_samples = samples_per_court + (1 if remainder > 0 else 0)
        remainder = max(0, remainder - 1)

        print(f"Generating {n_samples} samples for {court.model} court...")

        for i in range(n_samples):
            # 生成 3D 样本
            img, masks, cam = generate_3d_sample(renderer, image_size, rng)

            # 保存图像
            img_path = output_dir / "images" / f"{sample_idx:06d}.png"
            msk_path = output_dir / "masks" / f"{sample_idx:06d}.png"

            Image.fromarray(img).save(img_path)
            # 保存为单通道类别标签
            seg_label = np.argmax(masks, axis=0).astype(np.uint8)
            Image.fromarray(seg_label).save(msk_path)

            meta.append({
                "idx": sample_idx,
                "court_model": court.model,
                "camera_pos": list(cam["pos"]),
                "camera_look_at": list(cam["look_at"]),
                "focal_length": cam["focal_length"],
            })

            sample_idx += 1

            if sample_idx % 500 == 0:
                print(f"  Generated {sample_idx}/{num_samples}")

    # 保存元数据
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDataset saved to {output_dir}")
    print(f"Total samples: {sample_idx}")
    print(f"  FIBA: {sum(1 for m in meta if m['court_model'] == 'FIBA')}")
    print(f"  NBA:  {sum(1 for m in meta if m['court_model'] == 'NBA')}")


def generate_test_samples(
    output_dir: str | Path,
    courts: list[Court] | None = None,
    image_size: int = INPUT_SIZE,
) -> None:
    """生成少量测试样本用于可视化验证。"""
    if courts is None:
        courts = [FIBA, NBA]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for court in courts:
        mesh = CourtMesh3D.build(court)
        renderer = Court3DRenderer(mesh)

        # 生成 3 个固定视角 + 1 个随机视角
        test_cams = [
            {
                "pos": (18.0, 14.0, 14.0),
                "look_at": (0.0, 0.0, 0.0),
                "up": (0, 0, 1),
                "focal_length": 900.0,
                "image_size": (image_size, image_size),
            },
            {
                "pos": (-16.0, -10.0, 8.0),
                "look_at": (2.0, 1.0, 0.0),
                "up": (0, 0, 1),
                "focal_length": 1100.0,
                "image_size": (image_size, image_size),
            },
            {
                "pos": (2.0, 0.5, 22.0),
                "look_at": (0.0, 0.0, 0.0),
                "up": (0, 1, 0),
                "focal_length": 600.0,
                "image_size": (image_size, image_size),
            },
        ]

        for idx, cam in enumerate(test_cams):
            rgb, masks = renderer.render_sample(cam)

            img_path = output_dir / f"{court.model}_test_{idx}.png"
            mask_path = output_dir / f"{court.model}_test_{idx}_mask.png"

            Image.fromarray(rgb).save(img_path)
            mask_vis = colorize_masks(masks)
            Image.fromarray(mask_vis).save(mask_path)

            print(f"{court.model} test {idx}: {img_path}")

        # 随机视角
        rng = np.random.default_rng(42)
        img, masks, cam = generate_3d_sample(renderer, image_size, rng)
        img_path = output_dir / f"{court.model}_random.png"
        mask_path = output_dir / f"{court.model}_random_mask.png"

        Image.fromarray(img).save(img_path)
        mask_vis = colorize_masks(masks)
        Image.fromarray(mask_vis).save(mask_path)

        print(f"{court.model} random: {img_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic court line segmentation dataset")
    parser.add_argument("--num-samples", type=int, default=100, help="Total number of samples")
    parser.add_argument("--output", type=str, default="data/synthetic_seg", help="Output directory")
    parser.add_argument("--image-size", type=int, default=INPUT_SIZE, help="Image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-only", action="store_true", help="Only generate test samples")

    args = parser.parse_args()

    if args.test_only:
        generate_test_samples(args.output, image_size=args.image_size)
    else:
        generate_dataset(
            num_samples=args.num_samples,
            output_dir=args.output,
            image_size=args.image_size,
            seed=args.seed,
        )
