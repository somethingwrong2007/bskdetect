"""合并多个篮球检测数据集为统一训练集。

目录结构标准化：
merged_dataset/
  ├── images/
  │   ├── train/
  │   └── val/
  ├── labels/
  │   ├── train/
  │   └── val/
  └── dataset.yaml
"""

from __future__ import annotations

import shutil
from pathlib import Path


def merge_datasets(
    output_dir: str | Path = "data/ball_merged",
    datasets: list[dict] | None = None,
):
    """合并多个 YOLO 格式数据集。

    Args:
        output_dir: 输出目录
        datasets: 数据集列表，每项包含：
            - path: 数据集根目录
            - split_ratio: train/val 划分比例（默认 0.8）
            - prefix: 文件名前缀（避免冲突）
    """
    if datasets is None:
        # 默认合并已知数据集
        datasets = [
            {"path": "data/ball_synth_large", "prefix": "synth"},
            # TODO: 添加真实数据集路径（下载后取消注释）
            # {"path": "data/csdn_basketball", "prefix": "csdn"},
            # {"path": "data/large_basketball", "prefix": "large"},
        ]

    out = Path(output_dir)
    img_train = out / "images" / "train"
    img_val = out / "images" / "val"
    lbl_train = out / "labels" / "train"
    lbl_val = out / "labels" / "val"

    for d in [img_train, img_val, lbl_train, lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    idx = 0
    total_images = 0
    total_labels = 0

    for ds in datasets:
        ds_path = Path(ds["path"])
        prefix = ds.get("prefix", f"ds{idx}")
        split_ratio = ds.get("split_ratio", 0.8)

        if not ds_path.exists():
            print(f"[merge] WARNING: Dataset not found: {ds_path}, skipping")
            continue

        print(f"[merge] Processing {ds_path} (prefix={prefix})...")

        # 查找 images 和 labels 目录
        img_src = ds_path / "images"
        lbl_src = ds_path / "labels"

        if not img_src.exists() or not lbl_src.exists():
            # 尝试其他常见结构
            img_src_alt = ds_path / "JPEGImages"
            lbl_src_alt = ds_path / "Annotations"
            if img_src_alt.exists() and lbl_src_alt.exists():
                img_src = img_src_alt
                lbl_src = lbl_src_alt
            else:
                print(f"[merge] ERROR: Cannot find images/labels in {ds_path}")
                continue

        # 处理 train 和 val 子目录
        for split in ["train", "val"]:
            img_split_src = img_src / split
            lbl_split_src = lbl_src / split

            if not img_split_src.exists():
                # 如果没有分 train/val，则全部视为 train
                if split == "train":
                    img_split_src = img_src
                    lbl_split_src = lbl_src
                else:
                    continue

            if not lbl_split_src.exists():
                continue

            # 复制文件并重命名
            img_files = sorted(img_split_src.glob("*.jpg")) + sorted(img_split_src.glob("*.png"))
            lbl_files_map = {f.stem: f for f in lbl_split_src.glob("*.txt")}

            for img_file in img_files:
                stem = img_file.stem
                is_val = (split == "val")

                # 生成新文件名
                new_name = f"{prefix}_{stem}_{idx:06d}{img_file.suffix}"
                lbl_name = f"{prefix}_{stem}_{idx:06d}.txt"

                # 目标路径
                img_dst = img_val / new_name if is_val else img_train / new_name
                lbl_dst = lbl_val / lbl_name if is_val else lbl_train / lbl_name

                # 复制图像
                shutil.copy2(img_file, img_dst)
                total_images += 1

                # 复制标注（如果存在）
                if stem in lbl_files_map:
                    shutil.copy2(lbl_files_map[stem], lbl_dst)
                    total_labels += 1

                idx += 1

        print(f"[merge]   Copied {total_images} images, {total_labels} labels")

    # 生成 dataset.yaml
    yaml_content = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["ball"],
    }

    import yaml
    with open(out / "dataset.yaml", "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"\n[merge] Merged dataset summary:")
    print(f"  Total images: {total_images}")
    print(f"  Total labels: {total_labels}")
    print(f"  Output dir: {out}")
    print(f"  Train/Val split: ~80/20")

    return out


if __name__ == "__main__":
    merge_datasets()
