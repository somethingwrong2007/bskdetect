"""把 Roboflow 篮球检测数据过滤成单类球检测数据集。

优先按 data.yaml 的类别名查找 `ball` / `ball-in-basket`；旧社区数据集只提供
数字类别名时保留既有约定：0=Ball，1=Ball in Basket。可用命令行显式覆盖。

本脚本默认只保留球框，生成 `data/external/ball_real_yolo/`：
- images 用硬链接指向原始文件（同盘零拷贝，不删改原数据）
- labels 重新写：只留球框，类别号重映射为 0
- 丢弃没有任何球框的图（避免把只有球员/篮筐的图当正样本）
- data.yaml：nc=1，names=[ball]，绝对路径（本文件不入库，随 data/ 一起 gitignore）

`--include-basket-ball` 可以把篮筐里的球也并入球框（默认关）。

用法：
    python -m bsk.prep_ball_real --audit
    python -m bsk.prep_ball_real --src data/external/roboflow_nba_basketball_v4 \\
        --dst data/external/ball_nba_yolo --source-note "Roboflow NBA v4 (CC BY 4.0)"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

SPLITS = ("train", "valid", "test")
DEFAULT_SRC = Path("data/external/roboflow_basketball")
DEFAULT_DST = Path("data/external/ball_real_yolo")
BALL_CLASS = 0
BASKET_BALL_CLASS = 1


def resolve_classes(src: Path, ball_class: int | None = None,
                    basket_ball_class: int | None = None) -> tuple[int, int | None, dict]:
    import yaml

    metadata = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8")) or {}
    raw_names = metadata.get("names")
    if isinstance(raw_names, dict):
        names = {int(k): str(v) for k, v in raw_names.items()}
    elif isinstance(raw_names, list):
        names = {i: str(v) for i, v in enumerate(raw_names)}
    else:
        names = {}

    def find(candidates: set[str]) -> int | None:
        normalized = {name.strip().lower().replace("_", "-").replace(" ", "-")
                      for name in names.values()}
        if candidates & normalized:
            return next(i for i, name in names.items()
                        if name.strip().lower().replace("_", "-").replace(" ", "-") in candidates)
        return None

    numeric_names = bool(names) and all(name.isdigit() for name in names.values())
    resolved_ball = ball_class if ball_class is not None else find({"ball"})
    if resolved_ball is None and (not names or numeric_names):
        resolved_ball = BALL_CLASS
    if resolved_ball is None:
        raise ValueError(f"{src / 'data.yaml'} 中没有名为 ball 的类别，请用 --ball-class 指定")
    resolved_basket = basket_ball_class
    if resolved_basket is None:
        resolved_basket = find({"ball-in-basket", "in-basket"})
        if resolved_basket is None and numeric_names:
            resolved_basket = BASKET_BALL_CLASS if BASKET_BALL_CLASS in names else None
    return resolved_ball, resolved_basket, {"names": names, "roboflow": metadata.get("roboflow", {})}


def _read_boxes(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append([float(v) for v in parts])
    return rows


def audit(src: Path, ball_class: int | None = None,
          basket_ball_class: int | None = None) -> dict:
    """按 split 统计每类框数 / 出现图数 / 框尺寸，并报告类别映射。"""
    import numpy as np

    ball_class, basket_ball_class, metadata = resolve_classes(src, ball_class, basket_ball_class)
    names = metadata["names"]
    report: dict = {"class_mapping": {
        "ball": ball_class,
        "ball_in_basket": basket_ball_class,
        **metadata,
    }}
    for split in SPLITS:
        labels = sorted((src / split / "labels").glob("*.txt"))
        counts: Counter = Counter()
        imgs: Counter = Counter()
        wh: dict[int, list[tuple[float, float]]] = {}
        for p in labels:
            rows = _read_boxes(p)
            img = next((p.parent.parent / "images" / f"{p.stem}{ext}"
                        for ext in (".jpg", ".jpeg", ".png")
                        if (p.parent.parent / "images" / f"{p.stem}{ext}").exists()), None)
            width, height = 640, 640
            if img is not None:
                from PIL import Image
                with Image.open(img) as image:
                    width, height = image.size
            seen = set()
            for cls, _cx, _cy, w, h in rows:
                c = int(cls)
                counts[c] += 1
                seen.add(c)
                wh.setdefault(c, []).append((w * width, h * height))
            for c in seen:
                imgs[c] += 1
        split_report = {"images": len(labels), "classes": {}}
        for c in sorted(counts):
            arr = np.asarray(wh[c], dtype=np.float64)
            split_report["classes"][str(c)] = {
                "name": names.get(c),
                "boxes": int(counts[c]),
                "images": int(imgs[c]),
                "median_wh_px": [round(float(v), 1) for v in np.median(arr, axis=0)],
                "p95_wh_px": [round(float(v), 1) for v in np.percentile(arr, 95, axis=0)],
            }
        report[split] = split_report
    return report


def _link_or_copy(src: Path, dst: Path) -> str:
    """同卷优先硬链接（不占额外空间，原数据不动）；失败则复制。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def build(src: Path = DEFAULT_SRC, dst: Path = DEFAULT_DST,
          include_basket_ball: bool = False, ball_class: int | None = None,
          basket_ball_class: int | None = None, source_note: str | None = None) -> dict:
    ball_class, basket_ball_class, metadata = resolve_classes(src, ball_class, basket_ball_class)
    keep = {ball_class}
    if include_basket_ball:
        if basket_ball_class is None:
            raise ValueError("源数据没有 ball-in-basket 类别，不能使用 --include-basket-ball")
        keep.add(basket_ball_class)

    stats: dict = {"src": str(src.resolve()), "dst": str(dst.resolve()),
                   "keep_classes": sorted(keep),
                   "class_mapping": {"ball": ball_class, "ball_in_basket": basket_ball_class,
                                     **metadata},
                   "splits": {}}
    for split in SPLITS:
        img_dir, lbl_dir = src / split / "images", src / split / "labels"
        if not img_dir.exists():
            raise FileNotFoundError(f"找不到 {img_dir}")
        if dst.exists() and split == "train":
            # 重建时只清本脚本生成的三件套，避免误删外部目录
            for sub in ("images", "labels"):
                shutil.rmtree(dst / split / sub, ignore_errors=True)
        n_img = n_box = n_skip = 0
        way = Counter()
        for lbl in sorted(lbl_dir.glob("*.txt")):
            rows = [r for r in _read_boxes(lbl) if int(r[0]) in keep]
            if not rows:
                n_skip += 1
                continue
            img = None
            for ext in (".jpg", ".jpeg", ".png"):
                cand = img_dir / (lbl.stem + ext)
                if cand.exists():
                    img = cand
                    break
            if img is None:
                n_skip += 1
                continue
            way[_link_or_copy(img, dst / split / "images" / img.name)] += 1
            out_lbl = dst / split / "labels" / f"{lbl.stem}.txt"
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            out_lbl.write_text(
                "".join(f"0 {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n" for r in rows),
                encoding="utf-8",
            )
            n_img += 1
            n_box += len(rows)
        stats["splits"][split] = {
            "images": n_img, "boxes": n_box, "skipped_no_ball": n_skip,
            "method": dict(way),
        }

    yaml_path = dst / "data.yaml"
    note = source_note or f"Roboflow {src.name}"
    yaml_path.write_text(
        f"# 由 bsk/prep_ball_real.py 生成：{note}\n"
        f"# 单类：只保留原始 class {ball_class} = ball；类别号已重映射为 0。\n"
        f"path: {dst.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: ['ball']\n",
        encoding="utf-8",
    )
    stats["yaml"] = str(yaml_path.resolve())
    (dst / "build_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Roboflow 篮球 -> 单类球 YOLO 数据集")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--audit", action="store_true", help="只打印类别审计，不写数据")
    ap.add_argument("--include-basket-ball", action="store_true",
                    help="把 ball-in-basket 也并入球框（默认不混）")
    ap.add_argument("--ball-class", type=int, default=None,
                    help="覆盖球类别 ID；默认按 data.yaml 类别名查找")
    ap.add_argument("--basket-ball-class", type=int, default=None,
                    help="覆盖篮筐内球类别 ID")
    ap.add_argument("--source-note", default=None, help="写入生成 data.yaml 的来源说明")
    args = ap.parse_args(argv)

    if args.audit:
        print(json.dumps(audit(args.src, args.ball_class, args.basket_ball_class),
                         ensure_ascii=False, indent=2))
        return 0

    stats = build(args.src, args.dst, args.include_basket_ball, args.ball_class,
                  args.basket_ball_class, args.source_note)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())