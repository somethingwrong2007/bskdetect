"""构建职业比赛单类球数据集，FIBA 与 NBA 保留独立 test 统计。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SOURCES = (
    {"name": "fiba_u18_ball", "league": "FIBA", "src": Path("data/external/fiba_u18_ball"),
     "ball_class": 0, "square_to_16x9": False, "holdout_from_train": 0.10},
    {"name": "nba_official_10class", "league": "NBA", "src": Path("data/external/nba_official_10class"),
     "ball_class": 0, "square_to_16x9": True},
    {"name": "nba_ball_2000", "league": "NBA", "src": Path("data/external/nba_ball_2000"),
     "ball_class": 0, "square_to_16x9": True},
    {"name": "nba_game_720p_ball", "league": "NBA", "src": Path("data/external/nba_game_720p_ball"),
     "ball_class": 0, "square_to_16x9": True},
    {"name": "nba_v2_mixed", "league": "NBA", "src": Path("data/external/nba_v2_mixed"),
     "ball_class": 0, "square_to_16x9": False},
)
SPLITS = ("train", "valid", "test")


def _find_image(images: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        p = images / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _box_rows(label: Path, ball_class: int) -> list[list[float]]:
    rows = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5 and int(float(parts[0])) == ball_class:
            rows.append([float(v) for v in parts[1:]])
    return rows


def _write_image(src_path: Path, dst: Path, square_to_16x9: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if square_to_16x9:
        with Image.open(src_path) as src:
            if src.width == src.height and src.width > 320:
                width = min(src.width, 1280)
                height = int(round(width * 9 / 16))
                resized = src.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
                resized.save(dst, quality=95)
                return "resized"
    if dst.exists():
        dst.unlink()
    try:
        os.link(src_path, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src_path, dst)
        return "copy"


def _fiba_groups(src: Path, train_stems: list[str], ratio: float, seed: int) -> set[str]:
    groups = sorted({stem.split("_f", 1)[0] for stem in train_stems})
    n_test = max(1, int(round(len(groups) * ratio)))
    picked = np.random.default_rng(seed).choice(groups, size=n_test, replace=False)
    return set(picked.tolist())


def build(out: Path = Path("data/ball_pro_v1"), seed: int = 0) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{out}")
    report = {"out": str(out.resolve()), "sources": {}, "splits": {}}
    split_source_counts: dict[str, Counter] = defaultdict(Counter)

    for source in SOURCES:
        src_root: Path = source["src"]
        train_images = src_root / "train" / "images"
        train_stems = [p.stem for p in train_images.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png")] if train_images.exists() else []
        holdout = _fiba_groups(src_root, train_stems, source["holdout_from_train"], seed) \
            if source.get("holdout_from_train") else set()
        counts = Counter()
        for split in SPLITS:
            images = src_root / split / "images"
            labels = src_root / split / "labels"
            if not images.exists() or not labels.exists():
                continue
            for label in sorted(labels.glob("*.txt")):
                stem = label.stem
                rows = _box_rows(label, source["ball_class"])
                if not rows:
                    continue
                out_split = split
                if split == "train" and stem.split("_f", 1)[0] in holdout:
                    out_split = "test"
                img = _find_image(images, stem)
                if img is None:
                    continue
                name = f"{source['name']}__{stem}"
                out_img = out / out_split / "images" / f"{name}.jpg"
                out_lbl = out / out_split / "labels" / f"{name}.txt"
                method = _write_image(img, out_img, source["square_to_16x9"])
                out_lbl.parent.mkdir(parents=True, exist_ok=True)
                out_lbl.write_text(
                    "".join(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
                            for cx, cy, bw, bh in rows), encoding="utf-8")
                counts[f"{out_split}:{method}"] += 1
                split_source_counts[out_split][source["name"]] += 1
        report["sources"][source["name"]] = {
            "league": source["league"], "counts": dict(counts),
            "heldout_groups": len(holdout),
        }

    for split in SPLITS:
        report["splits"][split] = {
            "images": sum(split_source_counts[split].values()),
            "by_source": dict(split_source_counts[split]),
        }
    (out / "data.yaml").write_text(
        "# Generated by bsk.build_ball_pro from CC BY 4.0 / MIT Roboflow sources\n"
        f"path: {out.resolve().as_posix()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n"
        "nc: 1\nnames: ['ball']\n", encoding="utf-8")
    (out / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/ball_pro_v1"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.out, args.seed), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())