"""构建 RF-DETR-S 职业比赛 10 类联合数据集。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from bsk.build_ball_pro import _fiba_groups, _find_image, _write_image

DEFAULT_OUT = Path("data/rfdetr_pro_v1")
CLASS_NAMES = [
    "ball", "ball-in-basket", "number", "player", "player-in-possession",
    "player-jump-shot", "player-layup-dunk", "player-shot-block", "referee", "rim",
]
SOURCES = (
    {"name": "nba_official_10class", "src": Path("data/external/nba_official_10class"),
     "square_to_16x9": True, "mapping": None},
    {"name": "fiba_u18_ball", "src": Path("data/external/fiba_u18_ball"),
     "square_to_16x9": False, "mapping": {0: 0}, "holdout_from_train": 0.10},
    {"name": "nba_ball_2000", "src": Path("data/external/nba_ball_2000"),
     "square_to_16x9": True, "mapping": {0: 0}},
    {"name": "nba_game_720p_ball", "src": Path("data/external/nba_game_720p_ball"),
     "square_to_16x9": True, "mapping": {0: 0}},
    {"name": "nba_v2_mixed", "src": Path("data/external/nba_v2_mixed"),
     "square_to_16x9": False, "mapping": {0: 0, 2: 3}},
)
SPLITS = ("train", "valid", "test")


def _hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build(out: Path = DEFAULT_OUT, seed: int = 0) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{out}")
    report = {"out": str(out.resolve()), "sources": {}, "splits": {}}
    seen: dict[str, tuple[str, str]] = {}
    split_counts: dict[str, Counter] = {split: Counter() for split in SPLITS}

    for source in SOURCES:
        src_root: Path = source["src"]
        metadata = yaml.safe_load((src_root / "data.yaml").read_text(encoding="utf-8")) or {}
        source_names = metadata.get("names", [])
        mapping = source["mapping"]
        if mapping is None:
            if len(source_names) != len(CLASS_NAMES):
                raise ValueError(f"官方 10 类数据类别不匹配: {src_root}")
            mapping = {index: index for index in range(len(CLASS_NAMES))}
        else:
            for source_id in mapping:
                if source_id >= len(source_names):
                    raise ValueError(f"类别 ID 越界: {src_root} class={source_id}")

        train_images = src_root / "train" / "images"
        train_stems = [p.stem for p in train_images.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png")] if train_images.exists() else []
        holdout = _fiba_groups(src_root, train_stems, source["holdout_from_train"], seed) \
            if source.get("holdout_from_train") else set()
        counts: Counter = Counter()
        class_counts: Counter = Counter()

        for split in ("test", "valid", "train"):
            images, labels = src_root / split / "images", src_root / split / "labels"
            if not images.exists() or not labels.exists():
                continue
            for label in sorted(labels.glob("*.txt")):
                stem = label.stem
                rows = []
                for line in label.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    source_id = int(float(parts[0]))
                    if source_id in mapping:
                        rows.append([mapping[source_id], *map(float, parts[1:])])
                if not rows:
                    continue
                out_split = "test" if split == "train" and stem.split("_f", 1)[0] in holdout else split
                img = _find_image(images, stem)
                if img is None:
                    continue
                digest = _hash(img)
                if digest in seen:
                    counts[f"{out_split}:duplicate"] += 1
                    continue
                seen[digest] = (source["name"], out_split)
                name = f"{source['name']}__{stem}"
                dst_img = out / out_split / "images" / f"{name}.jpg"
                dst_lbl = out / out_split / "labels" / f"{name}.txt"
                method = _write_image(img, dst_img, source["square_to_16x9"])
                dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                dst_lbl.write_text(
                    "".join(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
                            for class_id, cx, cy, bw, bh in rows), encoding="utf-8")
                counts[f"{out_split}:{method}"] += 1
                split_counts[out_split][source["name"]] += 1
                for row in rows:
                    class_counts[CLASS_NAMES[int(row[0])]] += 1

        report["sources"][source["name"]] = {
            "league": "FIBA" if source["name"].startswith("fiba") else "NBA",
            "counts": dict(counts), "heldout_groups": len(holdout),
            "boxes": dict(class_counts),
        }

    for split in SPLITS:
        report["splits"][split] = {
            "images": sum(split_counts[split].values()),
            "by_source": dict(split_counts[split]),
        }
    (out / "data.yaml").write_text(
        "# Generated by bsk.build_nba_detector from professional basketball datasets\n"
        f"path: {out.resolve().as_posix()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(CLASS_NAMES)}\nnames: {json.dumps(CLASS_NAMES)}\n", encoding="utf-8")
    (out / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.out, args.seed), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
