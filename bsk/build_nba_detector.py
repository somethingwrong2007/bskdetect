"""把官方 NBA 10 类数据校正回 16:9，供 RF-DETR 训练球员与号码检测。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from bsk.build_ball_pro import _find_image, _write_image

DEFAULT_SRC = Path("data/external/nba_official_10class")
DEFAULT_OUT = Path("data/nba_10class_v1")
SPLITS = ("train", "valid", "test")


def build(src: Path = DEFAULT_SRC, out: Path = DEFAULT_OUT) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{out}")
    metadata = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8")) or {}
    names = metadata["names"]
    report = {"src": str(src.resolve()), "out": str(out.resolve()), "splits": {}}
    for split in SPLITS:
        images, labels = src / split / "images", src / split / "labels"
        counts: Counter = Counter()
        written = 0
        for label in sorted(labels.glob("*.txt")):
            img = _find_image(images, label.stem)
            if img is None:
                continue
            rows = [line.split() for line in label.read_text(encoding="utf-8").splitlines()
                    if len(line.split()) == 5]
            if not rows:
                continue
            dst_img = out / split / "images" / f"{label.stem}.jpg"
            dst_lbl = out / split / "labels" / f"{label.stem}.txt"
            _write_image(img, dst_img, square_to_16x9=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.write_text("".join(" ".join(row) + "\n" for row in rows), encoding="utf-8")
            for row in rows:
                counts[names[int(float(row[0]))]] += 1
            written += 1
        report["splits"][split] = {"images": written, "boxes": dict(counts)}
    (out / "data.yaml").write_text(
        "# Official Roboflow NBA 10-class data, aspect restored to 16:9\n"
        f"path: {out.resolve().as_posix()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(names)}\nnames: {json.dumps(names)}\n", encoding="utf-8")
    (out / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.src, args.out), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())