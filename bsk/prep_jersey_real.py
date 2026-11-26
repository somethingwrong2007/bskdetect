"""把官方 NBA 球衣号码 JSONL 转成保留官方 split 的训练清单。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_SRC = Path("data/external/jersey_numbers_ocr")
DEFAULT_OUT = Path("data/jersey_pro_v1")
SPLITS = ("train", "valid", "test")


def build(src: Path = DEFAULT_SRC, out: Path = DEFAULT_OUT) -> dict:
    records = {}
    seen_images: dict[str, str] = {}
    for split in SPLITS:
        rows = []
        for line in (src / split / "annotations.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            label = str(rec.get("suffix", "")).strip()
            image = str(rec.get("image", "")).strip()
            if not image or not re.fullmatch(r"\d{1,2}", label):
                continue
            if image in seen_images:
                if seen_images[image] != split:
                    raise ValueError(f"官方 split 泄漏：{image} 同时出现在 {seen_images[image]} 和 {split}")
                continue
            seen_images[image] = split
            rows.append({"crop": f"{split}/{image}", "label": int(label), "split": split})
        records[split] = rows

    out.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        path = out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8")
    report = {
        "src": str(src.resolve()), "out": str(out.resolve()),
        "splits": {split: len(rows) for split, rows in records.items()},
        "labels": sorted({r["label"] for rows in records.values() for r in rows}),
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
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