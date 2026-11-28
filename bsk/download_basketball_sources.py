"""下载并登记用于球、球员和球衣号码训练的职业篮球数据源。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bsk.download_ball_real import download

SOURCES = (
    {
        "name": "nba_official_10class",
        "workspace": "roboflow-jvuqo",
        "project": "basketball-player-detection-3-ycjdo",
        "version": 4,
        "format": "yolov8",
        "license": "CC BY 4.0",
        "roles": ["ball", "player", "number"],
        "note": "Roboflow 官方 NBA 教程数据，10 类转播标注",
    },
    {
        "name": "fiba_u18_ball",
        "workspace": "fibau18dataset",
        "project": "fiba-basketball",
        "version": 9,
        "format": "yolov8",
        "license": "CC BY 4.0",
        "roles": ["ball"],
        "note": "FIBA U18 官方比赛用球数据，单类 ball",
    },
    {
        "name": "fiba_3x3_u23_v1",
        "workspace": "3x3-basketball-player-detection",
        "project": "u23-world-cup-fiba-3x3-basketball",
        "version": 1,
        "format": "yolov8",
        "license": "CC BY 4.0",
        "roles": ["ball", "player"],
        "note": "FIBA U23 3x3 世界杯，ball/player",
    },
    {
        "name": "fiba_3x3_u23_v2",
        "workspace": "3x3-basketball-player-detection",
        "project": "u23-world-cup-fiba-3x3-basketball-2",
        "version": 2,
        "format": "yolov8",
        "license": "CC BY 4.0",
        "roles": ["ball", "player", "referee"],
        "note": "FIBA U23 3x3 世界杯，ball/player/referee",
    },
    {
        "name": "jersey_numbers_ocr",
        "workspace": "roboflow-jvuqo",
        "project": "basketball-jersey-numbers-ocr",
        "version": 7,
        "format": "jsonl",
        "license": "CC BY 4.0",
        "roles": ["jersey"],
        "note": "Roboflow 官方 2025 NBA 季后赛号码裁片，text-image-pairs",
    },
    {
        "name": "nba_ball_2000",
        "workspace": "basketball-llmyu",
        "project": "nba-ball",
        "version": 1,
        "format": "yolov8",
        "license": "MIT",
        "roles": ["ball"],
        "note": "NBA 转播球检测，单类 ball",
    },
    {
        "name": "nba_game_720p_ball",
        "workspace": "project-pakkl",
        "project": "basketball-in-nba-game-720p",
        "version": 1,
        "format": "yolov8",
        "license": "CC BY 4.0",
        "roles": ["ball"],
        "note": "NBA 720p 比赛画面，单类 basketball",
    },
    {
        "name": "nba_v2_mixed",
        "workspace": "nbashorts",
        "project": "nba-v2",
        "version": 1,
        "format": "yolov8",
        "license": "CC BY 4.0",
        "roles": ["ball", "player"],
        "note": "NBA 转播 ball/player/hoop 三类",
    },
    {
        "name": "nba_players_tirocinio",
        "workspace": "tirocinio-mxxnf",
        "project": "nba-players",
        "version": 4,
        "format": "yolov8",
        "license": "CC BY 4.0",
        "roles": ["player"],
        "note": "NBA 球员框数据；类别名为数字，下载后需复核语义",
    },
    {
        "name": "nba_players_betatracker",
        "workspace": "betatracker",
        "project": "nba-players-ronfv",
        "version": 4,
        "format": "yolov8",
        "license": "CC BY 4.0",
        "roles": ["player"],
        "note": "NBA Players/Referees 两类，带离线增强",
    },
)


def fetch(root: Path = Path("data/external"), only: set[str] | None = None) -> list[dict]:
    results = []
    for source in SOURCES:
        if only and source["name"] not in only:
            continue
        out = root / source["name"]
        result = download(out, source["workspace"], source["project"],
                          source["version"], source["format"])
        record = {**source,
                  "url": (f"https://universe.roboflow.com/{source['workspace']}/"
                          f"{source['project']}/dataset/{source['version']}"),
                  "path": str(out.resolve()), "status": result["status"]}
        (out / "source.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(record)
        print(json.dumps(record, ensure_ascii=False))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/external"))
    parser.add_argument("--only", nargs="*", default=None,
                        help="只下载指定 name，默认全部")
    args = parser.parse_args(argv)
    records = fetch(args.root, set(args.only) if args.only else None)
    print(json.dumps({"sources": records}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())