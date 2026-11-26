"""下载 Roboflow 职业比赛篮球检测数据，不保存或输出 API Key。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_OUT = Path("data/external/roboflow_nba_basketball_v4")


def _api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if key or os.name != "nt":
        return key
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as registry_key:
            key, _ = winreg.QueryValueEx(registry_key, "ROBOFLOW_API_KEY")
    except FileNotFoundError:
        return ""
    return str(key).strip()


def download(out: Path = DEFAULT_OUT, workspace: str = "roboflow-jvuqo",
             project: str = "basketball-player-detection-3-ycjdo",
             version: int = 4, fmt: str = "yolov8") -> dict:
    api_key = _api_key()
    if not api_key:
        raise RuntimeError("未配置 ROBOFLOW_API_KEY（当前进程或 Windows 用户环境变量）")

    out = Path(out)
    if (out / "data.yaml").exists():
        return {"status": "already_downloaded", "path": str(out.resolve())}
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{out}")

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    dataset = rf.workspace(workspace).project(project).version(version).download(fmt, location=str(out.resolve()))
    if fmt == "jsonl":
        if not any(out.rglob("*.jsonl")):
            raise RuntimeError(f"下载不完整，没有找到 JSONL：{out}")
    else:
        missing = [str(out / split / sub) for split in ("train", "valid")
                   for sub in ("images", "labels") if not (out / split / sub).exists()]
        if missing:
            raise RuntimeError(f"下载不完整，缺少：{', '.join(missing)}")
    return {"status": "downloaded", "path": str(out.resolve()),
            "workspace": workspace, "project": project, "version": version,
            "format": fmt,
            "location": str(getattr(dataset, "location", "") or "")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workspace", default="roboflow-jvuqo")
    parser.add_argument("--project", default="basketball-player-detection-3-ycjdo")
    parser.add_argument("--version", type=int, default=4)
    parser.add_argument("--format", default="yolov8")
    args = parser.parse_args(argv)
    print(json.dumps(download(args.out, args.workspace, args.project, args.version, args.format),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
