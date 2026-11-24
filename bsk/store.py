"""调试台的素材库与派生数据存取。

一台机器上会反复导入不同视频，所以所有产物按 vid 分目录放，避免多素材互相覆盖：
    data/derived/<vid>/pts.jsonl      逐帧 PTS 真值表（取帧定位用）
    data/derived/<vid>/shots.json     镜头切分与球场可见性门控
    data/derived/<vid>/frames.jsonl   俯视坐标对象（schema.FrameRecord）
    data/derived/<vid>/poses.jsonl    2D 骨架点（schema.Pose2D）
    data/derived/<vid>/segments.jsonl 分割轮廓（schema.Segment2D）

frames/poses/segments 三份分文件存放是刻意的：姿态和分割会反复重跑，俯视坐标是
标定后的稳定结论，混在一起重跑一次就把标定成果冲掉了。
"""

from __future__ import annotations

import dataclasses
import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bsk import schema
from bsk.seek import FramePicker, FrameTable

ROOT = Path("data/derived")
LIBRARY = ROOT / "library.json"


def slug(name: str) -> str:
    base = Path(name).stem.lower()
    base = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    return base[:40] or "video"


@dataclass
class Video:
    vid: str
    path: str
    name: str
    added_at: float = field(default_factory=time.time)
    status: str = "registered"  # registered / indexing / ready / error
    detail: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    nframes: int = 0
    duration_s: float = 0.0
    is_cfr: bool = True
    gop_mean: float = 0.0
    keys: int = 0
    codec: str = ""
    shots: int = 0
    court_shots: int = 0
    has_frames: bool = False
    has_poses: bool = False
    has_segments: bool = False

    @property
    def dir(self) -> Path:
        return ROOT / self.vid

    def manifest_path(self) -> Path:
        return self.dir / "pts.jsonl"

    def to_dict(self) -> dict:
        return asdict(self)


class Library:
    def __init__(self, root: Path = ROOT) -> None:
        self.root = root
        self.path = root / "library.json"
        self._lock = threading.Lock()
        self._cache: dict[str, FrameTable] = {}
        self._pickers: dict[str, FramePicker] = {}
        self._rows: dict[str, dict[str, dict[int, list]]] = {}
        self.videos: dict[str, Video] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        names = {f.name for f in dataclasses.fields(Video)}
        for row in raw.get("videos", []):
            self.videos[row["vid"]] = Video(**{k: v for k, v in row.items() if k in names})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"videos": [v.to_dict() for v in self.videos.values()]}
        with self._lock:
            self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    def add(self, path: str | Path) -> Video:
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        vid = slug(p.name)
        i = 2
        while vid in self.videos and Path(self.videos[vid].path) != p:
            vid = f"{slug(p.name)}-{i}"
            i += 1
        v = self.videos.get(vid) or Video(vid=vid, path=str(p), name=p.name)
        v.path = str(p)
        v.name = p.name
        v.status = "registered"
        v.dir.mkdir(parents=True, exist_ok=True)
        self.videos[vid] = v
        self._cache.pop(vid, None)
        self._pickers.pop(vid, None)
        self._rows.pop(vid, None)
        self.save()
        return v

    def get(self, vid: str) -> Video:
        try:
            return self.videos[vid]
        except KeyError as exc:
            raise KeyError(f"未登记的素材: {vid}") from exc

    def table(self, vid: str) -> FrameTable:
        v = self.get(vid)
        if vid not in self._cache:
            if not v.manifest_path().exists():
                raise FileNotFoundError(f"缺少 pts 真值表: {v.manifest_path()}")
            self._cache[vid] = FrameTable.load(v.manifest_path())
        return self._cache[vid]

    def picker(self, vid: str) -> FramePicker:
        if vid not in self._pickers:
            self._pickers[vid] = FramePicker(self.get(vid).path, self.table(vid))
        return self._pickers[vid]

    def _row_index(self, vid: str, kind: str, cls, key: str) -> dict[int, list]:
        store = self._rows.setdefault(vid, {})
        if kind not in store:
            bucket: dict[int, list] = {}
            f = self.get(vid).dir / key
            if f.exists():
                for row in schema.read_obj_jsonl(f, cls):
                    bucket.setdefault(row.frame, []).append(row)
            store[kind] = bucket
        return store[kind]

    def poses(self, vid: str) -> dict[int, list]:
        return self._row_index(vid, "poses", schema.Pose2D, "poses.jsonl")

    def segments(self, vid: str) -> dict[int, list]:
        return self._row_index(vid, "segments", schema.Segment2D, "segments.jsonl")

    def frame_records(self, vid: str) -> dict[int, schema.FrameRecord]:
        store = self._rows.setdefault(vid, {})
        if "frames" not in store:
            bucket: dict[int, schema.FrameRecord] = {}
            f = self.get(vid).dir / "frames.jsonl"
            if f.exists():
                for line in f.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rec = schema.loads(line)
                        bucket[rec.frame] = rec
            store["frames"] = bucket
        return store["frames"]

    def shots(self, vid: str) -> dict:
        f = self.get(vid).dir / "shots.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"stats": {}, "shots": []}

    def refresh_flags(self, vid: str) -> None:
        v = self.get(vid)
        d = v.dir
        v.has_frames = (d / "frames.jsonl").exists()
        v.has_poses = (d / "poses.jsonl").exists()
        v.has_segments = (d / "segments.jsonl").exists()
        data = self.shots(vid)
        v.shots = len(data.get("shots", []))
        v.court_shots = sum(1 for s in data.get("shots", []) if s.get("court_visible"))
        if (d / "pts.jsonl").exists():
            t = self.table(vid)
            v.nframes = t.nframes
            v.keys = len(t.keys)
        self.save()

    def reload(self, vid: str) -> None:
        self._rows.pop(vid, None)
        self.refresh_flags(vid)


__all__ = ["Library", "ROOT", "Video", "slug"]
