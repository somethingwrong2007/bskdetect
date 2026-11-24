"""逐帧数据契约：后端各阶段与前端唯一共享的结构。

坐标系约定（改动会污染所有历史数据，勿随意变更）：
- 单位米，原点在球场中心
- x 沿球场长边，正方向为 A 队进攻方向
- y 沿短边
- z 垂直向上，地面为 0；篮球触地时 z == BALL_RADIUS_M
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Literal, TextIO

BALL_RADIUS_M = 0.122
COURT_MODELS = ("FIBA", "NBA", "3x3")

Category = Literal["player", "ball", "referee", "coach", "bench", "rim", "backboard"]
Provenance = Literal["live", "refined", "human"]
Team = Literal["A", "B"]

# COCO-17：骨骼拓扑的唯一真源，前端画图与后端校正都引用这里
COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
COCO17_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
)
# 左右成对，供骨长对称性约束使用
COCO17_PAIRS = ((5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
POSE_SCHEMAS = {"coco17": COCO17_EDGES}


def pose_edges(schema: str) -> tuple[tuple[int, int], ...]:
    try:
        return POSE_SCHEMAS[schema]
    except KeyError as exc:
        raise ValueError(f"未知骨骼拓扑: {schema}") from exc


@dataclass(slots=True)
class Tracked:
    track_id: int
    category: Category
    frame: int
    x: float | None = None
    y: float | None = None
    z: float | None = None
    vx: float | None = None
    vy: float | None = None
    team: Team | None = None
    jersey: int | None = None
    jersey_conf: float = 0.0
    visible: bool = True
    conf: float = 1.0
    bbox_px: tuple[float, float, float, float] | None = None
    in_possession: bool = False
    provenance: Provenance = "live"
    edit_id: str | None = None


@dataclass(slots=True)
class Camera:
    """单帧相机解算结果。usable=False 时不得输出球场坐标。"""

    fx: float
    fy: float
    cx: float
    cy: float
    rotation: tuple[float, ...]
    translation: tuple[float, float, float]
    homography: tuple[float, ...]
    reproj_error_px: float
    usable: bool
    method: str = "court_keypoints"


@dataclass(slots=True)
class FrameRecord:
    video_id: str
    frame: int
    pts_us: int
    shot_id: int
    width: int
    height: int
    # 预留多视角：以后接第二路机位时，世界坐标 track 不变，只有观测来源要区分
    camera_id: str = "main"
    court_model: str = "FIBA"
    camera: Camera | None = None
    objects: list[Tracked] = field(default_factory=list)

    def court_coords(self) -> list[Tracked]:
        return [o for o in self.objects if o.visible and o.x is not None and o.y is not None]


@dataclass(slots=True)
class Pose2D:
    """单帧单人 2D 关键点，像素坐标；kpts[i] = [u, v, conf]，长度须等于拓扑点数。

    与 Tracked 分开存：姿态要反复重跑与人工拖动，而俯视坐标是稳定结论，
    两者生命周期不同，混在一条记录里会让每次重跑都覆盖掉标定成果。
    """

    video_id: str
    frame: int
    track_id: int
    kpts: list[list[float]]
    schema: str = "coco17"
    height_px: float = 0.0
    model: str = ""
    # 时序状态层标记：True 表示这帧骨架里含推测点（关节被遮挡/整人框丢失后维持），
    # 只是给人看的提示，坐标仍然是像素坐标；旧数据没有该字段，默认 False 兼容。
    occluded: bool = False
    provenance: Provenance = "live"
    edit_id: str | None = None

    def mean_conf(self) -> float:
        if not self.kpts:
            return 0.0
        return sum(k[2] for k in self.kpts) / len(self.kpts)


@dataclass(slots=True)
class Segment2D:
    """逐帧分割轮廓，存多边形而非掩码位图：人工可直接拖点改，且体积可控。"""

    video_id: str
    frame: int
    track_id: int
    category: Category
    polygon: list[list[float]]
    area_px: float = 0.0
    model: str = ""
    provenance: Provenance = "live"
    edit_id: str | None = None


def obj_dumps(record) -> str:
    return json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":"))


def obj_loads(cls, line: str):
    return cls(**json.loads(line))


def write_obj_jsonl(path: str | Path, records) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(obj_dumps(record) + "\n")
            n += 1
    return n


def read_obj_jsonl(path: str | Path, cls):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield obj_loads(cls, line)


def dumps(record: FrameRecord) -> str:
    return json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":"))


def loads(line: str) -> FrameRecord:
    data = json.loads(line)
    camera = data.pop("camera", None)
    objects = [Tracked(**o) for o in data.pop("objects", [])]
    return FrameRecord(camera=Camera(**camera) if camera else None, objects=objects, **data)


def write_jsonl(path: str, records: Iterator[FrameRecord] | list[FrameRecord]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(dumps(record) + "\n")


def read_jsonl(path: str) -> Iterator[FrameRecord]:
    with open(path, encoding="utf-8") as fh:
        yield from (loads(line) for line in fh if line.strip())


def merge_edits(machine: FrameRecord, overlay: list[Tracked]) -> FrameRecord:
    """重跑分析只替换 provenance != human 的对象，人工修改永远存活。"""
    kept = [o for o in overlay if o.provenance == "human"]
    human_ids = {o.track_id for o in kept}
    out = [o for o in machine.objects if o.track_id not in human_ids]
    out.extend(kept)
    out.sort(key=lambda o: (o.category != "player", o.track_id))
    return FrameRecord(**{**asdict(machine), "objects": out, "camera": machine.camera})


__all__ = [
    "BALL_RADIUS_M",
    "COURT_MODELS",
    "COCO17_EDGES",
    "COCO17_NAMES",
    "COCO17_PAIRS",
    "Camera",
    "FrameRecord",
    "Pose2D",
    "Segment2D",
    "Tracked",
    "dumps",
    "loads",
    "merge_edits",
    "obj_dumps",
    "obj_loads",
    "pose_edges",
    "read_jsonl",
    "read_obj_jsonl",
    "write_jsonl",
    "write_obj_jsonl",
]
