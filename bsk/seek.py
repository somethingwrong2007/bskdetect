"""按帧号随机取帧：靠关键帧索引 seek，不做整段重解。

调试台的核心访问模式是"拖到第 N 帧"，而 H.264 的 P/B 帧必须从上一个关键帧
向前解出来。所以这里用 pts 真值表二分定位目标帧所属的关键帧，seek 到该关键帧
（换算成 stream 时间基），再向前解到目标为止。

收敛判据用 pts 相等而不是帧号计数：seek 后解码器吐出的第一帧对应哪个帧号，
受 open-GOP 与重排序影响，数出来的偏移并不可信。time_base 一律从打开的
stream 现取，不缓存到真值表里，免得两份来源打架。
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


@dataclass(slots=True)
class FrameTable:
    """pts 真值表：下标 i 对应 frame=i+1，keys 为升序的关键帧帧号。"""

    pts_us: list[int]
    keys: list[int]

    @property
    def nframes(self) -> int:
        return len(self.pts_us)

    def clamp(self, frame: int) -> int:
        return max(1, min(int(frame), self.nframes))

    def key_before(self, frame: int) -> int:
        """目标帧之前（含自身）最近的关键帧帧号；没有就用 1，表示只能从头线性解。

        网上下的集锦常被截断到某个 GOP 中间，首帧不是关键帧：此时 keys[0] 远大于 1，
        按"最近前序关键帧"会跳到目标之后，反而解不出该帧。
        """
        i = bisect.bisect_right(self.keys, self.clamp(frame)) - 1
        if i < 0:
            return 1
        return self.keys[i]

    @classmethod
    def load(cls, manifest: str | Path) -> "FrameTable":
        pts: list[int] = []
        keys: list[int] = []
        with Path(manifest).open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                pts.append(int(row["pts_us"]))
                if row.get("is_key"):
                    keys.append(int(row["frame"]))
        if not pts:
            raise ValueError(f"pts 真值表为空: {manifest}")
        return cls(pts_us=pts, keys=keys or [1])


class FramePicker:
    """帧精确随机访问：每次调用开一次解码器，seek 到所属关键帧后向前解。"""

    def __init__(self, path: str | Path, table: FrameTable) -> None:
        self.path = str(path)
        self.table = table

    def get(self, frame: int, size: tuple[int, int] | None = None) -> tuple[int, int, np.ndarray]:
        """返回 (frame, pts_us, RGB)。"""
        return self.frames_at([frame], size=size)[0]

    def frames_at(
        self, wanted: list[int], size: tuple[int, int] | None = None
    ) -> list[tuple[int, int, np.ndarray]]:
        """一次 seek 覆盖一段邻近帧，按帧号升序返回。"""
        frames = sorted({self.table.clamp(f) for f in wanted})
        if not frames:
            return []

        container = av.open(self.path)
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            tb = Fraction(stream.time_base)

            out: list[tuple[int, int, np.ndarray]] = []
            i = 0
            while i < len(frames):
                # 同一关键帧区间内的目标合并到一次 seek，多余的就顺路解出来
                anchor = self.table.key_before(frames[i])
                j = i
                while j < len(frames) and self.table.key_before(frames[j]) == anchor:
                    j += 1
                out.extend(self._run(container, stream, tb, anchor, frames[i:j], size))
                i = j
            order = {f: k for k, f in enumerate(frames)}
            return sorted(out, key=lambda r: order[r[0]])
        finally:
            container.close()

    def _run(
        self,
        container,
        stream,
        tb: Fraction,
        anchor_key: int,
        targets: list[int],
        size: tuple[int, int] | None = None,
    ) -> list[tuple[int, int, np.ndarray]]:
        want = set(targets)
        top = max(targets)
        start_us = self.table.pts_us[anchor_key - 1]
        stop_us = self.table.pts_us[top - 1]
        start_units = int(round(start_us * 1e-6 / float(tb)))
        container.seek(max(start_units - 1, 0), any_frame=False, backward=True, stream=stream)

        found: list[tuple[int, int, np.ndarray]] = []
        for packet in container.demux(stream):
            for decoded in packet.decode():
                pts = decoded.pts if decoded.pts is not None else decoded.best_pts
                pts_us = int(round(pts * tb * 1e6))
                if pts_us > stop_us:
                    break
                if pts_us < start_us:
                    continue
                # 真值表按呈现顺序编号，解码顺序即呈现顺序，故可用累计计数定帧号
                frame_no = anchor_key + _offset(self.table, anchor_key, pts_us)
                if frame_no in want:
                    rgb = np.ascontiguousarray(decoded.to_rgb().to_ndarray())
                    if size is not None:
                        import cv2

                        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
                    found.append((frame_no, pts_us, rgb))
                    want.discard(frame_no)
            if not want:
                break
        if want:
            raise ValueError(f"未能解出帧 {sorted(want)}（关键帧 {anchor_key} 起解）")
        return found


def _offset(table: FrameTable, anchor_key: int, pts_us: int) -> int:
    return bisect.bisect_right(table.pts_us, pts_us, lo=anchor_key - 1) - anchor_key


__all__ = ["FramePicker", "FrameTable"]
