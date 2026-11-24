"""素材接入层：帧号基准、PTS 真值表、可变帧率判定与 CFR 化方案。

这里定下的是整条管线的"时间坐标系"，前后端都靠它对齐：
- frame 一律 1-based，且是解码后的呈现顺序序号
- pts_us 由 packet/frame 的 pts 乘 time_base 换算，绝不按 frame/fps 反算
- 复盘时前端用 pts_us 二分查帧，直播时后端落后就丢到最近关键帧
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av


@dataclass(slots=True)
class Probe:
    path: str
    codec: str
    profile: str | None
    pix_fmt: str
    width: int
    height: int
    avg_fps: float
    base_fps: float | None
    declared_frames: int
    duration_s: float
    bit_rate: int | None
    time_base: str
    scanned_frames: int
    keyframes: int
    gop_mean: float
    pts_delta_median_us: float
    pts_delta_min_us: float
    pts_delta_max_us: float
    is_cfr: bool
    cfr_share: float
    vfr_ratio: float
    pts_monotonic: bool
    # (帧间隔us, 出现次数) 前若干项：用来区分 time_base 量化抖动与真正的可变帧率
    delta_hist_us: tuple[tuple[int, int], ...] = ()


def _stream_info(stream: av.video.stream.VideoStream) -> tuple[float, float | None]:
    def _f(r) -> float | None:
        try:
            fr = Fraction(r)
            return float(fr) if fr else None
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    return _f(stream.average_rate) or 0.0, _f(stream.base_rate)


def probe(path: str | Path) -> Probe:
    """读取素材规格并实测帧率是否恒定，不信任容器里的声明值。"""
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        tb = Fraction(stream.time_base)
        avg_fps, base_fps = _stream_info(stream)
        dur_s = float(stream.duration * tb) if stream.duration else float(container.duration / av.time_base)

        pts_list: list[int] = []
        keyframes = 0
        prev_pts: int | None = None
        monotonic = True
        # 关键帧判定只能在 packet 层做：帧对象的 key_frame 属性在不同 PyAV 版本上不可靠
        for packet in container.demux(stream):
            is_key = bool(packet.is_keyframe)
            for frame in packet.decode():
                pts = frame.pts if frame.pts is not None else frame.best_pts
                if prev_pts is not None and pts < prev_pts:
                    monotonic = False
                pts_list.append(int(pts))
                prev_pts = pts
                if is_key:
                    keyframes += 1

        deltas = [int((b - a) * tb * 1e6) for a, b in zip(pts_list, pts_list[1:], strict=False)]
        pos = [d for d in deltas if d > 0]
        if not pos:
            raise ValueError("没有可用的帧时间戳")
        sorted_pos = sorted(pos)
        median_us = sorted_pos[len(sorted_pos) // 2]
        min_us, max_us = sorted_pos[0], sorted_pos[-1]
        # 帧间隔分布比"是否恒定"这个布尔值有用：time_base 量化误差只会在中位数附近
        # 抖 1~2 个 tick，真 VFR 会出现成比例的多个不同间隔值
        hist = sorted(Counter(deltas).items(), key=lambda kv: -kv[1])[:6]
        tol = max(2.0, 0.02 * median_us)
        cfr_share = sum(1 for d in pos if abs(d - median_us) <= tol) / len(pos)
        return Probe(
            path=str(path),
            codec=stream.codec_context.name,
            profile=getattr(stream.codec_context, "profile", None),
            pix_fmt=str(stream.format.name if stream.format else "?"),
            width=stream.width,
            height=stream.height,
            avg_fps=avg_fps,
            base_fps=base_fps,
            declared_frames=stream.frames,
            duration_s=dur_s,
            bit_rate=stream.bit_rate or container.bit_rate,
            time_base=str(stream.time_base),
            scanned_frames=len(pts_list),
            keyframes=keyframes,
            gop_mean=len(pts_list) / max(keyframes, 1),
            pts_delta_median_us=median_us,
            pts_delta_min_us=min_us,
            pts_delta_max_us=max_us,
            is_cfr=cfr_share >= 0.995,
            cfr_share=round(cfr_share, 5),
            vfr_ratio=max_us / median_us if median_us else 0.0,
            pts_monotonic=monotonic,
            delta_hist_us=tuple(hist),
        )
    finally:
        container.close()


def write_pts_manifest(path: str | Path, out: str | Path) -> dict:
    """逐帧真值表：前端按 pts_us 二分定位帧，复盘不靠播放器时钟。"""
    container = av.open(str(path))
    stream = container.streams.video[0]
    tb = Fraction(stream.time_base)
    rows = []
    idx = 0
    try:
        for packet in container.demux(stream):
            is_key = bool(packet.is_keyframe)
            for frame in packet.decode():
                idx += 1
                pts = frame.pts if frame.pts is not None else frame.best_pts
                rows.append(
                    {
                        "frame": idx,
                        "pts_us": int(round(pts * tb * 1e6)),
                        "is_key": is_key,
                    }
                )
    finally:
        container.close()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {"frames": len(rows), "keys": sum(r["is_key"] for r in rows), "path": str(out)}


_HWACCEL_OK: bool | None = None  # None=未探测；False=不可用


def _open_container(path: str | Path):
    """默认 CPU 解码；BSK_HWACCEL=1 时尝试 NVDEC 硬解（不可用自动回退）。

    实测 NVDEC 与同卡推理争抢带宽，流水线里反而更慢，所以默认关闭；
    高码率/4K 素材或多卡部署可通过环境变量开启。
    """
    global _HWACCEL_OK
    import av

    if os.environ.get("BSK_HWACCEL") == "1" and _HWACCEL_OK is not False:
        try:
            container = av.open(
                str(path),
                options={"hwaccel": "cuda", "hwaccel_output_format": "cuda"},
            )
            _HWACCEL_OK = True
            return container
        except Exception as exc:  # 驱动/编解码不支持时退回软件解码
            _HWACCEL_OK = False
            print(f"[ingest] NVDEC 不可用，回退 CPU 解码: {type(exc).__name__}: {exc}")
    return av.open(str(path))


def frames(
    path: str | Path,
    start: int = 1,
    limit: int | None = None,
    size: tuple[int, int] | None = None,
    stride: int = 1,
    seek_us: int | None = None,
) -> Iterator[tuple[int, int, "object"]]:
    """按呈现顺序产出 (frame_idx, pts_us, ndarray RGB)。

    size 指定时缩放到 (w,h)，供直方图类算法用；像素路径目前走 CPU，
    NVDEC 零拷贝后端接入后换实现但保持同样的三元组签名。

    seek_us 给出时先 seek 到该时间戳之前最近的关键帧再解，避免从片头整段重解；
    此时调用方必须同时传 start，因为跳过后无从得知全局帧号。真值表由 seek 模块负责，
    这里不反向依赖它。
    """
    import numpy as np

    container = _open_container(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    tb = Fraction(stream.time_base)
    skipping = seek_us is not None
    if skipping:
        units = int(round(seek_us * 1e-6 / float(tb)))
        container.seek(max(units - 1, 0), any_frame=False, backward=True, stream=stream)
    idx = start - 1
    emitted = 0
    try:
        for packet in container.demux(stream):
            for frame in packet.decode():
                pts = frame.pts if frame.pts is not None else frame.best_pts
                pts_us = int(round(pts * tb * 1e6))
                if skipping:
                    if pts_us < seek_us:
                        continue
                    skipping = False
                idx += 1
                if (idx - start) % stride:
                    continue
                if limit is not None and emitted >= limit:
                    return
                rgb = frame.to_rgb().to_ndarray()
                if size is not None:
                    import cv2

                    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
                yield idx, pts_us, np.ascontiguousarray(rgb)
                emitted += 1
    finally:
        container.close()


def gather(path: str | Path, indices: list[int], size: tuple[int, int] | None = None) -> dict:
    """单次解码取出多个指定帧，避免逐帧随机访问时的 O(n) 重复解码。"""
    want = set(int(i) for i in indices)
    out: dict[int, object] = {}
    if not want:
        return out
    top = max(want)
    for idx, _pts, arr in frames(path, start=1, limit=top, size=size):
        if idx in want:
            out[idx] = arr
            want.discard(idx)
            if not want:
                break
    return out


__all__ = ["Probe", "frames", "gather", "probe", "write_pts_manifest"]
