"""镜头切分与"这帧能不能出球场坐标"的门控特征。

集锦和真实转播里，特写、回放、慢镜、观众席、字幕叠层都不该产出俯视坐标。
门控判错的代价分两种：误杀战术镜头丢数据，误放垃圾镜头污染标定和 ID，
后者更难事后发现，所以阈值偏保守，宁可少出坐标。
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from bsk.ingest import frames

SCAN_SIZE = (160, 90)


@dataclass(slots=True)
class FrameFeature:
    frame: int
    pts_us: int
    hist_dist: float
    floor_ratio: float
    edge_density: float
    brightness: float
    court_fill: float = 0.0


@dataclass(slots=True)
class Shot:
    start_frame: int
    end_frame: int
    frames: int
    start_pts_us: int
    end_pts_us: int
    floor_ratio_p50: float
    court_fill_p50: float = 0.0
    edge_density_p50: float = 0.0
    brightness_p50: float = 0.0
    peak_hist_dist: float = 0.0
    court_visible: bool = False

    @property
    def mid_frame(self) -> int:
        return (self.start_frame + self.end_frame) // 2


@dataclass(slots=True)
class ShotReport:
    shots: list[Shot]
    thresh: float
    median_dist: float
    mad: float
    frames_scanned: int
    court_visible_frames: int = 0


def floor_mask(hsv: np.ndarray) -> np.ndarray:
    """以画面下中部的主色当地板参考色，返回全帧的地板像素掩膜。

    不同球场地板色差极大（浅枫木到深色染色），写死 HSV 区间换个场馆就失效，
    所以参考色从当前帧自取。
    """
    h, w = hsv.shape[:2]
    patch = hsv[int(h * 0.6) : h, int(w * 0.2) : int(w * 0.8)]
    h0, s0, v0 = (float(np.median(patch[:, :, c])) for c in range(3))
    hue = hsv[:, :, 0].astype(np.int16)
    dh = np.minimum(np.abs(hue - h0), 180 - np.abs(hue - h0))
    return (dh <= 9) & (np.abs(hsv[:, :, 1].astype(np.int16) - s0) <= 55) & (
        np.abs(hsv[:, :, 2].astype(np.int16) - v0) <= 70
    )


def _floor_ratio(hsv: np.ndarray) -> float:
    return float(floor_mask(hsv).mean())


def court_fill(hsv: np.ndarray) -> float:
    """画面下缘带里最大连通地板块占该带的比例，用作"这帧能不能标定"的门控。

    整帧地板占比在实测集锦上是 p25=0.05 / p50=0.25，写死 0.40 会把五分之四的真
    宽景镜头误杀；而连通块占比 p25=0.012 / p50=0.384，双峰分得很开：人群和球衣上的
    木色是碎斑，球场是一整片。碎斑过不了连通性这一关，比颜色阈值稳。
    """
    return _band_fill(floor_mask(hsv), hsv.shape)


def features(path: str | Path, start: int = 1, limit: int | None = None, stride: int = 1) -> Iterator[FrameFeature]:
    prev_hist = None
    for idx, pts_us, rgb in frames(path, start=start, limit=limit, size=SCAN_SIZE, stride=stride):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 12], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        dist = 0.0 if prev_hist is None else float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        prev_hist = hist
        luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hsv_mask = floor_mask(hsv)
        yield FrameFeature(
            frame=idx,
            pts_us=pts_us,
            hist_dist=dist,
            floor_ratio=float(hsv_mask.mean()),
            edge_density=float(cv2.Canny(luma, 60, 140).mean() / 255.0),
            brightness=float(luma.mean() / 255.0),
            court_fill=_band_fill(hsv_mask, hsv.shape),
        )


def _band_fill(mask: np.ndarray, shape: tuple[int, int]) -> float:
    h, w = shape[:2]
    band = mask[int(h * 0.55) : int(h * 0.94), int(w * 0.06) : int(w * 0.94)]
    n, lab = cv2.connectedComponents(band.astype(np.uint8))
    return float(np.bincount(lab.ravel())[1:].max() / band.size) if n > 1 else 0.0


def detect_shots(
    path: str | Path,
    limit: int | None = None,
    stride: int = 1,
    min_abs_cut: float = 0.22,
    mad_k: float = 6.0,
    min_shot_frames: int = 8,
    court_fill_min: float = 0.15,
) -> ShotReport:
    """直方图距离的滑动中位数做自适应硬切检测，再按镜头聚合门控特征。"""
    feats = list(features(path, limit=limit, stride=stride))
    if not feats:
        return ShotReport([], 0.0, 0.0, 0.0, 0)

    dists = np.array([f.hist_dist for f in feats])
    med = float(np.median(dists))
    mad = float(np.median(np.abs(dists - med)))
    thresh = max(min_abs_cut, med + mad_k * mad)

    bounds = [0]
    for i in range(1, len(feats)):
        if dists[i] >= thresh and feats[i].frame - feats[bounds[-1]].frame >= min_shot_frames:
            bounds.append(i)
    bounds.append(len(feats))

    shots = []
    for a, b in zip(bounds, bounds[1:], strict=False):
        seg = feats[a:b]
        if not seg:
            continue
        floor_p50 = float(np.median([f.floor_ratio for f in seg]))
        fill_p50 = float(np.median([f.court_fill for f in seg]))
        shots.append(
            Shot(
                start_frame=seg[0].frame,
                end_frame=seg[-1].frame,
                frames=len(seg),
                start_pts_us=seg[0].pts_us,
                end_pts_us=seg[-1].pts_us,
                floor_ratio_p50=floor_p50,
                court_fill_p50=fill_p50,
                edge_density_p50=float(np.median([f.edge_density for f in seg])),
                brightness_p50=float(np.median([f.brightness for f in seg])),
                peak_hist_dist=float(dists[max(a - 1, 0) : b].max(initial=0.0)),
                court_visible=fill_p50 >= court_fill_min,
            )
        )
    return ShotReport(
        shots=shots,
        thresh=thresh,
        median_dist=med,
        mad=mad,
        frames_scanned=len(feats),
        court_visible_frames=sum(1 for f in feats if f.court_fill >= court_fill_min),
    )


def write_shots(path: str | Path, report: ShotReport, extra: dict | None = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "thresh": report.thresh,
        "median_dist": report.median_dist,
        "mad": report.mad,
        "frames_scanned": report.frames_scanned,
        "shots": len(report.shots),
        "court_visible_shots": sum(1 for s in report.shots if s.court_visible),
        "court_visible_frames": report.court_visible_frames,
        **(extra or {}),
    }
    payload = {"stats": stats, "shots": [asdict(s) for s in report.shots]}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
