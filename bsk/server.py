"""调试台后端：导入视频、按帧号取烧录叠加图、按区间取测试数据。

单进程、零构建前端，路径只在服务端解析，浏览器只拿帧号。这样做的关键原因是
帧精确：把 <video> 的时钟交给浏览器去对齐数据帧号，会踩 mediaTime 与
presentedFrames 的坑，而调试台判断的恰恰是"这一帧的数据对不对"。
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from bsk import analyze, calibrate, court, ingest, render, schema
from bsk.auto_calibrate import auto_calibrate as _auto_calibrate
from bsk.perceive import shots as shotmod
from bsk.realtime_calibrate import calibrate_frame
from bsk.server_rt import (analyze_frame_realtime, render_frame_realtime,
                           get_cached, ensure_3d, frame_3d_payload,
                           engine_info, profile_allows_3d, rt_profile,
                           RealtimeCache)
from bsk.store import Library

lib = Library()
app = FastAPI(title="bskdetect 调试台", version="0.1")

_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_imgs: OrderedDict[tuple, bytes] = OrderedDict()
IMG_CACHE = 48
LAYER_KEYS = ("hud", "boxes", "pose", "seg", "floor", "motion", "trail")

_rt_cache: RealtimeCache = RealtimeCache(max_size=60)


def _parse_layers(raw: str) -> render.Layers:
    if not raw:
        return render.Layers()
    want = {s.strip() for s in raw.split(",") if s.strip()}
    known = {k for k in want if k in LAYER_KEYS}
    if not known:
        raise HTTPException(400, f"未知图层，可选: {','.join(LAYER_KEYS)}")
    return render.Layers(**{k: (k in known) for k in LAYER_KEYS})


def _index_job(vid: str) -> None:
    """后台建索引：规格 -> pts 真值表 -> 镜头切分 -> 缩略条。

    每一步都要整段解码，所以产物已存在就跳过：服务重启不该把几分钟的活重做一遍。
    """
    job = _jobs[vid]
    try:
        v = lib.get(vid)
        v.status = "indexing"
        if v.nframes == 0:
            job["stage"] = "probe"
            p = ingest.probe(v.path)
            v.width, v.height, v.fps, v.nframes = p.width, p.height, p.avg_fps, p.scanned_frames
            v.duration_s, v.is_cfr, v.gop_mean, v.codec = p.duration_s, p.is_cfr, p.gop_mean, p.codec
        if not v.manifest_path().exists():
            job["stage"] = "pts"
            info = ingest.write_pts_manifest(v.path, v.manifest_path())
            v.keys = info["keys"]
        if not (v.dir / "shots.json").exists():
            job["stage"] = "shots"
            report = shotmod.detect_shots(v.path, stride=3)
            shotmod.write_shots(v.dir / "shots.json", report, {"video": v.path})
        if not (v.dir / "strip.png").exists():
            job["stage"] = "strip"
            _build_strip(v)
        job["stage"] = "done"
        v.status = "ready"
        v.detail = ""
        lib.refresh_flags(vid)
        job["ok"] = True
    except Exception as exc:  # 后台线程里的异常必须回传给前端，否则界面永远停在 indexing
        v = lib.get(vid)
        v.status = "error"
        v.detail = f"{type(exc).__name__}: {exc}"
        job["ok"] = False
        job["error"] = v.detail
    finally:
        job["done"] = True
        job["secs"] = round(time.time() - job["t0"], 1)
        lib.save()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "videos": len(lib.videos), "t": time.strftime("%H:%M:%S"),
            "engine": engine_info().get("rtmo")}


class AddBody(BaseModel):
    path: str


def start_index(vid: str) -> dict:
    """已在跑或已就绪就不重复起线程，CLI 与 HTTP 共用这条入口。"""
    with _lock:
        job = _jobs.get(vid)
        if job and not job.get("done"):
            return job
        job = {"stage": "queued", "ok": None, "done": False, "t0": time.time()}
        _jobs[vid] = job
    threading.Thread(target=_index_job, args=(vid,), daemon=True).start()
    return job


@app.post("/api/videos")
def add_video(body: AddBody) -> dict:
    try:
        v = lib.add(body.path)
    except FileNotFoundError:
        raise HTTPException(404, f"路径不存在: {body.path}") from None
    if v.status != "ready":
        start_index(v.vid)
    return {"vid": v.vid, "status": v.status, "name": v.name}


@app.get("/api/videos")
def list_videos() -> dict:
    lib.root.mkdir(parents=True, exist_ok=True)
    out = []
    for v in lib.videos.values():
        d = v.to_dict()
        d["job"] = _jobs.get(v.vid, {}).get("stage")
        out.append(d)
    return {"videos": sorted(out, key=lambda d: -d["added_at"]), "derived_root": str(lib.root.resolve())}


@app.get("/api/videos/{vid}")
def video_meta(vid: str) -> dict:
    lib.refresh_flags(vid)
    v = lib.get(vid)
    d = v.to_dict()
    d["job"] = _jobs.get(vid)
    d["shots_json"] = lib.shots(vid).get("stats", {})
    d["keys"] = len(lib.table(vid).keys) if v.manifest_path().exists() else 0
    return d


@app.get("/api/videos/{vid}/frames")
def frame_data(vid: str, frm: int = Query(0, alias="from"), to: int = 0, limit: int = 400) -> dict:
    recs = lib.frame_records(vid)
    poses = lib.poses(vid)
    segs = lib.segments(vid)
    hi = to or (frm + limit)
    frames = sorted(f for f in set(recs) | set(poses) | set(segs) if frm <= f <= hi)[:limit]
    return {
        "vid": vid,
        "frames": [dataclasses.asdict(recs[f]) for f in frames if f in recs],
        "poses": {str(f): [dataclasses.asdict(p) for p in poses[f]] for f in frames if f in poses},
        "segments": {str(f): [dataclasses.asdict(s) for s in segs[f]] for f in frames if f in segs},
        "missing": [f for f in frames if f not in recs],
    }


_shot_cache: dict[str, tuple[int, list[dict]]] = {}


def shot_list(vid: str) -> list[dict]:
    """镜头列表补上稳定 id：Shot 本身不带编号，前端要靠 id 画刻度和跳转。"""
    mtime = int(time.time())
    cached = _shot_cache.get(vid)
    if cached and mtime - cached[0] < 5:
        return cached[1]
    data = lib.shots(vid)
    rows = [{**s, "id": i} for i, s in enumerate(data.get("shots", []))]
    _shot_cache[vid] = (mtime, rows)
    return rows


def _shot_of(shots: list[dict], frame: int) -> dict | None:
    for s in shots:
        if s["start_frame"] <= frame <= s["end_frame"]:
            return s
    return None


def _motion_mask(cur: np.ndarray, prev: np.ndarray | None) -> np.ndarray | None:
    if prev is None or prev.shape != cur.shape:
        return None
    g1 = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)
    g0 = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    d = cv2.absdiff(g1, g0)
    m = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0


def _ball_history(vid: str, n: int, shape: tuple, recs: dict, span: int = 12) -> list:
    """球在像素面的拖影：把前 span 帧的球框中心按当前图的缩放比例折算过来。"""
    out = []
    h, w = shape[:2]
    for f in range(max(1, n - span), n):
        r = recs.get(f)
        if not r:
            continue
        sx, sy = w / max(r.width, 1), h / max(r.height, 1)
        for o in r.objects:
            if o.category == "ball" and o.bbox_px:
                x0, y0, x1, y1 = o.bbox_px
                out.append((f, (x0 + x1) / 2 * sx, (y0 + y1) / 2 * sy))
    return out


def _scale_overlay(
    objects: list[schema.Tracked],
    poses: list[schema.Pose2D],
    segments: list[schema.Segment2D],
    sx: float,
    sy: float,
) -> tuple[list[schema.Tracked], list[schema.Pose2D], list[schema.Segment2D]]:
    """把记录分辨率下的坐标折算到当前解码图尺寸。

    frames.jsonl / poses.jsonl 里的坐标是在原始分辨率下算出来的，而
    frame.jpg 按请求的 width 解码；不折算会导致宽度不是原尺寸时叠加层整体偏移。
    """
    if abs(sx - 1.0) < 1e-3 and abs(sy - 1.0) < 1e-3:
        return objects, poses, segments
    objects = [
        dataclasses.replace(o, bbox_px=(o.bbox_px[0] * sx, o.bbox_px[1] * sy,
                                        o.bbox_px[2] * sx, o.bbox_px[3] * sy))
        for o in objects
    ]
    poses = [
        dataclasses.replace(p, kpts=[[u * sx, v * sy, c] for u, v, c in p.kpts],
                            height_px=p.height_px * sy)
        for p in poses
    ]
    segments = [
        dataclasses.replace(s, polygon=[[u * sx, v * sy] for u, v in s.polygon],
                            area_px=s.area_px * sx * sy)
        for s in segments
    ]
    return objects, poses, segments


_HUMAN_CATS = {"player", "referee", "coach", "bench"}


def _inside_court(polygon: np.ndarray, x: float, y: float, margin: float) -> bool:
    """点到球场多边形的有符号距离测试，margin 允许脚踩在边线外一点。"""
    return cv2.pointPolygonTest(polygon.reshape(-1, 1, 2), (float(x), float(y)), True) >= -margin


def _filter_by_court(
    objects: list[schema.Tracked],
    poses: list[schema.Pose2D],
    segments: list[schema.Segment2D],
    court_polygon: np.ndarray | None,
    margin: float,
) -> tuple[list[schema.Tracked], list[schema.Pose2D], list[schema.Segment2D]]:
    """按球场外框剔除观众/替补席：脚点（bbox 底边中点）在多边形外即丢弃。

    没有标定时原样返回，宁可多画也不误删。
    """
    if court_polygon is None:
        return objects, poses, segments
    kept = []
    for o in objects:
        if o.category not in _HUMAN_CATS or o.provenance == "human" or not o.bbox_px:
            kept.append(o)
            continue
        fx = (o.bbox_px[0] + o.bbox_px[2]) / 2
        fy = o.bbox_px[3]
        if _inside_court(court_polygon, fx, fy, margin):
            kept.append(o)
    ids = {o.track_id for o in kept}
    return (
        kept,
        [p for p in poses if p.track_id in ids],
        [s for s in segments if s.track_id in ids],
    )


def _overlay(vid: str, n: int, pts_us: int, rgb, table, spec: "render.Layers", prev=None) -> np.ndarray:
    recs = lib.frame_records(vid)
    rec = recs.get(n)
    shot = _shot_of(shot_list(vid), n)
    floor = shotmod.floor_mask(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)) if spec.floor else None
    motion = _motion_mask(rgb, prev) if spec.motion else None

    # 实时增量标定：每帧都更新 H 矩阵
    cal = calibrate_frame(vid, rgb)
    H_inv = np.array(cal.H_inv) if cal else None
    court_obj = court.COURTS.get(cal.court_model, court.FIBA) if cal else None

    # 记录坐标是原始分辨率，当前图按请求 width 解码，需要折算
    v = lib.get(vid)
    sx = rgb.shape[1] / max(v.width, 1)
    sy = rgb.shape[0] / max(v.height, 1)
    objects, poses, segments = _scale_overlay(
        rec.objects if rec else [],
        lib.poses(vid).get(n, []),
        lib.segments(vid).get(n, []),
        sx,
        sy,
    )

    # 按球场范围剔除观众/替补席（标定是实时的，多边形随摄像机运动更新）
    court_poly = None
    if cal is not None:
        court_poly = court.project_court_polygon(np.array(cal.H_inv, dtype=np.float64), court_obj)
    objects, poses, segments = _filter_by_court(
        objects, poses, segments, court_poly, margin=max(24.0, rgb.shape[1] * 0.025)
    )

    notes = []
    if not recs:
        notes.append("no frames.jsonl (M1/M2 not run)")
    if not lib.poses(vid):
        notes.append("no poses.jsonl (weights missing)")
    if cal:
        notes.append(f"calib: err={cal.reproj_error_px:.1f}px n={cal.keypoints_used}")
    else:
        notes.append("calib: FAILED")

    return render.render_frame(
        rgb,
        frame=n,
        pts_us=pts_us,
        nframes=table.nframes,
        shot_id=shot.get("id") if shot else None,
        court_visible=shot.get("court_visible") if shot else None,
        objects=objects,
        poses=poses,
        segments=segments,
        floor=floor,
        motion=motion,
        ball_history=_ball_history(vid, n, rgb.shape, recs),
        layers=spec,
        sources={"boxes": "frames.jsonl" if rec else "none", "pose": "poses.jsonl",
                 "seg": "segments.jsonl", "ball": "frames.jsonl", "camera": "realtime"},
        notes=notes,
        calib_H_inv=H_inv,
        court_model=court_obj,
    )


@app.get("/api/videos/{vid}/frame.jpg")
def frame_jpg(
    vid: str,
    n: int = Query(1, ge=1),
    layers: str = "hud,boxes,pose,trail",
    width: int = Query(0, ge=0, le=1920),
    q: int = Query(82, ge=40, le=98),
    realtime: int = Query(0, ge=0, le=1),
    ball: int = Query(0, ge=0, le=1),
    quality: str = Query("accurate", regex="^(fast|accurate)$"),
    court_model: str = Query("FIBA", regex="^(FIBA|NBA|3x3)$"),
    enable_3d: int = Query(0, ge=0, le=1),
) -> Response:
    v = lib.get(vid)
    if not (v.width and v.height):
        raise HTTPException(503, f"素材 {vid} 尚未完成探测，等索引结束")
    spec = _parse_layers(layers)
    key = (vid, n, layers, width, q, realtime, ball, court_model, quality)
    with _lock:
        if key in _imgs:
            _imgs.move_to_end(key)
            return Response(_imgs[key], media_type="image/jpeg")

    table = lib.table(vid)
    n = table.clamp(n)
    size = (width, max(1, int(width * v.height / max(v.width, 1)))) if width else None
    got = lib.picker(vid).frames_at([n - 1, n] if (spec.motion and n > 1) else [n], size=size)
    picked = {f: (pts, img) for f, pts, img in got}
    if n not in picked:
        raise HTTPException(404, f"帧 {n} 解不出来")
    pts_us, rgb = picked[n]
    prev = picked.get(n - 1, (0, None))[1]

    if realtime:
        # 使用优化的实时管线（8GB 档 + 请求要求时才做 3D）
        buf = render_frame_realtime(
            vid, n, rgb, pts_us, spec, width or rgb.shape[1], q,
            enable_3d=bool(enable_3d) and profile_allows_3d(),
            enable_calib=False,
            run_ball=bool(ball),
            backend=quality,
            court_model=court_model,
        )
    else:
        buf = render.to_jpeg(_overlay(vid, n, pts_us, rgb, table, spec, prev), q)
    
    with _lock:
        _imgs[key] = buf
        _imgs.move_to_end(key)
        if len(_imgs) > IMG_CACHE:
            _imgs.popitem(last=False)
    return Response(buf, media_type="image/jpeg", headers={"Cache-Control": "no-store", "X-Realtime": str(realtime)})


@app.get("/api/videos/{vid}/stream.mjpeg")
def stream_mjpeg(
    vid: str,
    frm: int = Query(1, alias="from", ge=1),
    count: int = Query(600, ge=1, le=20000),
    stride: int = Query(1, ge=1, le=10),
    layers: str = "hud,boxes,pose,trail",
    width: int = Query(960, ge=240, le=1920),
    q: int = Query(72, ge=40, le=95),
    fps: float = Query(0.0, ge=0.0, le=120),
):
    """顺序播放走线性解码 + multipart 推流。

    逐帧跳转每次都要从关键帧重解一整个 GOP（实测本机素材 GOP≈150），播放时
    这个代价会被放大 150 倍；线性解码一遍则只付一次。
    """
    v = lib.get(vid)
    if not (v.width and v.height):
        raise HTTPException(503, f"素材 {vid} 尚未完成探测，等索引结束")
    spec = _parse_layers(layers)
    table = lib.table(vid)
    start = table.clamp(frm)
    target_fps = fps or v.fps or 29.97
    anchor = table.key_before(start)
    size = (width, max(1, int(width * v.height / max(v.width, 1))))
    delay = stride / target_fps

    def pump():
        prev = None
        emitted = 0
        for idx, pts_us, rgb in ingest.frames(
            v.path, start=anchor, size=size, stride=1, seek_us=table.pts_us[anchor - 1]
        ):
            if idx < start:
                continue
            if (idx - start) % stride:
                prev = rgb
                continue
            if emitted >= count or idx > table.nframes:
                break
            try:
                img = _overlay(vid, idx, pts_us, rgb, table, spec, prev)
            except Exception as exc:  # 单帧画崩不该掐断整个流
                print(f"[stream] frame {idx}: {type(exc).__name__}: {exc}")
                prev = rgb
                continue
            buf = render.to_jpeg(img, q)
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(buf) + buf + b"\r\n"
            if delay:
                time.sleep(delay)
            prev = rgb
            emitted += 1
    yield b"--frame--\r\n"

    return StreamingResponse(pump(), media_type="multipart/x-mixed-replace; boundary=frame")


class _RealtimePlayer:
    """顺序解码游标：播放时逐帧取图，避免每帧都从关键帧重解一个 GOP。

    播放位置连续时直接读下一个解码帧；跳帧/回退/换宽度则重建游标。
    """

    _pool: dict[tuple[str, int], "_RealtimePlayer"] = {}

    def __init__(self, vid: str, width: int):
        v = lib.get(vid)
        self.vid = vid
        self.width = width
        self.table = lib.table(vid)
        self.size = (width, max(1, int(width * v.height / max(v.width, 1))))
        self.gen = None
        self.next_idx: int | None = None
        # 浏览器与调试脚本可能并发请求同一游标；生成器不支持并发迭代
        self._lock = threading.Lock()

    @classmethod
    def get(cls, vid: str, width: int) -> "_RealtimePlayer":
        key = (vid, width)
        p = cls._pool.get(key)
        if p is None:
            p = cls._pool[key] = cls(vid, width)
        return p

    def frame(self, n: int):
        """返回 (pts_us, rgb)；无法解出时返回 None（并发请求串行化）。

        多个播放端共用游标时，另一个请求可能已把生成器推进到 n 之后；
        首次取不到时重建游标再试一次，避免单次游标竞争直接让前端停播。
        """
        with self._lock:
            for _attempt in range(2):
                got = self._frame_locked(n)
                if got is not None:
                    return got
                self.gen = None
                self.next_idx = None
            return None

    def _frame_locked(self, n: int):
        v = lib.get(self.vid)
        # 游标失效（回退、跳太远、刚创建）就重新定位到最近关键帧
        if self.gen is None or self.next_idx is None or not (self.next_idx <= n <= self.next_idx + 90):
            anchor = self.table.key_before(n)
            self.gen = ingest.frames(
                v.path, start=anchor, size=self.size, stride=1,
                seek_us=self.table.pts_us[anchor - 1],
            )
            self.next_idx = None
        for idx, pts_us, rgb in self.gen:
            if idx < n:
                continue
            self.next_idx = idx + 1
            return (pts_us, rgb) if idx == n else None
        return None


def _rt_meta_header(vid: str, n: int, quality: str) -> dict:
    """把该帧分析摘要放进响应头，前端无需再发一次 JSON 请求。

    注意：响应头只放"摘要"（几十字节）。10 人 × 17 点的 3D 坐标是 4~6KB，
    绝不能塞进头里——那会把默认 2D 请求也撑慢，3D 数据走 /rt_3d JSON。
    """
    r = get_cached(vid, n, quality)
    if r is None:
        return {}
    objs = []
    for o in r.objects:
        if not o.bbox_px:
            continue
        item = {"id": o.track_id, "cat": o.category, "conf": round(float(o.conf), 2),
                "box": [int(v) for v in o.bbox_px]}
        if o.x is not None and o.y is not None:
            item["x"] = round(float(o.x), 3)
            item["y"] = round(float(o.y), 3)
        objs.append(item)
    objs = objs[:24]
    st3 = r.status_3d or {}
    st2d = r.status_court2d or {}
    meta = {
        "frame": n,
        "players": sum(1 for o in r.objects if o.category == "player"),
        "balls": sum(1 for o in r.objects if o.category == "ball"),
        "poses": len(r.poses),
        "backend": quality,
        "engine": (engine_info().get("rtmo") or "none"),
        "objects": objs,
        "court2d": {k: st2d.get(k) for k in ("ok", "reason", "n", "rejected", "reproj_px", "age_s") if k in st2d},
        "3d": {"ok": bool(st3.get("ok")), "n": len(r.players_3d),
               "why": st3.get("reason", "disabled")},
        "profile": rt_profile(),
    }
    return {"X-Rt-Data": json.dumps(meta, separators=(",", ":"))}


@app.get("/api/videos/{vid}/rt_meta.json")
def rt_meta_json(
    vid: str,
    n: int = Query(1, ge=1),
    quality: str = Query("accurate", regex="^(fast|accurate)$"),
) -> Response:
    """只读缓存返回该帧分析摘要（供前端低频率刷新面板，不触发计算）。"""
    table = lib.table(vid)
    n = table.clamp(n)
    r = get_cached(vid, n, quality)
    if r is None:
        return Response(status_code=204)
    header = _rt_meta_header(vid, n, quality)
    return Response(header.get("X-Rt-Data", "{}"), media_type="application/json")


@app.get("/api/videos/{vid}/rt_3d")
def rt_3d_json(
    vid: str,
    n: int = Query(1, ge=1),
    quality: str = Query("accurate", regex="^(fast|accurate)$"),
    court_model: str = Query("", regex="^(|FIBA|NBA|3x3)$"),
) -> Response:
    """当前帧的 3D 骨架（米制世界坐标）。

    只读 RealtimeCache：同一帧的 2D 由 rt_frame.jpg 触发分析，本接口不再跑检测，
    所以 2D/3D 一定来自同一份 FrameResult。没算好时返回 status=no_frame，
    前端下一帧重试即可（可增量、stream-friendly）。
    """
    v = lib.get(vid)
    if not (v.width and v.height):
        return Response(status_code=503, content=json.dumps({"error": "素材尚未完成索引"}),
                        media_type="application/json")
    n = lib.table(vid).clamp(n)
    payload = frame_3d_payload(vid, n, quality, court_model or None)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return Response(
        body, media_type="application/json",
        headers={"Cache-Control": "no-store", "X-Rt-3D": payload["status"]},
    )


@app.get("/api/videos/{vid}/rt_frame.jpg")
def rt_frame_jpg(
    vid: str,
    n: int = Query(1, ge=1),
    layers: str = "hud,boxes,pose,trail",
    width: int = Query(960, ge=240, le=1920),
    q: int = Query(72, ge=40, le=95),
    ball: int = Query(0, ge=0, le=1),
    quality: str = Query("accurate", regex="^(fast|accurate)$"),
    court_model: str = Query("FIBA", regex="^(FIBA|NBA|3x3)$"),
    enable_3d: int = Query(0, ge=0, le=1),
) -> Response:
    """实时播放帧：播到哪算到哪（顺序解码 + 按需分析 + 渲染）。

    enable_3d=1 时该帧顺带做 3D 提升（仅 8GB 档；4GB 档忽略）。前端进入 3D 视图后
    把这个参数带上，/rt_3d 就能直接读到同一帧的缓存结果。
    """
    v = lib.get(vid)
    if not (v.width and v.height):
        raise HTTPException(503, "素材尚未完成索引")
    table = lib.table(vid)
    n = table.clamp(n)
    spec = _parse_layers(layers)
    want_3d = bool(enable_3d) and profile_allows_3d()

    key = (vid, n, layers, width, q, ball, court_model, quality, "rt", int(want_3d))

    with _lock:
        if key in _imgs:
            _imgs.move_to_end(key)
            buf = _imgs[key]
            return Response(buf, media_type="image/jpeg",
                            headers={"Cache-Control": "no-store",
                                     **_rt_meta_header(vid, n, quality)})

    player = _RealtimePlayer.get(vid, width)
    got = player.frame(n)
    if got is None:
        raise HTTPException(404, f"帧 {n} 解不出来")
    pts_us, rgb = got

    buf = render_frame_realtime(
        vid, n, rgb, pts_us, spec, width, q,
        enable_3d=want_3d, enable_calib=None, run_ball=bool(ball),
        backend=quality, court_model=court_model,
    )
    with _lock:
        _imgs[key] = buf
        _imgs.move_to_end(key)
        if len(_imgs) > IMG_CACHE:
            _imgs.popitem(last=False)
    return Response(buf, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store",
                             **_rt_meta_header(vid, n, quality)})


STRIP_TILES = 120


def _build_strip(v, tiles: int = STRIP_TILES) -> Path:
    """线性解码一遍生成缩略条并落盘：逐帧 seek 要付 GOP×tiles 次解码，播放条会卡死。"""
    out = v.dir / "strip.png"
    table = lib.table(v.vid)
    step = max(1, round(table.nframes / tiles))
    tw, th = 96, 54
    picked: list[tuple[int, np.ndarray]] = []
    for idx, _pts, rgb in ingest.frames(v.path, stride=step, size=(tw, th)):
        picked.append((idx, rgb))
    if not picked:
        raise RuntimeError("缩略条一帧都没解出来")
    canvas = np.full((th, tw * len(picked), 3), 20, np.uint8)
    for i, (_f, img) in enumerate(picked):
        canvas[:, i * tw : (i + 1) * tw] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out), canvas)
    out.with_suffix(".json").write_text(
        json.dumps({"tiles": len(picked), "step": picked[1][0] - picked[0][0] if len(picked) > 1 else step,
                    "tw": tw, "first": picked[0][0]}, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


@app.get("/api/videos/{vid}/strip.png")
def strip_ep(vid: str, regen: int = Query(0, ge=0, le=1)) -> FileResponse:
    v = lib.get(vid)
    out = v.dir / "strip.png"
    if regen or not out.exists():
        _build_strip(v)
    lib.table(vid)
    return FileResponse(out, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/videos/{vid}/strip.json")
def strip_meta(vid: str) -> dict:
    v = lib.get(vid)
    f = v.dir / "strip.json"
    if not f.exists():
        _build_strip(v)
    data = json.loads(f.read_text(encoding="utf-8"))
    table = lib.table(vid)
    data["nframes"] = table.nframes
    data["keys"] = table.keys[:2000]
    return data


GUI = Path(__file__).resolve().parent.parent / "gui"


@app.get("/api/videos/{vid}/shots")
def shots_ep(vid: str) -> dict:
    rows = shot_list(vid)
    return {"vid": vid, "stats": lib.shots(vid).get("stats", {}), "shots": rows}


@app.get("/api/videos/{vid}/calibration")
def get_calibration(vid: str) -> dict:
    v = lib.get(vid)
    cal_path = v.dir / "calibration.json"
    if not cal_path.exists():
        return {"vid": vid, "calibrated": False}
    cal = calibrate.Calibration.load(cal_path)
    return {
        "vid": vid,
        "calibrated": True,
        "court_model": cal.court_model,
        "reproj_error_px": cal.reproj_error_px,
        "keypoints_used": cal.keypoints_used,
        "timestamp": cal.timestamp,
    }


class CalibrateReq(BaseModel):
    court_model: str = "FIBA"
    keypoints: list[dict] = []


@app.post("/api/videos/{vid}/calibrate")
def calibrate_ep(vid: str, req: CalibrateReq) -> dict:
    v = lib.get(vid)
    if not req.keypoints:
        raise HTTPException(400, "需要提供关键点")
    
    pixel_pts = [(kp["u"], kp["v"]) for kp in req.keypoints]
    court_pts = [(kp["x"], kp["y"]) for kp in req.keypoints]
    
    cal = calibrate.calibrate(pixel_pts, court_pts)
    cal.court_model = req.court_model
    cal.save(v.dir / "calibration.json")
    
    return {
        "vid": vid,
        "calibrated": True,
        "court_model": cal.court_model,
        "reproj_error_px": cal.reproj_error_px,
        "keypoints_used": cal.keypoints_used,
    }


class AutoCalibrateReq(BaseModel):
    frame: int = 1
    court_model: str = "FIBA"


@app.post("/api/videos/{vid}/auto_calibrate")
def auto_calibrate_ep(vid: str, req: AutoCalibrateReq) -> dict:
    v = lib.get(vid)
    if not (v.width and v.height):
        raise HTTPException(503, "素材尚未完成索引")
    table = lib.table(vid)
    n = table.clamp(req.frame)
    got = list(lib.picker(vid).frames_at([n]))
    if not got:
        raise HTTPException(404, f"帧 {n} 解不出来")
    _pts_us, rgb = got[0][1], got[0][2]

    court_obj = court.COURTS.get(req.court_model, court.FIBA)
    cal = _auto_calibrate(rgb, court=court_obj)
    if cal is None or not cal.is_plausible(rgb.shape[1], rgb.shape[0]):
        raise HTTPException(422, "自动标定失败：单应计算不可信（球场角点塌缩或超出范围）")
    cal.save(v.dir / "calibration.json")
    return {
        "vid": vid,
        "calibrated": True,
        "court_model": cal.court_model,
        "reproj_error_px": cal.reproj_error_px,
        "keypoints_used": cal.keypoints_used,
    }


class AnalyzeReq(BaseModel):
    frm: int = 1
    to: int | None = None
    stride: int = 1
    det_thr: float = 0.3
    with_masks: bool = True


@app.post("/api/videos/{vid}/analyze")
def analyze_ep(vid: str, req: AnalyzeReq) -> dict:
    v = lib.get(vid)
    if v.width == 0 or v.height == 0:
        raise HTTPException(503, "视频还在索引中，稍等")
    table = lib.table(vid)
    shots = shot_list(vid)
    if not shots:
        raise HTTPException(400, "未检测到镜头，无法跑分析")
    to = req.to or table.nframes
    if req.frm < 1 or to > table.nframes or req.frm > to:
        raise HTTPException(400, f"帧范围无效: {req.frm}-{to}, 视频共 {table.nframes} 帧")
    job_id = f"{vid}_analyze_{int(time.time())}"
    _jobs[job_id] = {"vid": vid, "status": "running", "stage": "init", "progress": 0.0, "result": None}

    def _run():
        try:
            def prog(cur, start, end):
                _jobs[job_id]["progress"] = (cur - start) / max(end - start, 1)
            res = analyze.run(
                v.path, table, shots, v.dir,
                frm=req.frm, to=to, stride=req.stride,
                det_thr=req.det_thr, with_masks=req.with_masks,
                video_id=vid, progress=prog,
            )
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = {
                "frames": res.frames,
                "poses": res.poses,
                "segments": res.segments,
                "ms_per_frame": res.ms_per_frame,
                "ep": res.ep,
            }
            lib.reload(vid)
        except Exception as exc:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["result"] = {"error": str(exc)}

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "status": "running"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(404, "任务不存在")
    return {"job_id": job_id, **_jobs[job_id]}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(GUI / "index.html")


@app.get("/gui/{name:path}")
def gui_file(name: str) -> FileResponse:
    """前端静态文件；允许子目录（gui/vendor/three.min.js 等本地依赖）。"""
    f = (GUI / name).resolve()
    if GUI not in f.parents or not f.exists() or not f.is_file():
        raise HTTPException(404, name)
    return FileResponse(f)


__all__ = ["app", "lib"]
