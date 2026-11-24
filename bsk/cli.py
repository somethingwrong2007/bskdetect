"""命令行入口：素材体检、帧时间戳表、镜头切分与抽查拼图。"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import cv2
import numpy as np

from bsk import ingest
from bsk.perceive import shots as shotmod

TILE = (320, 180)
CAPTION_H = 34


def _cmd_probe(args) -> int:
    print(json.dumps(dataclasses.asdict(ingest.probe(args.video)), ensure_ascii=False, indent=1))
    return 0


def _cmd_manifest(args) -> int:
    info = ingest.write_pts_manifest(args.video, args.out)
    print(json.dumps(info, ensure_ascii=False))
    return 0


def _montage(report: shotmod.ShotReport, video: str, out: Path, max_shots: int) -> int:
    picks = report.shots[:max_shots]
    if not picks:
        return 0
    wanted = [s.mid_frame for s in picks]
    got = ingest.gather(video, wanted, size=TILE)

    cols = 5
    rows = math.ceil(len(picks) / cols)
    tw, th = TILE
    canvas = np.full((rows * (th + CAPTION_H), cols * tw, 3), 24, np.uint8)
    for i, shot in enumerate(picks):
        img = got.get(shot.mid_frame)
        if img is None:
            continue
        rgb = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        r, c = divmod(i, cols)
        y0 = r * (th + CAPTION_H)
        canvas[y0 : y0 + th, c * tw : (c * tw) + tw] = bgr
        tag = f"{shot.start_frame}-{shot.end_frame} c{shot.court_fill_p50:.2f}"
        cv2.putText(canvas, tag, (c * tw + 4, y0 + th + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1)
        mark = "COURT-OK" if shot.court_visible else "reject"
        cv2.putText(
            canvas, mark, (c * tw + 4, y0 + th + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            (90, 200, 90) if shot.court_visible else (70, 70, 220), 1,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return len(picks)


def _cmd_shots(args) -> int:
    report = shotmod.detect_shots(
        args.video,
        limit=args.limit,
        stride=args.stride,
        court_fill_min=args.fill_min,
        min_shot_frames=args.min_frames,
    )
    out = shotmod.write_shots(args.out, report, {"video": args.video})
    stats = {
        "shots": len(report.shots),
        "frames_scanned": report.frames_scanned,
        "cut_thresh": round(report.thresh, 4),
        "median_hist_dist": round(report.median_dist, 4),
        "court_visible_shots": sum(1 for s in report.shots if s.court_visible),
        "json": str(out),
    }
    if args.montage:
        stats["montage_tiles"] = _montage(report, args.video, Path(args.montage), args.max_shots)
        stats["montage"] = args.montage
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    return 0


def _cmd_add(args) -> int:
    from bsk.store import Library

    v = Library().add(args.video)
    print(json.dumps({"vid": v.vid, "path": v.path, "dir": str(v.dir)}, ensure_ascii=False))
    return 0


def _cmd_serve(args) -> int:
    import uvicorn

    from bsk import server

    if args.video:
        v = server.lib.add(args.video)
        server.start_index(v.vid)
        print(f"[serve] 已登记并后台建索引: {v.vid} <- {v.path}")
    print(f"[serve] http://{args.host}:{args.port}/  (调试台)")
    uvicorn.run(server.app, host=args.host, port=args.port, log_level="warning", reload=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bsk", description="bskdetect 素材与镜头体检")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="实测容器规格与帧率是否恒定")
    p.add_argument("video")
    p.set_defaults(fn=_cmd_probe)

    p = sub.add_parser("add", help="把素材登记进调试台素材库")
    p.add_argument("video")
    p.set_defaults(fn=_cmd_add)

    p = sub.add_parser("serve", help="启动浏览器调试台")
    p.add_argument("--video", default=None, help="启动即导入并后台建索引")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8760)
    p.set_defaults(fn=_cmd_serve)

    p = sub.add_parser("manifest", help="导出逐帧 PTS 真值表")
    p.add_argument("video")
    p.add_argument("out")
    p.set_defaults(fn=_cmd_manifest)

    p = sub.add_parser("shots", help="镜头切分与球场可见性门控")
    p.add_argument("video")
    p.add_argument("--out", default="outputs/shots.json")
    p.add_argument("--montage", default=None, help="抽查拼图 PNG，一次看几十个镜头")
    p.add_argument("--stride", type=int, default=5, help="隔 N 帧扫描一次，省时用")
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 帧")
    p.add_argument("--min-frames", type=int, default=8)
    p.add_argument("--fill-min", type=float, default=0.15, help="下缘带最大连通地板块占比阈值")
    p.add_argument("--max-shots", type=int, default=40)
    p.set_defaults(fn=_cmd_shots)

    ns = ap.parse_args(argv)
    return ns.fn(ns)


if __name__ == "__main__":
    raise SystemExit(main())
