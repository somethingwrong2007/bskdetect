"""研究对照：MediaPipe BlazePose GHUM（Apache-2.0，唯一许可干净的学习法 3D 人体模型）
在项目真实转播帧上的可用性 / 延迟 / 输出量级基准。

**研究脚本，不进生产链路**。mediapipe 装在 .tools/mp（隔离目录，不动 .venv）。

为什么测它：MediaPipe Pose Landmarker 是公开的"图像 → 33 关键点 + 世界坐标(米、
骨盆原点)"模型，模型卡明示 Apache-2.0；但同一张模型卡也写了两条 out-of-scope：
"Multiple people in an image" 与 "People too far away from the camera (>4m)"，
以及"Applications requiring metric accurate depth"。本项目是 10~30m 的多人转播，
必须实测它在这两条边界外的表现。

用法：
    .venv/Scripts/python.exe docs/research/pose_mp_bench.py --frames 24
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
import ssl
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".tools" / "mp"))

TASK_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_{m}/float16/1/pose_landmarker_{m}.task")
TASK_PATH = Path("data/models/pose_landmarker_{m}.task")

# MediaPipe 33 点 → COCO17 的子集（BlazePose 索引 → COCO 索引）
MP2COCO = {0: 0, 2: 1, 5: 2, 7: 3, 8: 4, 11: 5, 12: 6, 13: 7, 14: 8,
           15: 9, 16: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16}
COCO17_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle")
EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6))


MODEL = "full"


def ensure_model() -> Path:
    path = Path(str(TASK_PATH).format(m=MODEL))
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        d = urllib.request.urlopen(str(TASK_URL).format(m=MODEL), timeout=90, context=ctx).read()
        path.write_bytes(d)
        print(f"[下载] {path} ({len(d)/1e6:.1f} MB)")
    return path


def load_landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    opt = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3)
    return mp, vision.PoseLandmarker.create_from_options(opt)


def to_coco(lms33):
    """33 个 (x,y,z) → COCO17 子集 (17,3)，缺失点填 NaN。"""
    out = np.full((17, 3), np.nan)
    for mp_i, c_i in MP2COCO.items():
        if mp_i < len(lms33):
            out[c_i] = lms33[mp_i]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--model", default="full", choices=["lite", "full", "heavy"])
    args = ap.parse_args()
    global MODEL
    MODEL = args.model

    from eval_pose_filter import collect, VID
    from bsk import ingest
    from bsk.store import Library
    from bsk.perceive.pose_3d import Pose3DLifter
    from bsk.court import FIBA
    import cv2

    mp, lm = load_landmarker()
    print(f"[MediaPipe] {mp.__version__} 模型 {ensure_model().name}")

    frame_ids, obs, w, h = collect(args.start, args.frames, 960)
    lib = Library()
    v = lib.get(VID)
    table = lib.table(VID)
    anchor = table.key_before(args.start)
    rgbs = {}
    for idx, _p, rgb in ingest.frames(v.path, start=anchor, size=(w, h), stride=1,
                                      seek_us=table.pts_us[anchor - 1]):
        if idx < args.start:
            continue
        if idx >= args.start + args.frames:
            break
        rgbs[idx] = np.asarray(rgb)
    print(f"[数据] {len(frame_ids)} 帧，{sum(len(o) for o in obs)} 个人体观测")

    # 标定：项目 calibration.json（当前不可信，只做量级检查）
    H = None
    try:
        import json
        d = json.loads(Path("data/derived/harden-55/calibration.json").read_text("utf-8"))
        H = np.array(d["H"], dtype=np.float64)
    except Exception:
        pass
    geo = Pose3DLifter(court=FIBA, image_size=(w, h))

    rows = []
    lat = []
    for fid, ob in zip(frame_ids, obs):
        rgb = rgbs.get(fid)
        if rgb is None:
            continue
        for tid, (box, kpts) in list(ob.items())[:4]:
            x0, y0, x1, y1 = [float(x) for x in box]
            bw, bh = x1 - x0, y1 - y0
            mx, my = 0.25 * bw, 0.25 * bh
            cx0 = int(max(0, x0 - mx)); cy0 = int(max(0, y0 - my))
            cx1 = int(min(w - 1, x1 + mx)); cy1 = int(min(h - 1, y1 + my))
            if cx1 - cx0 < 24 or cy1 - cy0 < 24:
                continue
            crop = rgb[cy0:cy1, cx0:cx1]
            # MediaPipe 内部 ROI 假定方形输入（否则 NORM_RECT 投影会偏），补成正方形
            ch, cw = crop.shape[:2]
            side = int(max(ch, cw) * 1.15)
            px0 = (side - cw) // 2
            py0 = (side - ch) // 2
            crop = cv2.copyMakeBorder(crop, py0, side - ch - py0, px0, side - cw - px0,
                                      cv2.BORDER_REPLICATE)
            crop = np.ascontiguousarray(crop)
            t0 = time.perf_counter()
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
            res = lm.detect(img)
            ms = (time.perf_counter() - t0) * 1000
            lat.append(ms)
            if not res.pose_world_landmarks:
                rows.append(dict(fid=fid, tid=tid, ok=False, ms=ms, box_h=bh))
                continue
            lms = res.pose_world_landmarks[0]                      # 世界坐标（米，骨盆原点）
            world = np.array([[p.x, p.y, p.z, getattr(p, "visibility", 1.0)] for p in lms])
            coco = to_coco(world[:, :3])
            # 归一化图像坐标 → 像素（用于和 RTMO 2D 对照）
            img_lms = res.pose_landmarks[0] if res.pose_landmarks else None
            pix = None
            if img_lms is not None:
                pix = np.array([[p.x * side + cx0 - px0, p.y * side + cy0 - py0]
                                for p in img_lms])
            k = np.asarray(kpts, dtype=np.float64)
            # 2D 一致性：两个检测器在同一人体上的同名字典点差（像素，按 bbox 高度归一）
            d2 = np.nan
            if pix is not None:
                idx = [(11, 5), (12, 6), (23, 11), (24, 12), (25, 13), (26, 14),
                       (27, 15), (28, 16), (13, 7), (14, 8), (15, 9), (16, 10)]
                diffs = [np.linalg.norm(pix[a] - k[b, :2]) for a, b in idx
                         if np.isfinite(pix[a]).all() and np.isfinite(k[b, :2]).all()]
                if diffs:
                    d2 = float(np.median(diffs)) / max(bh, 1.0)
            # 世界坐标的骨长（米）：BlazePose 自己声称的"米制"
            bones = {}
            for name, (a, b) in (("肩宽", (11, 12)), ("躯干", (23, 24)),
                                 ("大腿", (23, 25)), ("小腿", (25, 27)),
                                 ("上臂", (11, 13)), ("前臂", (13, 15))):
                bones[name] = float(np.linalg.norm(world[a, :3] - world[b, :3]))
            body_h = float(world[:, 2].max() - world[:, 2].min())     # y 向上? 见下
            rows.append(dict(fid=fid, tid=tid, ok=True, ms=ms, box_h=bh, bones=bones,
                             d2=d2, world=world[:, :3].copy(), coco=coco,
                             body_h=body_h, pix=pix, k=k))
    ok = [r for r in rows if r["ok"]]
    print(f"\n[结果] 成功 {len(ok)}/{len(rows)}；延迟 中位 {np.median(lat):.1f} ms"
          f"（p90 {np.percentile(lat, 90):.1f} ms，含 25% 外扩裁剪的整帧解码不含）")
    if not ok:
        return
    print(f"  人体框高 中位 {np.median([r['box_h'] for r in rows]):.0f}px"
          f"（10m 外转播里通常 60~250px）")
    for name in ("肩宽", "躯干", "大腿", "小腿", "上臂", "前臂"):
        vals = np.array([r["bones"][name] for r in ok])
        print(f"  BlazePose 世界坐标 {name}: 中位 {np.median(vals):.3f}m "
              f"p10 {np.percentile(vals,10):.3f} p90 {np.percentile(vals,90):.3f}")
    print("  （真人参考：肩宽≈0.40m、髋宽≈0.25m、大腿≈0.42m、小腿≈0.43m、"
          "上臂≈0.30m、前臂≈0.26m；骨盆原点系里躯干是 y 轴，不能直接用")
    d2 = np.array([r["d2"] for r in ok if np.isfinite(r["d2"])])
    if len(d2):
        print(f"  MediaPipe vs RTMO 同名 2D 点偏差：中位 {100*d2.mean():.1f}% 框高"
              f"（>10% 说明裁剪/尺度下两个检测器分歧大）")
    # 骨盆原点系里的“腿长”（y 向上则 foot y 为负）
    legs = np.array([abs(r["world"][[27, 28], 1].mean()) for r in ok])
    print(f"  |骨盆→踝|（世界坐标 y）: 中位 {np.median(legs):.2f}m（真人≈0.85m）")
    # 与几何法做"形状"对照：把两者都转到骨盆原点后比较腿长/腕高
    if H is not None:
        diffs = []
        for r in ok:
            res = geo.lift_result(r["k"], None, H, r["tid"], 0)
            geo.reset()
            if res is None:
                continue
            g = res.kpts3d - 0.5 * (res.kpts3d[11] + res.kpts3d[12])
            diffs.append(float(np.linalg.norm(g[15] - g[16])))
        if diffs:
            print(f"  几何法同批样本 |左踝−右踝|（骨盆原点）中位 {np.median(diffs):.2f}m")


if __name__ == "__main__":
    main()