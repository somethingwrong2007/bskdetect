"""WebGPU 球场线数据：采集规格、几何真值与数据集后处理。"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageDraw

from bsk.calib_seg_real import sample_camera, world_to_image_homography
from bsk.court import COURTS, Court, all_court_keypoints, court_dims
from bsk.court_3d import Court3DRenderer, CourtMesh3D
from bsk.team_palettes import TEAM_PALETTES

LEVELS = np.array([0, 51, 102, 153, 204, 255], dtype=np.int16)
VIS_COLORS = np.array([
    [0, 0, 0], [255, 255, 255], [255, 220, 60],
    [80, 220, 80], [80, 140, 255], [220, 80, 220],
], dtype=np.uint8)


def _arc(cx, cy, radius, a0, a1, n=96):
    angles = np.linspace(a0, a1, n + 1)
    return np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)])


def _dashes(cx, cy, radius, a0, a1, dash=0.38, gap=0.28):
    total = abs(a1 - a0) * radius
    direction = 1.0 if a1 >= a0 else -1.0
    result = []
    pos = 0.0
    while pos < total:
        end = min(pos + dash, total)
        result.append(_arc(cx, cy, radius, a0 + direction * pos / radius,
                           a0 + direction * end / radius, max(3, math.ceil(dash / 0.02))))
        pos += dash + gap
    return result


def line_primitives(court: Court) -> list[dict]:
    hl, hw, lw = court.half_length, court.half_width, court.line_width
    edge = court.line_half
    primitives = [
        {"id": "boundary", "class_id": 1, "world_xy": np.array([
            [-hl - edge, -hw - edge], [hl + edge, -hw - edge],
            [hl + edge, hw + edge], [-hl - edge, hw + edge], [-hl - edge, -hw - edge],
        ])},
        {"id": "center_line", "class_id": 4, "world_xy": np.array([
            [0, -hw - edge - court.center_line_extension], [0, hw + edge + court.center_line_extension],
        ])},
        {"id": "center_circle", "class_id": 5, "world_xy": _arc(0, 0, court.center_circle_center_radius, 0, 2 * math.pi, 128)},
    ]
    keypoints = {
        "mid_left": (0.0, -hw - edge), "mid_right": (0.0, hw + edge),
        "center": (0.0, 0.0),
    }
    for side, sign in (("A", 1.0), ("B", -1.0)):
        bl = sign * hl
        rim_x, _ = court.rim_pos(side)
        ft_x = bl - sign * court.free_throw_center_dist
        lane = court.paint_width / 2 + court.line_half
        corner_y = hw - court.three_pt_corner_center_dist
        radius3 = court.three_pt_center_radius
        dx = math.sqrt(max(0.0, radius3 ** 2 - corner_y ** 2))
        arc_x = rim_x - sign * dx
        half = math.asin(min(1.0, corner_y / radius3))
        restricted = court.restricted_center_radius
        back_x = bl - sign * court.backboard_from_baseline
        ft_radius = court.free_throw_circle_center_radius
        primitives.extend([
            {"id": f"{side}_paint", "class_id": 3, "world_xy": np.array([
                [bl, -lane], [ft_x, -lane], [ft_x, lane], [bl, lane], [bl, -lane],
            ])},
            {"id": f"{side}_three_corner_left", "class_id": 2, "world_xy": np.array([[bl, -corner_y], [arc_x, -corner_y]])},
            {"id": f"{side}_three_corner_right", "class_id": 2, "world_xy": np.array([[bl, corner_y], [arc_x, corner_y]])},
            {"id": f"{side}_three_arc", "class_id": 2,
             "world_xy": _arc(rim_x, 0, radius3, math.pi - half, math.pi + half, 128) if sign > 0
             else _arc(rim_x, 0, radius3, -half, half, 128)},
            {"id": f"{side}_restricted_arc", "class_id": 3,
             "world_xy": _arc(rim_x, 0, restricted, math.pi / 2, 3 * math.pi / 2, 64) if sign > 0
             else _arc(rim_x, 0, restricted, -math.pi / 2, math.pi / 2, 64)},
            {"id": f"{side}_restricted_left", "class_id": 3, "world_xy": np.array([[back_x, -restricted], [rim_x, -restricted]])},
            {"id": f"{side}_restricted_right", "class_id": 3, "world_xy": np.array([[back_x, restricted], [rim_x, restricted]])},
        ])
        if court.free_throw_dashed_inside:
            solid = _arc(ft_x, 0, ft_radius, math.pi / 2, 3 * math.pi / 2, 64) if sign > 0 \
                else _arc(ft_x, 0, ft_radius, -math.pi / 2, math.pi / 2, 64)
            dashed = _dashes(ft_x, 0, ft_radius, -math.pi / 2, math.pi / 2) if sign > 0 \
                else _dashes(ft_x, 0, ft_radius, math.pi / 2, 3 * math.pi / 2)
            primitives.append({"id": f"{side}_ft_solid", "class_id": 3, "world_xy": solid})
            for i, points in enumerate(dashed):
                primitives.append({"id": f"{side}_ft_dash_{i:02d}", "class_id": 3, "world_xy": points})
        else:
            primitives.append({"id": f"{side}_ft_circle", "class_id": 3,
                               "world_xy": _arc(ft_x, 0, ft_radius, 0, 2 * math.pi, 128)})
        keypoints.update({
            f"{side}_baseline_left": (bl, -hw - edge),
            f"{side}_baseline_right": (bl, hw + edge),
            f"{side}_three_left": (bl, -corner_y),
            f"{side}_three_right": (bl, corner_y),
            f"{side}_ft_left": (ft_x, -lane),
            f"{side}_ft_right": (ft_x, lane),
            f"{side}_restricted_left": (rim_x, -restricted),
            f"{side}_restricted_right": (rim_x, restricted),
        })
    return primitives, keypoints


def project_points(points: np.ndarray, homography: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ph = np.column_stack([points, np.ones(len(points))]) @ np.asarray(homography).T
    depth = ph[:, 2]
    valid = np.abs(depth) > 1e-9
    uv = ph[:, :2] / np.where(valid, depth, 1.0)[:, None]
    return uv, valid & np.all(np.isfinite(uv), axis=1)


def geometry_truth(court: Court, homography: np.ndarray, label: np.ndarray) -> dict:
    h, w = label.shape
    distances = {cid: cv2.distanceTransform((label != cid).astype(np.uint8), cv2.DIST_L2, 3)
                 for cid in range(1, 6)}
    primitives, keypoint_world = line_primitives(court)
    output = []
    visible_samples = total_samples = 0
    for primitive in primitives:
        world = primitive["world_xy"]
        pixel, valid = project_points(world, homography)
        indices = np.linspace(0, len(world) - 1, min(len(world), 160)).astype(int)
        sampled_visible = []
        for uv, front in zip(pixel[indices], valid[indices]):
            x, y = uv
            inside = front and 0 <= x < w and 0 <= y < h
            visible = inside and distances[primitive["class_id"]][int(y), int(x)] <= 3.0
            sampled_visible.append(bool(visible))
        fraction = float(np.mean(sampled_visible)) if sampled_visible else 0.0
        visible_samples += int(sum(sampled_visible))
        total_samples += len(sampled_visible)
        output.append({
            "id": primitive["id"], "class_id": primitive["class_id"],
            "world_xy": world.round(6).tolist(), "pixel_xy": pixel.round(4).tolist(),
            "visible_fraction": round(fraction, 5),
        })
    keypoints = []
    for name, world in keypoint_world.items():
        pixel, valid = project_points(np.array([world]), homography)
        x, y = pixel[0]
        xi, yi = int(round(x)), int(round(y))
        inside = bool(valid[0] and 0 <= xi < w and 0 <= yi < h)
        visible = inside and any(distances[cid][yi, xi] <= 3.0 for cid in range(1, 6))
        keypoints.append({"name": name, "world_xy": list(world), "pixel_xy": pixel[0].round(4).tolist(),
                          "in_frame": inside, "visible": visible})
    classes = sorted({p["class_id"] for p in output if p["visible_fraction"] > 0.02})
    visible_keypoints = sum(k["visible"] for k in keypoints)
    return {
        "line_primitives": output,
        "keypoints": keypoints,
        "visible_classes": classes,
        "visible_keypoints": visible_keypoints,
        "visible_line_fraction": round(visible_samples / max(total_samples, 1), 5),
        "calibratable_gt": len(classes) >= 2 and visible_keypoints >= 4,
    }


def generate_specs(output: Path, n: int, size: tuple[int, int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    renderers = {model: Court3DRenderer(CourtMesh3D.build(court)) for model, court in COURTS.items() if model in ("FIBA", "NBA")}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for idx in range(n):
            court = COURTS["FIBA" if idx % 2 == 0 else "NBA"]
            team_pool = [t for t in TEAM_PALETTES if t["model"] == court.model]
            team = team_pool[(idx // 2) % len(team_pool)]
            mode = sample_camera(size, court, rng)["mode"]
            for _ in range(24):
                camera = sample_camera(size, court, rng, mode=mode)
                homography = world_to_image_homography(renderers[court.model], camera)
                keypoints = all_court_keypoints(court)
                world = np.array([kp.pos for kp in keypoints], dtype=np.float64)
                pixel, valid = project_points(world, homography)
                h, w = size
                inside = valid & (pixel[:, 0] >= 0) & (pixel[:, 0] < w) & (pixel[:, 1] >= 0) & (pixel[:, 1] < h)
                min_inside = 8 if camera["mode"] == "midcourt" else 6
                rim_visible = any(kp.name.endswith("_rim") and ok for kp, ok in zip(keypoints, inside))
                if (
                    inside.sum() >= min_inside
                    and np.ptp(pixel[inside], axis=0).min() > 20
                    and (camera["mode"] != "midcourt" or rim_visible)
                ):
                    break
            split_draw = float(rng.random())
            split = "val" if split_draw < 0.10 else "test" if split_draw < 0.20 else "train"
            spec = {
                "idx": idx, "split": split, "court_model": court.model, "team": team["name"],
                "camera_mode": camera["mode"], "camera": {
                    "pos": list(camera["pos"]), "look_at": list(camera["look_at"]),
                    "up": list(camera["up"]), "focal_length": camera["focal_length"],
                },
            }
            stream.write(json.dumps(spec, ensure_ascii=False) + "\n")
    print(f"specs={n} size={size[1]}x{size[0]} seed={seed} output={output}")


def _decode_label(raw_rgb: np.ndarray) -> np.ndarray:
    """兼容旧红色通道编码与新的亮色调色板编码。"""
    rgb = np.asarray(raw_rgb, dtype=np.uint8)
    if rgb.ndim == 2:
        red = rgb
    else:
        rgb = rgb[..., :3]
        if int(rgb[..., 1:].max(initial=0)) <= 2:
            red = rgb[..., 0]
        else:
            r = rgb[..., 0].astype(np.int32)
            g = rgb[..., 1].astype(np.int32)
            b = rgb[..., 2].astype(np.int32)
            label = np.zeros(r.shape, dtype=np.uint8)
            best = np.full(r.shape, 3 * 255 * 255 + 1, dtype=np.int32)
            for cid, color in enumerate(VIS_COLORS.astype(np.int32)):
                dist = (r - color[0]) ** 2 + (g - color[1]) ** 2 + (b - color[2]) ** 2
                take = dist < best
                best[take] = dist[take]
                label[take] = cid
            return label
    return np.abs(red.astype(np.int16)[..., None] - LEVELS).argmin(axis=2).astype(np.uint8)


def _save_label(path: Path, label: np.ndarray) -> None:
    palette = VIS_COLORS.reshape(-1).tolist() + [0] * (768 - 18)
    image = Image.fromarray(label, mode="P")
    image.putpalette(palette)
    try:
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    except Exception:
        profile = None
    image.save(path, **({"icc_profile": profile} if profile else {}))


def finalize(raw_dir: Path, out_dir: Path) -> None:
    records = [json.loads(line) for line in (raw_dir / "meta.jsonl").read_text(encoding="utf-8").splitlines() if line]
    models = ("FIBA", "NBA")
    for model in models:
        for split in ("train", "val", "test"):
            for kind in ("images", "masks"):
                (out_dir / model / split / kind).mkdir(parents=True, exist_ok=True)
    split_meta = {model: {split: [] for split in ("train", "val", "test")} for model in models}
    skipped = []
    for record in records:
        idx = record["idx"]
        raw = np.asarray(Image.open(raw_dir / "labels" / f"{idx:06d}.png").convert("RGB"))
        label = _decode_label(raw)
        boundary_px = int((label == 1).sum())
        court = COURTS[record["court_model"]]
        truth = geometry_truth(court, np.array(record["H_world_to_px"]), label)
        if not truth["visible_classes"]:
            skipped.append(idx)
            continue
        record.update(truth)
        record["boundary_px"] = boundary_px
        record["bench_occlusion_suspected"] = record["camera_mode"] in ("courtside", "midcourt") and truth["visible_line_fraction"] < 0.92
        record["target_cm"] = 10.0
        split = record["split"]
        model = record["court_model"]
        model_dir = out_dir / model
        image_name = f"{idx:06d}.jpg"
        mask_name = f"{idx:06d}.png"
        source = raw_dir / "images" / image_name
        target = model_dir / split / "images" / image_name
        if target.exists():
            target.unlink()
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        _save_label(model_dir / split / "masks" / mask_name, label)
        split_meta[model][split].append(record)
    counts = {}
    for model in models:
        config = {
            "court_model": model,
            "court_dims": court_dims(COURTS[model]),
            "classes": ["background", "boundary", "three_point", "paint", "center_line", "center_circle"],
            "renderer": "three-webgpu-r170", "target_error_m": 0.10,
            "eval_grid_step_m": 1.0,
        }
        (out_dir / model / "dataset.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        for split, meta in split_meta[model].items():
            (out_dir / model / split / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            counts[f"{model}/{split}"] = len(meta)
    print(f"finalized={len(records) - len(skipped)} skipped={skipped} {json.dumps(counts)}")


def audit(out_dir: Path) -> None:
    all_records = []
    model_counts = {}
    for model in ("FIBA", "NBA"):
        records = []
        for split in ("train", "val", "test"):
            records.extend(json.loads((out_dir / model / split / "meta.json").read_text(encoding="utf-8")))
        all_records.extend(records)
        model_counts[model] = len(records)
    records = all_records
    modes = Counter(r["camera_mode"] for r in records)
    bench = sum(r["bench_occlusion_suspected"] for r in records)
    calibratable = sum(r["calibratable_gt"] for r in records)
    boundary = [int(r.get("boundary_px", 0)) for r in records]
    boundary_present = sum(v > 20 for v in boundary) / max(len(boundary), 1)
    assert records and all(r["visible_classes"] for r in records)
    assert all(r["H_world_to_px"] and len(r["line_primitives"]) >= 15 for r in records)
    sheet = Image.new("RGB", (960, 12 * 205), (24, 28, 35))
    draw = ImageDraw.Draw(sheet)
    for cell, record in enumerate(records[:12]):
        idx = record["idx"]
        split = record["split"]
        model = record["court_model"]
        image = Image.open(out_dir / model / split / "images" / f"{idx:06d}.jpg").convert("RGB").resize((240, 135))
        mask = Image.open(out_dir / model / split / "masks" / f"{idx:06d}.png").convert("RGB").resize((240, 135))
        x, y = (cell % 4) * 240, (cell // 4) * 205
        sheet.paste(image, (x, y))
        sheet.paste(mask, (x, y + 135))
        draw.text((x + 4, y + 171), f"{idx:06d} {record['court_model']} {record['camera_mode']}", fill="white")
    sheet.save(out_dir / "audit.jpg", quality=92)
    print(json.dumps({
        "samples": len(records), "models": model_counts, "camera_modes": modes, "bench_occlusion_suspected": bench,
        "calibratable_gt": calibratable,
        "boundary_present_ratio": round(boundary_present, 4),
        "boundary_px_p10": float(np.percentile(boundary, 10)),
        "visible_line_fraction_p10": float(np.percentile([r["visible_line_fraction"] for r in records], 10)),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    specs = sub.add_parser("specs")
    specs.add_argument("--out", type=Path, required=True)
    specs.add_argument("--n", type=int, default=12000)
    specs.add_argument("--size", default="540x960")
    specs.add_argument("--seed", type=int, default=20261125)
    final = sub.add_parser("finalize")
    final.add_argument("--raw", type=Path, required=True)
    final.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("audit")
    check.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "specs":
        h, w = (int(v) for v in args.size.lower().split("x"))
        generate_specs(args.out, args.n, (h, w), args.seed)
    elif args.command == "finalize":
        finalize(args.raw, args.out)
    else:
        audit(args.data)


if __name__ == "__main__":
    main()
