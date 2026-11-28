"""Prepare Roboflow court labels, train the project's 16-point model, and export ONNX.

The public ``basketball-court-detection-2`` dataset uses 33 numbered landmarks.
This module keeps only the 16 landmarks required by ``court_keypoints.KEYPOINT_IDS``.
The mapping was recovered by aligning all 850 images to one projective court
coordinate system and matching the resulting geometry to the FIBA contract.

Usage::

    python -m bsk.train_court_keypoints prepare --source data/roboflow/.../extracted \
      --output data/derived/court_keypoints16_official
    python -m bsk.train_court_keypoints regroup \
      --source data/derived/court_keypoints16_official \
      --output data/derived/court_keypoints16_grouped
    python -m bsk.train_court_keypoints train \
      --data data/derived/court_keypoints16_grouped
    python -m bsk.train_court_keypoints export --checkpoint data/models/court_keypoints16.pt

Only the existing training extra (torch/torchvision/onnx) is required. Dataset
images, checkpoints and exported weights stay under ignored data directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from bsk.court import FIBA
from bsk.perceive.court_keypoints import (
    KEYPOINT_IDS,
    KEYPOINT_SPECS,
    WORLD_POINTS,
    COURT_SEGMENTS,
    CourtKeypointTracker,
    DetectedKeypoint,
    KeypointConfig,
)

INPUT_HW = (640, 640)
OUTPUT_HW = (320, 320)
OUTPUT_SCALE = 2
DEFAULT_DATA = Path("data/derived/court_keypoints16_grouped")
DEFAULT_MODEL = Path("data/models/court_keypoints16")

# Dataset court coordinates: x runs from the left baseline to the right baseline;
# y runs from the near sideline to the far sideline.  The names below are the
# source COCO keypoint names, in the exact order required by KEYPOINT_IDS.
SOURCE_TO_TARGET = {
    "corner_A_near": "01",
    "corner_B_near": "08",
    "corner_A_far": "41",
    "corner_B_far": "34",
    "mid_near": "19",
    "mid_far": "23",
    "ft_A_near": "28",
    "ft_A_far": "30",
    "ft_B_near": "12",
    "ft_B_far": "14",
    "three_corner_A_near": "35",
    "three_corner_A_far": "40",
    "three_corner_B_near": "02",
    "three_corner_B_far": "07",
    "three_apex_A": "26",
    "three_apex_B": "16",
}
SOURCE_KEYPOINTS = (
    "01", "02", "04", "05", "07", "08", "09", "10", "11", "12", "13", "14",
    "15", "16", "17", "19", "21", "23", "25", "26", "27", "28", "29", "30",
    "31", "32", "33", "34", "35", "37", "38", "40", "41",
)
# Roboflow filenames contain the source game, optional quarter, clip timestamp,
# extracted frame number and a Roboflow hash. Frames from one game must never be
# distributed across train/valid/test, even when their file hashes differ.
VIDEO_NAME_RE = re.compile(r"^(.*?)(?:-q[1-4])?-\d+_\d+-\d+_\d+_mp4-")


@dataclass(frozen=True)
class Sample:
    path: Path
    keypoints: np.ndarray  # (16, 3), COCO x/y/visibility in source pixels
    source_hw: tuple[int, int]
    metric_keypoints: np.ndarray | None = None  # original sparse human labels


def _keypoint_category(data: dict) -> dict:
    return next(category for category in data["categories"] if "keypoints" in category)


def convert_split(source: Path, destination: Path) -> dict:
    """Convert one COCO split and copy its images into an ignored derived directory."""
    annotation_path = source / "_annotations.coco.json"
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    category = _keypoint_category(data)
    source_names = tuple(category["keypoints"])
    if source_names != SOURCE_KEYPOINTS:
        raise ValueError(f"unexpected 33-point schema in {annotation_path}: {source_names}")
    source_index = {name: index for index, name in enumerate(source_names)}
    selected = [source_index[SOURCE_TO_TARGET[name]] for name in KEYPOINT_IDS]

    destination.mkdir(parents=True, exist_ok=True)
    annotations = []
    for annotation in data["annotations"]:
        raw = np.asarray(annotation["keypoints"], np.float32).reshape(-1, 3)
        if len(raw) != len(source_names):
            raise ValueError(f"annotation {annotation['id']} has {len(raw)} keypoints")
        points = raw[selected].copy()
        # Off-screen points must not teach the model a high-confidence prediction
        # at a clipped image border.  COCO visibility 0 is the runtime absence cue.
        width, height = annotation.get("width", 0), annotation.get("height", 0)
        if not width or not height:
            image = next(image for image in data["images"] if image["id"] == annotation["image_id"])
            width, height = image["width"], image["height"]
        outside = ((points[:, 0] < 0) | (points[:, 0] >= width) |
                   (points[:, 1] < 0) | (points[:, 1] >= height))
        points[outside] = (0, 0, 0)
        item = {key: value for key, value in annotation.items() if key != "keypoints"}
        item["keypoints"] = points.reshape(-1).tolist()
        item["num_keypoints"] = int(np.count_nonzero(points[:, 2] > 0))
        annotations.append(item)

    for image in data["images"]:
        src = source / image["file_name"]
        dst = destination / image["file_name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        if cv2.imread(str(dst)) is None:
            raise ValueError(f"cannot decode copied image: {dst}")

    output = {
        "info": {
            "description": "FIBA 16-point contract converted from Roboflow basketball-court-detection-2 v22",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "source": "https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2",
            "changes": "Selected and reordered 16 of 33 keypoints; off-screen points set to visibility 0.",
            "mapping": SOURCE_TO_TARGET,
        },
        "licenses": data.get("licenses", []),
        "images": data["images"],
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "court",
            "supercategory": "basketball-court",
            "keypoints": list(KEYPOINT_IDS),
        }],
    }
    (destination / "_annotations.coco.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"split": source.name, "images": len(output["images"]),
            "annotations": len(annotations), "destination": str(destination)}


def prepare(args: argparse.Namespace) -> None:
    source, destination = Path(args.source), Path(args.output)
    summaries = [convert_split(source / split, destination / split)
                 for split in ("train", "valid", "test")]
    (destination / "dataset.json").write_text(
        json.dumps({"input_hw": list(INPUT_HW), "keypoint_ids": list(KEYPOINT_IDS),
                    "mapping": SOURCE_TO_TARGET, "splits": summaries},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def _render_synthetic_court(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Render one full-court broadcast-like frame with exact projective labels."""
    size = INPUT_HW[0]
    style = int(rng.integers(0, 3))
    if style == 0:
        quad = np.asarray([[65, 135], [575, 105], [615, 535], [25, 555]], np.float32)
    elif style == 1:
        quad = np.asarray([[175, 55], [465, 55], [600, 575], [40, 575]], np.float32)
    else:
        quad = np.asarray([[85, 65], [555, 65], [575, 565], [65, 565]], np.float32)
    quad += rng.uniform(-22, 22, quad.shape).astype(np.float32)
    world_corners = np.asarray([
        [-FIBA.half_length, -FIBA.half_width], [FIBA.half_length, -FIBA.half_width],
        [FIBA.half_length, FIBA.half_width], [-FIBA.half_length, FIBA.half_width],
    ], np.float32)
    matrix = cv2.getPerspectiveTransform(world_corners, quad)

    def project(points):
        return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2),
                                        matrix).reshape(-1, 2)

    floor = np.asarray(rng.uniform([92, 118, 145], [142, 165, 190]), np.float32)
    image = np.zeros((size, size, 3), np.float32)
    image[:] = rng.uniform(18, 55)
    cv2.fillConvexPoly(image, quad.astype(np.int32), floor.tolist(), cv2.LINE_AA)
    # Long floorboards and a faint center logo make the surface less template-like.
    for y in np.linspace(-FIBA.half_width, FIBA.half_width, 34):
        shade = rng.uniform(-13, 13)
        cv2.line(image, tuple(project([[-FIBA.half_length, y]])[0].astype(int)),
                 tuple(project([[FIBA.half_length, y]])[0].astype(int)),
                 tuple(int(value) for value in np.clip(floor + shade, 0, 255)), 1, cv2.LINE_AA)
    for x in np.linspace(-FIBA.half_length, FIBA.half_length, 22):
        cv2.line(image, tuple(project([[x, -FIBA.half_width]])[0].astype(int)),
                 tuple(project([[x, FIBA.half_width]])[0].astype(int)),
                 tuple(int(value) for value in np.clip(floor * 0.93, 0, 255)), 1, cv2.LINE_AA)

    white = tuple(int(value) for value in rng.uniform(205, 245, 3))
    line_width = int(rng.integers(2, 5))
    for start, end in COURT_SEGMENTS:
        a, b = project([start, end])
        distance = float(np.linalg.norm(b - a))
        pieces = max(1, int(distance / 45))
        for piece in range(pieces):
            if rng.random() < 0.07:
                continue
            p0 = a + (b - a) * (piece / pieces)
            p1 = a + (b - a) * ((piece + 0.94) / pieces)
            cv2.line(image, tuple(p0.astype(int)), tuple(p1.astype(int)),
                     white, line_width, cv2.LINE_AA)
    for center, radius in (((0.0, 0.0), 1.8),
                           ((FIBA.rim_pos("A")[0], 0.0), 1.8),
                           ((FIBA.rim_pos("B")[0], 0.0), 1.8)):
        angles = np.linspace(0, 2 * np.pi, 80)
        circle = project(np.column_stack([center[0] + radius * np.cos(angles),
                                          center[1] + radius * np.sin(angles)]))
        cv2.polylines(image, [circle.astype(np.int32)], False, white, line_width, cv2.LINE_AA)

    pixels = project(WORLD_POINTS)
    occluded = np.zeros(len(KEYPOINT_IDS), bool)
    for _ in range(int(rng.integers(0, 5))):
        world = np.asarray([rng.uniform(-14, 14), rng.uniform(-12, 12)], np.float32)
        foot = project([world])[0]
        scale = float(rng.uniform(0.75, 1.25))
        cv2.ellipse(image, tuple(foot.astype(int)), (int(9 * scale), int(4 * scale)),
                    0, 0, 360, (25, 25, 25), -1, cv2.LINE_AA)
        jersey = tuple(int(value) for value in rng.uniform(150, 245, 3))
        cv2.ellipse(image, (int(foot[0]), int(foot[1] - 18 * scale)),
                    (int(7 * scale), int(13 * scale)), 0, 0, 360, jersey, -1, cv2.LINE_AA)
        cv2.circle(image, (int(foot[0]), int(foot[1] - 33 * scale)),
                   int(5 * scale), (175, 125, 95), -1, cv2.LINE_AA)
        occluded |= np.linalg.norm(pixels - foot, axis=1) < 24 * scale

    # Mild glare, sensor noise and blur approximate broadcast-domain variation.
    glare = rng.uniform(-0.10, 0.16)
    gradient = np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :, None]
    image = image * (1.0 + glare * gradient) + rng.uniform(-10, 10)
    image += rng.normal(0, rng.uniform(1.0, 4.0), image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < size) &
              (pixels[:, 1] >= 0) & (pixels[:, 1] < size) & ~occluded)
    points = np.column_stack([pixels, inside.astype(np.float32) * 2.0]).astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), points


def generate_synthetic(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    root = Path(args.output) / args.split
    root.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for index in range(args.count):
        image, points = _render_synthetic_court(rng)
        file_name = f"synthetic_{index:05d}.jpg"
        cv2.imwrite(str(root / file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        images.append({"id": index, "file_name": file_name,
                       "width": INPUT_HW[1], "height": INPUT_HW[0]})
        annotations.append({"id": index, "image_id": index, "category_id": 1,
                            "bbox": [0, 0, INPUT_HW[1], INPUT_HW[0]], "area": INPUT_HW[0] * INPUT_HW[1],
                            "iscrowd": 0, "keypoints": points.reshape(-1).tolist(),
                            "num_keypoints": int(np.count_nonzero(points[:, 2] > 0))})
    output = {"info": {"description": "Synthetic FIBA full-court keypoint supervision",
                       "license": "Apache-2.0"},
              "images": images, "annotations": annotations,
              "categories": [{"id": 1, "name": "court", "supercategory": "basketball-court",
                              "keypoints": list(KEYPOINT_IDS)}]}
    (root / "_annotations.coco.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    (Path(args.output) / "dataset.json").write_text(json.dumps(
        {"input_hw": list(INPUT_HW), "keypoint_ids": list(KEYPOINT_IDS),
         "split": args.split, "synthetic_images": args.count, "seed": args.seed},
        indent=2), encoding="utf-8")
    print(f"[synthetic] {args.count} images -> {root}")


def load_split(root: Path, split: str) -> list[Sample]:
    root = root / split
    data = json.loads((root / "_annotations.coco.json").read_text(encoding="utf-8"))
    category = _keypoint_category(data)
    if tuple(category["keypoints"]) != KEYPOINT_IDS:
        raise ValueError(f"{root} keypoint order does not match the runtime contract")
    images = {image["id"]: image for image in data["images"]}
    samples = []
    for annotation in data["annotations"]:
        image = images[annotation["image_id"]]
        sparse = annotation.get("source_keypoints", annotation["keypoints"])
        samples.append(Sample(root / image["file_name"],
                              np.asarray(annotation["keypoints"], np.float32).reshape(-1, 3),
                              (int(image["height"]), int(image["width"])),
                              np.asarray(sparse, np.float32).reshape(-1, 3)))
    if not samples:
        raise ValueError(f"no samples in {root}")
    return samples


def _fit_label_homography(points: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    """Fit world→source-pixels from sparse human labels and reject bad geometry."""
    visible = ((points[:, 2] > 0) & np.isfinite(points[:, :2]).all(axis=1) &
               (points[:, 0] >= 0) & (points[:, 1] >= 0))
    indices = np.flatnonzero(visible)
    errors = np.empty(0, np.float32)
    if len(indices) < 4:
        return None, errors
    # Four mathematically valid correspondences can still be collinear. Such a
    # fit has zero residual but explodes when extrapolating the missing points.
    max_triangle_area = 0.0
    for first, second, third in combinations(indices.tolist(), 3):
        a, b, c = WORLD_POINTS[[first, second, third]]
        area = abs(float((b[0] - a[0]) * (c[1] - a[1]) -
                         (b[1] - a[1]) * (c[0] - a[0]))) / 2.0
        max_triangle_area = max(max_triangle_area, area)
    if max_triangle_area < 40.0:
        return None, errors
    world = WORLD_POINTS[indices].astype(np.float32)
    pixels = points[indices, :2].astype(np.float32)
    method = 0 if len(indices) == 4 else cv2.RANSAC
    # A zero RANSAC threshold can reject exact four-point fits because float32
    # reprojection is a fraction of a pixel rather than mathematically zero.
    threshold = 1.0 if len(indices) == 4 else 8.0
    matrix, inliers = cv2.findHomography(world, pixels, method, threshold, maxIters=1000)
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return None, errors
    keep = inliers.reshape(-1) > 0
    # A homography is homogeneous: its condition number must be checked after
    # scale normalization, otherwise a perfectly good pixel translation alone
    # can make the raw 3x3 matrix appear ill-conditioned.
    normalized = matrix / np.linalg.norm(matrix)
    if int(keep.sum()) < 4 or np.linalg.cond(normalized) > 1e4:
        return None, errors
    projected = cv2.perspectiveTransform(world[keep].reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected - pixels[keep], axis=1)
    # Four labels determine H exactly, so only multi-label frames have a residual
    # check. The threshold is deliberately loose enough for hand annotation but
    # rejects wrong semantic mappings before dense pseudo-labels are generated.
    if len(errors) > 4 and (float(np.median(errors)) > 6.0 or
                            float(np.percentile(errors, 90)) > 12.0):
        return None, errors
    return matrix, errors


def _densify_split_single(source: Path, destination: Path) -> dict:
    """Single-frame fallback for labels that already have 2D court support."""
    data = json.loads((source / "_annotations.coco.json").read_text(encoding="utf-8"))
    images = {image["id"]: image for image in data["images"]}
    annotations = []
    residuals: list[float] = []
    visibility_hist: dict[int, int] = {}
    skipped = 0
    destination.mkdir(parents=True, exist_ok=True)
    for annotation in data["annotations"]:
        image = images[annotation["image_id"]]
        width, height = int(image["width"]), int(image["height"])
        sparse = np.asarray(annotation.get("source_keypoints", annotation["keypoints"]),
                            np.float32).reshape(-1, 3).copy()
        outside = ((sparse[:, 0] < 0) | (sparse[:, 0] >= width) |
                   (sparse[:, 1] < 0) | (sparse[:, 1] >= height))
        sparse[outside] = (0, 0, 0)
        matrix, errors = _fit_label_homography(sparse)
        if matrix is None:
            skipped += 1
            continue
        projected = cv2.perspectiveTransform(
            WORLD_POINTS.reshape(-1, 1, 2).astype(np.float32), matrix
        ).reshape(-1, 2)
        inside = ((projected[:, 0] >= 0) & (projected[:, 0] < width) &
                  (projected[:, 1] >= 0) & (projected[:, 1] < height) &
                  np.isfinite(projected).all(axis=1))
        dense = np.column_stack([projected, inside.astype(np.float32) * 2.0])
        # Keep direct human annotations where they exist; fill only absent points.
        fill = (sparse[:, 2] == 0) & inside
        dense[fill, :2] = projected[fill]
        residuals.extend(float(value) for value in errors)
        visible = int(np.count_nonzero(dense[:, 2] > 0))
        visibility_hist[visible] = visibility_hist.get(visible, 0) + 1
        item = {key: value for key, value in annotation.items() if key != "keypoints"}
        item.update({"keypoints": dense.reshape(-1).tolist(),
                     "source_keypoints": sparse.reshape(-1).tolist(),
                     "num_keypoints": visible,
                     "source_num_keypoints": int(np.count_nonzero(sparse[:, 2] > 0)),
                     "label_densified": True})
        annotations.append(item)
        target = destination / image["file_name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        src = source / image["file_name"]
        if not target.exists():
            shutil.copy2(src, target)

    accepted_images = {item["image_id"] for item in annotations}
    output = {
        "info": {
            "description": "Geometry-densified FIBA 16-point labels with sparse human labels retained",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "source": "https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2",
            "changes": "Absent in-frame points derived from a FIBA homography fitted to human labels; "
                       "original sparse labels retained in source_keypoints for evaluation.",
            "mapping": SOURCE_TO_TARGET,
        },
        "licenses": data.get("licenses", []),
        "images": [image for image in data["images"] if image["id"] in accepted_images],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "court", "supercategory": "basketball-court",
                        "keypoints": list(KEYPOINT_IDS)}],
    }
    (destination / "_annotations.coco.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"split": source.name, "accepted": len(annotations), "skipped": skipped,
            "label_residual_median_px": float(np.median(residuals)) if residuals else None,
            "label_residual_p90_px": float(np.percentile(residuals, 90)) if residuals else None,
            "dense_visibility": dict(sorted(visibility_hist.items()))}


def densify_split(source: Path, destination: Path) -> dict:
    """Complete sparse labels by aligning nearby frames from the same source video.

    A single frame often labels only one sideline, which cannot determine a
    homography. Offline training-data preparation may use the whole source clip:
    SIFT/RANSAC aligns neighbouring frames on the static court plane, and labels
    visible in different frames are transferred to a common reference frame.
    Runtime inference remains strictly causal and never reads future frames.
    """
    data = json.loads((source / "_annotations.coco.json").read_text(encoding="utf-8"))
    images = {image["id"]: image for image in data["images"]}
    records = []
    for annotation in data["annotations"]:
        image = images[annotation["image_id"]]
        width, height = int(image["width"]), int(image["height"])
        sparse = np.asarray(annotation.get("source_keypoints", annotation["keypoints"]),
                            np.float32).reshape(-1, 3).copy()
        outside = ((sparse[:, 0] < 0) | (sparse[:, 0] >= width) |
                   (sparse[:, 1] < 0) | (sparse[:, 1] >= height))
        sparse[outside] = (0, 0, 0)
        records.append({"image": image, "annotation": annotation, "sparse": sparse,
                        "path": source / image["file_name"]})

    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(video_group(record["image"]["file_name"]), []).append(index)
    for indices in groups.values():
        indices.sort(key=lambda index: records[index]["image"]["file_name"])

    sift = cv2.SIFT_create(nfeatures=1800)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    features: dict[int, tuple[np.ndarray | None, np.ndarray | None]] = {}
    for index, record in enumerate(records):
        frame = cv2.imread(str(record["path"]), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            features[index] = (None, None)
            continue
        keypoints, descriptors = sift.detectAndCompute(frame, None)
        features[index] = (np.asarray([kp.pt for kp in keypoints], np.float32)
                           if keypoints else None, descriptors)

    annotations = []
    residuals: list[float] = []
    visibility_hist: dict[int, int] = {}
    skipped = 0
    aligned_pairs = 0
    destination.mkdir(parents=True, exist_ok=True)
    for indices in groups.values():
        for position, index in enumerate(indices):
            record = records[index]
            image, sparse = record["image"], record["sparse"]
            width, height = int(image["width"]), int(image["height"])
            ref_points, ref_descriptors = features[index]
            pools: list[list[np.ndarray]] = [[] for _ in KEYPOINT_IDS]
            support = np.zeros(len(KEYPOINT_IDS), np.int32)
            if ref_descriptors is not None:
                neighbours = indices[max(0, position - 8):position] + indices[position + 1:position + 9]
                for other in neighbours:
                    if other == index:
                        continue
                    other_points, other_descriptors = features[other]
                    if other_descriptors is None or len(other_descriptors) < 12:
                        continue
                    matches = matcher.knnMatch(ref_descriptors, other_descriptors, k=2)
                    good = []
                    for pair in matches:
                        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                            good.append(pair[0])
                    if len(good) < 12:
                        continue
                    src = np.float32([other_points[match.trainIdx] for match in good])
                    dst = np.float32([ref_points[match.queryIdx] for match in good])
                    matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.5, maxIters=1000)
                    if matrix is None or inliers is None:
                        continue
                    inlier_count = int(inliers.sum())
                    if inlier_count < 12 or inlier_count / len(good) < 0.45:
                        continue
                    aligned_pairs += 1
                    visible = np.flatnonzero(records[other]["sparse"][:, 2] > 0)
                    if not len(visible):
                        continue
                    pixels = records[other]["sparse"][visible, :2].astype(np.float32)
                    transferred = cv2.perspectiveTransform(
                        pixels.reshape(-1, 1, 2), matrix).reshape(-1, 2)
                    for keypoint, xy in zip(visible, transferred):
                        if (np.isfinite(xy).all() and 0 <= xy[0] < width and
                                0 <= xy[1] < height):
                            pools[int(keypoint)].append(xy)
                            support[int(keypoint)] += 1

            combined = sparse.copy()
            for keypoint in range(len(KEYPOINT_IDS)):
                if combined[keypoint, 2] > 0 or support[keypoint] < 2:
                    continue
                candidates = np.asarray(pools[keypoint], np.float32)
                median = np.median(candidates, axis=0)
                mad = np.median(np.linalg.norm(candidates - median, axis=1))
                if mad <= 6.0 and np.max(np.linalg.norm(candidates - median, axis=1)) <= 15.0:
                    combined[keypoint] = (median[0], median[1], 2.0)

            matrix, errors = _fit_label_homography(combined)
            if matrix is None:
                skipped += 1
                continue
            projected = cv2.perspectiveTransform(
                WORLD_POINTS.reshape(-1, 1, 2).astype(np.float32), matrix).reshape(-1, 2)
            if (not np.isfinite(projected).all() or
                    np.max(np.linalg.norm(projected, axis=1)) > 4.0 * math.hypot(width, height)):
                skipped += 1
                continue
            inside = ((projected[:, 0] >= 0) & (projected[:, 0] < width) &
                      (projected[:, 1] >= 0) & (projected[:, 1] < height))
            dense = np.column_stack([projected, inside.astype(np.float32) * 2.0])
            fill = (sparse[:, 2] == 0) & inside
            dense[fill, :2] = projected[fill]
            residuals.extend(float(value) for value in errors)
            visible_count = int(np.count_nonzero(dense[:, 2] > 0))
            if visible_count < 4:
                skipped += 1
                continue
            visibility_hist[visible_count] = visibility_hist.get(visible_count, 0) + 1
            item = {key: value for key, value in record["annotation"].items() if key != "keypoints"}
            item.update({"keypoints": dense.reshape(-1).tolist(),
                         "source_keypoints": sparse.reshape(-1).tolist(),
                         "num_keypoints": visible_count,
                         "source_num_keypoints": int(np.count_nonzero(sparse[:, 2] > 0)),
                         "label_densified": True,
                         "temporal_support": support.tolist()})
            annotations.append(item)
            target = destination / image["file_name"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(record["path"], target)

    accepted_images = {item["image_id"] for item in annotations}
    output = {
        "info": {
            "description": "FIBA 16-point labels densified with same-video planar alignment",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "source": "https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2",
            "changes": "Sparse human labels transferred between nearby same-video frames with "
                       "SIFT/RANSAC; missing in-frame points projected from FIBA geometry. "
                       "Original labels retained in source_keypoints for evaluation.",
            "mapping": SOURCE_TO_TARGET,
        },
        "licenses": data.get("licenses", []),
        "images": [image for image in data["images"] if image["id"] in accepted_images],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "court", "supercategory": "basketball-court",
                        "keypoints": list(KEYPOINT_IDS)}],
    }
    (destination / "_annotations.coco.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"split": source.name, "accepted": len(annotations), "skipped": skipped,
            "aligned_frame_pairs": aligned_pairs,
            "label_residual_median_px": float(np.median(residuals)) if residuals else None,
            "label_residual_p90_px": float(np.percentile(residuals, 90)) if residuals else None,
            "dense_visibility": dict(sorted(visibility_hist.items()))}


def densify(args: argparse.Namespace) -> None:
    source, destination = Path(args.source), Path(args.output)
    summaries = [densify_split(source / split, destination / split)
                 for split in ("train", "valid", "test")]
    manifest = {"input_hw": list(INPUT_HW), "keypoint_ids": list(KEYPOINT_IDS),
                "mapping": SOURCE_TO_TARGET, "label_method": "fiba_homography_densified",
                "metric_labels": "original_sparse_human", "splits": summaries}
    (destination / "dataset.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def video_group(file_name: str) -> str:
    """Return the source game identifier encoded in a Roboflow frame filename."""
    match = VIDEO_NAME_RE.match(file_name)
    if match is None or not match.group(1):
        raise ValueError(f"cannot identify source video from filename: {file_name}")
    return match.group(1)


def assign_video_groups(groups: dict[str, list], seed: int = 17) -> dict[str, list[str]]:
    """Greedily balance frame counts while keeping each source game intact."""
    total_groups = len(groups)
    if total_groups < 3:
        raise ValueError("at least three source videos are required for train/valid/test")
    valid_groups = max(1, round(total_groups * 0.17))
    test_groups = max(1, round(total_groups * 0.17))
    capacities = {"train": total_groups - valid_groups - test_groups,
                  "valid": valid_groups, "test": test_groups}
    total_frames = sum(len(records) for records in groups.values())
    targets = {"train": round(total_frames * 0.72),
               "valid": round(total_frames * 0.14), "test": 0}
    targets["test"] = total_frames - targets["train"] - targets["valid"]
    assigned = {split: [] for split in capacities}
    counts = {split: 0 for split in capacities}

    def order_key(item: tuple[str, list]) -> tuple:
        name, records = item
        digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
        return -len(records), digest, name

    for name, records in sorted(groups.items(), key=order_key):
        eligible = [split for split in capacities if len(assigned[split]) < capacities[split]]
        if not eligible:
            raise RuntimeError("video group assignment exhausted split capacities")
        split = max(eligible, key=lambda name: (targets[name] - counts[name], -len(assigned[name])))
        assigned[split].append(name)
        counts[split] += len(records)
    return assigned


def regroup(source: Path, destination: Path, seed: int = 17) -> dict:
    """Re-split converted COCO data by source game, not by individual frame."""
    records = []
    original_counts = {}
    for split in ("train", "valid", "test"):
        root = source / split
        data = json.loads((root / "_annotations.coco.json").read_text(encoding="utf-8"))
        category = _keypoint_category(data)
        if tuple(category["keypoints"]) != KEYPOINT_IDS:
            raise ValueError(f"{root} keypoint order does not match the runtime contract")
        images = {image["id"]: image for image in data["images"]}
        annotations = {annotation["image_id"]: annotation for annotation in data["annotations"]}
        if set(images) != set(annotations):
            raise ValueError(f"{root} images and annotations do not match")
        original_counts[split] = len(images)
        for image_id, image in images.items():
            records.append({"origin": split, "image": image, "annotation": annotations[image_id],
                            "path": root / image["file_name"]})

    groups: dict[str, list] = {}
    for record in records:
        groups.setdefault(video_group(record["image"]["file_name"]), []).append(record)
    assignment = assign_video_groups(groups, seed)
    owner = {name: split for split, names in assignment.items() for name in names}
    if len(owner) != len(groups):
        raise RuntimeError("a source video was assigned to more than one split")

    summaries = []
    for split in ("train", "valid", "test"):
        output = destination / split
        output.mkdir(parents=True, exist_ok=True)
        selected = [record for record in records if owner[video_group(record["image"]["file_name"])] == split]
        images, annotations = [], []
        for new_id, record in enumerate(selected):
            image = dict(record["image"])
            image["id"] = new_id
            image["original_split"] = record["origin"]
            images.append(image)
            annotation = {key: value for key, value in record["annotation"].items()
                          if key not in ("id", "image_id")}
            annotation.update({"id": new_id, "image_id": new_id, "category_id": 1})
            annotations.append(annotation)
            target = output / image["file_name"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if record["path"].resolve() != target.resolve():
                shutil.copy2(record["path"], target)
        output_data = {
            "info": {
                "description": "FIBA 16-point contract, grouped by source game video",
                "license": "CC BY 4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "source": "https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2",
                "changes": "Selected/reordered 16 of 33 keypoints; off-screen points set to visibility 0; "
                           "official frame split replaced by source-video grouped split.",
            },
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "court", "supercategory": "basketball-court",
                            "keypoints": list(KEYPOINT_IDS)}],
        }
        (output / "_annotations.coco.json").write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append({"split": split, "images": len(images), "videos": len(assignment[split]),
                          "video_names": assignment[split], "destination": str(output)})

    manifest = {"input_hw": list(INPUT_HW), "keypoint_ids": list(KEYPOINT_IDS),
                "mapping": SOURCE_TO_TARGET, "split_method": "source_video_grouped",
                "seed": seed, "original_split_counts": original_counts, "splits": summaries}
    (destination / "dataset.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def regroup_command(args: argparse.Namespace) -> None:
    manifest = regroup(Path(args.source), Path(args.output), args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _resize_sample(sample: Sample) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode {sample.path}")
    height, width = image.shape[:2]
    image = cv2.resize(image, INPUT_HW, interpolation=cv2.INTER_LINEAR)
    points = sample.keypoints.copy()
    points[:, 0] *= INPUT_HW[1] / width
    points[:, 1] *= INPUT_HW[0] / height
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), points


def augment(image: np.ndarray, points: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Photometric and small projective augmentation; never guesses landmark semantics."""
    height, width = image.shape[:2]
    if rng.random() < 0.8:
        margin = float(rng.uniform(3, 14))
        source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        target = source + rng.uniform(-margin, margin, (4, 2)).astype(np.float32)
        matrix = cv2.getPerspectiveTransform(source, target)
        image = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT_101)
        points[:, :2] = cv2.perspectiveTransform(points[:, :2].reshape(-1, 1, 2), matrix).reshape(-1, 2)
    if rng.random() < 0.7:
        # Source frames are square crops, while the runtime directly resizes
        # 16:9 broadcast frames to 640x640. Independent x/y scales make the
        # training geometry less dependent on the source crop aspect ratio.
        scale_x = float(rng.uniform(0.72, 1.08))
        scale_y = float(rng.uniform(0.92, 1.08))
        tx, ty = rng.uniform(-8, 8, 2)
        matrix = np.float32([[scale_x, 0, width * (1 - scale_x) / 2 + tx],
                             [0, scale_y, height * (1 - scale_y) / 2 + ty]])
        image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        points[:, :2] = cv2.transform(points[:, :2].reshape(-1, 1, 2), matrix).reshape(-1, 2)
    if rng.random() < 0.8:
        image = image.astype(np.float32)
        image *= float(rng.uniform(0.78, 1.22))
        image += float(rng.uniform(-18, 18))
        contrast = float(rng.uniform(0.82, 1.18))
        image = (image - 127.5) * contrast + 127.5
        if rng.random() < 0.35:
            hsv = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * rng.uniform(0.75, 1.25), 0, 255)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
        image = np.clip(image, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    outside = ((points[:, 0] < 0) | (points[:, 0] >= width) |
               (points[:, 1] < 0) | (points[:, 1] >= height))
    points[outside] = (0, 0, 0)
    return image, points


class CourtDataset:
    """Lazy image loading keeps worker processes small on Windows."""

    def __init__(self, samples: list[Sample], training: bool, seed: int):
        self.samples = samples
        self.training = training
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image, points = _resize_sample(self.samples[index])
        if self.training:
            rng = np.random.default_rng(self.seed + index + int(np.random.randint(0, 2 ** 31 - 1)))
            image, points = augment(image, points, rng)
        metric_keypoints = (self.samples[index].metric_keypoints
                            if self.samples[index].metric_keypoints is not None
                            else self.samples[index].keypoints)
        metric_points = metric_keypoints.copy()
        source_height, source_width = self.samples[index].source_hw
        metric_points[:, 0] *= INPUT_HW[1] / source_width
        metric_points[:, 1] *= INPUT_HW[0] / source_height
        return (image, points[:, :2].astype(np.float32), points[:, 2].astype(np.float32),
                np.asarray(self.samples[index].source_hw, np.int32),
                metric_points[:, :2].astype(np.float32), metric_points[:, 2].astype(np.float32))


def decode_heatmaps(heatmaps, torch):
    """Decode a 5x5 softargmax around each peak for sub-pixel coordinates.

    The wider window is intentional: two equal adjacent Gaussian peaks represent
    a half-pixel centre. A 3x3 window anchored by argmax would drop one side of
    that symmetric pair and systematically shift the decoded coordinate.
    """
    from torch.nn import functional as F

    batch, keypoints, height, width = heatmaps.shape
    radius = 2
    size = radius * 2 + 1
    flat = heatmaps.flatten(2)
    index = flat.argmax(dim=-1)
    peak_x = index % width
    peak_y = torch.div(index, width, rounding_mode="floor")
    offsets = torch.arange(-radius, radius + 1, device=heatmaps.device,
                           dtype=torch.long)
    offset_x = offsets.repeat(size)
    offset_y = offsets.repeat_interleave(size)
    # Gather only the 25 values around each peak. This avoids materializing an
    # unfold tensor of every spatial location and keeps one index per peak.
    padded = F.pad(heatmaps, (radius, radius, radius, radius), value=-1e4)
    padded_flat = padded.flatten(2)
    patch_index = ((peak_y[:, :, None] + radius + offset_y) * (width + size - 1)
                   + peak_x[:, :, None] + radius + offset_x)
    gathered = padded_flat.gather(2, patch_index)
    weights = torch.softmax(gathered, dim=-1)
    dx = (weights * offsets.repeat(size)).sum(dim=-1)
    dy = (weights * offsets.repeat_interleave(size)).sum(dim=-1)
    # Return the raw heatmap logit. Callers apply sigmoid exactly once when
    # forming confidence; sigmoid here would silently square the peak term.
    peak = flat.gather(2, index.unsqueeze(-1)).squeeze(-1)
    x_coord = ((index % width).float() + dx) * OUTPUT_SCALE
    y_coord = (torch.div(index, width, rounding_mode="floor").float() + dy) * OUTPUT_SCALE
    return torch.stack([x_coord, y_coord], dim=-1), peak


def build_torch(architecture: str = "tiny", pretrained: bool = True):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class ConvBlock(nn.Sequential):
        def __init__(self, cin: int, cout: int):
            super().__init__(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.SiLU(),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.SiLU(),
            )

    class CourtNet(nn.Module):
        """Compact from-scratch UNet retained for compatibility and ablations."""

        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
                                      nn.BatchNorm2d(24), nn.SiLU())
            self.enc1 = ConvBlock(24, 24)
            self.down2 = nn.Conv2d(24, 40, 3, stride=2, padding=1, bias=False)
            self.enc2 = ConvBlock(40, 40)
            self.down3 = nn.Conv2d(40, 64, 3, stride=2, padding=1, bias=False)
            self.enc3 = ConvBlock(64, 64)
            self.down4 = nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False)
            self.enc4 = ConvBlock(96, 96)
            self.dec3 = ConvBlock(160, 64)
            self.dec2 = ConvBlock(104, 40)
            self.dec1 = ConvBlock(64, 24)
            self.heatmap_head = nn.Conv2d(24, len(KEYPOINT_IDS), 1)
            self.presence_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 16))

        def forward(self, x):
            x = x.float() / 127.5 - 1.0
            stem = self.stem(x)
            e1 = self.enc1(stem)
            e2 = self.enc2(self.down2(e1))
            e3 = self.enc3(self.down3(e2))
            e4 = self.enc4(self.down4(e3))
            d3 = self.dec3(torch.cat([F.interpolate(e4, size=e3.shape[-2:],
                                                     mode="bilinear", align_corners=False), e3], 1))
            d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[-2:],
                                                     mode="bilinear", align_corners=False), e2], 1))
            d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:],
                                                     mode="bilinear", align_corners=False), e1], 1))
            heatmaps = self.heatmap_head(d1)
            return heatmaps, self.presence_head(e4)

    class ResNetCourtNet(nn.Module):
        """ImageNet-pretrained ResNet-18 encoder with a multi-scale heatmap decoder."""

        def __init__(self):
            super().__init__()
            from torchvision.models import ResNet18_Weights, resnet18

            backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
            self.enc1 = backbone.layer1
            self.enc2 = backbone.layer2
            self.enc3 = backbone.layer3
            self.enc4 = backbone.layer4
            self.dec3 = ConvBlock(768, 256)
            self.dec2 = ConvBlock(384, 128)
            self.dec1 = ConvBlock(192, 64)
            self.heatmap_head = nn.Conv2d(64, len(KEYPOINT_IDS), 1)
            self.presence_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 16))

        def forward(self, x):
            # Training batches arrive as uint8. ImageNet constants must stay
            # float32: deriving their dtype from x would truncate std to zero
            # and turn every channel into inf/NaN before the first convolution.
            mean = torch.tensor((0.485, 0.456, 0.406), device=x.device,
                                dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor((0.229, 0.224, 0.225), device=x.device,
                               dtype=torch.float32).view(1, 3, 1, 1)
            x = (x.float() / 255.0 - mean) / std
            stem = self.stem(x)
            e1 = self.enc1(stem)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
            d3 = self.dec3(torch.cat([F.interpolate(e4, size=e3.shape[-2:],
                                                     mode="bilinear", align_corners=False), e3], 1))
            d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[-2:],
                                                     mode="bilinear", align_corners=False), e2], 1))
            d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:],
                                                     mode="bilinear", align_corners=False), e1], 1))
            heatmaps = F.interpolate(self.heatmap_head(d1), scale_factor=2.0,
                                     mode="bilinear", align_corners=False)
            return heatmaps, self.presence_head(e4)

    class ExportNet(nn.Module):
        def __init__(self, model: CourtNet):
            super().__init__()
            self.model = model

        def forward(self, x):
            heatmaps, presence = self.model(x)
            xy, peak = decode_heatmaps(heatmaps, torch)
            confidence = torch.sigmoid(presence) * torch.sigmoid(peak)
            return torch.cat([xy, confidence.unsqueeze(-1)], dim=-1)

    if architecture == "tiny":
        return CourtNet, ExportNet
    if architecture == "resnet18":
        return ResNetCourtNet, ExportNet
    raise ValueError(f"unknown court architecture: {architecture}")


def make_targets(points, visibility, torch, sigma: float = 1.8):
    batch, keypoints = points.shape[:2]
    height, width = OUTPUT_HW
    grid_y, grid_x = torch.meshgrid(torch.arange(height, device=points.device),
                                    torch.arange(width, device=points.device), indexing="ij")
    x = points[:, :, 0] / OUTPUT_SCALE
    y = points[:, :, 1] / OUTPUT_SCALE
    gaussian = torch.exp(-(((grid_x[None, None] - x[:, :, None, None]) ** 2 +
                            (grid_y[None, None] - y[:, :, None, None]) ** 2) / (2 * sigma ** 2)))
    # COCO visibility is 0=unlabeled, 1=occluded, 2=visible. Both presence BCE
    # and the heatmap target need a binary present/absent label, not raw 0/1/2.
    target_visibility = (visibility > 0).to(dtype=gaussian.dtype)
    return gaussian * target_visibility[:, :, None, None], target_visibility


def loss_fn(heatmaps, presence, targets, visibility, torch):
    weights = 1.0 + 39.0 * targets
    heat = ((torch.sigmoid(heatmaps) - targets) ** 2 * weights).mean()
    present = torch.nn.functional.binary_cross_entropy_with_logits(presence, visibility)
    return heat + 0.25 * present, heat, present


def decode_for_metrics(heatmaps, presence, torch):
    xy, peak = decode_heatmaps(heatmaps, torch)
    confidence = torch.sigmoid(presence) * torch.sigmoid(peak)
    return xy, confidence


def evaluate(model, loader, device, torch) -> dict:
    model.eval()
    thresholds = (5, 10, 20)
    correct = {threshold: 0 for threshold in thresholds}
    gated_correct = {threshold: 0 for threshold in thresholds}
    visible_count = 0
    detected_visible = 0
    absent = 0
    false_positive = 0
    confidence_sum = 0.0
    image_count = 0
    published_h = 0
    accurate_h = 0
    ground_truth_h = 0
    reprojection_errors = []
    corner_errors = []
    runtime_cfg = KeypointConfig()
    solver = object.__new__(CourtKeypointTracker)
    solver.cfg = runtime_cfg
    with torch.no_grad():
        for batch in loader:
            images, points, visibility, source_hw = batch[:4]
            metric_points = batch[4] if len(batch) > 4 else points
            metric_visibility = batch[5] if len(batch) > 5 else visibility
            images = images.to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
            points = points.to(device, non_blocking=True)
            visibility = visibility.to(device, non_blocking=True)
            metric_points = metric_points.to(device, non_blocking=True)
            metric_visibility = metric_visibility.to(device, non_blocking=True)
            heatmaps, presence = model(images)
            predicted, confidence = decode_for_metrics(heatmaps, presence, torch)
            distance = torch.linalg.vector_norm(predicted - metric_points, dim=-1)
            mask = metric_visibility > 0
            detected = confidence >= runtime_cfg.min_conf
            visible_count += int(mask.sum())
            detected_visible += int((mask & detected).sum())
            confidence_sum += float(confidence[mask].sum())
            for threshold in thresholds:
                correct[threshold] += int((mask & (distance <= threshold)).sum())
                gated_correct[threshold] += int((mask & detected & (distance <= threshold)).sum())
            absent_mask = metric_visibility == 0
            absent += int(absent_mask.sum())
            false_positive += int((absent_mask & detected).sum())
            image_count += int(images.shape[0])

            # Publication and accuracy are different metrics. A four-point fit
            # can have near-zero inlier residual while being geometrically wrong,
            # so compare every published H with the H implied by human labels.
            predicted_np = predicted.detach().cpu().numpy()
            confidence_np = confidence.detach().cpu().numpy()
            metric_np = metric_points.detach().cpu().numpy()
            metric_visibility_np = metric_visibility.detach().cpu().numpy()
            source_hw_np = source_hw.detach().cpu().numpy()
            for batch_index in range(images.shape[0]):
                source_height, source_width = (int(value) for value in source_hw_np[batch_index])
                scale_x = source_width / float(INPUT_HW[1])
                scale_y = source_height / float(INPUT_HW[0])
                detections = [
                    DetectedKeypoint(KEYPOINT_IDS[index],
                                     (float(xy[0] * scale_x), float(xy[1] * scale_y)),
                                     float(conf), KEYPOINT_SPECS[index][1])
                    for index, (xy, conf) in enumerate(zip(predicted_np[batch_index],
                                                           confidence_np[batch_index]))
                    if conf >= runtime_cfg.min_conf and
                    0 <= xy[0] < INPUT_HW[1] and 0 <= xy[1] < INPUT_HW[0]
                ]
                H, _H_inv, solved, error = solver._solve_labeled(
                    detections, (source_height, source_width), runtime_cfg.min_matches)
                if H is None:
                    continue
                publish_conf = CourtKeypointTracker._publication_confidence(solved)
                if publish_conf < runtime_cfg.min_publish_conf:
                    continue
                published_h += 1
                reprojection_errors.append(float(error))

                metric_sample = np.column_stack([metric_np[batch_index],
                                                metric_visibility_np[batch_index]])
                truth_world_to_pixel, residuals = _fit_label_homography(metric_sample)
                if truth_world_to_pixel is None:
                    continue
                ground_truth_h += 1
                to_input = np.diag([INPUT_HW[1] / source_width,
                                    INPUT_HW[0] / source_height, 1.0])
                from_input = np.linalg.inv(to_input)
                # H maps source pixels to world. Convert only its input side:
                # source = from_input @ input, so H_input = H @ from_input.
                # Pre-multiplying by to_input would incorrectly scale world units.
                predicted_input_to_world = H @ from_input
                predicted_world_to_input = np.linalg.inv(predicted_input_to_world)
                projected = cv2.perspectiveTransform(
                    WORLD_POINTS.reshape(-1, 1, 2), predicted_world_to_input
                ).reshape(-1, 2)
                truth = cv2.perspectiveTransform(
                    WORLD_POINTS.reshape(-1, 1, 2), truth_world_to_pixel
                ).reshape(-1, 2)
                errors = np.linalg.norm(projected - truth, axis=1)
                corner_errors.extend(float(value) for value in errors)
                if float(np.median(errors)) <= 20.0 and float(np.percentile(errors, 90)) <= 35.0:
                    accurate_h += 1

    denominator = max(1, visible_count)
    return {"visible": visible_count,
            "pck5": correct[5] / denominator,
            "pck10": correct[10] / denominator,
            "pck20": correct[20] / denominator,
            "gated_pck5": gated_correct[5] / denominator,
            "gated_pck10": gated_correct[10] / denominator,
            "gated_pck20": gated_correct[20] / denominator,
            "presence_recall": detected_visible / denominator,
            "mean_conf": confidence_sum / denominator,
            "absent_fpr": false_positive / max(1, absent),
            "images": image_count,
            "homography_publish_rate": published_h / max(1, image_count),
            "homography_publish_accuracy": accurate_h / max(1, published_h),
            "homography_evaluated_accuracy": accurate_h / max(1, ground_truth_h),
            # Accuracy rate is measured over all images, not only the subset
            # with enough sparse labels to fit a reference homography. This
            # prevents a single good publication from selecting a bad model.
            "homography_accurate_rate": accurate_h / max(1, image_count),
            "homography_evaluable_rate": ground_truth_h / max(1, image_count),
            # A published H without enough human labels for a reference is
            # unknown, not proven false. Only evaluable references contribute
            # to the false-publication rate; accurate_rate still uses all images.
            "homography_false_publish_rate": (ground_truth_h - accurate_h) / max(1, ground_truth_h),
            "homography_reproj_median_px": float(np.median(reprojection_errors))
            if reprojection_errors else None,
            "homography_corner_median_px": float(np.median(corner_errors)) if corner_errors else None,
            "homography_corner_p90_px": float(np.percentile(corner_errors, 90)) if corner_errors else None}


def checkpoint_score(real: dict, synthetic: dict | None = None) -> float:
    """Balance real-domain localisation with independently rendered geometry."""
    if synthetic is None:
        return (real["gated_pck10"] + 0.25 * real["homography_accurate_rate"]
                - 0.05 * real["absent_fpr"]
                - 0.25 * real.get("homography_false_publish_rate", 0.0))
    return (real["gated_pck10"] + 0.35 * synthetic["gated_pck10"]
            + 0.60 * synthetic["homography_accurate_rate"]
            - 0.05 * real["absent_fpr"] - 0.05 * synthetic["absent_fpr"]
            - 0.15 * synthetic["homography_false_publish_rate"]
            - 0.25 * real.get("homography_false_publish_rate", 0.0))


def train(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and torch.version.cuda and torch.version.cuda.startswith("11"):
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 12:
            print(f"[warn] PyTorch CUDA 11.x reports sm_{capability[0]}{capability[1]} incompatibility; "
                  "a short training probe will determine whether this wheel can run.")
    root = Path(args.data)
    train_samples = load_split(root, "train")
    valid_samples = load_split(root, "valid")
    synthetic_count = 0
    if args.synthetic_data:
        synthetic_samples = load_split(Path(args.synthetic_data), "train")
        synthetic_count = len(synthetic_samples)
        train_samples = train_samples + synthetic_samples
        print(f"[data] added {synthetic_count} synthetic full-court samples")
    train_set = CourtDataset(train_samples, True, args.seed)
    valid_set = CourtDataset(valid_samples, False, args.seed)
    loader_args = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                       persistent_workers=args.workers > 0)
    synthetic_valid_loader = None
    if args.synthetic_valid:
        synthetic_valid_samples = load_split(Path(args.synthetic_valid), "valid")
        synthetic_valid_set = CourtDataset(synthetic_valid_samples, False, args.seed)
        synthetic_valid_loader = DataLoader(synthetic_valid_set, batch_size=args.batch,
                                            shuffle=False, **loader_args)
        print(f"[data] geometry validation={len(synthetic_valid_samples)} "
              "rendered full-court images")
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, **loader_args)
    valid_loader = DataLoader(valid_set, batch_size=args.batch, shuffle=False, **loader_args)

    CourtNet, _ = build_torch(args.architecture, args.pretrained)
    model = CourtNet().to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(train_set) / args.batch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_state = None
    start_epoch = int(state.get("epoch", 0)) if args.checkpoint else 0
    if args.checkpoint and "metrics" in state:
        previous_metrics = state["metrics"]
        previous_synthetic = state.get("synthetic_valid_metrics")
        best_score = checkpoint_score(previous_metrics, previous_synthetic)
        # Preserve the incoming selection checkpoint until a newly evaluated
        # epoch genuinely improves the current scoring function.
        best_state = {key: value.detach().cpu().clone()
                      for key, value in model.state_dict().items()}
        print(f"[resume] incoming_epoch={state.get('epoch', '?')} "
              f"selection_score={best_score:.5f}", flush=True)
    stale = 0
    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"[model] {params / 1e6:.3f}M parameters; train={len(train_set)} valid={len(valid_set)} "
          f"device={device} input={INPUT_HW} heatmap={OUTPUT_HW}")

    for epoch in range(args.epochs):
        started = time.time()
        model.train()
        totals = np.zeros(3, np.float64)
        for images, points, visibility, _source_hw, *_metric in train_loader:
            images = images.to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
            points = points.to(device, non_blocking=True)
            visibility = visibility.to(device, non_blocking=True)
            targets, target_visibility = make_targets(points, visibility, torch)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                heatmaps, presence = model(images)
                loss, heat, present = loss_fn(heatmaps, presence, targets, target_visibility, torch)
            if not all(torch.isfinite(value).all() for value in
                       (images, points, visibility, heatmaps, presence, loss)):
                raise RuntimeError(
                    f"non-finite training value at epoch {epoch + 1}; check the "
                    "CUDA/PyTorch/cuDNN combination and input preprocessing"
                )
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            totals += np.array([float(loss), float(heat), float(present)])
        metrics = evaluate(model, valid_loader, device, torch)
        synthetic_metrics = (evaluate(model, synthetic_valid_loader, device, torch)
                             if synthetic_valid_loader is not None else None)
        score = checkpoint_score(metrics, synthetic_metrics)
        synthetic_log = ""
        if synthetic_metrics is not None:
            synthetic_log = (f" Syn_pck10={synthetic_metrics['pck10']:.4f}"
                             f" Syn_gated={synthetic_metrics['gated_pck10']:.4f}"
                             f" Syn_Hpub={synthetic_metrics['homography_publish_rate']:.4f}"
                             f" Syn_Hacc={synthetic_metrics['homography_accurate_rate']:.4f}"
                             f" Syn_corner_med={synthetic_metrics['homography_corner_median_px'] or 0:.1f}"
                             f" Syn_falseH={synthetic_metrics['homography_false_publish_rate']:.4f}")
        print(f"[epoch {epoch + 1:03d}/{args.epochs}] loss={totals[0] / steps_per_epoch:.5f} "
              f"heat={totals[1] / steps_per_epoch:.5f} presence={totals[2] / steps_per_epoch:.5f} "
              f"pck5={metrics['pck5']:.4f} pck10={metrics['pck10']:.4f} "
              f"pck20={metrics['pck20']:.4f} gated_pck10={metrics['gated_pck10']:.4f} "
              f"H_publish={metrics['homography_publish_rate']:.4f} "
              f"H_accurate={metrics['homography_accurate_rate']:.4f} "
              f"H_corner_med={metrics['homography_corner_median_px'] or 0:.1f} "
              f"absent_fpr={metrics['absent_fpr']:.4f}{synthetic_log} "
              f"time={time.time() - started:.1f}s", flush=True)
        # Keep the newest weights separately so an interrupted run can resume
        # without losing epochs that did not beat the selection score yet.
        current_state = {key: value.detach().cpu().clone()
                         for key, value in model.state_dict().items()}
        torch.save({"model": current_state, "keypoint_ids": list(KEYPOINT_IDS),
                    "input_hw": list(INPUT_HW), "architecture": args.architecture,
                    "synthetic_train_images": synthetic_count,
                    "metrics": metrics, "synthetic_valid_metrics": synthetic_metrics,
                    "checkpoint_score": score, "epoch": start_epoch + epoch + 1},
                   output.with_name(f"{output.stem}_last.pt"))
        if score > best_score + 1e-5:
            best_score = score
            best_state = current_state
            torch.save({"model": best_state, "keypoint_ids": list(KEYPOINT_IDS),
                        "input_hw": list(INPUT_HW), "architecture": args.architecture,
                        "synthetic_train_images": synthetic_count,
                        "metrics": metrics, "synthetic_valid_metrics": synthetic_metrics,
                        "checkpoint_score": score, "epoch": start_epoch + epoch + 1}, output)
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[early-stop] no validation improvement for {stale} epochs")
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    print(f"[checkpoint] {output} score={best_score:.5f}")


def evaluate_checkpoint(args: argparse.Namespace) -> dict:
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_split(Path(args.data), args.split)
    dataset = CourtDataset(samples, False, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    state = torch.load(args.checkpoint, map_location=device)
    architecture = state.get("architecture", "tiny") if isinstance(state, dict) else "tiny"
    CourtNet, _ = build_torch(architecture, pretrained=False)
    model = CourtNet().to(device)
    model.load_state_dict(state.get("model", state))
    metrics = evaluate(model, loader, device, torch)
    metrics.update({"split": args.split, "images": len(samples), "checkpoint": str(args.checkpoint)})
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def export(args: argparse.Namespace) -> dict:
    import onnx
    import onnxruntime as ort
    import torch

    checkpoint = Path(args.checkpoint)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        checkpoint_ids = state.get("keypoint_ids")
        if checkpoint_ids is not None and tuple(checkpoint_ids) != KEYPOINT_IDS:
            raise ValueError("checkpoint keypoint_ids do not match the runtime contract")
        checkpoint_hw = state.get("input_hw")
        if checkpoint_hw is not None and tuple(checkpoint_hw) != INPUT_HW:
            raise ValueError("checkpoint input_hw does not match the runtime contract")
    architecture = state.get("architecture", "tiny") if isinstance(state, dict) else "tiny"
    CourtNet, ExportNet = build_torch(architecture, pretrained=False)
    model = CourtNet()
    model.load_state_dict(state.get("model", state))
    model.eval()
    wrapper = ExportNet(model).eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, *INPUT_HW, dtype=torch.float32)
    torch.onnx.export(wrapper, dummy, str(output), opset_version=17,
                      input_names=["input"], output_names=["keypoints"],
                      dynamic_axes={"input": {0: "batch"}, "keypoints": {0: "batch"}})
    onnx.checker.check_model(str(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    if model_input.type != "tensor(float)" or tuple(model_input.shape[1:]) != (3, *INPUT_HW):
        raise RuntimeError(f"unexpected ONNX input contract: {model_input.type} {model_input.shape}")
    for batch in (1, 2):
        result = session.run(None, {"input": np.zeros((batch, 3, *INPUT_HW), np.float32)})[0]
        if (result.shape != (batch, len(KEYPOINT_IDS), 3)
                or not np.isfinite(result).all()
                or not ((result[..., :2] >= 0) & (result[..., :2] < INPUT_HW[1])).all()
                or not ((result[..., 2] >= 0) & (result[..., 2] <= 1)).all()):
            raise RuntimeError(f"invalid ONNX output for batch {batch}: {result.shape}")
    # Zero inputs cannot catch an incorrect normalization, decoder or confidence
    # path. Compare the exported graph with the raw PyTorch model on nonzero data.
    generator = torch.Generator().manual_seed(20261128)
    probe = torch.randint(0, 256, (2, 3, *INPUT_HW), dtype=torch.uint8, generator=generator)
    with torch.no_grad():
        probe_heatmaps, probe_presence = model(probe)
        reference_xy, reference_confidence = decode_for_metrics(
            probe_heatmaps, probe_presence, torch)
    exported = session.run(None, {"input": probe.float().numpy()})[0]
    coordinate_error = float(np.max(np.abs(exported[..., :2] - reference_xy.numpy())))
    confidence_error = float(np.max(np.abs(exported[..., 2] - reference_confidence.numpy())))
    if coordinate_error > 0.1 or confidence_error > 1e-3:
        raise RuntimeError(
            f"PyTorch/ONNX parity failed: coordinate={coordinate_error}, "
            f"confidence={confidence_error}"
        )
    sidecar = {
        "input_hw": list(INPUT_HW), "preprocess": "rgb_0_255",
        "coordinate_space": "input_pixels", "mean": [0, 0, 0], "std": [1, 1, 1],
        "keypoint_ids": list(KEYPOINT_IDS), "opset": 17,
        "source": "Roboflow basketball-court-detection-2 v22, CC BY 4.0",
        "source_url": "https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "changes": "Selected/reordered 16 of 33 keypoints; added projective full-court synthetic "
                   "training images; original sparse human labels retained for evaluation.",
        "source_mapping": SOURCE_TO_TARGET,
        "architecture": architecture,
        "synthetic_train_images": state.get("synthetic_train_images", 0),
        "training": state.get("metrics", {}),
        "synthetic_valid": state.get("synthetic_valid_metrics"),
        "checkpoint_score": state.get("checkpoint_score"),
        "export_parity": {"coordinate_max_abs_diff_px": coordinate_error,
                          "confidence_max_abs_diff": confidence_error},
        "epoch": state.get("epoch"),
    }
    output.with_suffix(".json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"[onnx] {output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)")
    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="convert 33-point COCO to the 16-point contract")
    prepare_parser.add_argument("--source", required=True, type=Path)
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_DATA)
    prepare_parser.set_defaults(func=prepare)

    regroup_parser = sub.add_parser("regroup", help="replace frame-level splits with source-video grouped splits")
    regroup_parser.add_argument("--source", required=True, type=Path,
                                help="root containing converted train/valid/test COCO splits")
    regroup_parser.add_argument("--output", type=Path,
                                default=Path("data/derived/court_keypoints16_grouped"))
    regroup_parser.add_argument("--seed", type=int, default=17)
    regroup_parser.set_defaults(func=regroup_command)

    densify_parser = sub.add_parser("densify", help="complete sparse labels from the FIBA court geometry")
    densify_parser.add_argument("--source", required=True, type=Path,
                                help="root containing sparse grouped train/valid/test COCO splits")
    densify_parser.add_argument("--output", type=Path,
                                default=Path("data/derived/court_keypoints16_dense"))
    densify_parser.set_defaults(func=densify)

    synthetic_parser = sub.add_parser("synthetic", help="render full-court FIBA training images")
    synthetic_parser.add_argument("--output", type=Path,
                                  default=Path("data/derived/court_keypoints16_synthetic"))
    synthetic_parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    synthetic_parser.add_argument("--count", type=int, default=1200)
    synthetic_parser.add_argument("--seed", type=int, default=17)
    synthetic_parser.set_defaults(func=generate_synthetic)

    train_parser = sub.add_parser("train", help="train the 16-point heatmap model")
    train_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    train_parser.add_argument("--synthetic-data", type=Path,
                              help="optional rendered full-court training split")
    train_parser.add_argument("--synthetic-valid", type=Path,
                              help="independent rendered split for geometry-aware checkpoint selection")
    train_parser.add_argument("--output", type=Path, default=DEFAULT_MODEL.with_suffix(".pt"))
    train_parser.add_argument("--checkpoint", type=Path)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch", type=int, default=8)
    train_parser.add_argument("--workers", type=int, default=4)
    train_parser.add_argument("--architecture", choices=["tiny", "resnet18"], default="resnet18")
    train_parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                              help="use ImageNet weights for the ResNet-18 encoder")
    train_parser.add_argument("--lr", type=float, default=2e-3)
    train_parser.add_argument("--seed", type=int, default=17)
    train_parser.add_argument("--patience", type=int, default=12)
    train_parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                              help="disabled by default; enable only after validating finite FP16 training")
    train_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    train_parser.set_defaults(func=train)

    evaluate_parser = sub.add_parser("evaluate", help="evaluate a checkpoint on a converted COCO split")
    evaluate_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    evaluate_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL.with_suffix(".pt"))
    evaluate_parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    evaluate_parser.add_argument("--batch", type=int, default=8)
    evaluate_parser.add_argument("--seed", type=int, default=17)
    evaluate_parser.add_argument("--report", type=Path)
    evaluate_parser.set_defaults(func=evaluate_checkpoint)

    export_parser = sub.add_parser("export", help="export checkpoint to the runtime ONNX contract")
    export_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL.with_suffix(".pt"))
    export_parser.add_argument("--output", type=Path, default=DEFAULT_MODEL.with_suffix(".onnx"))
    export_parser.set_defaults(func=export)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
