"""球场关键点回归数据契约与解码。

关键点由精确 H_world_to_px 投影生成，并用 finalize 的可见性结果监督，
避免让模型学习被替补席/人物遮挡的伪标签。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from bsk.court import Court

KEYPOINT_NAMES = (
    "A_baseline_left", "A_baseline_right", "B_baseline_right", "B_baseline_left",
    "mid_left", "mid_right",
    "A_corner_left", "A_corner_right", "B_corner_right", "B_corner_left",
    "A_ft_left", "A_ft_right", "B_ft_right", "B_ft_left",
    "A_rim", "B_rim",
)


def world_keypoints(court: Court) -> np.ndarray:
    from bsk.court import all_court_keypoints

    points = {kp.name: np.asarray(kp.pos, dtype=np.float64) for kp in all_court_keypoints(court)}
    return np.stack([points[name] for name in KEYPOINT_NAMES])


def project_keypoints(record: dict, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    from bsk.calib_seg_webgpu import project_points

    court_model = record["court_model"]
    from bsk.court import COURTS

    world = world_keypoints(COURTS[court_model])
    H = np.asarray(record["H_world_to_px"], dtype=np.float64)
    src_h, src_w = record["image_size"]
    h, w = size
    scale = np.array([[w / src_w, 0.0, 0.0], [0.0, h / src_h, 0.0], [0.0, 0.0, 1.0]])
    pixel, valid = project_points(world, scale @ H)
    visibility = {k["name"]: bool(k.get("visible", False)) for k in record.get("keypoints", [])}
    visible = valid & np.array([visibility.get(name, False) for name in KEYPOINT_NAMES])
    return pixel.astype(np.float32), visible


def decode_heatmaps(heatmaps: torch.Tensor, threshold: float = 0.25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回像素坐标、可见性、最大响应。"""
    if heatmaps.ndim == 2:
        heatmaps = heatmaps.unsqueeze(0)
    b, k, h, w = heatmaps.shape
    flat = heatmaps.flatten(2)
    probs = F.softmax(flat, dim=-1)
    xs = torch.linspace(0, w - 1, w, device=heatmaps.device, dtype=heatmaps.dtype)
    ys = torch.linspace(0, h - 1, h, device=heatmaps.device, dtype=heatmaps.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    u = (probs * xx.flatten()).sum(-1)
    v = (probs * yy.flatten()).sum(-1)
    response = flat.max(-1).values
    coords = torch.stack([u, v], dim=-1).cpu().numpy()
    response = response.cpu().numpy()
    return coords, response >= threshold, response


class CourtKeypointDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Path, size: tuple[int, int] = (540, 960), augment: bool = False):
        self.data_dir = Path(data_dir)
        self.size = size
        self.augment = augment
        self.records = json.loads((self.data_dir / "meta.json").read_text(encoding="utf-8"))
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = cv2.imread(str(self.data_dir / "images" / f"{record['idx']:06d}.jpg"))
        if image is None:
            raise FileNotFoundError(record["idx"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = self.size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        with Image.open(self.data_dir / "masks" / f"{record['idx']:06d}.png") as mask_image:
            mask = np.asarray(mask_image, dtype=np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        points, visible = project_keypoints(record, self.size)
        if self.augment and np.random.rand() < 0.5:
            image = image[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
            points[:, 0] = w - 1 - points[:, 0]
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return (torch.from_numpy(image.transpose(2, 0, 1)),
                torch.from_numpy(mask.astype(np.int64)),
                torch.from_numpy(points), torch.from_numpy(visible))


__all__ = ["KEYPOINT_NAMES", "CourtKeypointDataset", "decode_heatmaps", "project_keypoints", "world_keypoints"]
