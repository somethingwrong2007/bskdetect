"""球场关键点模型推理与 RANSAC 单应拟合。"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from bsk import calibrate
from bsk.calib_keypoint import KEYPOINT_NAMES, decode_heatmaps, world_keypoints
from bsk.calib_seg_net import CourtLineSegNet
from bsk.court import COURTS
from bsk.line_calib import _orient_homography


class KeypointCalibrator:
    def __init__(self, weights: str | Path, court_model: str = "FIBA",
                 size: tuple[int, int] = (540, 960), device: str | None = None,
                 conf_thr: float = 0.5):
        self.size = size
        self.court = COURTS[court_model]
        self.world = world_keypoints(self.court)
        self.conf_thr = conf_thr
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = CourtLineSegNet(num_keypoints=len(KEYPOINT_NAMES), pretrained=False).to(self.device)
        checkpoint = torch.load(weights, map_location=self.device, weights_only=False)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.model.load_state_dict(state)
        self.model.eval()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @torch.no_grad()
    def predict(self, rgb: np.ndarray, prior: tuple[np.ndarray, str] | None = None):
        started = time.time()
        h, w = self.size
        image = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        _seg, heatmaps, _coords, vis_logits = self.model(tensor)
        pixel_model, _heat_visible, _heat_response = decode_heatmaps(heatmaps, threshold=-1e9)
        pixel_model = pixel_model[0]
        response = torch.sigmoid(vis_logits[0]).detach().cpu().numpy()
        visible = response >= self.conf_thr
        if int(visible.sum()) < 4:
            return None, {"ok": False, "reason": "too_few_keypoints", "visible": int(visible.sum()),
                          "response": response.tolist(), "ms": (time.time() - started) * 1000}
        src_h, src_w = rgb.shape[:2]
        scale = np.array([[src_w / w, 0.0, 0.0], [0.0, src_h / h, 0.0], [0.0, 0.0, 1.0]])
        pixel = (scale @ np.column_stack([pixel_model, np.ones(len(pixel_model))]).T).T[:, :2]
        try:
            H_inv, mask = cv2.findHomography(self.world[visible], pixel[visible], cv2.USAC_MAGSAC, 4.0,
                                             maxIters=5000, confidence=0.999)
        except cv2.error:
            H_inv, mask = cv2.findHomography(self.world[visible], pixel[visible], cv2.RANSAC, 4.0,
                                             maxIters=5000, confidence=0.999)
        if H_inv is None or mask is None or int(mask.sum()) < 4:
            return None, {"ok": False, "reason": "homography_failed", "visible": int(visible.sum()),
                          "response": response.tolist(), "ms": (time.time() - started) * 1000}
        H_inv = _orient_homography(H_inv, self.court, prior[0] if prior else None)
        try:
            H = np.linalg.inv(H_inv)
        except np.linalg.LinAlgError:
            return None, {"ok": False, "reason": "singular", "ms": (time.time() - started) * 1000}
        projected = (H_inv @ np.column_stack([self.world[visible], np.ones(int(visible.sum()))]).T).T
        projected = projected[:, :2] / projected[:, 2:3]
        error = float(np.mean(np.linalg.norm(projected - pixel[visible], axis=1)))
        calibration = calibrate.Calibration(
            H=H.tolist(), H_inv=H_inv.tolist(), reproj_error_px=error,
            keypoints_used=int(mask.sum()), court_model=self.court.model, timestamp=time.time(),
        )
        calibration.source = "keypoint"
        if not calibration.is_plausible(src_w, src_h):
            return None, {"ok": False, "reason": "implausible", "reproj_px": error,
                          "visible": int(visible.sum()), "ms": (time.time() - started) * 1000}
        return calibration, {
            "ok": True, "reason": "ok", "visible": int(visible.sum()),
            "inliers": int(mask.sum()), "reproj_px": error,
            "pixel": pixel.tolist(), "visible_mask": visible.tolist(), "response": response.tolist(),
            "ms": (time.time() - started) * 1000,
        }


__all__ = ["KeypointCalibrator"]
