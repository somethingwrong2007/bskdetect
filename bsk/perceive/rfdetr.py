"""RF-DETR-S ONNX 推理封装，用于实时球员候选筛选。"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


_PRO_WEIGHTS = Path("data/models/rfdetr_small_pro_10class.onnx")
_LEGACY_WEIGHTS = Path("data/models/rfdetr_small_nba_10class.onnx")
_ENV_WEIGHTS = os.environ.get("BSK_RFDETR_WEIGHTS", "").strip()
DEFAULT_WEIGHTS = (Path(_ENV_WEIGHTS) if _ENV_WEIGHTS
                   else (_PRO_WEIGHTS if _PRO_WEIGHTS.exists() else _LEGACY_WEIGHTS))


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    conf: float
    bbox: tuple[float, float, float, float]


class RfDetrDetector:
    """输出 query 框与逐类 logits；只依赖当前帧，不缓存未来检测。"""

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device: str = "auto",
                 conf_thr: float = 0.4, nms_thr: float = 0.5) -> None:
        self.path = Path(weights)
        self.conf_thr = conf_thr
        self.nms_thr = nms_thr
        self.session = None
        self.ep = None
        self.error: str | None = None
        self._ms = 0.0
        self._n = 0
        sidecar = self.path.with_suffix(".json")
        if not self.path.exists():
            self.error = f"权重不存在: {self.path}"
            return
        if not sidecar.exists():
            self.error = f"缺少 sidecar: {sidecar}"
            return
        try:
            self.meta = json.loads(sidecar.read_text(encoding="utf-8"))
            self.size = int(self.meta["input_size"])
            self.class_names = list(self.meta["class_names"])
            self.label_offset = int(self.meta.get("label_offset", 1))
            self.mean = np.asarray(self.meta["mean"], np.float32)[None, None, :]
            self.std = np.asarray(self.meta["std"], np.float32)[None, None, :]
            from bsk.perceive.persons import _add_nvidia_dll_dirs, _add_torch_dll_dir

            _add_torch_dll_dir()
            _add_nvidia_dll_dirs()
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.log_severity_level = 3
            providers = ["CPUExecutionProvider"]
            if device in ("auto", "cuda"):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(self.path), options, providers=providers)
            self.ep = self.session.get_providers()[0]
        except Exception as exc:
            self.error = f"RF-DETR ONNX 会话创建失败: {type(exc).__name__}: {exc}"

    @property
    def ready(self) -> bool:
        return self.session is not None

    def _letterbox(self, rgb: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = rgb.shape[:2]
        scale = min(self.size / w, self.size / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.size, self.size, 3), np.uint8)
        x0, y0 = (self.size - nw) // 2, (self.size - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas, scale, x0, y0

    def predict(self, rgb: np.ndarray, conf: float | None = None,
                max_det: int = 100) -> list[Detection]:
        if self.session is None:
            raise RuntimeError(self.error or "RF-DETR 未就绪")
        threshold = self.conf_thr if conf is None else conf
        image, scale, pad_x, pad_y = self._letterbox(rgb)
        x = image.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        t0 = time.perf_counter()
        dets, logits = self.session.run(None, {self.session.get_inputs()[0].name: x})
        self._ms += (time.perf_counter() - t0) * 1000.0
        self._n += 1
        dets, logits = np.asarray(dets)[0], np.asarray(logits)[0]
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        flat = probs.reshape(-1)
        order = np.argsort(flat)[::-1][:300]
        h, w = rgb.shape[:2]
        by_class: dict[int, list[tuple[float, tuple[float, float, float, float]]]] = {}
        for index in order:
            score = float(flat[index])
            if score <= threshold:
                break
            label = int(index % logits.shape[1])
            if label < self.label_offset or label >= len(self.class_names):
                continue
            query = int(index // logits.shape[1])
            cx, cy, bw, bh = (float(v) * self.size for v in dets[query])
            x0, y0 = (cx - bw / 2 - pad_x) / scale, (cy - bh / 2 - pad_y) / scale
            x1, y1 = (cx + bw / 2 - pad_x) / scale, (cy + bh / 2 - pad_y) / scale
            x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            class_id = label - self.label_offset
            by_class.setdefault(class_id, []).append((score, (x0, y0, x1, y1)))
        out: list[Detection] = []
        for class_id, rows in by_class.items():
            boxes = np.asarray([[x0, y0, x1 - x0, y1 - y0]
                                for _s, (x0, y0, x1, y1) in rows], np.float32)
            scores = np.asarray([s for s, _b in rows], np.float32)
            keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), threshold, self.nms_thr)
            for i in np.asarray(keep).reshape(-1):
                score, box = rows[int(i)]
                out.append(Detection(class_id, self.class_names[class_id], score, box))
        out.sort(key=lambda d: -d.conf)
        return out[:max_det]

    @property
    def mean_ms(self) -> float:
        return self._ms / self._n if self._n else 0.0


__all__ = ["Detection", "RfDetrDetector"]
