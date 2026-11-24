"""篮球检测器：YOLOv8n 实时检测。

性能目标：
- 推理速度 < 10ms/frame on GPU
- mAP@0.5 > 0.90
- 支持小目标检测（篮球直径通常 20-60px）
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np


class BallDetection(NamedTuple):
    """单次球检测结果。"""
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    conf: float
    track_id: int | None = None  # 可选：跟踪 ID


class BallDetector:
    """篮球检测器封装。

    懒加载模型，首次调用时初始化。
    """

    def __init__(self, model_path: str | Path | None = None, device: str = "cuda", cache_dir: str | Path | None = None):
        """
        Args:
            model_path: 模型权重路径（None 则用默认训练结果）
            device: 推理设备 ("cuda" / "cpu")
            cache_dir: 模型缓存目录
        """
        self.model_path = model_path
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._class_id = 0       # 自训单类模型：ball=0
        self._is_coco = False

    def _load_model(self):
        """懒加载 YOLO 模型。"""
        from ultralytics import YOLO
        import torch

        # Fix PyTorch 2.6 weights_only issue
        old_load = torch.load
        def safe_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return old_load(*args, **kwargs)
        torch.load = safe_load

        try:
            env_path = os.environ.get("BSK_BALL_WEIGHTS", "").strip()
            candidates = [
                Path(env_path) if env_path else None,
                self.cache_dir / "ball_yolov8n.pt",
                Path("runs/detect/ball_det-3/weights/best.pt"),
                Path("runs/detect/ball_det-2/weights/best.pt"),
                Path("runs/detect/ball_det/weights/best.pt"),
            ]
            if self.model_path and Path(self.model_path).exists():
                model_path = str(self.model_path)
                self._class_id, self._is_coco = 0, False
            else:
                found = next((p for p in candidates if p is not None and p.exists()), None)
                if found is not None:
                    model_path = str(found)
                    self._class_id, self._is_coco = 0, False
                    print(f"[ball] weights: {model_path}")
                else:
                    # 只有在完全没有自训权重时才退回 COCO；COCO 的 ball 是 class 32
                    print("[warn] Ball detector weights missing; fallback COCO YOLOv8n (sports ball=32)")
                    model_path = "yolov8n.pt"
                    self._class_id, self._is_coco = 32, True

            self._model = YOLO(model_path)
            self._model.to(self.device)
        finally:
            torch.load = old_load

    def __call__(self, rgb: np.ndarray, thr: float = 0.3,
                 max_det: int = 3) -> list[BallDetection]:
        """检测图像中的篮球。

        Args:
            rgb: RGB 图像 [H, W, 3]
            thr: 置信度阈值

        Returns:
            检测结果列表（按置信度降序）
        """
        if self._model is None:
            self._load_model()

        # YOLO 推理（自动 resize 到 640x640）
        results = self._model(rgb, verbose=False, conf=thr, iou=0.5)

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                if conf < thr:
                    continue

                # 自训单类模型 ball=0；COCO 回退时 sports ball=32，绝不能把 person 当球
                cls = int(boxes.cls[i])
                if cls != self._class_id:
                    continue

                # 提取 bbox (xyxy 格式)
                xyxy = boxes.xyxy[i].cpu().numpy()
                x0, y0, x1, y1 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

                detections.append(BallDetection(
                    bbox=(x0, y0, x1, y1),
                    conf=conf,
                ))

        # 按置信度排序
        detections.sort(key=lambda d: d.conf, reverse=True)

        # 单场比赛通常只有一颗球：过滤掉尺寸/形状明显不像球的框，
        # 再用橙色占比压掉球衣/广告/记分牌等误检，最后保留前 max_det 个。
        h, w = rgb.shape[:2]
        filtered: list[BallDetection] = []
        for det in detections:
            x0, y0, x1, y1 = det.bbox
            bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
            if max(bw, bh) < 6 or min(bw, bh) / max(bw, bh) < 0.45:
                continue
            if max(bw, bh) > 0.18 * max(h, w):
                continue
            crop = rgb[max(0, int(y0)):min(h, int(y1)),
                       max(0, int(x0)):min(w, int(x1))]
            if crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            orange = ((hh <= 32) & (ss >= 70) & (vv >= 35)).mean()
            if orange < 0.12:
                continue
            filtered.append(det)
            if len(filtered) >= max(1, int(max_det)):
                break
        return filtered

    def visualize(self, rgb: np.ndarray, detections: list[BallDetection]) -> np.ndarray:
        """在图像上绘制检测结果。

        Args:
            rgb: RGB 图像
            detections: 检测结果列表

        Returns:
            绘制后的图像
        """
        img = rgb.copy()
        for det in detections:
            x0, y0, x1, y1 = (int(v) for v in det.bbox)
            conf = det.conf

            # 绘制边界框（橙色）
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 140, 0), 2, cv2.LINE_AA)

            # 绘制标签
            label = f"ball {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x0, y0 - th - 6), (x0 + tw, y0), (255, 140, 0), -1)
            cv2.putText(img, label, (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return img


# ── 全局单例（懒加载）──────────────────────────────────────

_instance: BallDetector | None = None


def get_ball_detector(device: str = "cuda") -> BallDetector:
    """获取全局球检测器实例。"""
    global _instance
    if _instance is None or _instance.device != device:
        _instance = BallDetector(device=device)
    return _instance


if __name__ == "__main__":
    # 测试：检测单张图像
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m bsk.perceive.ball <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {img_path}")
        sys.exit(1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = get_ball_detector()

    print("[ball] Detecting...")
    dets = detector(rgb, thr=0.3)
    print(f"[ball] Found {len(dets)} balls:")
    for i, det in enumerate(dets):
        print(f"  #{i}: bbox={det.bbox}, conf={det.conf:.3f}")

    # 可视化
    vis = detector.visualize(rgb, dets)
    out_path = "test_ball_detection.jpg"
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"[ball] Saved visualization: {out_path}")
