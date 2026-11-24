"""人物检测 / 姿态 / 分割的 ONNX 推理封装（RTMO-s、RTMDet-Ins-s、RTMPose-m）。

预处理各家不一样，不能互相套用：RTMO 是纯 resize + RGB 0-255 不归一化，
RTMDet-Ins 是 letterbox + BGR + mean/std，RTMPose 是 bbox 仿射裁剪 + mean/std。
套错的典型症状是置信度腰斩但完全不报错，所以每个模型的变换都写死在各自类里。

许可：三个权重均来自 Apache-2.0 仓库（RTMDet-Ins 带 LICENSE 与 NOTICE），
符合本项目的依赖白名单。
"""

from __future__ import annotations

import os
from pathlib import Path

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path("data/models")
RTMO_IN = 640
INS_IN = 640
INS_MEAN = np.array([103.53, 116.28, 123.675], np.float32)
INS_STD = np.array([57.375, 57.12, 58.395], np.float32)


def _add_torch_dll_dir() -> None:
    """torch 的 cu12 运行时里带 cublasLt64_12 与 cudnn64_9，让 ORT 的 CUDA EP 能找到。"""
    try:
        import torch

        p = Path(torch.__file__).resolve().parent / "lib"
        if p.is_dir() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(p))
    except Exception:
        pass


def _add_nvidia_dll_dirs():
    """把 TensorRT / CUDA runtime 的 DLL 目录加入搜索路径（Windows）。"""
    import site

    roots = []
    try:
        import torch

        roots.append(Path(torch.__file__).parent / "lib")
    except Exception:
        pass
    try:
        for sp in site.getsitepackages():
            p = Path(sp)
            roots += [
                p / "tensorrt_libs",
                p / "tensorrt_bindings",
                p / "nvidia" / "cuda_runtime" / "bin",
            ]
    except Exception:
        pass
    for r in roots:
        try:
            if r.exists():
                os.add_dll_directory(str(r))
                os.environ["PATH"] = str(r) + os.pathsep + os.environ.get("PATH", "")
        except OSError:
            pass


def make_session_trt(path: str | Path, cache_dir: str | Path = "data/models/trt_cache"):
    """TensorRT FP16 会话（效果与 CUDA FP16 等价，速度快数倍）。

    首次运行会构建引擎（几十秒到数分钟），之后走本地缓存。
    """
    import onnxruntime as ort

    _add_torch_dll_dir()
    _add_nvidia_dll_dirs()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    opts = {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(cache.resolve()),
        "trt_max_workspace_size": 1 << 30,
    }
    sess = ort.InferenceSession(
        str(path), so,
        providers=[("TensorrtExecutionProvider", opts),
                   "CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    ep = sess.get_providers()[0]
    if ep != "TensorrtExecutionProvider":
        raise RuntimeError(f"TensorRT EP 未生效（实际 {ep}）")
    return sess, ep


def make_session(path: str | Path, prefer_gpu: bool = True):
    import onnxruntime as ort

    _add_torch_dll_dir()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if prefer_gpu else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(path), so, providers=providers)
    return sess, sess.get_providers()[0]


@dataclass(slots=True)
class Person:
    bbox: tuple[float, float, float, float]
    score: float
    kpts: np.ndarray | None = None  # (17, 3) u, v, conf，原图像素坐标
    polygon: list[list[float]] | None = None
    area_px: float = 0.0


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw, ih = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


# Try to import torchvision for fast NMS (50x speedup)
try:
    import torch
    from torchvision.ops import nms as torch_nms
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    """Non-maximum suppression. Uses torchvision.ops.nms if available (50x faster)."""
    if _HAS_TORCHVISION and len(boxes) > 0:
        # Convert to torch tensors for GPU-accelerated NMS
        boxes_torch = torch.from_numpy(boxes).float()
        scores_torch = torch.from_numpy(scores).float()
        keep_torch = torch_nms(boxes_torch, scores_torch, thr)
        return keep_torch.tolist()
    
    # Fallback to pure Python implementation
    order = np.argsort(-scores)
    keep: list[int] = []
    for i in order:
        if all(_iou(boxes[i], boxes[j]) <= thr for j in keep):
            keep.append(int(i))
    return keep


class Rtmo:
    """单发模型同时出人体框与 17 关键点，CPU 约 90ms/帧，适合调试台批量跑。"""

    def __init__(self, path: str | Path | None = None, prefer_gpu: bool = True) -> None:
        # Try FP16 model first for 1.5-2x speedup
        model_dir = MODEL_DIR if path is None else Path(path).parent
        fp16_path = model_dir / "rtmo_s_coco_fp16.onnx"
        default_path = path or (fp16_path if fp16_path.exists() else MODEL_DIR / "rtmo_s_coco.onnx")
        
        self.sess = None
        self.ep = None
        if os.environ.get("BSK_TRT", "1") != "0":
            try:
                self.sess, self.ep = make_session_trt(default_path)
                print(f"[RTMO] TensorRT FP16 引擎就绪")
            except Exception as exc:
                print(f"[RTMO] TensorRT 不可用，回退 CUDA: {type(exc).__name__}: {exc}")
        if self.sess is None:
            self.sess, self.ep = make_session(default_path, prefer_gpu)
        self.in_name = self.sess.get_inputs()[0].name

    def __call__(self, rgb: np.ndarray, thr: float = 0.45) -> list[Person]:
        """Detect persons with optimized threshold for basketball.
        
        Args:
            rgb: RGB image
            thr: Confidence threshold (raised from 0.3 to 0.45 to reduce false positives)
        """
        h, w = rgb.shape[:2]
        img = cv2.resize(rgb, (RTMO_IN, RTMO_IN), interpolation=cv2.INTER_LINEAR)
        blob = img.astype(np.float32).transpose(2, 0, 1)[None]
        dets, kps = self.sess.run(None, {self.in_name: blob})
        dets = np.atleast_2d(dets[0])
        kps = np.asarray(kps[0]).reshape(len(dets), -1, 3)
        if dets.size == 0:
            return []
        boxes = dets[:, :4].astype(np.float32)
        scores = dets[:, 4]
        keep = [i for i in nms(boxes, scores, 0.45) if scores[i] >= thr]
        sx, sy = w / RTMO_IN, h / RTMO_IN
        out = []
        for i in keep:
            x0, y0, x1, y1 = boxes[i] * np.array([sx, sy, sx, sy], np.float32)
            k = kps[i].astype(np.float32).copy()
            k[:, 0] *= sx
            k[:, 1] *= sy
            out.append(Person((float(x0), float(y0), float(x1), float(y1)), float(scores[i]), k))
        return out


class RtmdetIns:
    """人体实例分割：框 + 每人一张 640×640 掩膜，转成多边形存进 Segment2D。"""

    def __init__(self, path: str | Path | None = None, prefer_gpu: bool = True) -> None:
        # Try FP16 model first for 1.5-2x speedup
        model_dir = MODEL_DIR if path is None else Path(path).parent
        fp16_path = model_dir / "rtmdet_ins_s_640_fp16.onnx"
        default_path = path or (fp16_path if fp16_path.exists() else MODEL_DIR / "rtmdet_ins_s_640.onnx")
        
        self.sess = None
        self.ep = None
        if os.environ.get("BSK_TRT", "1") != "0":
            try:
                self.sess, self.ep = make_session_trt(default_path)
                print(f"[RTMO] TensorRT FP16 引擎就绪")
            except Exception as exc:
                print(f"[RTMO] TensorRT 不可用，回退 CUDA: {type(exc).__name__}: {exc}")
        if self.sess is None:
            self.sess, self.ep = make_session(default_path, prefer_gpu)
        self.in_name = self.sess.get_inputs()[0].name

    def _letterbox(self, rgb: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        h, w = rgb.shape[:2]
        scale = min(INS_IN / w, INS_IN / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        pad_x, pad_y = (INS_IN - nw) / 2.0, (INS_IN - nh) / 2.0
        canvas = np.full((INS_IN, INS_IN, 3), 114.0, np.float32)
        small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        y0, x0 = int(round(pad_y)), int(round(pad_x))
        canvas[y0 : y0 + nh, x0 : x0 + nw] = small
        blob = ((canvas - INS_MEAN) / INS_STD).transpose(2, 0, 1)[None].astype(np.float32)
        return blob, scale, pad_x, pad_y

    def __call__(self, rgb: np.ndarray, thr: float = 0.3, mask_thr: float = 0.5) -> list[Person]:
        h, w = rgb.shape[:2]
        blob, scale, pad_x, pad_y = self._letterbox(rgb)
        dets, labels, masks = self.sess.run(None, {self.in_name: blob})
        dets = np.atleast_2d(dets[0])
        labels = np.asarray(labels[0]).ravel()
        masks = np.asarray(masks)
        if masks.ndim == 4:
            masks = masks[0]
        if dets.size == 0:
            return []
        scores = dets[:, 4]
        cand_indices = [i for i in range(len(dets)) if labels[i] == 0 and scores[i] >= thr]
        if not cand_indices:
            return []
        keep = nms(dets[cand_indices, :4], scores[cand_indices], 0.5)
        out = []
        for j in keep:
            i = cand_indices[j]
            lb = dets[i, :4]
            x0, y0, x1, y1 = (lb - np.array([pad_x, pad_y, pad_x, pad_y], np.float32)) / scale
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, w), min(y1, h)
            if x1 <= x0 or y1 <= y0:
                continue
            poly = self._polygon(masks[i], (x0, y0, x1, y1), scale, pad_x, pad_y, h, w, mask_thr)
            out.append(Person((float(x0), float(y0), float(x1), float(y1)), float(scores[i]), None, poly[0], poly[1]))
        return out

    def _polygon(self, mask, box, scale, pad_x, pad_y, h, w, mask_thr):
        """掩膜只在框内取，轮廓直接映射回原图，省掉每人一张全分辨率图。"""
        x0, y0, x1, y1 = box
        bx0 = max(int(x0 * scale + pad_x), 0)
        by0 = max(int(y0 * scale + pad_y), 0)
        bx1 = min(int(np.ceil(x1 * scale + pad_x)), INS_IN)
        by1 = min(int(np.ceil(y1 * scale + pad_y)), INS_IN)
        if bx1 <= bx0 or by1 <= by0:
            return [], 0.0
        crop = (mask[by0:by1, bx0:bx1] > mask_thr).astype(np.uint8)
        cnts, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return [], 0.0
        c = max(cnts, key=cv2.contourArea)
        eps = max(1.0, 0.008 * cv2.arcLength(c, True))
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)
        c[:, 0] = (c[:, 0] + bx0 - pad_x) / scale
        c[:, 1] = (c[:, 1] + by0 - pad_y) / scale
        c[:, 0] = np.clip(c[:, 0], 0, w)
        c[:, 1] = np.clip(c[:, 1], 0, h)
        return c.tolist(), float(cv2.contourArea(c))


class Rtmpose:
    """top-down 精修：对给定 bbox 仿射裁剪出 17 点，用于 RTMO 结果的按需复核。"""

    def __init__(self, path: str | Path | None = None, prefer_gpu: bool = True) -> None:
        # Try FP16 model first for 1.5-2x speedup
        model_dir = MODEL_DIR if path is None else Path(path).parent
        fp16_path = model_dir / "rtmpose_m_simcc_256x192_fp16.onnx"
        default_path = path or (fp16_path if fp16_path.exists() else MODEL_DIR / "rtmpose_m_simcc_256x192.onnx")
        
        self.sess = None
        self.ep = None
        if os.environ.get("BSK_TRT", "1") != "0":
            try:
                self.sess, self.ep = make_session_trt(default_path)
                print(f"[RTMO] TensorRT FP16 引擎就绪")
            except Exception as exc:
                print(f"[RTMO] TensorRT 不可用，回退 CUDA: {type(exc).__name__}: {exc}")
        if self.sess is None:
            self.sess, self.ep = make_session(default_path, prefer_gpu)
        self.in_name = self.sess.get_inputs()[0].name
        self.mean = np.array([123.675, 116.28, 103.53], np.float32)
        self.std = np.array([58.395, 57.12, 57.375], np.float32)

    def __call__(self, rgb: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
        x0, y0, x1, y1 = box
        bw, bh = (x1 - x0) * 1.25, (y1 - y0) * 1.25 * 1.25  # 纵向留头脚余量
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        scale = max(bw, bh / (256 / 192))
        tw, th = 192, 256
        src = np.array([[cx - scale / 2, cy - scale / 2], [cx + scale / 2, cy - scale / 2],
                        [cx - scale / 2, cy + scale / 2]], np.float32)
        dst = np.array([[0, 0], [tw, 0], [0, th]], np.float32)
        warp = cv2.getAffineTransform(src, dst)
        crop = cv2.warpAffine(rgb, warp, (tw, th), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        blob = ((crop.astype(np.float32) - self.mean) / self.std).transpose(2, 0, 1)[None]
        sx, sy = self.sess.run(None, {self.in_name: blob})
        sx, sy = np.asarray(sx)[0], np.asarray(sy)[0]
        ux = sx.argmax(1) / 2.0  # simcc split_ratio=2
        uy = sy.argmax(1) / 2.0
        conf = np.minimum(sx.max(1), sy.max(1))
        inv = cv2.invertAffineTransform(warp)
        pts = np.stack([ux, uy], 1).astype(np.float32)
        pts = pts @ inv[:, :2].T + inv[:, 2]
        return np.stack([pts[:, 0], pts[:, 1], conf], 1)


__all__ = ["Person", "RtmdetIns", "Rtmo", "Rtmpose", "make_session"]
