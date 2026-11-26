"""深度学习标定推理接口：封装模型加载、预处理、后处理。

用法：
    infer = CalibInference('models/court_calib/best_model.pth')
    keypoints, H = infer.predict(image)
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from pathlib import Path

from bsk.calib_net import CourtCalibNet, heatmaps_to_keypoints
from bsk.court import FIBA, Court
from bsk.calibrate import Calibration


class CalibInference:
    """标定推理封装。"""

    def __init__(
        self,
        model_path: str | Path,
        court: Court = FIBA,
        device: str = "cuda",
        input_size: tuple[int, int] = (256, 256),
    ):
        """
        Args:
            model_path: 模型权重路径
            court: 球场几何模型
            device: 'cuda' 或 'cpu'
            input_size: 模型输入尺寸 (H, W)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.court = court
        self.input_size = input_size

        # 加载模型
        self.model = CourtCalibNet(num_keypoints=8, pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 定义 8 个关键点的米制坐标（与训练时一致）
        self.keypoints_m = self._define_court_keypoints()

        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")

    def _define_court_keypoints(self) -> np.ndarray:
        """定义 8 个关键点的米制坐标。"""
        c = self.court
        ft_hw = c.paint_width / 2 + c.line_half
        ft_off = c.free_throw_center_dist
        return np.array([
            # 4 个角点
            [-c.half_length, -c.half_width],
            [c.half_length, -c.half_width],
            [c.half_length, c.half_width],
            [-c.half_length, c.half_width],
            # 4 个罚球区外角
            [-c.half_length + ft_off, -ft_hw],
            [-c.half_length + ft_off, ft_hw],
            [c.half_length - ft_off, -ft_hw],
            [c.half_length - ft_off, ft_hw],
        ], dtype=np.float32)

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """预测单帧的关键点和单应矩阵。

        Args:
            image: [H, W, 3] BGR 或 RGB 图像

        Returns:
            keypoints_px: [8, 2] 像素坐标
            H: [3, 3] 单应矩阵（图像 → 球场），或 None（计算失败）
        """
        # 预处理
        input_tensor = self._preprocess(image)

        # 推理
        with torch.no_grad():
            heatmaps = self.model(input_tensor)
            keypoints_norm = heatmaps_to_keypoints(heatmaps, self.input_size)

        # 后处理
        keypoints_px = keypoints_norm[0].cpu().numpy()  # [8, 2]

        # 计算单应矩阵
        H = self._compute_homography(keypoints_px)

        return keypoints_px, H

    def predict_calibration(self, image: np.ndarray) -> Calibration | None:
        """预测并返回 Calibration 对象。"""
        keypoints_px, H = self.predict(image)

        if H is None:
            return None

        H_inv = np.linalg.inv(H)

        # 计算重投影误差
        reproj_error = self._compute_reprojection_error(keypoints_px, H)

        import time
        return Calibration(
            H=H.tolist(),
            H_inv=H_inv.tolist(),
            reproj_error_px=reproj_error,
            keypoints_used=8,
            court_model=self.court.model,
            timestamp=time.time(),
        )

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """图像预处理：缩放、归一化、转 Tensor。"""
        # BGR -> RGB
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 缩放到输入尺寸
        image = cv2.resize(image, (self.input_size[1], self.input_size[0]))

        # HWC -> CHW, 归一化到 [0, 1]
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0

        # 转 Tensor, 添加 batch 维度
        return torch.from_numpy(image).unsqueeze(0).to(self.device)

    def _compute_homography(self, keypoints_px: np.ndarray) -> np.ndarray | None:
        """从关键点计算单应矩阵（DLT 算法）。"""
        # 图像点 -> 球场点
        H, mask = cv2.findHomography(
            keypoints_px.reshape(-1, 1, 2),
            self.keypoints_m.reshape(-1, 1, 2),
            cv2.RANSAC,
            5.0,
        )
        return H

    def _compute_reprojection_error(
        self, keypoints_px: np.ndarray, H: np.ndarray
    ) -> float:
        """计算重投影误差（像素）。"""
        # 球场点 -> 图像点
        H_inv = np.linalg.inv(H)
        court_h = np.hstack([self.keypoints_m, np.ones((8, 1))]).T  # [3, 8]
        proj = H_inv @ court_h  # [3, 8]
        proj = proj[:2] / proj[2:]  # [2, 8]
        proj = proj.T  # [8, 2]

        # 计算平均误差
        error = np.linalg.norm(keypoints_px - proj, axis=1).mean()
        return float(error)


# 便捷函数
_infer_cache: dict[str, CalibInference] = {}


def get_inference(model_path: str | Path, device: str = "cuda") -> CalibInference:
    """获取或创建推理实例（缓存）。"""
    key = str(model_path)
    if key not in _infer_cache:
        _infer_cache[key] = CalibInference(model_path, device=device)
    return _infer_cache[key]


def predict_frame(image: np.ndarray, model_path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """便捷接口：单帧预测。"""
    infer = get_inference(model_path)
    return infer.predict(image)


if __name__ == "__main__":
    # 测试
    infer = CalibInference("models/court_calib_test/best_model.pth")

    # 创建测试图像
    test_img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    keypoints, H = infer.predict(test_img)
    print(f"Keypoints shape: {keypoints.shape}")
    print(f"H shape: {H.shape if H is not None else None}")

    cal = infer.predict_calibration(test_img)
    if cal:
        print(f"Calibration: error={cal.reproj_error_px:.2f}px")
