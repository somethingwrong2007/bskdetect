"""训练篮球检测器：YOLOv8n 从合成数据学习。

架构选择：
- YOLOv8n（nano）：3.2M 参数，~5ms/frame on GPU
- 单类别检测（ball only）
- 输入分辨率 640x640（篮球是小目标，不需要高分辨率）

训练策略：
1. 先用合成数据预训练（10k+ 样本）
2. 再用真实视频标注微调（待人工标注后）
3. 数据增强：Mosaic + Mixup + HSV

性能目标：
- mAP@0.5 > 0.85（合成数据上应接近1.0）
- 推理速度 < 10ms/frame on RTX 3060
"""

from __future__ import annotations

from pathlib import Path


def train_ball_detector(
    data_dir: str | Path = "data/ball_synth",
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    pretrained: bool = True,
):
    """训练球检测模型。

    Args:
        data_dir: 数据集目录（含 dataset.yaml）
        epochs: 训练轮数
        batch_size: 批次大小
        imgsz: 输入图像尺寸
        pretrained: 是否使用 COCO 预训练权重
    """
    from ultralytics import YOLO
    import torch
    from pathlib import Path

    # Fix PyTorch 2.6 weights_only issue
    old_load = torch.load
    def safe_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return old_load(*args, **kwargs)
    torch.load = safe_load

    try:
        # 检查 CUDA
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[train_ball] Using device: {device}")

        # 转换数据路径为绝对路径（避免 ultralytics tarfile 检测问题）
        data_path = Path(data_dir).resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        # ultralytics 需要 YAML 文件路径，不是目录
        yaml_path = data_path / "dataset.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"dataset.yaml not found: {yaml_path}")

        # 加载 YOLOv8n（COCO 预训练）
        model = YOLO("yolov8n.pt") if pretrained else YOLO("yolov8n.yaml")
    finally:
        torch.load = old_load

    # 训练配置（使用 YAML 文件路径）
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        name="ball_det",
        device=device,
        # 超参数优化
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        # 数据增强
        hsv_h=0.015,  # HSV-Hue 增强
        hsv_s=0.7,    # HSV-Saturation
        hsv_v=0.4,    # HSV-Value
        degrees=0.0,  # 不旋转（球是圆形）
        translate=0.1,
        scale=0.5,    # 尺度变化（模拟远近）
        shear=0.0,
        perspective=0.0,
        flipud=0.0,   # 不垂直翻转（重力方向固定）
        fliplr=0.5,   # 水平翻转
        mosaic=1.0,   # Mosaic 增强
        mixup=0.1,    # Mixup 增强
        copy_paste=0.0,
        # 验证
        val=True,
        patience=10,  # early stopping
        save_period=10,
    )

    # 保存最佳模型
    best_model = Path("runs/detect/ball_det/weights/best.pt")
    if best_model.exists():
        print(f"[train_ball] Best model saved to: {best_model}")

    return results


def export_to_onnx(model_path: str | Path = "runs/detect/ball_det/weights/best.pt",
                   imgsz: int = 640):
    """导出为 ONNX 格式（用于部署）。"""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    onnx_path = model.export(format="onnx", imgsz=imgsz, dynamic=False)
    print(f"[train_ball] Exported to ONNX: {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    import sys

    # 默认参数
    data_dir = "data/ball_synth"
    epochs = 100
    batch_size = 16

    # 命令行覆盖
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        epochs = int(sys.argv[2])
    if len(sys.argv) > 3:
        batch_size = int(sys.argv[3])

    print(f"[train_ball] Training ball detector:")
    print(f"  Data: {data_dir}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")

    # 训练
    results = train_ball_detector(data_dir, epochs, batch_size)

    # 导出
    export_to_onnx()

    print("[train_ball] Done!")
