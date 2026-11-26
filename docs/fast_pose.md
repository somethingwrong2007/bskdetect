# 实时姿态与检测链路

当前实时链路只使用许可清晰的 ONNX 模型，不依赖 Ultralytics/YOLO。

## 模型分工

- **RTMO-s**：每帧输出人体框与 17 点 COCO 关键点，支持 TensorRT FP16。
- **RF-DETR-S 10 类**：输出 `ball`、`ball-in-basket`、`number`、球员动作子类、`referee`、`rim`。
- **RTMDet-Ins-s**：离线/辅助实例分割，不参与 YOLO 替代。

RF-DETR 每帧只推理一次，球员框、号码框和球框共享结果；球员框通过 IoU 与 RTMO 关键点配对。

## 运行方式

```bash
.venv/Scripts/python.exe -u -m uvicorn bsk.server:app --host 127.0.0.1 --port 8000
```

默认请求使用 `quality=accurate`。旧 API 的 `quality=fast` 作为兼容别名，仍映射到同一 RTMO + RF-DETR 链路。

## 相关开关

- `BSK_PLAYER_DETECTOR=pose`：保留 RTMO 人体框，不使用 RF-DETR 球员框覆盖。
- `BSK_JERSEY=0`：关闭号码识别。
- `BSK_RFDETR_WEIGHTS`：显式指定 RF-DETR ONNX；未指定时优先使用职业比赛联合训练权重。
- `BSK_TRT=0`：关闭 RTMO TensorRT，回退 CUDA/CPU ONNX Runtime。
- `BSK_BALL_WEIGHTS` 已移除；球检测权重由 RF-DETR sidecar 配置。
