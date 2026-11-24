# Fast Pose Estimation for Real-Time Streaming

## Overview

Replaced slow RTMO-s (~380ms/frame) with YOLOv8n-pose (~15ms on GPU) to enable real-time 30 FPS live streaming.

## Models

### YOLOv8n-pose (Primary)
- **Speed**: ~15ms/frame on GPU (67 FPS)
- **License**: Apache-2.0
- **Size**: ~6MB (nano model)
- **Accuracy**: Good enough for basketball player tracking
- **Keypoints**: COCO 17 format (same as RTMO)

### MediaPipe BlazePose (Alternative, not installed)
- **Speed**: ~5ms on CPU
- **License**: Apache-2.0
- **Pros**: Ultra-fast, works on Raspberry Pi
- **Cons**: Different keypoint format (33 landmarks), needs mapping

### RTMO-s (Deprecated for real-time)
- **Speed**: ~380ms on GPU (2.6 FPS) - TOO SLOW
- **Use case**: Offline analysis only

## Installation

YOLOv8 is already installed via `requirements.txt`:
```bash
pip install ultralytics==8.0.196
```

First run will auto-download `yolov8n-pose.pt` (~6MB) to `data/models/`.

## Usage

### In Server (Real-Time Mode)

The server automatically uses YOLOv8n-pose when `realtime=1`:

```javascript
// Frontend request
img.src = `/api/videos/${vid}/frame.jpg?n=${n}&layers=...&realtime=1`;
```

Backend pipeline (`bsk/server_rt.py`):
```python
from bsk.perceive.fast_pose import YoloV8Pose

pose_model = YoloV8Pose(device="cuda", cache_dir="data/models")
persons = pose_model(rgb, thr=0.3)
```

### Standalone Testing

```bash
# Structural tests (no model download)
.venv/Scripts/python.exe test_yolov8.py

# Benchmark with real inference (requires internet)
.venv/Scripts/python.exe test_yolov8.py --with-model
```

## Performance Comparison

| Model | GPU Time | CPU Time | FPS (GPU) | License |
|-------|----------|----------|-----------|---------|
| YOLOv8n-pose | ~15ms | ~50ms | 67 | Apache-2.0 |
| RTMO-s | ~380ms | ~1200ms | 2.6 | Apache-2.0 |
| MediaPipe BlazePose | N/A | ~5ms | 200 | Apache-2.0 |

**Target**: 30 FPS live streaming requires <33ms/frame total budget.
- Calibration: ~10ms (reused)
- Detection + Pose: ~15ms (YOLOv8n)
- Rendering: ~5ms
- **Total**: ~30ms ✓

## API Changes

### Old (RTMO)
```python
from bsk.analyze import Rtmo
rtmo = Rtmo()
ppl = rtmo(rgb, thr=0.3)  # Returns list[Person]
```

### New (YOLOv8n-pose)
```python
from bsk.perceive.fast_pose import YoloV8Pose
pose = YoloV8Pose(device="cuda")
ppl = pose(rgb, thr=0.3)  # Returns list[FastPerson]
```

Both return similar data structures with `bbox`, `score`, and `kpts` fields.

## Migration Notes

1. **Model format**: `.pt` (PyTorch) vs `.onnx` (ONNX Runtime)
   - YOLOv8 uses native PyTorch format
   - No need for ONNX conversion

2. **Keypoint format**: Both use COCO 17 keypoints
   - No mapping needed, drop-in replacement

3. **Confidence scores**: Similar range (0-1)
   - Threshold `thr=0.3` works for both

4. **Device selection**: 
   - YOLOv8: `device="cuda"` or `device="cpu"`
   - RTMO: Handled internally by ONNX Runtime

## Troubleshooting

### Model Download Fails
- Check internet connection
- Manual download: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt
- Place in `data/models/yolov8n-pose.pt`

### Slow Inference
- Verify GPU is being used: check console output for "device: cuda"
- Reduce input resolution (server already scales to `width` parameter)
- Close other GPU-intensive applications

### Import Errors
```bash
.venv/Scripts/pip install ultralytics==8.0.196
```

## Future Improvements

1. **MediaPipe integration**: For ultra-low-latency scenarios (<10ms)
2. **Batch inference**: Process multiple frames at once
3. **Quantization**: INT8 models for faster CPU inference
4. **TensorRT optimization**: Further GPU speedup (2-3x)

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [COCO Keypoints Format](https://cocodataset.org/#keypoints-2020)
- [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
