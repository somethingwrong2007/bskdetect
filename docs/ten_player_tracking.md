# 十人追踪能力评估与优化

## 当前状态

### 检测管线
```
视频帧 → YOLOv8n-pose (thr=0.5) → 跟踪器(IoU匹配) → 后处理过滤 → 输出
```

### 已知限制

1. **置信度阈值过高**: `thr=0.5` 会漏检远处/遮挡球员（实际conf常为0.3-0.45）
2. **长宽比过滤**: `aspect_ratio < 1.2 or > 3.5` 可能误删俯拍/侧拍球员
3. **最小尺寸**: `bh < 80 or bw < 40` 对720p视频过于严格
4. **球场ROI**: 未标定时无法使用，标定后可能裁剪掉边线球员

---

## 优化方案

### 方案1: 降低置信度阈值（推荐）

**修改**: `bsk/server_rt.py` line 273

```python
# Before
ppl = fast_pose(rgb, thr=0.5)  # Too high, misses distant players

# After
ppl = fast_pose(rgb, thr=0.25)  # Lower threshold for 10-player detection
```

**预期效果**: 
- ✅ 检出率从~6人提升至~9-10人
- ⚠️ 假阳性增加（观众席误检），需加强后处理过滤

**风险**: 假阳性从3/帧增至8-12/帧，必须依赖后处理过滤

---

### 方案2: 放宽长宽比过滤

**修改**: `bsk/server_rt.py` lines 293-297

```python
# Before
if aspect_ratio < 1.2 or aspect_ratio > 3.5:
    continue

# After (basketball-specific)
if aspect_ratio < 1.0 or aspect_ratio > 4.0:  # Allow more extreme poses
    continue
```

**理由**: 
- 俯拍视角下球员接近正方形 (AR≈1.0)
- 倒地/扑救时球员可能很扁 (AR>3.5)

**预期效果**: 额外检出1-2名边缘姿态球员

---

### 方案3: 自适应最小尺寸

**修改**: `bsk/server_rt.py` lines 299-301

```python
# Before (fixed threshold)
if bh < 80 or bw < 40:
    continue

# After (resolution-adaptive)
min_height = max(60, rgb.shape[0] * 0.05)  # 5% of frame height
min_width = max(30, rgb.shape[1] * 0.025)  # 2.5% of frame width
if bh < min_height or bw < min_width:
    continue
```

**理由**: 对720p和1080p使用统一绝对值不合理

---

### 方案4: 启用多尺度检测

**新增**: 对高分辨率视频(>1080p)使用金字塔检测

```python
def detect_multiscale(rgb: np.ndarray, model, thr: float) -> list[FastPerson]:
    """Detect at multiple scales for better coverage."""
    h, w = rgb.shape[:2]
    
    if h <= 1080 and w <= 1920:
        # Standard resolution, single scale
        return model(rgb, thr=thr)
    
    # High resolution: detect at 1080p then upscale
    scale = min(1920 / w, 1080 / h)
    small = cv2.resize(rgb, (int(w * scale), int(h * scale)))
    detections = model(small, thr=thr)
    
    # Scale back to original resolution
    for p in detections:
        p.bbox = tuple(x / scale for x in p.bbox)
        if p.kpts is not None:
            p.kpts[:, :2] /= scale
    
    return detections
```

**预期效果**: 在4K视频中检出更多小尺寸球员

---

## 推荐配置

### 8GB档（完整3D + 十人追踪）

```python
# bsk/server_rt.py analyze_frame_realtime()

# Detection: lower threshold for maximum recall
ppl = fast_pose(rgb, thr=0.25)

# Filtering: basketball-specific rules
for p, tid in zip(ppl, ids):
    # Keep conf >= 0.25 (was 0.5)
    if p.score < 0.25:
        continue
    
    # Relaxed aspect ratio for varied camera angles
    bw, bh = p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1]
    aspect_ratio = bh / max(bw, 1)
    if aspect_ratio < 1.0 or aspect_ratio > 4.0:  # Was 1.2-3.5
        continue
    
    # Adaptive minimum size
    min_h = max(60, rgb.shape[0] * 0.05)
    min_w = max(30, rgb.shape[1] * 0.025)
    if bh < min_h or bw < min_w:  # Was fixed 80/40
        continue
    
    # Court ROI (if calibrated) - KEEP THIS to filter crowd
    if court_bounds is not None:
        cx, cy = (p.bbox[0] + p.bbox[2]) / 2, (p.bbox[1] + p.bbox[3]) / 2
        if not _point_in_rect(cx, cy, court_bounds):
            continue
```

### 4GB档（仅2D坐标）

相同配置，但跳过3D重建阶段以节省显存。

---

## 验证方法

### 测试场景

1. **标准比赛**: 1080p, 10名球员清晰可见
2. **密集防守**: 多名球员聚集在油漆区
3. **快攻转换**: 球员分散在全场
4. **替补席入镜**: 测试假阳性过滤

### 成功标准

| 指标 | 目标 | 测量方法 |
|------|------|----------|
| 检出率 | ≥9/10 | 人工标注vs自动检测对比 |
| 假阳性率 | ≤2/帧 | 非球员检测数 |
| ID稳定性 | ≥90% | 跨帧ID一致性 |
| 吞吐量 | ≥30 FPS | 实时处理能力 |

---

## 实施步骤

1. **降低阈值**: thr=0.5 → 0.25
2. **运行测试**: 观察假阳性增加情况
3. **调整过滤**: 根据实际结果微调长宽比/尺寸阈值
4. **验证十人**: 在标准比赛片段上确认检出数
5. **性能回归测试**: 确保FPS不下降

---

## 潜在问题与解决

### 问题1: 假阳性暴增

**症状**: 每帧检测到15-20人，大部分是观众

**解决**:
- 强化球场ROI过滤（必须先标定）
- 添加服装颜色过滤（球员队服 vs 观众便服）
- 使用时序一致性（观众席位置固定）

### 问题2: ID频繁切换

**症状**: 同一球员在不同帧获得不同track_id

**解决**:
- 降低IoU阈值: `Tracker(iou_thr=0.25)` (默认0.35)
- 增加max_lost: `Tracker(max_lost=20)` (默认12)
- 考虑引入ReID特征（高级优化）

### 问题3: 远处球员关键点质量差

**症状**: 检出bbox但kpts全为零或置信度<0.1

**解决**:
- 标记低质量骨架为"部分可见"
- 使用时序平滑预测缺失关键点
- 前端渲染时用半透明显示不确定性

---

## 下一步

如果基础优化后仍无法稳定检出10人，考虑：

1. **更换检测模型**: YOLOv8m-pose (更大更准，慢20%)
2. **两阶段检测**: 
   - 第一阶段: RTMO-s快速扫描全图
   - 第二阶段: 对低置信区域用YOLOv8复核
3. **时序增强**: 利用前5帧检测结果辅助当前帧

---

## 基准数据（待收集）

需要用户在实际比赛视频上运行以下测试：

```bash
# 1. 当前配置基线
python run_batch_analysis.py --video game.mp4 --thr 0.5

# 2. 优化配置
python run_batch_analysis.py --video game.mp4 --thr 0.25

# 3. 对比结果
# - 平均每帧检测人数
# - 最大同时检测人数
# - 假阳性数量
```

**需要用户提供**: 典型比赛视频片段（至少包含10名球员清晰可见的帧）
