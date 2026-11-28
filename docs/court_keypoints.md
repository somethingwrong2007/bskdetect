# FIBA 球场 16 点在线标定

## 功能与坐标契约

`bsk/perceive/court_keypoints.py` 从当前 RGB 帧检测球场线交点，匹配 FIBA 米制坐标，
使用像素空间 RANSAC 求世界→像素单应，再反转为项目统一的 `Calibration`：

- `H`：图像像素 `(u, v)` → 球场世界 `(x, y)`，单位米。
- `H_inv`：球场世界 → 图像像素。
- 世界原点在球场中心，x 沿长边，y 沿短边；尺寸只引用 `bsk/court.py::FIBA`。
- `valid=False` 时 `H`、`H_inv` 均为 `null`，不会发布旧机位矩阵。

16 个主标签固定为：4 个外框角点、2 个中线端点、4 个罚球线端点、4 个三分线底角
拐点、2 个三分线弧顶。限制区矩形另有 8 个辅助几何角点，参与边线支撑但不占模型标签，
因此不会把“限制区四角”重复计数或挤掉其他必需点位。

标签顺序如下，训练导出和 sidecar 必须保持一致：

```text
corner_A_near, corner_B_near, corner_A_far, corner_B_far,
mid_near, mid_far,
ft_A_near, ft_A_far, ft_B_near, ft_B_far,
three_corner_A_near, three_corner_A_far,
three_corner_B_near, three_corner_B_far,
three_apex_A, three_apex_B
```

## 训练数据与模型

训练使用 Roboflow 公开数据集
[Basketball Court Detection 2 v22](https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2)，
共 850 张图像。源站官方帧级划分为 train/valid/test = 610/124/116，但同视频的相邻帧
会跨划分出现，不能作为独立泛化验收。本项目按文件名中的源视频（比赛）重新分组为
train/valid/test = 626/112/112，分别来自 12/3/3 个视频，三个划分之间没有共享视频。
原数据为 33 点 COCO Keypoint，许可证为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。
源图经 Roboflow 直接 stretch 到 576×576，而运行时通常把 16:9 转播帧直接 resize 到
640×640；训练因此加入独立 x/y 缩放增强来缓解域差，但这不等同于真实转播域验证。
本项目的修改是：按几何对应关系选取并重排 16 个点，将越界点改为 visibility=0，
并在训练时把 COCO visibility 0/1/2 归一为“缺失/存在”二分类。版权图像、派生数据、
checkpoint、ONNX 权重和日志均不提交到仓库。

原数字标签到运行时标签的映射如下；映射由多帧单应对齐恢复，再用标准球场几何校验，
不能按 skeleton 中的旧模板下标猜测：

```text
corner_A_near=01          corner_B_near=08
corner_A_far=41           corner_B_far=34
mid_near=19               mid_far=23
ft_A_near=28              ft_A_far=30
ft_B_near=12              ft_B_far=14
three_corner_A_near=35    three_corner_A_far=40
three_corner_B_near=02    three_corner_B_far=07
three_apex_A=26           three_apex_B=16
```

`bsk/train_court_keypoints.py` 只使用项目现有训练依赖，不引入额外模型框架：

```bash
# 1. 校验源 33 点 schema，生成固定 16 点顺序的派生 COCO 数据
python -m bsk.train_court_keypoints prepare \
  --source data/roboflow/basketball-court-detection-2/v22/extracted \
  --output data/derived/court_keypoints16_official

# 2. 按源视频重新分组，消除同视频相邻帧跨 train/valid/test 的泄漏
python -m bsk.train_court_keypoints regroup \
  --source data/derived/court_keypoints16_official \
  --output data/derived/court_keypoints16_grouped --seed 17

# 3. 生成带精确投影标签的完整全场合成训练图；不使用 test
python -m bsk.train_court_keypoints synthetic \
  --count 1200 --split train --output data/derived/court_keypoints16_synthetic --seed 17

# 另生成不同随机种子的合成 valid，仅用于完整 16 点几何准确性验收
python -m bsk.train_court_keypoints synthetic \
  --count 300 --split valid --output data/derived/court_keypoints16_synthetic --seed 2026

# 4. 预训练 ResNet-18 + 真实稀疏标签 + 合成全场监督
#    验证集只用于选择最佳 checkpoint，test 不参与训练或选模型
python -m bsk.train_court_keypoints train \
  --data data/derived/court_keypoints16_grouped \
  --synthetic-data data/derived/court_keypoints16_synthetic \
  --synthetic-valid data/derived/court_keypoints16_synthetic \
  --architecture resnet18 --pretrained \
  --output data/models/court_keypoints16_resnet.pt \
  --epochs 40 --batch 8 --workers 4 --no-amp --device cuda

# 训练每轮另存 court_keypoints16_resnet_last.pt；中断后可从最新权重热启动。
# --output 始终保留验证集选择出的最佳权重，不会被较晚的差轮覆盖。

# 5. 分别复验真实 valid 与独立合成 valid；确认几何与误检后才进行最终 test
python -m bsk.train_court_keypoints evaluate \
  --data data/derived/court_keypoints16_grouped \
  --checkpoint data/models/court_keypoints16_resnet.pt \
  --split valid --report outputs/court_keypoints_resnet_valid.json
python -m bsk.train_court_keypoints evaluate \
  --data data/derived/court_keypoints16_synthetic \
  --checkpoint data/models/court_keypoints16_resnet.pt \
  --split valid --report outputs/court_keypoints_resnet_synthetic_valid.json

# 6. 先导出到独立路径，并在真实 valid 上完成 CUDA/因果序列验证
python -m bsk.train_court_keypoints export \
  --checkpoint data/models/court_keypoints16_resnet.pt \
  --output data/models/court_keypoints16_resnet.onnx

# 7. 全部模型选择与 valid 验证结束后，才按源视频隔离的 test split 最终验收
python -m bsk.train_court_keypoints evaluate \
  --data data/derived/court_keypoints16_grouped \
  --checkpoint data/models/court_keypoints16_resnet.pt \
  --split test --report outputs/court_keypoints_resnet_test.json
```

保留约 0.570M 参数的轻量 UNet 用于兼容旧 checkpoint 和消融；新训练默认使用
ImageNet 预训练 ResNet-18 编码器和多尺度解码器（约 14.284M 参数）。两种模型均为
640×640 RGB 输入、320×320 的 16 通道 heatmap 和 16 维 presence logits。导出图在
峰值周围执行 5×5 softargmax，直接输出亚像素坐标 `(B,16,3)` 的 `x,y,confidence`；
confidence 为 presence 与 heatmap peak 的 sigmoid 乘积。

真实数据每张通常只有 4～8 个选中标签，单帧四点拟合又会因共线产生巨大外推误差。
`densify` 提供同视频 SIFT/RANSAC 的离线多帧标签迁移，但实测覆盖不足，不作为默认
训练数据；`synthetic` 用完整 FIBA 几何生成全场合成图，所有标签由同一精确单应投影，
用于补齐非共线监督。COCO 原始稀疏标签另存为 `source_keypoints`，验收仍只相信人工真值。

ResNet 新训练使用支持 sm_120 的 PyTorch CUDA 12.8 wheel。曾出现的首轮 NaN 已定位为
训练 batch 为 `uint8`，而 ImageNet 归一化常数误用了输入 dtype，导致标准差截断为零；
现在常数固定为 float32，并加入真实 `uint8` batch 的前向/反向回归。训练循环会在输入、
输出或 loss 非有限时立即中止，不保存伪收敛 checkpoint；默认仍为 FP32/`--no-amp`，
AMP 需另行验证。

验收同时报告 640×640 模型输入尺度下的原始 PCK@5/10/20、`confidence >= 0.35` 的
gated PCK、presence recall 和缺失点误检率。`homography_publish_rate` 会把坐标按
运行时方式折算回原图，调用同款 `_solve_labeled`，并共用“内点占比 × 平均置信度”的
发布门限；只有求解成功且发布置信度达到 `0.55` 时计数。求解前要求像素凸包和世界三角面积
支撑；五点及以上还会做 leave-one-out 稳定性检查。`homography_publish_rate` 与
`homography_publish_accuracy` 分开报告：后者把预测矩阵投影到全部 16 个 FIBA 点，与稀疏
人工标签拟合的参考矩阵比较，要求 16 点误差中位数不超过 20px、p90 不超过 35px。
`homography_accurate_rate` 以全部验收图片为分母，避免只发布一张正确矩阵就得到虚假高分。
`homography_false_publish_rate` 只以“能用人工标签拟合参考 H”的图片为分母；参考 H 不足的
已发布结果记为不可评估，不冒充错误发布；`homography_evaluated_accuracy` 则只统计这些可评估
参考。checkpoint 选择同时使用真实验证 gated PCK@10、独立合成 valid 的 gated PCK@10 与全图
单应准确率，并惩罚错误发布和缺失点误检；真实 test 始终不参与训练、选模型或调参。
原始 PCK 仍作为定位质量指标，但不单独代表在线标定成功率。

## 启用

```bat
REM auto：权重存在且契约合法时启用；否则继续使用 Shi-Tomasi 特征点线标定
set BSK_COURT_KP=auto
set BSK_COURT_KP_WEIGHTS=data/models/court_keypoints16.onnx
set BSK_COURT_KP_EVERY=5

REM 关闭
set BSK_COURT_KP=0

REM 1：显式启用经典 Shi-Tomasi 特征点/边线兜底（无模型时较慢，仅用于排障/离线验证）
set BSK_COURT_KP=1
```

ONNX 模型旁边需要同名 `.json` sidecar，例如 `court_keypoints16.json`：

```json
{
  "input_hw": [640, 640],
  "preprocess": "rgb_0_255",
  "coordinate_space": "input_pixels",
  "mean": [0, 0, 0],
  "std": [1, 1, 1],
  "keypoint_ids": [
    "corner_A_near", "corner_B_near", "corner_A_far", "corner_B_far",
    "mid_near", "mid_far",
    "ft_A_near", "ft_A_far", "ft_B_near", "ft_B_far",
    "three_corner_A_near", "three_corner_A_far",
    "three_corner_B_near", "three_corner_B_far",
    "three_apex_A", "three_apex_B"
  ]
}
```

支持输出 `(1,16,3)`、`(16,3)` 或 48 个展平值，三项依次为 `x,y,confidence`。
`coordinate_space` 可为 `input_pixels` 或 `normalized`；`preprocess` 可为
`rgb_0_255` 或 `gray_clahe`。标签顺序、输入尺寸或输出形状不合法时拒绝加载，
不会猜测计算图。权重和 sidecar 放 `data/models/`，不要提交版权帧或大权重。

Windows 独立 GPU 推理环境使用 `.venv-court-gpu`：Python 3.13、ONNX Runtime
1.29.0 CUDA 12 feed、CUDA Runtime 12.8、cuDNN 9，不安装 PyTorch。环境内
`requirements-court-gpu.txt` 固定了 wheel 与 SHA256；PyPI 同版本 wheel 可能指向
CUDA 13，不能按包名盲目替换。除 cuBLAS/cuDNN/NVRTC 外还必须安装 cuFFT 11.4.1.4
及其 NVJitLink 12.9.86 依赖，否则 ORT 初始化会报 `cufft64_11.dll` 缺失。共享会话
注册表会在建 CUDA session 前把 NVIDIA wheel 的 DLL 目录加入搜索路径并调用
`onnxruntime.preload_dlls`，无 torch 时也能直接加载。运行时必须检查
`CourtKeypointModel.ep == "CUDAExecutionProvider"`，不能只看包已安装。

球衣与球场关键点通过 `bsk/perceive/onnx_sessions.py` 共享进程级会话注册表、
provider 选择和生命周期；不同权重仍使用不同 `InferenceSession`。经典边线兜底使用
`bsk/perceive/court_line_features.py` 的线程安全 OpenCV 实现：默认从画面下半部
学习主场地颜色，结合与场地相连的主白线簇估计 ROI，把 mask 直接传给 Shi-Tomasi
角点检测，再做 HSV 白线和局部强梯度双重过滤；全部有效特征点按“方向角 + 法向距离”
聚类，每个点簇独立执行 RANSAC 和 MAD 离群剔除。不依赖 Canny/Hough 连续线段，
也不引入深度学习。

当前经典兜底默认 `quality_level=0.025`、最少 5 个簇内点、原图 80px 最小线长，
两两无限线求交后输出 `intersection_keypoints`。标定侧临时允许最少 4 个对应点，
默认每 3 帧检测一次，其余帧只复用历史 H；该路径优先保证能产出单应矩阵。

### 假特征点与假边线强化

- **固定地面 ROI**：默认使用 `DEFAULT_FIXED_COURT_POLYGON` 归一化多边形，直接传给
  `goodFeaturesToTrack(mask=...)`，ROI 外不提取角点。显式 `roi=` 优先；可用
  `use_fixed_roi=False` 恢复自动 ROI，或通过 `fixed_roi_polygon=` / `--roi-polygon`
  按实际转播机位覆盖。
- **场地颜色与前景过滤**：从画面下半部学习主场地颜色，并用同一颜色且与下半部
  连通的区域收紧固定 ROI；角点邻域必须有足够场地颜色证据。对场地外的块状前景
  做开运算和连通域筛选，排除白色球衣、广告块及观众席纹理，细白线不会被当作
  前景删除。颜色模型不可用时自动退回原固定多边形，不阻止灰度/合成帧标定。
- **邻域颜色证据**：HSV `V >= white_threshold` 且 `S <= max_saturation` 先形成白线
  证据图，再按 `white_neighborhood_size`（默认 5×5）计算角点邻域白线像素比例，
  低于 `min_point_evidence`（默认 0.35）直接剔除，不再只看角点中心单像素。
- **梯度证据**：Sobel 幅值在 `gradient_neighborhood_size`（默认 5×5）窗口取局部
  最大值，并按 ROI 内 95 分位归一化；低于 `min_gradient_strength`（默认 0.10）的点
  不进入聚类。可选 `player_mask=` 还能在聚类前剔除球员区域内的角点。
- **最小簇点数**：`min_cluster_points` 默认 5，与 RANSAC 的 `min_line_points`
  （默认 8）分离；簇内点数不足时整簇丢弃。
- **拟合后校验**：`min_line_length` 默认 80 像素，同时约束内点跨度和 ROI 内最终
  连续可见线段长度；默认角度白名单为水平/垂直 ±35°，过滤明显斜线，未知机位可清空。
- **重复线合并**：角度差不超过 `merge_angle_deg`（默认 8°）、可见支撑段双向中位
  残差不超过 `distance_threshold × merge_distance_factor`（默认 3 倍）的直线合并，
  对内点并集重新拟合，而不是简单丢弃后一条线。
- **无限线求交**：所有通过校验的拟合线两两求无限延长线交点；近平行线跳过，交点
  必须在 ROI 内，到两条源线距离均不超过 12px，并按 25px 半径聚类去重。
- **输出与可视化**：`LineObservation(points, segments)` 接口不变，新增
  `intersection_keypoints`，元素与 DL 输出同为 `DetectedKeypoint`。经典交点没有
  语义标签，`world=None`，由有界 RANSAC 匹配 FIBA 点位，不伪造语义。红点为有效
  内点，蓝线为拟合边线，绿圈为交点。

## 独立边线特征调试

不启动视频服务、不加载模型，也可以直接对单帧运行：

```bash
# 默认固定地面 ROI；红点=白线特征点，蓝线=拟合线，绿圈=无限线交点
python detect_court_lines.py frame.jpg \
  --max-corners 1200 --quality-level 0.025 \
  --distance-threshold 5 --min-distance 4 \
  --min-cluster-points 5 --min-line-length 80 \
  --white-threshold 145 --max-saturation 90 \
  --white-neighborhood-size 5 --min-point-evidence 0.35 \
  --gradient-neighborhood-size 5 --min-gradient-strength 0.10 \
  --min-line-points 8 --ransac-iterations 300 \
  --intersection-max-line-distance 12 --intersection-cluster-distance 25 \
  --output frame_lines.jpg --json

# 按机位覆盖固定多边形（坐标为相对宽高的 0~1 值）
python detect_court_lines.py frame.jpg \
  --roi-polygon 0.10,0.23,0.90,0.23,0.97,0.92,0.03,0.92 \
  --output frame_fixed_roi.jpg

# 显式球场 mask 优先于固定 ROI；mask 必须与输入图同尺寸
python detect_court_lines.py frame.jpg --roi-mask court_mask.png --output frame_lines.jpg

# 可选：再排除球员区域，mask 中非零像素内的角点不参与聚类
python detect_court_lines.py frame.jpg \
  --roi-mask court_mask.png --player-mask player_mask.png \
  --output frame_without_players.jpg

# 保留的单目标线调试分支；给出 target-line 时自动关闭自动 ROI
python detect_court_lines.py frame.jpg \
  --target-line 0,440,750,310 \
  --output frame_target_line.jpg --json

# 关闭固定 ROI 并使用全图自动聚类
python detect_court_lines.py frame.jpg --no-fixed-roi --no-use-roi-auto

# 已知机位角度时启用角度白名单；度数为无向直线角度，按 [0,180) 处理
python detect_court_lines.py frame.jpg \
  --line-angles 10,100 --angle-prior-tolerance 8 \
  --merge-angle 8 --merge-distance-factor 3 \
  --output frame_angle_filtered.jpg
```

自动 ROI 优先使用当前帧下半部的主场地颜色证据，只保留大面积、靠近画面下方的
场地连通域，再与上方主白线簇合并；颜色不明显时才退回细长白线连通域凸包。这样
看台白字、广告牌和纯色地面不会仅因位于两条线之间就被填入 ROI。自动估计仍是
启发式：俯角很大、场地被大面积遮挡或场地/看台颜色相近时，应传入人工或上游
球场 mask；显式 mask 支持 API 的 `roi=` 参数和命令行 `--roi-mask`，优先级最高。
拟合线的输出端点会裁剪到 ROI 内最长连续段，避免蓝线外延穿过被屏蔽区域。

自动模式的前置过滤为：角点邻域 HSV 高亮度、低饱和白线证据，加上 5×5 邻域 Sobel 强梯度。
纯色地面、低对比纹理和缺少白线/场地边界的角点会被抑制；与白线颜色和梯度都相似的
球衣、广告纹理仍可能需要严格球场 mask 或可选球员 mask 排除。随后使用点对投票
把特征点按角度和法向距离分簇，少于 `min_cluster_points` 的簇直接丢弃；每个簇单独
RANSAC，拟合后再校验最小线长、可选角度先验和 ROI 内连续长度，并合并邻近重复线。
输出图使用红色最终有效内点、蓝色最终校验通过的拟合线，JSON 包含每条线的端点、
方向、法线、自身内点、全部有效内点和统计信息。`stats` 中的 `roi_source`、
`floor_roi_used`、`raw_corners`、`filtered_corners`、`white_rejected`、
`gradient_rejected`、`player_rejected`、`obstacle_rejected`、`floor_rejected`、
`rejected_clusters`、`merged_lines`、`short_lines_rejected`、
`angle_lines_rejected` 和 `roi_lines_rejected` 可用于排查过滤效果。

`--target-line` 使用原图像素坐标 `x1,y1,x2,y2`，保留原来的目标线优先 RANSAC
调试流程，与自动 ROI 模式不冲突；同时提供时以 target-line 分支为准，显式 `roi=`
仍然有效。`--max-corners`、`--quality-level`、`--distance-threshold` 分别控制角点
数量、Shi-Tomasi 质量阈值和点到直线的距离阈值。

## 镜头状态与降级

- `initial`：首个观测或分辨率改变，重新求解。
- `static`：直方图、归一化 MAD 和相位相关均表明机位稳定。
- `pan` / `zoom`：使用过去 H 作为先验，平移时逐帧累积光流位移，并对参考点投影做
  EMA 平滑；不预取未来帧。
- `hard_cut`：清空关键点缓存、旧 H、旧线标定跟踪器和球场 ROI，当帧强制重新检测/
  RANSAC。旧机位结果不会继续用于球员剔除或 3D。
- 深度学习点位最少允许 4 个，但必须通过非共线、像素凸包和留一法稳定支撑检查；
  共线四点或外推不稳定的五点组合直接拒绝。只有同时存在至少两个非平行线族且
  线段支撑达标时，才允许 `keypoints+lines`；有可信历史机位时可用 `line_aid` 做小幅平移修正。
- 无历史机位的局部特写无法唯一确定全场坐标，返回 `valid=False`，不猜 H。
- NaN/Inf、越界点、低置信点、退化和不可信单应都会被拒绝。

## 实时接口

先请求实时帧触发分析，再读取缓存，JSON 接口本身不会再次推理：

```text
GET /api/videos/{vid}/rt_frame.jpg?n=100&quality=accurate
GET /api/videos/{vid}/court_keypoints.json?n=100&quality=accurate
GET /api/videos/{vid}/rt_meta.json?n=100&quality=accurate
```

`court_keypoints.json` 返回全部 16 个标签的像素坐标、FIBA 世界坐标、置信度、内点标记、
`H`、`H_inv`、镜头模式、失效原因和统计。`rt_meta.json` 只放简短摘要，避免撑大响应头。

## 测试与性能边界

```bash
.venv-jersey/Scripts/python.exe -m pytest -q test_train_court_keypoints.py test_court_keypoints.py test_court_line_features.py test_onnx_sessions.py test_jersey.py
```

合成测试覆盖坐标/RANSAC 方向、隔帧复用、硬切清空、摇移平滑、遮挡、低置信、NaN、
局部特写、分辨率变化、多视频/线程隔离、边线辅助和只读 API；边线模块另覆盖自动 ROI
排除广告牌、严格多边形 mask、邻域 HSV 过滤、梯度过滤、可选球员 mask、最小簇点数、
最小线长、角度先验、邻近重复线合并、最终内点可视化和命令行参数。训练器另覆盖
33→16 标签顺序、越界点清理、源视频分组、COCO visibility 二值化、PCK 计数、gated PCK、
原图尺度单应发布和内点置信门禁、几何感知 checkpoint 选择、非零输入 PyTorch/ONNX 一致性、
动态 batch 与导出输出契约。测试不依赖真实权重；真实数据指标与运行时延迟在下方单独记录。

本机合成帧上，镜头硬切判定路径约 0.65ms/帧，结构化缓慢摇移约 0.95ms/帧；经典 Shi-Tomasi + 自动 ROI +
角度距离聚类兜底在 1080p 合成全场约 2 秒级，明显慢于原 Hough，因此默认 `auto`
不会在缺权重时把它塞进直播链路；实时链路仍按 `every_n` 隔帧推理。

## 本轮训练与验收结果（2026-11-27）

按源视频隔离的 grouped split 完成 40 轮 FP32 训练，最佳 checkpoint 为第 30 轮，
验证集选择分数 `0.63635`。最终只在 112 张、3 个未参与训练/选模型的 test 视频上验收一次：

| 指标 | test 结果 |
| --- | ---: |
| 可见关键点 PCK@5 / @10 / @20 | 41.43% / 47.78% / 53.49% |
| `confidence >= 0.35` gated PCK@10 | 45.24% |
| presence recall | 78.41% |
| 缺失点误检率 | 0.43% |
| 可发布单应率 | **0.00%** |
| 单应重投影误差中位数 | 无成功样本 |

0% 不是评估器漏传世界坐标：修复后的精确合成点回归可以正常发布 H。对 test 的诊断显示，
112 张中有 108 张在 `0.35` 门限下不足 6 个检测；其余 4 张虽然达到 6 个检测，但 RANSAC
仅得到 4–5 个内点，仍不满足在线发布的 6 点支撑门限。该数据集包含大量局部特写，
不能据此把原始 PCK 当作全场标定成功率，也不能宣称模型已满足实时全场标定需求。
后续应补充完整全场、少遮挡的转播训练数据，并重新做源视频隔离验收；不要用 test 调参。

导出模型为 `data/models/court_keypoints16.onnx`（约 2.19 MiB）及同名 sidecar。已验证
opset 17、输入/输出名称、动态 batch 1/2、`(B,16,3)` 形状和标签顺序。真实非零帧上
PyTorch 与 ONNX 的最大坐标差为 `0 px`，最大置信度差约 `4.47e-7`。

运行时延迟边界：

- PyTorch CUDA FP32 模型前向中位数约 `6.83ms`（RTX 5060 Laptop，50 次同步计时）；
  不含 resize、预处理和后处理，且本机 CUDA 11.8 wheel 对 sm_120 有兼容性警告。
- 当前 ONNX Runtime 只有 `AzureExecutionProvider` / `CPUExecutionProvider`，没有 CUDA EP。
  CPU EP 零输入前向中位数约 `202.90ms`；真实帧完整 `CourtKeypointModel.predict`
  中位数约 `264.47ms`，按每 5 帧推理摊销仍约 `52.89ms/帧`，**未达到 ≤2ms/帧或 30fps 目标**。
- 两张真实转播抽帧中，模型首帧返回 `low_confidence`；第二帧由经典边线兜底成功标定。
  这验证了降级链路，但不代表学习模型已可用。CPU-only 环境下建议保持 `BSK_COURT_KP=0`，
  待 GPU ONNX Runtime 和更高发布率模型就绪后再启用，避免拖慢实时帧率。

真实连续轨迹级准确率、高置信错号率、长视频遮挡/反光/硬切统计仍未完成；本次结果只覆盖
独立 test 图片、两张真实抽帧和合成回归。根目录项目回归为 `118 passed, 2 deselected`；
排除的两个既有测试分别是缺少 `video_path` fixture 的 `test_ten_player_detection` 和未导入
`cv2` 的 `test_ransac_tuning`。回归覆盖异常清 H、硬切当帧重标成功仍清旧线状态并停 worker、
隔帧复用持续检查置信度衰减、并发新结果保护、原图尺度单应发布及导出契约。

## ResNet-18 强化训练与验收结果（2026-11-28）

本轮将模型升级为 ImageNet 预训练 ResNet-18 + 多尺度解码器，使用 626 张真实训练图、1200 张
独立种子生成的完整全场合成训练图，以及不同种子生成的 300 张合成 valid。模型选择只看真实
valid 与独立合成 valid；112 张真实 test 在训练、选 checkpoint、导出、CUDA 和因果序列验证全部
结束后仅评估一次，没有根据 test 再调模型或阈值。最终选择累计第 36 轮，几何感知选择分数为
`1.6825469254977294`；累计第 40 轮仅另存为 `_last.pt`，没有替换选择结果。

真实 valid 的逐点定位明显提升，但全场单应仍受数据构图限制：

| 真实 valid 指标 | 旧轻量模型 | ResNet-18 |
| --- | ---: | ---: |
| 可见点 PCK@5 / @10 / @20 | 60.28% / 68.01% / 74.96% | 80.68% / 89.64% / 93.82% |
| gated PCK@5 / @10 / @20 | 56.41% / 63.68% / 69.71% | 77.59% / 86.09% / 90.26% |
| presence recall | 85.16% | 93.35% |
| 缺失点误检率 | 0.873% | 0.349% |
| 单应发布率 | 0.00% | 0.00% |

独立合成 valid 验证完整几何监督确实生效：300 张全部发布 H，其中 254 张能由人工标签拟合
参考 H，254 个可评估 H 全部满足 16 点误差中位数 ≤20px、p90 ≤35px；其余 46 张为不可评估，
不计作错误发布。发布准确率按全部图片计为 84.67%，按可评估参考计为 100%，错误发布率为 0；
16 点误差中位数/p90 为 0.265px/0.568px，gated PCK@10 为 99.77%。该结果证明全场几何映射能力，
不等于真实转播域已经可用。

真实 valid 的失败归因显示：112 帧中 109 帧的 RANSAC 矩阵未通过全场合理性检查，2 帧支撑不稳定，
1 帧发布置信度不足。65 帧虽能拟合人工参考 H，其中只有 2 个参考矩阵自身通过现有全场合理性
门禁；65 个模型 RANSAC 矩阵与参考矩阵均未达到准确标准。这是半场、局部特写和稀疏点造成的
全场外推不确定性，不能通过下调置信度或放宽门禁强行“提高发布率”。

唯一一次真实 test 验收结果如下，必须与 valid 提升分开看待：

| 真实 test 指标 | 旧轻量模型 | ResNet-18 |
| --- | ---: | ---: |
| 可见点 PCK@5 / @10 / @20 | 41.43% / 47.78% / 53.49% | 54.60% / 66.83% / 74.60% |
| gated PCK@5 / @10 / @20 | 40.16% / 45.24% / 49.21% | 38.25% / 46.67% / 50.32% |
| presence recall | 78.41% | 52.86% |
| 平均置信度 | 0.5532 | 0.3930 |
| 缺失点误检率 | 0.430% | 0.258% |
| 单应发布率 / 可发布准确率 | 0.00% / 无成功样本 | 0.00% / 无成功样本 |
| 可评估参考 H 比例 | 未统计 | 0.00% |

ResNet 在 test 上的原始定位精度提高，但 presence recall 明显下降，且 test 没有可评估参考 H，
不能宣称真实泛化或在线标定已经解决。后续需要补充完整全场、稳定机位和更密集真实标签的数据，
重新按源视频隔离验证；不得用本次 test 结果继续调参。

### 导出、GPU 与因果链路

新模型只导出到独立文件 `data/models/court_keypoints16_resnet.onnx`（约 54.50 MiB）及
`court_keypoints16_resnet.json`，默认运行时权重 `court_keypoints16.onnx` 未替换。导出为 opset 17，
真实非零 uint8 输入上 PyTorch/ONNX 最大坐标差 `1.91e-6px`、置信度差 `3.31e-8`；独立 GPU 环境
验证 CUDA EP 实际启用、动态 batch 1/2、`(B,16,3)` float32 输出、有限值及 batch 与逐帧结果一致。

RTX 5060 Laptop、无并发训练时，新模型同步 `session.run` 中位数/p90 为 9.65ms/9.92ms，完整
`CourtKeypointModel.predict` 为 13.71ms/14.79ms；旧模型分别为 7.30ms/7.71ms 和
10.81ms/11.37ms。新模型用约 2.9ms 的完整推理增量换取显著更好的定位精度。按每 3 帧推理摊销，
新模型原始模型调用约 4.57ms/帧，不代表整个应用已达到 30fps。

真实 valid 上按源视频分组、严格因果顺序运行 model-only tracker：新旧模型均为 112 帧、0 个有效 H、
108 次推理。分组样本混合了多个短片且大量相邻抽帧内容不连续，共识别 105 次硬切，因此这不是
连续转播视频的轨迹级验收。运行时另修复了一个链路缺陷：模型给出足够数量检测点但单应求解失败时，
现在仍会尝试经典边线、语义+边线融合和无标签边线兜底，不再提前返回 `low_confidence`；所有路径
保持因果，不读取未来帧。

仓库回归结果为 `138 passed, 2 deselected`。两个排除项仍是既有问题：`test_ten_player_detection`
缺少 `video_path` fixture，`test_quick_wins.py::test_ransac_tuning` 未导入 `cv2`。新增/强化回归覆盖
模型求解失败后的经典兜底、几何感知 checkpoint 选择、原图尺度单应方向、不可评估 H 的计数、
非零输入导出一致性、动态 batch、硬切清 H、NaN 与局部特写拒绝。

本地对照图 `outputs/court_keypoints_resnet_real_comparison.jpg` 展示了真实 valid 抽帧：绿色为人工
标签、红色为模型检测。完整数据、权重和输出只保留在本地，不提交版权帧或模型文件。
