# bskdetect

篮球转播视频的**实时分析调试台**：球员检测、跟踪、2D/3D 骨架、球场标定、按需实时渲染。
目标是把转播画面变成可计算的比赛数据（俯视坐标、动作、事件），并保持直播级时延。

> 许可：Apache-2.0；依赖白名单仅允许 MIT / Apache-2.0 / BSD（禁用 AGPL/GPL，Ultralytics 列为待替换项）。

## 特性一览

- **按需实时管线**：播放到哪一帧算哪一帧，不预取未来帧（后续要接直播流）。
- **准确档**：RTMO-s（ONNX + **TensorRT FP16**，单帧推理约 3–4ms）同出人体框与 17 点 COCO 关键点；
  **快速档**：YOLOv8n-pose（低配机器）。
- **骨架时序状态层**：在线 One Euro 滤波 + 置信度迟滞门控 + 遮挡预测补位（≤0.5s）+ 恢复混合过渡 +
  骨长约束 + 迟滞化左右镜像。实测抖动能量 −72%、大跳变 −73%、滤波开销约 1ms/帧。
- **实时渲染**：骨架/框/地板/运动层批量绘制，服务端烧录 JPEG；960 宽端到端约 20–30ms/帧，
  1280 宽约 32–37ms/帧（瓶颈是浏览器解码，前端会自动降档保 30fps）。
- **增量标定**：球场线分割 → 单应矩阵（pixel↔court）用于 3D 提升与球场范围剔除；
  **标定不可信时自动回退“不剔除”**，宁可不删也不误删球员。
- **数据与模型自造**：3D 球场渲染器合成训练数据（多机位、多地板颜色、遮挡带），
  不依赖来路不明的数据集授权。

## 快速开始

环境：Python 3.12（Windows/macOS/Linux 均可），建议 GPU（CUDA 12 + cuDNN 9；TensorRT 可选）。

```bash
git clone https://github.com/somethingwrong2007/bskdetect
cd bskdetect
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .          # Windows
# source .venv/bin/activate && pip install -e .        # macOS/Linux

# 把测试视频放到 data/raw/ 下（不入库），然后启动调试台
.venv/Scripts/python.exe -u -m uvicorn bsk.server:app --host 0.0.0.0 --port 8000
```

打开 <http://localhost:8000> ：左侧导入视频 → 中间播放/拖动 → 右侧实时统计与检测表。
播放栏可切换**质量档（准确/快速）**与速度；实时档支持 HUD/检测框/姿态/地板/运动图层。

模型权重放在 `data/models/`（不入库）。首次启用 TensorRT 会构建引擎（约数分钟），
之后走 `data/models/trt_cache` 秒开；`BSK_TRT=0` 可强制回退 CUDA。
排障开关：`BSK_HWACCEL=1` 启用 NVDEC 硬解；`BSK_POSE_FILTER=0` 关闭骨架时序层。

## 架构

```
                  ┌──────────── 实时播放（前端逐帧拉取，30fps 节拍）────────────┐
视频文件/直播帧 ──► ingest（顺序解码游标）──► 检测+姿态(RTMO/YOLO, GPU)
                  │        └─ 过滤/跟踪(track_id) ──► 骨架时序状态层 ──► 球场标定/ROI
                  │                                   └─ 2D 渲染(批量绘制) ──► JPEG
                  └──────────── /api/videos/{vid}/rt_frame.jpg?n=.. ──────────┘
```

- **帧精确**：帧号 1-based，时间戳来自容器 PTS 真值表（`bsk/ingest.py`），前端不用 `<video>` 时钟对齐。
- **实时语义**：只在请求的播放位置做分析；同屏多人流式解码由 `_RealtimePlayer` 串行化；
  前端按实际帧耗时自适应降档，落后时丢帧但保持墙钟同步。
- **双档显存 profile**：4GB 只出俯视坐标点、不加载姿态；8GB 启用三维骨架与 3D 展示（本机浏览器渲染）。

## 模块说明

| 路径 | 职责 |
|---|---|
| `bsk/ingest.py` | 解码、帧号/PTS 真值表、可变帧率判定；可选 NVDEC |
| `bsk/perceive/persons.py` | RTMO-s 人体检测+17 点关键点（ONNX/TensorRT），窗口化 NMS |
| `bsk/perceive/fast_pose.py` | YOLOv8n-pose 快速档（低配/高帧率） |
| `bsk/perceive/pose_filter.py` | 骨架时序状态层：抗抖动、遮挡预测、恢复过渡、骨长约束 |
| `bsk/perceive/ball.py` | 篮球检测（自训 YOLOv8n，默认 `data/models/ball_yolov8n.pt`；真实域数据微调待接入） |
| `bsk/perceive/shots.py` | 地板/镜头辅助 |
| `bsk/analyze.py` | 离线批量分析（区间 JSONL，可与既有结果合并续跑）；跟踪器 |
| `bsk/server.py` | 调试台 API、静态前端、实时解码游标、图像缓存 |
| `bsk/server_rt.py` | 实时单帧分析 + 渲染 + JPEG；模型懒加载/预热 |
| `bsk/render.py` | 2D 叠加渲染（框/骨架/分割/地板/运动/HUD） |
| `bsk/render_3d.py` | 2D→3D 骨架提升、透视与时序平滑（3D 展示用） |
| `bsk/court.py` | 球场几何唯一真源（FIBA/NBA/3x3），含像素投影工具 |
| `bsk/court_3d.py` | 3D 球场渲染器（合成训练数据：类别图/纯地板/线层/多机位） |
| `bsk/calibrate.py` / `realtime_calibrate.py` | 单应标定与增量跟踪；`is_plausible` 几何合理性校验 |
| `bsk/calib_seg_*.py` / `line_calib.py` | 球场线分割模型、合成数据生成、分割线→单应拟合（**进行中，第一优先级**） |
| `bsk/roi_tracking.py` | 球场 ROI 掩码与静态场景优化 |
| `bsk/schema.py` | 数据模型（Tracked/Pose2D/Segment2D/FrameRecord） |
| `bsk/store.py` | 素材库：索引、缩略条、派生数据加载 |
| `bsk/synth_ball.py` / `train_ball.py` | 3D 物理合成篮球数据与训练脚本 |
| `bsk/async_pipeline.py` | 生产-消费异步流水线（实验性） |
| `gui_terminal/` | 前端（React + Tailwind，零构建；实时帧与离线渲染两条管线，见 `gui_terminal/README.md`） |
| `eval_pose_filter.py` | 骨架时序层评估：抖动/延迟/遮挡/开销 |
| `docs/` | 设计说明、性能与标定方案、协作任务需求等 |

## 性能基线（本机 RTX 级 GPU，960 宽、准确档）

| 环节 | 耗时 |
|---|---|
| 解码（CPU，1080p→960） | ~2ms |
| RTMO-s 推理（TensorRT FP16） | ~3–4ms |
| 过滤/跟踪/骨架时序层 | ~3ms |
| 2D 渲染（批量绘制） | ~2–3ms |
| JPEG 编码 | ~3ms |
| **服务端合计（中位）** | **约 20–30ms/帧** |

1280 宽时服务端约 32–37ms，浏览器解码另需 ~10ms，端到端约 24fps；前端会自动降到 1120/960
维持 30fps（可在播放栏手动改回 1280）。

## 当前状态与路线图

**已完成**
- 实时检测/跟踪/骨架链路、RTMO TensorRT 加速、骨架时序状态层、图层修复、并发与缓存改造；
- 批量分析与调试台前端；合成数据生成器（球场线/篮球）。

**进行中（优先级从高到低）**
1. **球场范围标定**（第一优先级）：分割线→单应→球场 ROI 剔除替补席/观众；当前真实帧拟合尚未验收，
   期间标定不可信会自动关闭剔除以保护球员；
2. **球员号码检测**：协作任务，需求见 [`docs/jersey_number_detection.md`](docs/jersey_number_detection.md)；
3. 篮球检测真实域数据微调与实时接入（合成权重已具备；Roboflow 真实数据待导入）；
4. 直播流接入（帧到即算、无预取）。

**技术债**
- Ultralytics（AGPL）仅用于快速档 YOLOv8n-pose，需替换为 Apache-2.0 方案；
- 分割层/轨迹层在实时档暂不可用（已从图层栏隐藏）。

## 文档索引

- [`docs/jersey_number_detection.md`](docs/jersey_number_detection.md) — 号码检测需求与验收标准
- [`docs/ball_datasets.md`](docs/ball_datasets.md) — 篮球检测公开数据与授权说明
- [`docs/CALIB_NET_README.md`](docs/CALIB_NET_README.md) — 球场线分割标定网络
- [`docs/crowd_filtering.md`](docs/crowd_filtering.md) — 观众剔除方案
- [`docs/performance_roadmap.md`](docs/performance_roadmap.md) — 性能路线与测试结果

## 数据来源与引用

本项目**当前不随仓库分发任何第三方数据集或模型权重**；`data/raw/`、`data/derived/`、
`data/models/`、`checkpoints/`、`runs/` 均不入库。下面列出调研、候选和已使用来源，
“引用”不等于默认已采用；以各自页面上的最新许可为准。

### 篮球检测（Roboflow）

| 资源 | 用途 | 许可状态 |
|---|---|---|
| [Roboflow Sports](https://github.com/roboflow/sports) | 体育视觉工具与数据入口 | 代码 **MIT**；其链接的数据集许可独立 |
| [Roboflow Basketball Detection](https://universe.roboflow.com/basketball-6vyfz/basketball-detection-srfkd) | 真实篮球/球员/篮筐检测，补充球检测真实域 | 页面许可需在下载前确认；第三方项目引用为 **CC BY 4.0** |
| [Roboflow basketball court keypoint detection](https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2) | 篮球场关键点，候选用于标定 | 页面许可需确认（本次抓取被 403 拦截） |

如果后续合并 Roboflow 数据或由其训练的权重，必须：

- 保留数据集要求的署名与引用；
- 在 `NOTICE`/数据集说明中记录来源和许可，不把 CC BY 数据重新声明为 MIT/Apache；
- 不把原始图片、标注或受限权重提交进本仓库；
- 重新评估数据来源与比赛画面的版权链。

### 球场标定与跟踪（候选/对照）

| 资源 | 用途 | 许可状态 |
|---|---|---|
| [DeepSportRadar](https://github.com/DeepSportradar/camera-calibration-challenge) | 篮球相机标定/分割对照 | 数据集 **CC BY-NC-SA 4.0**，不可商用/不可再分发 |
| [SportsMOT](https://github.com/MCG-NJU/SportsMOT) | 篮球/足球/排球多目标跟踪对照 | **CC BY-NC 4.0**，不可商用 |
| [TrackID3x3](https://github.com/open-starlab/TrackID3x3) | 3x3 篮球跟踪/姿态对照 | 代码 **Apache-2.0**；数据集许可需单独确认 |
| [CourtKeyNet](https://github.com/adithyanraj03/CourtKeyNet) | 球场关键点架构参考（羽毛球） | 代码 **MIT**；不含篮球权重 |

### 本项目的自造数据

- 球场线/3D 场景：`bsk/court_3d.py`、`bsk/calib_seg_real.py`；
- 篮球物理轨迹与渲染：`bsk/synth_ball.py`、`bsk/gen_ball_dataset.py`；
- 这些数据由本项目生成，不依赖第三方数据集授权；生成脚本与渲染参数随代码一起维护。

### 参考文档

- [`docs/ball_datasets.md`](docs/ball_datasets.md) — 篮球检测数据与授权汇总
- [`docs/research/pose_2d_to_3d.md`](docs/research/pose_2d_to_3d.md) — 2D→3D 模型与权重许可调研
- [`docs/research/court_detection_code.md`](docs/research/court_detection_code.md) — 球场标定开源代码调研

## 协作约定

- 从 `master` 切功能分支（如 `feat/jersey-number`），小步提交；提交信息用中文，写清动机与指标；
- 不提交 `data/`、`checkpoints/`、`runs/`、`outputs/`、`*.log`、临时图片；
- 新增依赖必须在白名单（MIT/Apache-2.0/BSD）内；
- 实时链路所有逻辑必须因果/在线，禁止预取未来帧；
- 引入第三方数据/权重前先更新本节，并在 PR 中附许可证据。

## 许可证

Apache-2.0，见 [`LICENSE`](LICENSE)。