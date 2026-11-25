# 功能需求：球员号码检测（Jersey Number Recognition）

> 状态：已认领 · 优先级：高 · 归属模块：`bsk/perceive/jersey.py` + `bsk/server_rt.py` 接入
> 分支 `feat/jersey-number`；采片/标注/训练/评估的操作说明见 `docs/jersey_usage.md`。
> 仓库：https://github.com/somethingwrong2007/bskdetect （Apache-2.0）

## 1. 背景与目标

项目是篮球转播视频的实时分析系统：当前已具备人体检测+17 点骨架（RTMO-s，TensorRT FP16，
约 3ms/帧）、跟踪（track_id）与实时渲染（960–1280 宽下服务端约 20–32ms/帧，前端锁定 30fps）。

本需求的目标：**从球员躯干区域识别球衣号码**，并把它稳定地挂到 `track_id` 上，供后续统计
（犯规/得分归属）、人工纠错和 3D 展示使用。

成功标准（详见 §6）：单帧可读裁片 top-1 ≥ 90%，整段轨迹投票后的最终号码准确率 ≥ 95%，
号码不闪烁，且不破坏现有 30fps 实时预算。

## 2. 现有接口（已就绪，直接复用）

- 数据模型：`bsk/schema.py::Tracked` 已有字段
  - `jersey: int | None = None`
  - `jersey_conf: float = 0.0`
  填上后，`bsk/render.py::draw_boxes` 会自动在框标签里显示号码（`#12 play A 23`）。
- 跟踪：`bsk/analyze.py::Tracker.assign(boxes)` 给出逐帧 `track_id`；
  实时路径在 `bsk/server_rt.py::analyze_frame_realtime` 内完成“检测→过滤→跟踪→骨架”。
- 视频素材：`data/raw/harden_55.mp4`（vid `harden-55`，1920x1080，NBA 转播，8574 帧）。
- 实时服务：`.venv/Scripts/python.exe -u -m uvicorn bsk.server:app --host 0.0.0.0 --port 8000`
  单帧接口 `GET /api/videos/harden-55/rt_frame.jpg?n=..&layers=hud,boxes,pose&width=960&quality=accurate`。
- 评估可参考 `eval_pose_filter.py` 的取数方式（`bsk.ingest.frames` 顺序解码 + 与线上一致的过滤规则）。

## 3. 功能需求

### 3.1 输入
- 每帧的 `Tracked` 列表（含 bbox、track_id、team 可选）。号码识别从原图按 bbox 裁躯干区域：
  - x: `[x0 + 0.15w, x1 - 0.15w]`，y: `[y0 + 0.10h, y0 + 0.55h]`（以框左上为原点），
    实际比例可按验证结果微调；裁片统一缩放到模型输入尺寸。
- 采样策略：**不是每帧都跑**。每个 track 每 5 帧取一次（可配），人数多时按框面积排序取前 N；
  遮挡/侧身（关键点肩宽过窄、可见性低）时不采信。

### 3.2 输出
- `Tracked.jersey`（整数）与 `Tracked.jersey_conf`（0–1）。
- 输出必须是**时序融合后**的稳定结论，而不是逐帧 OCR 直出：
  - 每个 track 维护号码计数/对数似然（例如 logits EMA 或票数）；
  - 至少累计 ≥3 次一致观测且置信 ≥0.6 才发布；发布后允许在高置信新证据下更正一次，
    但整条轨迹的号码切换次数 ≤1（防闪烁）；
  - track 丢失超过 1s 或换队（若团队标注可用）时重置状态。

### 3.3 失败处理
- 号码不可读（背影、遮挡、裁片过小）→ 保持 `jersey=None`，不得瞎猜；
- 明显冲突（同队同号码）→ 置信度打折并记录日志，留人工纠错。

## 4. 技术建议（不强制，请自行论证）

1. **模型**：优先“小模型 + 强时序”而不是大 OCR。
   - 推荐 A：MobileNetV3-small 主干 + CTC 的 CRNN（输入 64×160，1–2 位数字），导出 ONNX，
     可复用项目现有的 ONNX→TensorRT 路径（`bsk/perceive/persons.py::make_session_trt`）；
   - 推荐 B：直接把号码当“两位数字分类”（10×10 + 个位数情形），类别少、收敛快；
   - 备选：PaddleOCR / EasyOCR（均 Apache-2.0，允许），但注意延迟与依赖体积。
   - **禁止新增 AGPL/GPL 依赖**（项目白名单：MIT / Apache-2.0 / BSD；Ultralytics 已列入待替换）。
2. **数据**：不要依赖来路不明的数据集授权。建议两条腿：
   - 真实裁片：用现有 tracker 自动导出 crop + track_id + 帧号清单，人工标注 300–800 张
     （覆盖 0–9、两位号码、不同球衣底色/字体/主客场）；
   - 合成增强：把渲染的数字按透视/光照/褶皱/运动模糊贴进真实裁片背景（参考
     `bsk/calib_seg_real.py` 的合成思路），解决长尾数字与低分辨率；
   - 弱标签提效：先用 OCR 打伪标签，只人工复核 top-N，控制标注成本。
3. **颜色/对比度**：球衣底色多变（本项目素材主场青色/客场白色），先做通道对比度归一化
   （灰度 + CLAHE 或按底色自适应）再识别，避免模型只记住某一种配色。
4. **批处理**：一帧内多个 crop 拼 batch 一次前向，减少 launch 开销。

## 5. 接入点与约束

- 新建 `bsk/perceive/jersey.py`，暴露类似：
  ```python
  class JerseyRecognizer:
      def __init__(self, weights: str | Path, device: str = "cuda", every_n: int = 5): ...
      def recognize(self, vid: str, n: int, rgb, objects) -> None:
          """就地写入 objects 里各 Tracked 的 jersey/jersey_conf（含时序融合）。"""
  ```
- 在 `bsk/server_rt.py::analyze_frame_realtime` 的 poses 构建之后调用（默认 `BSK_JERSEY=1` 开启，
  `BSK_JERSEY=0` 可关闭排障）；只对 `backend="accurate"` 生效。
- **性能预算**：新增 ≤2ms/帧（平均）。采样 + batch + 小模型应当远低于此；上线前给出实测。
- **实时约束**：所有逻辑必须是因果/在线的，禁止预取未来帧（后续要接直播流）。
- 不修改：`bsk/calib*.py`、`bsk/court_3d.py`、`bsk/line_calib.py`、`bsk/calib_seg_*.py`、`gui/`；
  `bsk/perceive/persons.py` 的 TensorRT 配置不要动；如需改 `server_rt.py` 请保持接口签名兼容。
- 状态隔离：缓存按 `(video_id, track_id)` 存放，多视频/多后端不得串号。

## 6. 验收标准与评估

在 `harden-55` 上至少抽 3 段（含快攻、半场阵地、特写）做验证：

| 指标 | 目标 |
|---|---|
| 单帧可读裁片 top-1 | ≥ 90% |
| 轨迹级最终号码准确率（投票后） | ≥ 95% |
| 号码闪烁（每条轨迹切换次数） | ≤ 1 |
| 高置信错号（conf>0.6 且错误） | < 2% |
| 新增延迟（平均/帧） | ≤ 2ms |
| 对现有端到端帧率影响 | 960 宽仍 ≥ 30fps |

提供评估脚本（如 `eval_jersey.py`）：裁片级混淆矩阵、轨迹级准确率、延迟统计；
保存 3–5 张可视化对比图（框标签带号码）到仓库外或 `.tmp_*.jpg`，不要提交大文件。

## 7. 交付物

1. `bsk/perceive/jersey.py`（识别 + 时序融合 + 采集/训练辅助函数）；
2. 训练/导出脚本（如 `bsk/train_jersey.py`），模型权重放 `data/models/`（不入库）；
3. `eval_jersey.py` + 一页结果报告（指标、失败案例分析）；
4. `server_rt.py` 的最小接入改动（默认开启、环境变量可关）；
5. `docs/` 下补一段使用说明（如何采样、标注、训练、评估）。

## 8. 建议里程碑（小步提交，每个 PR/commit 可独立回滚）

| 里程碑 | 内容 | 产出 | 当前状态 |
|---|---|---|---|
| M1 | 裁片采集管线 + 清单 | `jersey_samples/` 清单与导出脚本 | 代码就绪（`bsk/perceive/jersey.py::collect_crops`），等 `data/raw` 素材跑 |
| M2 | 数据（真实标注 + 合成增强）与基线模型 | 裁片级 top-1 ≥ 85% | 合成基线已达成；真实标注 top-1 未测 |
| M3 | 时序融合 + 实时接入（开关） | 轨迹级 ≥ 95%，延迟 ≤ 2ms | 融合逻辑 + 单测 + `BSK_JERSEY` 接入就绪；轨迹级精度与 GPU 延迟未测 |
| M4 | 验证报告 + 可视化 + 文档 | 验收指标全绿 | `eval_jersey.py` 与 `docs/jersey_usage.md` 就绪，指标待真实数据 |

## 9. 本机环境与合作方式

- Windows 10/11 + Git Bash；Python 用仓库自带 `.venv`（GPU：CUDA 12 / cuDNN 9 / TensorRT 10.13
  已安装，ONNX Runtime 1.22）。
- 拉代码：`git clone https://github.com/somethingwrong2007/bskdetect`；从 `master` 切
  `feat/jersey-number` 分支开发，commit 用中文、说明动机与指标；完成 M1/M2 可先提 PR 讨论。
- 服务端 8000 端口常驻；如端口被占先 `netstat -ano | grep :8000` 确认再重启。
- 有疑问看：`docs/` 现有文档、`eval_pose_filter.py`（评估口径参考）、
  `bsk/perceive/pose_filter.py`（时序状态层的写法可复用/对照）。