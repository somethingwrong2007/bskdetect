# 球衣号码识别：使用说明（采样 / 标注 / 训练 / 导出 / 评估）

> 需求与验收口径见 `docs/jersey_number_detection.md`。本页只讲怎么跑。
> 对应分支：`feat/jersey-number`。

## 0. 一分钟版

```bash
# 1) 采片（需要 data/raw 视频 + RTMO 权重）
.venv/Scripts/python.exe -m bsk.perceive.jersey --vid harden-55 --from 400 --nframes 600
# 2) 人工填 jersey_samples/labels.csv 的 label 列（-1 = 看不清/无号码）
# 3) 训练 + 导出 ONNX（三件套落在 data/models/，都不入库）
.venv/Scripts/python.exe -m bsk.train_jersey --manifest jersey_samples/labels.csv \
    --finetune data/models/jersey_small.pt --epochs 30
# 4) 评估
.venv/Scripts/python.exe eval_jersey.py --mode manifest
```

服务端的实时接入不用额外操作：`quality=accurate` 时自动带上号码，`BSK_JERSEY=0` 可整块关掉排障。

## 1. 这套东西长什么样

号码识别不是一个 OCR，而是「两位数字分类 + 时序融合」，链路三段：

```
躯干裁片 (crop_torso)  →  小 CNN 双头分类 (JerseyModel)  →  概率 EMA 融合 (JerseyRecognizer)
   几何 + 骨架门控           十位 10 + 个位 10 + 空白 2         ≥3 次一致观测且 conf≥0.6 才发布
```

- **模型输出 22 个 logits**：`[0:10]` 十位（索引 0 表示「没有十位」）、`[10:20]` 个位、
  `[20:22]` 有号码/空白。十位×个位组合到号码是双射（`_COMPOSITE_TO_NUM`），
  比 100 类 softmax 少一个数量级参数，且个位数不会被拆成「1?」和「?1」。
- **输入是 3×64×96 灰度复制图**：预处理走 CLAHE，球衣底色（主场青 / 客场白）在灰度 + 局部
  对比度归一化后基本不影响结果，模型学不到「青色衣服 = 某队」这类捷径。
- **训练与推理共用同一个 `gray_input()`**（`bsk/perceive/jersey.py`），不存在预处理漂移；
  sidecar `.json` 里记了输入尺寸，加载时不一致直接拒绝，防止拿错权重建会话。
- **融合层是纯在线的**：只用过去和当前帧，不预取未来帧，符合直播流约束。

主要旋钮都在 `JerseyConfig`（`bsk/perceive/jersey.py`）：采样 `every_n=5`、单帧最多
`max_crops=6` 人、发布门槛 `min_obs=3` / `min_conf=0.60`、每条轨迹最多 `max_switch=1` 次
号码切换（用完即锁定，不再花算力）、轨迹消失 `gap_reset_s=1.0` 秒后状态清零。

## 2. M1 采片：把真实裁片和 track_id 一起存下来

```bash
.venv/Scripts/python.exe -m bsk.perceive.jersey \
    --vid harden-55 --from 400 --nframes 600 --width 960 \
    --out jersey_samples --every-n 3 --max-per-track 6
```

顺序解码 → 检测 → 跟踪 → 躯干裁片，一趟跑完就把 crop 与 `track_id` 对齐存盘，比事后重新解码
便宜得多。产物：

- `jersey_samples/crops/<vid>/f%05d_t<id>.png` — 裁片原图（未做 CLAHE，标注时看清原样）
- `jersey_samples/manifest.jsonl` — 每张图的 frame/track_id/bbox/尺寸，以及被门控丢掉的原因
  （`dropped=true`，调 `min_shoulder_frac` 时能直接复盘误杀率）
- `jersey_samples/labels.csv` — 给人工填的模板，列 `crop,frame,track_id,label`
- `jersey_samples/summary.json` — 采到多少张、丢了多少张

标注规则：`label` 填 0–99 的整数；**看不清就填 -1**（空白类），不要猜——猜错的样本比缺失的
样本毒得多。目标覆盖 300–800 张，尽量摊开：0–9 个位数、两位号码、主客场两种底色、侧身与遮挡
各留一点。三段素材（快攻 / 阵地 / 特写）都要采，只采特写会让远端小裁片完全没训练信号。

`jersey_samples/` 已在 `.gitignore` 里：裁片是转播画面，**绝不能进仓库**。

## 3. M2 训练与导出

```bash
# 纯合成（零标注，验证管线 + 出基线权重）
python -m bsk.train_jersey --synth --per-number 120 --epochs 30

# 合成预训 + 真实标注微调（推荐路线）
python -m bsk.train_jersey --manifest jersey_samples/labels.csv \
    --finetune data/models/jersey_small.pt --epochs 30

# 只要真实数据
python -m bsk.train_jersey --manifest jersey_samples/labels.csv --epochs 30
```

要点：

- 合成数据由 `bsk/jersey_synth.py` 在内存里生成（真字体 + 透视 + 褶皱 + 遮挡 + 运动模糊 +
  JPEG 压缩），`--seed` 控制，留出评估必须换一个种子（见 §4）。
- 主干是 5 个深度可分离块 + 3 个分类头，**0.08M 参数**，64×96 输入。换 MobileNetV3-small
  要多花 30× 参数和更多 TRT 构建时间，而 100 类 + 空白的信息量根本用不满；输入输出契约
  （`1x3x64x96` → `Nx22`）不变，以后提精度直接换主干即可。
- 三个头的 loss 用 `ignore_index=-100` 处理空白样本：空白只训 blank 头，不污染数字头。
- 产物三件套，都在 `data/models/`、都不入库：`jersey_small.pt`（权重，供 `--finetune`）、
  `jersey_small.onnx`（推理）、`jersey_small.json`（sidecar：输入尺寸、头部布局、指标）。
- 导出走 `opset 17`、batch 维动态。TensorRT 侧要固定 profile，按 `docs` 里 `persons.py`
  的写法给 `min/opt/max` 三档（例如 1/6/8 张裁片）；不建 TRT 引擎也能跑，ONNX Runtime
  CPU 兜底，只是慢。
- `--device cuda` 需要 CUDA 12 轮子。**CUDA 11.x 轮子在 sm_120（Blackwell，如 RTX 50 系）
  上 conv 反向会直接挂死**，脚本检测到这个组合会自动降到 CPU 并打警告，别当成卡死。
- 训练依赖装在 `pip install -e ".[train]"`（含 `onnx`，`torch.onnx.export` 缺它会报
  `Module onnx is not installed`）。

## 4. 评估

```bash
.venv/Scripts/python.exe eval_jersey.py --mode synth              # 合成留出集
.venv/Scripts/python.exe eval_jersey.py --mode manifest            # 人工标注裁片（真验收）
.venv/Scripts/python.exe eval_jersey.py --mode fusion              # 轨迹级发布率/切换次数
.venv/Scripts/python.exe eval_jersey.py --mode video               # 端到端（需 data/raw + RTMO）
.venv/Scripts/python.exe eval_jersey.py --mode all --per-number 3  # 一次跑完，报告落 .tmp_
```

`--mode synth/fusion` 只要 `onnxruntime` 就能跑，没有 CUDA/TensorRT 时用 CPU 执行器也一样出精度
指标（延迟数字除外）；`--no-trt` 可强制不走 TensorRT。

各模式报的数（`--mode all` 一次跑完，结果落 `.tmp_jersey_report.json`）：

- `synth` / `manifest` → 裁片级 top-1（只看可读裁片，对应验收 ≥90%）、十位/个位分开的混淆
  矩阵、空白召回与误判成号码的比率、高置信错号率（conf>0.6 且错，对应 <2%）、单裁片毫秒数。
  **留出集种子必须和训练种子不同**（默认 `--seed 991` vs 训练 `--seed 7/11`），否则是在背题。
- `fusion` → 拿真实裁片序列驱动融合层：发布率、轨迹级准确率、每条轨迹最大切换次数（≤1）、
  首次发布耗时中位数、摊到每帧的毫秒数。这一项回答的是「延迟 ≤2ms/帧」和「不闪烁」。
  没标号码的裁片只驱动融合、不计入准确率，所以 `--gt` 不给也能看出发布与抖动行为。
- `gating` → 门控通过率，以及被 `shoulder_low_conf` / `side_or_back` / `too_small` 各挡掉多少。
- `video` → 端到端跑一段视频，把号码写进渲染标签，存 `.tmp_jersey_*.jpg` 对比图（不提交）。

单元测试：`python test_jersey.py`，12 组用例覆盖裁片几何、门控、头解码、发布门限、切换配额
与锁定、轨迹丢失重置与跨视频隔离、同队冲突打折、「读不出就留 None」、采样节拍与 `max_crops`、
权重缺失时 no-op、sidecar 尺寸守卫、ONNX 与 torch 数值一致性。**融合层两个真实 bug（配额用
完不锁定、空场后状态永不回收）就是这组用例逼出来的**，改采样/融合参数时先跑它。

## 5. 实时接入与排障

`bsk/server_rt.py::analyze_frame_realtime` 在骨架构建之后、渲染之前调 `recognize()`，就地写
`Tracked.jersey` / `jersey_conf`，`bsk/render.py::draw_boxes` 自动把号码显示成 `#23`。

- 开关：`BSK_JERSEY=0` 关闭（默认开）。只对 `quality=accurate` 生效，快速档不跑。
- 异常兜底：`recognize()` 整块包在 try/except 里，号码是附加信息，坏了只打一行日志，
  不能把 30fps 链路拖死。
- 权重不在 / 加载失败：首次构造时打一次
  `[jersey] 号码识别未启用：…（训练导出：python -m bsk.train_jersey；关闭：BSK_JERSEY=0）`，
  之后静默 no-op——不会每帧重试建会话。
- 延迟：超时告警行里带 `jersey=` 毫秒数，直接对照需求 §6 的「新增 ≤2ms/帧」。
  `JerseyRecognizer.stats()` 另给出 `crop_ms`/`infer_ms` 的均值与累计裁片数。
- 状态隔离：融合状态按 `(video_id, track_id)` 存，切视频不串号；`reset(vid)` 可手动清空。

## 6. 当前状态与已知限制

合成基线的实测数字与失败案例分析在 **`docs/jersey_results_20261124.md`**，这里只说清
**哪些数是实测的、哪些还不是**：

- 融合层行为（发布、切换 ≤1、锁定、重置、冲突打折）由 `test_jersey.py` 用桩模型完整验证过，
  加上真实 ONNX 的一致性用例（torch/ONNX max|Δ|=0），这部分不依赖真实素材，可以信。
- 合成留出集裁片级 top-1 = **0.890**（训练内验证 0.892，没有背题）、空白召回 0.993、
  空白误报 0/300；M2 的「≥85%」在合成分布上达成。
- **合成 ≠ 验收**：真实转播画面的字体、糊法、褶皱和合成器不是一个分布，`docs` 里 ≥90% / ≥95%
  这两条必须靠 M1 人工标注的 300–800 张裁片（`--mode manifest` / `--mode fusion`）才能出数。
  目前 `data/raw` 为空、没有标注，本机跑不出这两个数；高置信错号在合成上是 **6.8%**（目标
  <2%），修法也是补真实样本 + 校准，不是调阈值。
- 延迟：CPU 上整帧新增 0.10–0.36ms（p95 ≤1.14ms），已经满足「≤2ms/帧」的预算；但需求口径
  是 CUDA 12 + TensorRT 目标机上的实测数，仍需上线前复测。
- `min_shoulder_frac=0.16` 等门控阈值来自几何推理，真实素材上要重标（`--mode gating`）；
  调大误杀背影，调小会让侧身帧污染融合。
- 发布率对「可读帧占比」很敏感（0.55→1.00 时 0.65→0.825，而准确率反向 0.92→0.76），真实
  素材上如果面板长期空号，先动 `min_conf` 或空白票的衰减方式，不要动 `every_n`。
