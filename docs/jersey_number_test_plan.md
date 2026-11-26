# 球衣号码识别迭代测试方案

## 1. 测试目标

验证本次号码识别迭代在功能、兼容性、实时性、异常恢复和并发场景下满足要求，重点覆盖：

- 肩髋关键点正常裁片与异常降级分支。
- NaN、Inf、低置信关键点过滤。
- 门控失败和人数超限后的采样节拍释放。
- 裁片质量加权融合。
- 空白帧不计有效观测，防止连续模糊误发号码。
- CLAHE 线程本地实例。
- 全局 ONNX 会话复用与不同视频轨迹状态隔离。

本方案是待执行验收方案，不代表所有用例已经在当前代码上通过。测试中发现与需求不符的结果应按缺陷处理，不得通过放宽断言来让用例通过。

## 2. 验收基线

### 2.1  correctness 指标

| 指标 | 基线 | 验收要求 |
|---|---:|---|
| 原有回归用例 | 13 个 | 13 个全部通过，不删除、不跳过、不放宽原断言 |
| 600 张合成样本可读号码 top1 | 89.00% | ≥89.00% |
| 合成样本空白召回率 | 99.33% | ≥99.33% |
| 合成样本高置信错误率 | 6.79% | ≤6.79% |
| 合成样本空白误报数 | 0 | ≤0 |
| 40 条仿真轨迹发布率 | 65.00% | ≥65.00% |
| 已发布轨迹准确率 | 92.31% | ≥92.31% |
| 单条轨迹最多切换次数 | 1 | ≤1 |

合成样本的十位、个位准确率可继续记录，当前参考值分别为 93.33%、93.67%，作为辅助观察项。

### 2.2 性能指标

| 指标 | 基线/目标 | 验收要求 |
|---|---:|---|
| 单裁片处理平均耗时 | 0.155ms | 同机同配置下 ≤0.155ms |
| `2025-bsk1` 240 帧号码识别增量开销 | 1.87ms/帧 | 同机同配置下 ≤1.87ms/帧 |
| 需求文档单帧新增预算 | 2ms | ≤2ms |
| 960 宽实时播放 | 30fps | ≥30fps |

性能对比必须在同一台机器、同一模型、同一运行时配置下完成。 correctness 指标不允许劣于基线；耗时指标取 3 次运行的 median，不使用单次最优值。

### 2.3 真实视频口径

`2025-bsk1` 没有人工号码真值，因此只验证：

- 240 帧处理完成，无异常中断。
- 已发布号码在时间上稳定，单轨迹切换不超过 1 次。
- `jersey_conf` 始终为有限数值且落在 0–1。
- 不同视频、不同 track 不串号。
- 增量耗时满足性能门槛。

不得把“发布了几个号码”“看起来像不像”当作真实准确率。

## 3. 测试环境与数据

### 3.1 环境

- 项目根目录：`bskdetect`。
- Python：`.venv-jersey/Scripts/python.exe`。
- 模型：`data/models/jersey_small.onnx` 与同名 `.json`。
- 可选模型：`data/models/jersey_small.pt`，用于 torch/ONNX 对齐。
- 真实视频：`2025-bsk1`。
- 合成评估固定参数：`--per-number 3 --seed 991`。
- 真实视频固定片段：`--segments 1,120`，每段最多 120 帧，共 240 帧。
- 基线对比使用 CPU 路径，避免 TensorRT 引擎和 provider 变化干扰结果。

PowerShell 示例：

```powershell
$env:BSK_TRT = "0"
.\.venv-jersey\Scripts\python.exe -m pytest -q test_jersey.py
.\.venv-jersey\Scripts\python.exe eval_jersey.py --mode synth --no-trt --per-number 3 --seed 991
.\.venv-jersey\Scripts\python.exe eval_jersey.py --mode fusion --no-trt --per-number 3 --seed 991 --tracks 40
.\.venv-jersey\Scripts\python.exe eval_jersey.py --mode video --no-trt --vid 2025-bsk1 --segments 1,120
```

Git Bash 等价命令：

```bash
BSK_TRT=0 .venv-jersey/Scripts/python.exe -m pytest -q test_jersey.py
BSK_TRT=0 .venv-jersey/Scripts/python.exe eval_jersey.py --mode synth --no-trt --per-number 3 --seed 991
BSK_TRT=0 .venv-jersey/Scripts/python.exe eval_jersey.py --mode fusion --no-trt --per-number 3 --seed 991 --tracks 40
BSK_TRT=0 .venv-jersey/Scripts/python.exe eval_jersey.py --mode video --no-trt --vid 2025-bsk1 --segments 1,120
```

### 3.2 通用测试夹具

关键点使用 COCO-17，`kpts[i] = [x, y, confidence]`。

```python
bbox = (20.0, 20.0, 120.0, 280.0)

k_front = np.zeros((17, 3), dtype=np.float64)
k_front[5] = [45.0, 60.0, 0.9]
k_front[6] = [95.0, 60.0, 0.9]
k_front[11] = [45.0, 180.0, 0.9]
k_front[12] = [95.0, 180.0, 0.9]

k_side = np.zeros((17, 3), dtype=np.float64)
k_side[5] = [68.0, 60.0, 0.9]
k_side[6] = [72.0, 60.0, 0.9]
```

概率夹具：

```python
p23 = np.zeros(100, dtype=np.float64)
p23[23] = 1.0

p28 = np.zeros(100, dtype=np.float64)
p28[28] = 1.0

blank_num = np.zeros(100, dtype=np.float64)
blank_prob = 1.0
```

推理桩优先复用 `test_jersey.py` 中的 `StubModel`、`logits_for()`、`make_recognizer()`。需要验证并发时，桩模型必须只读共享 `table`，不能在 `predict()` 中修改共享状态。

### 3.3 建议新增测试文件

- `test_jersey_optimization.py`：新增几何、采样、融合单元用例。
- `test_jersey_integration.py`：服务接入、会话复用、状态隔离、异常兜底。
- `test_jersey_concurrency.py`：CLAHE、ONNX、多视频并发。
- `test_jersey_regression.py`：基线指标与接口契约回归。
- `bench_jersey.py`：性能与真实视频稳定性报告。

原有 `test_jersey.py` 的 13 个测试函数保留为固定回归集。

## 4. 单元测试用例

| 测试类型 | 测试输入 | 预期结果 |
|---|---|---|
| 单元 UT-01 模型输入输出契约 | 构造 4 张 64×96 RGB 裁片，执行 `preprocess()`；将 4×22 logits 输入 `logits_to_dist()` | 输入形状为 `(4,3,64,96)`，dtype 为 `float32`，数值范围在 `[-1,1]`；号码分布形状为 `(4,100)`，空白概率形状为 `(4,)`；每个样本 `pnum.sum()+pblank≈1` |
| 单元 UT-02 无关键点回退 | `bbox=(20,20,120,280)`，`kpts=None`；调用 `crop_torso()` | 不拒绝；裁片框与原 `torso_bbox()` 一致，即 `(32,42,108,163)`；不访问肩髋点 |
| 单元 UT-03 肩髋正常分支 | 使用上述 `bbox` 和 `k_front` | 使用骨架收紧胸腹框，预期框为 `(35,72,105,146)`，裁片形状约 `(74,70,3)`；几何质量为 1.0；不得退回原始人体框 |
| 单元 UT-04 髋点缺失降级 | 肩点正常且高置信；`kpts` 长度仅为 12，或髋点置信度为 0 | 不得使用错误髋线；按需求回退到原始 `torso_bbox()` 框 `(32,42,108,163)`；裁片仍可送模型 |
| 单元 UT-05 髋点几何异常降级 | 髋点位于肩线上方、髋点距离超过人体框 80%、左右髋点几乎重合 | 不得生成异常胸腹框；回退原始 `torso_bbox()`；不抛异常 |
| 单元 UT-06 肩点低置信 | 左肩或右肩 confidence=0.29；另一肩正常 | 返回 `shoulder_low_conf`，不进入模型推理；confidence=0.30 时按边界值处理，不应因浮点误差误拒绝 |
| 单元 UT-07 侧身/背影 | 使用 `k_side`，肩宽/框宽约 0.04 | 返回 `side_or_back`；`score` 小于 `min_shoulder_frac`；不进入模型推理 |
| 单元 UT-08 肩髋 NaN/Inf | 分别在左肩 x、右肩 y、左髋 x、右髋 y 上设置 `NaN` 或 `Inf` | 返回 `keypoint_nonfinite`；不得生成裁片、不得调用模型、不得发布号码 |
| 单元 UT-09 关键点结构异常 | 依次传入 `[]`、长度 6、长度 17 但只有 2 列、二维数组含对象类型 | 不抛未捕获异常；无效结构应安全拒绝并给出明确原因 |
| 单元 UT-10 越界与最小尺寸 | 人体框部分越出画面；再构造躯干框恰好 18×24 和 17×24 | 越界框夹取后不越界；恰好 18×24 可裁；小于任一最小尺寸返回 `too_small` 或 `out_of_frame` |
| 单元 UT-11 预处理确定性 | 同一裁片连续执行 `gray_input()` 20 次；再与另一线程结果比较 | 输出完全一致；形状为 `(64,96)`，dtype 为 `uint8`；无随机状态泄漏 |
| 单元 UT-12 号码解码 | 构造 0–99 和空白样本的 logits | 0–99 全部可解码；空白样本 `pblank>0.5`；分布归一 |
| 单元 UT-13 质量加权融合 | 两个 `_TrackState` 先吃入 `p23`，再分别吃入 `p28`：一个 `quality=1.0`，一个 `quality=0.1` | 低质量矛盾观测对后验改变更小；`low_quality.post[23] > high_quality.post[23]`；不得出现质量权重为 0 时完全无法学习的情况 |
| 单元 UT-14 空白不计观测 | 连续 20 次调用 `update(blank_num, 1.0, ...)` | `obs` 始终为 0；`post` 为全 0；`jersey=None`；不因空白帧满足 `min_obs` |
| 单元 UT-15 空白后首个数字 | 先 10 次空白，再首次输入 `p23` | 首个数字观测直接初始化 `post=p23`，`obs=1`；不得被之前的全零后验按 EMA 稀释 |
| 单元 UT-16 有效观测门限 | `min_obs=3`；先 2 次 `p23`，再 20 次空白，最后第 3 次 `p23` | 前 22 次不发布；第 3 次有效数字观测后才允许发布；发布事件中的 `obs=3` |
| 单元 UT-17 快速抖动与锁定 | 连续 10 帧 23，再连续 20 帧 28，`max_switch=1` | 首次发布不算切换；23→28 最多切换 1 次；切换配额用完后锁定，不再改动；锁定后不再推理 |
| 单元 UT-18 门控失败释放节拍 | `every_n=5`；第 1 帧侧身门控失败，第 2 帧恢复正面 | 第 1 帧不调用模型且不推进 `last_frame`；第 2 帧立即调用模型；之后第 3–6 帧不调用，第 7 帧再次调用 |
| 单元 UT-19 人数超限释放节拍 | `max_crops=3`；第 1 帧 8 人，记录被淘汰 track；第 2 帧让该 track 成为高优先级候选 | 第 1 帧只推理 3 人；被淘汰 track 的 `last_frame` 不更新；第 2 帧可立即补采，不得等待完整 5 帧 |
| 单元 UT-20 批次边界 | `max_crops=6`、`batch_size=2`，6 个有效候选 | 共调用模型 3 次，每批依次为 2、2、2；总裁片数 6；无候选丢失 |

## 5. 集成测试用例

| 测试类型 | 测试输入 | 预期结果 |
|---|---|---|
| 集成 IT-01 全局会话复用 | 清空 `server_rt._jersey` 与 `_jersey_model`；依次调用 `_get_jersey("videoA","accurate")`、`_get_jersey("videoB","accurate")` | 返回两个不同的 `JerseyRecognizer`；两个 recognizer 的 `model` 是同一个对象；`JerseyModel` 只创建一次 |
| 集成 IT-02 同 track 跨视频隔离 | videoA、videoB 都使用 `track_id=1`；A 固定输出 23，B 固定输出 28，各运行 10 帧 | A 最终为 23，B 最终为 28；事件中的 `vid` 正确；`recA._states` 与 `recB._states` 不共享；重置 A 不影响 B |
| 集成 IT-03 仅准确档启用 | 分别调用 `_get_jersey("v","accurate")` 和 `_get_jersey("v","fast")` | accurate 返回 recognizer 或不可用时的 `None`；fast 始终返回 `None`，不创建模型、不写号码 |
| 集成 IT-04 环境开关 | 新进程中设置 `BSK_JERSEY=0` 后导入服务并请求 accurate 档 | 号码识别整体关闭；不创建 recognizer；对象 `jersey=None`、`jersey_conf=0`；其他分析链路正常 |
| 集成 IT-05 权重缺失降级 | 将模型路径指向不存在的文件，执行识别 | `ready=False`；不抛异常；不写 `jersey`；服务日志给出训练/导出提示 |
| 集成 IT-06 sidecar 不匹配 | 构造真实 ONNX 和 `input_hw=[32,32]` 的 sidecar | recognizer 不可用；拒绝加载；不执行错误尺寸推理 |
| 集成 IT-07 旧接口兼容 | 使用旧调用方式 `JerseyRecognizer(weights, cfg, device)`、`recognize(vid,n,rgb,objects,poses)`、`update(pnum,pblank,n,cfg)` | 全部可调用且行为不变；新增 `model=` 和 `quality=` 为可选参数，不破坏旧调用方 |
| 集成 IT-08 因果性 | 用相同前 N 帧分别运行“只喂 N 帧”和“喂 N+50 帧”两组 | 两组在前 N 帧的发布、切换、事件完全一致；未来帧不得影响当前结果 |
| 集成 IT-09 推理异常不拖垮帧 | mock 检测/姿态依赖，仅让 `model.predict()` 抛一次异常 | 当前帧仍返回有效 `FrameResult`；号码字段保持未发布；日志包含异常类型；下一次正常调用可恢复，不污染后续轨迹状态 |
| 集成 IT-10 实时服务端到端 | 启动服务，请求 `2025-bsk1` 的 accurate 档连续 240 帧 | HTTP 请求成功；无 5xx；号码识别不改变现有对象字段结构；`jersey` 为整数或 `null`，`jersey_conf` 为 0–1 的有限数 |

## 6. 回归测试用例

| 测试类型 | 测试输入 | 预期结果 |
|---|---|---|
| 回归 RG-01 原有 13 项 | 执行 `pytest -q test_jersey.py` | 13 个测试函数全部通过；不得跳过、删除或弱化原断言 |
| 回归 RG-02 合成裁片指标 | 固定执行 `eval_jersey.py --mode synth --no-trt --per-number 3 --seed 991`，连续 3 次 | `top1_readable≥0.8900`；`blank_recall≥0.9933`；`high_conf_error_rate≤0.0679`；`blank_false_positive=0`；3 次 correctness 结果一致 |
| 回归 RG-03 仿真轨迹指标 | 固定执行 `--mode fusion --tracks 40`，连续 3 次 | `publish_rate≥0.6500`；`track_accuracy_among_published≥0.9231`；`switches_max≤1`；3 次结果一致 |
| 回归 RG-04 真实视频稳定性 | `2025-bsk1`，`--segments 1,120`，连续 3 次 | 每次处理 240 帧；每轨迹切换≤1；无 NaN/Inf 置信度；无跨视频串号；平均增量耗时≤1.87ms |
| 回归 RG-05 接口契约快照 | 对 recognizer、`Tracked` 序列化结果、服务响应做字段快照 | 旧字段名、类型、空值语义不变；不得新增必填字段；`jersey` 仍可为 `null` |
| 回归 RG-06 确定性回归 | 相同 seed、相同模型、相同输入连续运行两次 | 除耗时字段外，号码、置信度、事件、切换次数完全一致 |
| 回归 RG-07 模型对齐 | 有 `.pt` 时对 16 张随机裁片比较 torch 与 ONNX | logits 最大差值<0.05，argmax 一致；无 `.pt` 时明确记录跳过，不计为模型精度通过 |

## 7. 性能测试用例

| 测试类型 | 测试输入 | 预期结果 |
|---|---|---|
| 性能 PF-01 单裁片平均耗时 | 600 张合成裁片，CPU，batch=32，预热 10 个 batch，测 3 轮 | 每轮 `ms_per_crop_mean≤0.155ms`；3 轮 median≤0.155ms |
| 性能 PF-02 单裁片 P95 | 同 PF-01，记录 `ms_per_crop_p95` | 记录 P95；若 P95>2×均值，需排查尾延迟；不得用均值掩盖明显长尾 |
| 性能 PF-03 真实视频增量耗时 | `2025-bsk1` 240 帧，accurate 档，其他模块配置与基线一致 | `ms_per_frame_mean≤1.87ms`；同时满足需求预算≤2ms |
| 性能 PF-04 多人最坏场景 | 构造 10 个有效球员，`max_crops=6`，连续 300 帧 | 每帧最多 6 个裁片；无无限增长队列；号码模块平均耗时≤2ms；超过时记录 P95 和系统负载 |
| 性能 PF-05 960 宽实时帧率 | 960 宽连续播放 300 帧，姿态/跟踪开启，3D 和分割按默认关闭 | 预热 30 帧后，剩余 270 帧平均帧率≥30fps；若未达到，分别报告检测、姿态、号码、渲染耗时 |
| 性能 PF-06 会话复用收益 | 创建 20 个不同视频 recognizer | `JerseyModel` 实例数为 1；模型只加载一次；无重复 ONNX session 创建日志 |
| 性能 PF-07 内存稳定性 | 单进程连续处理 10000 张裁片，前 1000 张预热 | 无持续单调增长；预热后 RSS 增量≤50MB；线程结束后无线程数持续增长 |

性能报告必须记录：CPU/GPU 型号、Python 版本、ONNX Runtime provider、模型文件哈希、线程数、batch 大小、运行次数和 median。

## 8. 异常与边界测试用例

| 测试类型 | 测试输入 | 预期结果 |
|---|---|---|
| 异常 EB-01 全黑帧 | 全 0 的 RGB 帧，肩髋点正常；使用空白推理桩连续 30 帧 | 无异常；`obs=0`；不发布号码；预处理结果有限 |
| 异常 EB-02 全白帧 | 全 255 的 RGB 帧，肩髋点正常；使用空白推理桩连续 30 帧 | 无异常；`obs=0`；不发布号码； CLAHE 不产生 NaN/Inf |
| 异常 EB-03 肩部遮挡 | 一侧肩 confidence=0.1，另一侧正常 | 拒绝该次观测；不调用模型；下一帧肩恢复后可立即采样 |
| 异常 EB-04 髋部遮挡 | 肩正常，单侧或双侧髋 confidence<0.30 | 不使用低置信髋点；回退原始人体框；不拒绝本来可读的号码 |
| 异常 EB-05 号码区遮挡 | 肩髋正常，号码区域被人手/队友覆盖；模型判空白 | 几何裁片可保留；融合层不增加有效 `obs`；不发布猜测号码 |
| 异常 EB-06 空输入 | `objects=None`、`objects=[]`、`poses=None`、`poses=[]` | 不抛异常；不调用模型；不改变已有轨迹状态 |
| 异常 EB-07 非法人体框 | `bbox=None`、零宽、负宽、全部 NaN、全部 Inf | `recognize()` 跳过该对象；直接调用 `crop_torso()` 时应安全返回失败原因，不得抛未捕获异常 |
| 异常 EB-08 最小尺寸边界 | 躯干区域恰好 18×24，以及 17×24、18×23 | 恰好达到阈值可裁；任一维度小于阈值不推理 |
| 异常 EB-09 非有限关键点端到端 | 在 17×3 关键点任意位置放入 NaN/Inf，连续 20 帧 | 每帧都安全拒绝；模型调用次数为 0；不发布号码；服务不返回 5xx |
| 异常 EB-10 快速号码抖动 | 23→28→23→28 高速交替 100 帧 | 单轨迹切换≤1；切换后锁定；输出不出现 2 次以上闪号 |
| 异常 EB-11 轨迹中断与换队 | 发布后消失超过 `gap_reset_s`；以及同 track 的 team 从 A 变为 B | 超期状态被回收；换队后旧号码释放；重新出现不沿用旧号 |
| 异常 EB-12 模型输出 NaN/Inf | mock `predict()` 返回含 NaN/Inf 的 22 维 logits | 不得发布号码；不得让轨迹 `post` 变为 NaN；记录异常并继续处理后续帧。此用例用于验证模型输出兜底 |
| 异常 EB-13 非法质量权重 | 直接调用融合层，`quality` 分别为 -1、2、NaN、Inf | 有限且越界的值应夹取到 0–1；NaN/Inf 应按 0 或拒绝处理；不得污染后验。此用例用于验证内部健壮性 |

## 9. 并发测试用例

| 测试类型 | 测试输入 | 预期结果 |
|---|---|---|
| 并发 CC-01 CLAHE 结果一致性 | 生成 200 张固定裁片，先串行计算基线；8 线程各处理 200 次，线程启动/结束交错 | 每张并行结果与串行基线 `array_equal`；无异常；无灰度错位、花屏或数值漂移 |
| 并发 CC-02 CLAHE 线程本地 | 8 个线程分别记录 `_CLAHE.instance` 的 id，并重复访问 | 不同线程实例 id 不同；同一线程重复使用同一实例；实例数不超过工作线程数 |
| 并发 CC-03 预处理并发 | 8 线程同时对同一批裁片执行 `preprocess()` | 输出形状、dtype、数值与串行完全一致；无共享临时数组污染 |
| 并发 CC-04 共享 ONNX 推理 | 一个 `JerseyModel`，8 线程各执行 100 个 batch，每 batch 8 张 | 所有输出形状正确且有限；同一输入的并行输出与串行结果 `allclose`；无 session 崩溃；不重复创建 session |
| 并发 CC-05 单例创建竞态 | 16 线程同时调用 `_get_jersey`，一半请求 videoA，一半请求 videoB | 每个 `(vid,backend)` 只有一个 recognizer；全局只有一个 model；无半初始化对象；无重复模型加载 |
| 并发 CC-06 多视频同 track 并发 | 4 个视频同时运行，均使用 `track_id=1`，分别固定输出 23、28、35、46 | 各视频最终号码正确；事件 `vid` 正确；一个视频重置不影响其他视频；无状态串线 |
| 并发 CC-07 同视频同帧幂等 | 20 个线程同时对同一 recognizer、同一视频、同一帧、同一对象调用 `recognize()` | 只能累计一次有效观测、只能触发一次推理批次；重复请求不得重复发布或增加切换次数。可在 recognizer 内串行化，也可由服务层保证并在集成测试中覆盖 |
| 并发 CC-08 多视频压力 | 8 个视频各连续 300 帧，共享一个 ONNX model，每个视频独立 recognizer | 无未捕获异常；无跨视频串号；model 实例数仍为 1；记录吞吐、P95 和线程数，结束后线程数回到基线 |

### 并发测试的重点风险

共享 ONNX 会话只说明“模型推理可以并发”，不代表 `JerseyRecognizer._states` 的修改是线程安全的。CC-07 必须单独验证同一视频的重复请求；不能只验证不同视频并发后就宣称 recognizer 线程安全。

## 10. 执行顺序

1. 跑原有 13 项回归，确认基础行为未破坏。
2. 跑 UT-01～UT-20，覆盖所有新增分支。
3. 跑 IT-01～IT-10，验证服务接入、会话复用和异常兜底。
4. 跑 EB-01～EB-13，覆盖空帧、遮挡、非法输入和抖动。
5. 跑 CC-01～CC-08，覆盖 CLAHE、ONNX 和多视频并发。
6. 跑 RG-02～RG-04 和 PF-01～PF-07，对比基线。
7. 最后执行一次完整实时播放验收，确认 960 宽≥30fps。

新增测试文件就位后，可执行：

```powershell
.\.venv-jersey\Scripts\python.exe -m pytest -q `
  test_jersey.py `
  test_jersey_optimization.py `
  test_jersey_integration.py `
  test_jersey_concurrency.py `
  test_jersey_regression.py
```

## 11. 通过标准

出现以下任一情况，本次迭代不通过：

- 原有 13 项回归任一失败。
- 新增正常分支、异常降级、节拍释放、空白防误发、轨迹隔离用例失败。
- 合成 correctness 指标任一劣于基线。
- 真实视频出现跨视频串号、单轨迹切换超过 1 次或置信度 NaN/Inf。
- 同视频同帧并发导致重复计数、重复发布或状态损坏。
- 平均号码识别增量耗时超过 2ms，或同机对比劣于 1.87ms 基线。
- 960 宽实时播放低于 30fps，且号码模块是主要瓶颈。
- 真实视频测试报告错误地声称“准确率通过”。

## 12. 测试证据与缺陷记录

每轮至少保存以下本地证据，不提交到仓库：

- 测试命令、运行时间、Git commit、工作区是否干净。
- Python、NumPy、OpenCV、ONNX Runtime 版本与 provider。
- 模型文件 SHA256。
- pytest 完整输出。
- 3 次合成、融合、性能评估的 JSON 报告。
- 真实视频 240 帧的发布/切换事件清单。
- 并发测试的线程数、异常堆栈、CPU/内存峰值。
- 失败用例对应的最小输入和截图；截图仅用于本地排障。

缺陷等级建议：

- P0：串号、误发、状态损坏、服务崩溃、 correctness 明显回退。
- P1：新增分支不符合需求、性能超过预算、并发重复计数。
- P2：日志缺失、原因字符串不清晰、非关键统计字段异常。

## 13. 执行前特别关注

以下用例可能暴露当前实现缺口，不应提前放宽：

- UT-04、UT-05 要求髋点缺失或异常时回退原始人体框；需要确认实现没有用低置信髋点估算腰线。
- EB-12、EB-13 要求模型输出和质量权重非有限时不得污染后验；当前版本需要有明确兜底。
- CC-07 要求同视频同帧并发幂等；共享模型会话不能替代轨迹状态锁或请求层去重。
