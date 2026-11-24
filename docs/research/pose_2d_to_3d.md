# 2D→3D 骨架/人体模型调研与真实基准（篮球单目转播）

> 调研日期：2026-11-24
> 范围：为 `bsk/perceive/pose_3d.py` 几何基线寻找**许可干净、可实时、能落到球场米制世界坐标**的替代/增强方案。
> 约束：项目 Apache-2.0；依赖白名单 MIT / Apache-2.0 / BSD；禁 AGPL/GPL；权重须许可清晰、可商用/可再分发（禁 CC BY-NC、研究-only、无 LICENSE）；实时因果（不预取未来帧）；8GB 档才启用 3D。
> 方法：实际读取仓库 `LICENSE` 文件 / GitHub API 元数据 / HuggingFace 模型卡 / 数据集 EULA；**对许可允许的候选真实下载权重并跑基准**；对许可不允许的候选只读不下载。

---

## 0. 结论先行

### 0.1 有没有"许可允许商用/再分发且权重可用"的 2D→3D 模型？

**没有。**

- 所有主流 COCO17/H36M→3D 提升器（VideoPose3D / MotionBERT / MixSTE / PoseFormerV2 / MotionAGFormer / MeTRAbs / mmpose 的 RTMPose3D 系）**权重都训练在 Human3.6M / MPI-INF-3DHP / 3DPW 上**；这三者的条款分别是"仅学术、不可转授"（H36M EULA，已核实原文）、
  "非商业科研、未经书面许可不得再分发"（3DPW，已核实原文）、研究-only（3DHP，未核实原文）。
  → 权重既不能商用也不能放进 Apache-2.0 仓库**再分发**；即使只"内部使用"，也在灰色地带。
- 代码许可宽松并不能救权重：MotionBERT / MotionAGFormer 是 **Apache-2.0 代码**，PoseFormerV2 / MeTRAbs / OSX 是 **MIT 代码**，
  但权重许可由训练数据决定；MeTRAbs 甚至在 README 里明说 "The models can only be used for **non-commercial** purposes due to the licensing of the used training datasets"。
- VideoPose3D **代码本身就是 CC BY-NC 4.0**（已核实 `LICENSE` 文件全文），连"借鉴代码"都要小心；MixSTE 继承它的代码且**仓库没有任何 LICENSE 文件**（默认保留所有权利）。
- 人体网格路线（OSX / SMPLer-X / SMPL / SMPL-X / STAR）全部踩在 SMPL-X 的非商业许可或 S-Lab 非商业许可上（SMPL-X / STAR / SMPLer-X 已核实原文，SMPL 按同族条款推定）。
- **唯一例外**：MediaPipe **BlazePose GHUM**——模型卡明写 `LICENSED UNDER Apache License, Version 2.0`（已核实 PDF 原文）。
  但它是"图像 → 33 点 + 世界坐标"的端到端单人模型，模型卡明确 out-of-scope："多人"、">4 m 距离"、"需要米制精确深度的应用"。
  我们在真实转播帧上实测：**28–38 ms/人（CPU）**、米制尺度系统性偏小 ~20%、单人框 108 px 时 10% 失败 —— **不适合本项目**（详见 §5.8）。

### 0.2 推荐哪条路

**两条腿走，都不要第三方权重：**

1. **主线：继续用几何法（`pose_3d.py`）作为"米制锚"**——它给出根节点/脚点/身高，误差主要在"上半身前后深度、手腕"。
2. **增强：自训一个 576K 参数、因果 TCN 的"COCO17 + 相机射线 → 相机系根相对米制骨架"小模型**，只用**自造合成数据**训练（本项目比例表 + 随机广播机位，
   零第三方数据/权重），端到端 **0.52 ms/人/帧**（ORT CPU，含特征化与落地对齐）、显存 2.9 MB。
   实测（同一批 83 个合成样本，见 §5.3）：**几何 0.279 m → 几何根 + 学习形状 0.224~0.227 m（−19%）**，
   **上半身"前后深度"误差 0.258 → 0.173 m（−33%）**，**手腕 0.369 → 0.260 m（−30%）**；
   代价是踝关节基本持平（0.206 → 0.210~0.225 m）；**遮挡下只要训练时做 55% 整肢丢失增强，也能做到 0.292 m（几何 0.280）**（§5.4）。
3. **人体表示**：要网格就用 **MHR（Apache-2.0，已核实）**；否则继续程序化胶囊/关节骨架（零许可风险）。

### 0.3 三条硬结论

| 问题 | 答案 |
|---|---|
| 第三方 COCO17→3D 权重能用吗 | **全部不能用**（H36M/3DPW/3DHP 条款 + 部分仓库无 LICENSE） |
| SMPL/SMPL-X/STAR 与 Apache-2.0 兼容吗 | **不兼容**（非商业科研专用，需注册/签署；STAR 同为非商业）。替代：**MHR（Apache-2.0）**、自建胶囊骨架、CC0 网格 |
| 学习法值得做吗 | **值得，但只作为"形状先验"，米制/根节点仍由几何法给**；且必须自训，不能引权重 |

---

## 1. 我们的现状基线（对照用）

| 环节 | 事实 |
|---|---|
| 输入 | RTMO/YOLOv8-pose 的 COCO17（`kpts(17,3)` 像素+conf） |
| 相机 | `H`(court→pixel) 单应分解 → `K,R,t,C`（主点=图像中心、方形像素假设） |
| 输出 | 17×3 米制世界坐标（球场中心原点、z 向上），逐关节在**自己的像素射线**上、骨长严格保持 |
| 合成闭环 | MPJPE **0.279 m**（p90 0.47）、根 0.207、踝 0.206、腕 0.369（p90 0.80）、开销 0.237 ms/人/帧（H 变化时 0.352） |
| 已知弱项 | 上半身前后深度（沿光轴不可观测，靠时序连续性）；手臂/手腕；蹲下/腾空时根节点；H 坏了全盘不可信 |
| 真实帧 | 标定链路当前不可信 → `calibration.json` 的 H 反解出"骨盆在地面下 0.48 m"，**真实米制精度目前无法评估**（本次复测确认） |

---

## 2. 候选对比总表

> 许可栏为**实读**结果（附录 A 有原文摘录与链接）。"权重"列指官方预训练权重是否可下载。
> 精度列为论文数字（非本场景）；"篮球域可迁移性"是我们基于输入形态/训练域的判断，不是实测。

| 候选 | 代码 LICENSE（已核实） | 预训练权重 / 训练数据 | 输入 → 输出 | 公开精度（论文） | 实时性（量级） | 与球场 H 对齐方式 | 结论 |
|---|---|---|---|---|---|---|---|
| **VideoPose3D** (FAIR) | **CC BY-NC 4.0** | `pretrained_h36m_cpn.bin`（AWS）；H36M | 2D 轨迹窗口（243 帧）→ 根相对 3D（相机系，无米制） | H36M P1 **46.8 mm** / P2 36.5 | 极小模型（~0.4M），但 243 帧窗口=**8 秒未来** | 无（根相对）；需自行接 H | **禁止**（代码+权重都非商用） |
| **MotionBERT** | **Apache-2.0** | OneDrive + HF `walterzhu/MotionBERT`；H36M-SH / 3DPW / 3DHP | 17 点（**H36M 格式**）序列 ≤243 帧 → 根相对 3D | H36M **37.2 mm**（ft）/ 3DPW MPJPE 81.2 | 162 MB / Lite 61 MB；GPU 数十 ms/窗口 | 无；需转 COCO→H36M 并自接 H | **只可借鉴**（代码 Apache，权重训练数据受限） |
| **MixSTE** | **无 LICENSE 文件**（ARR）；且基于 VideoPose3D | 百度网盘/Google Drive；H36M / 3DHP | 243 帧窗口 → 根相对 3D | H36M ~45.6 mm（论文，未逐项核实） | 33M 参数级，seq2seq，重 | 无 | **禁止**（无许可 + CC BY-NC 上游） |
| **PoseFormerV2** | **MIT** | Google Drive；H36M / 3DHP | 27–243 帧（频域压缩）→ 根相对 3D | H36M **45.2 mm**（README 权重文件名 `27_243_45.2.bin`） | 轻（~1.0 GFLOPs 档），仍要 27+ 帧 | 无 | **只可借鉴**（权重受限） |
| **MotionAGFormer** | **Apache-2.0** | Google Drive；H36M / 3DHP | 27/81/243 帧 → 根相对 3D | H36M ~38.4 mm（论文，未逐项核实） | 2.2M(XS)/11.7M(B)/19M(L)；B 档 48.3 GFLOPs/窗口 | 无 | **只可借鉴**（权重受限） |
| **MeTRAbs** | **MIT** | TF Hub / `bit.ly/metrabs_l`；COCO+MPII+H36M+3DHP+3DPW 等 28 个数据集 | 图像/裁剪 → **绝对 3D（含相机内参时可给世界坐标）** | 3DPW / CMU-Panoptic 强 | MobileNetV3 版较快，但 TensorFlow SavedModel | **自带"给定标定就出世界坐标"**，与我们思路一致 | **禁止权重**（README 明示 non-commercial）；可借鉴其"绝对尺度 + 截断鲁棒"设计 |
| **OSX** | **MIT** | Google Drive；AGORA/COCO/3DPW/H36M…，输出 SMPL-X | 图像 → SMPL-X 参数（含弱透视相机） | AGORA 榜 SMPL-X 第一（2023-03） | ViT-L/B，慢 | 有相机分支，但需 SMPL-X 资产 | **禁止**（SMPL-X 非商业 + 权重无明示许可） |
| **SMPLer-X** | **S-Lab License 1.0（非商业）** | 官方 ckpt；混合大规模数据 + SMPL-X | 图像 → SMPL-X | — | ViT-H，重 | 同上 | **禁止** |
| **Pose2Sim** | **BSD-3-Clause** | 无自有 3D 权重（用 RTMPose/OpenPose + OpenSim） | **多机位**三角化 → OpenSim IK | 运动学角度误差论文级 | 慢（OpenSim），离线 | 多机位世界系 | **代码可引入，但对单目无帮助** |
| **SportsPose** | **仓库无 LICENSE**（ARR）；数据集许可未核实 | 数据集网站下载；SMPL 标注 | 多为"多视角一致性"训练方法 | — | — | — | **仅思路**（方法值得读，代码/权重不可用） |
| **mmpose RTMPose3D / video_pose_lift** | **Apache-2.0**（框架） | 模型配置全部位于 `configs/body_3d_keypoint/*/h36m/`；H36M | 2D 序列 → 根相对 3D | H36M ~45 mm 级 | 轻（ONNX 友好） | 无 | **代码可引入，权重不可用**（H36M） |
| **MediaPipe BlazePose GHUM** | **Apache-2.0**（代码 + 模型卡 `LICENSED UNDER Apache License, Version 2.0`） | `pose_landmarker_{lite,full,heavy}.task`（已实测下载） | **图像/裁剪** → 33 点（图像系）+ **world landmarks（骨盆原点，声称米）** | 论文 PDJ 93.6%（自建验证集） | **实测 28–38 ms/人 CPU**（见 §6.3） | 可给"根相对米制" → 仍需 H 才能落到球场 | **许可唯一可用，但场景 out-of-scope（多人/>4 m/米制深度），不建议** |
| **OpenVINO human-pose-estimation-3d-0001**（及 HF 转档 INT4） | OMZ 为 Apache-2.0；HF 转档自标 MIT | 权重可下；**训练数据未核实**（疑似 H36M 系） | 图像 → 单人气相机系 3D | — | 快（Lightweight OpenPose 系） | 无 | **未核实 → 不得作为引入依据** |
| **本项目几何法** | Apache-2.0（自有） | 无权重 | COCO17 + H → 米制世界坐标 | 合成闭环 0.279 m | **0.237 ms/人/帧** | 原生 | **主线** |

**一句话**：能"引代码"的有 MotionBERT / MotionAGFormer / PoseFormerV2 / mmpose / Pose2Sim（思路或框架层）；
能"引权重"的**只有 MediaPipe BlazePose**，而它恰好在本场景 out-of-scope；篮球/运动专项的 3D 权重一个都没有。

---

## 3. 逐候选：输入输出、能否落地球场米制、失败模式

### 3.1 提升器（root-relative）类的共性

VideoPose3D / MotionBERT / MixSTE / PoseFormerV2 / MotionAGFormer / mmpose 的 3D 提升配置**同构**：

- 输入：**2D 关键点序列（H36M 17 点顺序）**，坐标经"根关节减均值 + 缩放"归一化；
- 输出：**相机坐标系下、根（骨盆）相对**的 3D，**没有米制尺度**（网络没见过米的定义，只学过 H36M 的骨长分布）；
- 结构上必须吃**时间窗口**（27–243 帧）：243 帧 @30fps = **8.1 秒**，与"实时因果"直接冲突；27 帧 = 0.9 秒，仍要 **延迟 26 帧**。
  可切一半做因果（只用过去帧），但精度会掉（论文里的数字都是对称窗口调的）。
- 要落到球场米制，必须自己补三件事，**没有一件是模型给的**：
  1. **尺度**：单目 + 已知球场 → 用脚点射线∩地面反推（我们已有），或用人体高度先验；
  2. **朝向/平移**：根相对骨架的方向在相机系里，需用相机 R 转到世界，再解一个 3 自由度平移；
  3. **地面约束**：脚必须踩 z=0，否则"整体漂浮"。

**失败模式（这一类的通病）**：
- 窗口越长越依赖"未来帧"，因果化后退化；
- 训练域是"室内、3–5 m、单人、全身可见"，转播是"10–30 m、多人、常被遮挡/出画、变焦"；
- 归一化后的输入**丢掉了尺度**，同一姿势在 10 m 与 30 m 处输入几乎相同 → 深度只能靠骨长先验猜；
- 输出无米制 → 一旦接错尺度，整副骨架沿射线滑走（我们实测过：自由解尺度会跑到 20 倍以上，见 §4.1）。

### 3.2 绝对尺度类（MeTRAbs / OSX / SMPLer-X / BlazePose）

- MeTRAbs：**唯一把"绝对尺度 + 截断鲁棒"作为核心卖点**的方法，输入裁剪图像，输出"根相对 + 绝对尺度"；官方支持给了标定就出世界坐标。
  设计上与我们最贴（metric-scale heatmap + "给定标定就出世界坐标"），值得按这个方向自研。
  但**权重 non-commercial**，只能当参考文献。
- OSX / SMPLer-X：图像 → SMPL-X 参数（含弱透视相机）。输出是**网格**，要 17 关节得做回归/取顶点；
  需要 SMPL-X 模型文件（非商业），权重无明示许可 → 双保险禁止。同族的 4D-Humans / CLIFF / CameraHMR / NLF 结论相同（MIT/Apache 代码 + SMPL 权重 + 混合训练数据）。
- BlazePose：图像裁剪 → 33 点 + world landmarks（骨盆原点，米）。**单人、≤4 m、非米制精确**（模型卡自述）。

### 3.3 多机位类（Pose2Sim / SportsPose）

- Pose2Sim：**多机位**三角化 + OpenSim IK，产出的是"运动学量（关节角）"，不是单目深度；
  代码 BSD-3 可以引入，但我们只有单机位转播 → **零收益**。可借鉴的是它的"标定→三角化→IK"流程与 trc 输出约定。
- SportsPose：核心贡献是"用多视角一致性约束训练单目模型"（CVPRW 2025）。这个方法**非常适合我们**
  （我们有单应 H 与球场几何，可以构造弱监督），但代码/权重无许可 → **只借鉴思路**。

---

## 4. 与球场 H 对齐的工程做法（把 root-relative 变成米制）

这是"引入任何学习法"都必须自己写的一层。我们实测了三种做法。

### 4.1 做法 A：自由解（X₀ 与尺度 s 一起解）——**不可用**

约束"关节落在自己的像素射线上"：
`min_{X₀,s} Σ wᵢ‖(I − dᵢdᵢᵀ)(X₀ + s·pᵢ − C)‖²`，对 (X₀, s) 线性。

**实测**：20 m 外、2 px 噪声下"尺度—深度"耦合几乎是零空间 → s 解到 **21–28 倍**，MPJPE 25 m 级。
**结论**：单目远距离的绝对尺度不是"解"出来的，是"先验"给的（脚点/身高/模型自带的米制）。

### 4.2 做法 B：脚锚 + 射线正则（`place_on_court`，推荐）

1. 双踝像素射线 ∩ 地面 z=0 → 脚锚（这一步几何法本来就做得很准，误差厘米级）；
2. 平移骨架，使"预测姿态里的踝中点"落在脚锚上方 `ANKLE_H`；
3. 用全部**可见**关节的射线做 γ 正则的最小二乘微调（γ=0.35），**不引入深度自由解**；
4. 触地修正：最低踝不低于地面。

**实测**：单独用学习法 + 做法 B → 0.281 m（与几何 0.279 持平），踝 0.162（优于几何 0.206）。

### 4.3 做法 C：几何根 + 学习形状（`hybrid`，最优）

用几何法给的骨盆位置做整体平移，只采用学习法的**关节相对几何**：
`X = (X₀^geo − root(pred)) + s(=1)·pᵢ`

**实测**（变体 A）：0.224 m（−20% vs 几何），上半身前后深度 −33%，手腕 −30%，踝 +9%；变体 B 为 0.227 m / −32% / −25% / +2%。
**结论**：**米制锚交给几何法，形状交给学习法**，这是本场景的正确分工。

**共同失败模式**：
- 脚点不可观测（出画/被挡/腾空）→ 脚锚失效，必须有"框底兜底 + 只用比例"的降级；
- H 本身错 → 三者同时错，且错得"自洽"（重投影仍然≈0），**没有任何自检能发现**——必须用标定侧的可信度门控；
- 学习法把腿预测短了 3 cm，脚锚一贴地就把全身抬高 3 cm（误差沿链传播）→ 训练时加骨长损失（我们加了，腿长偏差从 −0.11 m 收到 −0.018 m）。

---

## 5. 真实基准（本机实测）

环境：Windows 11 / RTX 5070 12.8 GB / torch 2.7.0+cu128 / onnxruntime-gpu 1.22 / Python 3.12。

### 5.1 复现现有几何基线

```bash
.venv/Scripts/python.exe eval_pose_3d.py --cams 24
# [合成闭环] 83 个有效样本（24 机位 × 4 姿势，σ=1.5px）
#   总体：MPJPE 0.279m(p90 0.47) 根 0.207m 踝 0.206m 身高 -3.0% 重投影 1.8px
#   开销：缓存命中 0.237 ms/人/帧；H 每帧变化 0.352 ms/人/帧
```

### 5.2 自训小模型（唯一可交付的"学习法"）

```bash
# 训练（自造合成数据，24k 段 × 16 帧；无任何第三方权重/数据集）
.venv/Scripts/python.exe docs/research/pose_2d_to_3d_bench.py train --seqs 24000 --epochs 250
#   数据生成 105 s；训练 156 s（RTX 5070）；参数 576K
#   域内验证（相机系根相对）rel MPJPE 0.111 m，手腕 0.185 m，腿长偏差 -0.018 m

# 推荐变体 B：训练时 55% 概率整肢/单点丢失（遮挡增强）
.venv/Scripts/python.exe docs/research/pose_2d_to_3d_bench.py train --seqs 24000 --epochs 200 \
    --drop 0.55 --ckpt data/models/pose_lift_tcn_occ.pt
#   域内 rel MPJPE 0.121 m，手腕 0.205 m，腿长偏差 -0.027 m

# 闭环对照 + 遮挡 + 时序 + 延迟/ONNX（--ckpt 切换变体）
.venv/Scripts/python.exe docs/research/pose_2d_to_3d_bench.py eval --cams 24
.venv/Scripts/python.exe docs/research/pose_2d_to_3d_bench.py eval --cams 24 --ckpt data/models/pose_lift_tcn_occ.pt
.venv/Scripts/python.exe docs/research/pose_2d_to_3d_bench.py temp
.venv/Scripts/python.exe docs/research/pose_2d_to_3d_bench.py speed
```

### 5.3 闭环对照数字（同一批 83 个合成样本，世界坐标米制）

训练了两个变体（同一结构，只有"训练时整肢丢失概率"不同）：

- **A**：`drop_p=0.25`，250 epoch → `data/models/pose_lift_tcn.pt`
- **B（推荐）**：`drop_p=0.55`，200 epoch → `data/models/pose_lift_tcn_occ.pt`

| 方案 | MPJPE | p90 | 根 | 踝 | 腕 | 上半身前向 | 全向 |
|---|---|---|---|---|---|---|---|
| 几何（现有基线） | 0.279 | 0.37 | 0.207 | **0.206** | 0.369 | 0.258 | 0.220 |
| 学习 A（脚锚做法 B） | 0.281 | 0.33 | 0.273 | **0.162** | 0.311 | 0.232 | 0.228 |
| **混合 A（几何根 + 学习形状）** | **0.224** | **0.29** | 0.207 | 0.225 | **0.260** | **0.173** | **0.170** |
| 学习 B（脚锚做法 B） | 0.282 | 0.33 | 0.281 | **0.161** | 0.318 | 0.235 | 0.232 |
| **混合 B（推荐）** | **0.227** | 0.29 | 0.207 | 0.210 | 0.277 | 0.175 | 0.169 |

（"上半身前向"= 肩/肘/腕沿相机水平主轴的误差，专门衡量"前后深度"；"全向"= 所有关节沿水平主轴的误差）

### 5.4 遮挡（左肩/肘/腕 2D 全丢）——训练时遮挡增强是关键

| 方案 | MPJPE | 根 | 踝 | 腕 | 上半身前向 |
|---|---|---|---|---|---|
| 几何 | 0.280 | 0.222 | **0.206** | 0.458 | 0.204 |
| 学习 A（drop 0.25） | 0.508 | 0.279 | 0.329 | 0.785 | 0.457 |
| 混合 A | 0.418 | 0.222 | 0.510 | 0.673 | 0.300 |
| 混合 A + 逐关节门控 | 0.408 | 0.222 | 0.510 | 0.570 | 0.226 |
| 学习 B（drop 0.55） | 0.356 | 0.338 | 0.189 | 0.502 | 0.278 |
| **混合 B（drop 0.55）** | **0.292** | 0.222 | 0.254 | 0.437 | **0.191** |
| 混合 B + 逐关节门控 | 0.315 | 0.222 | 0.254 | 0.480 | 0.205 |

两个必须记住的结论：
1. **遮挡增强训练极其有效**：同结构只把训练时的整肢丢失概率从 0.25 提到 0.55，遮挡下的混合方案从 **0.418 → 0.292**（几何 0.280），
   踝误差从 0.510 回到 0.254。不做遮挡增强，"学习法"在遮挡下必然比几何法差（0.508 vs 0.280）。
2. **"缺失点交回几何法"的门控不是无条件有效**：对欠训练的模型 A 有帮助（0.418→0.408，腕 0.673→0.570），
   对训练充分的模型 B **反而变差**（0.292→0.315），因为此时几何法对缺失关节的处理（射线推到骨盆）比学习法的推断更差。
   → 门控要按"A/B 实测"决定，不能拍脑袋默认开。

### 5.5 时序（因果窗口的收益，40 段随机机位 × 24 帧，变体 A）

| 方案 | 首帧（无历史） | 稳态（第 8 帧后） |
|---|---|---|
| 几何（depth EMA τ=0.1 s） | 0.315 | 0.519 |
| 学习（TCN，16 帧因果窗） | 0.263 | 0.281 |

几何法在**部分机位会把深度 EMA 锁死在错误的分支上**（用项目自带 `temporal_test` 复测四个种子：
0.213→0.197 / 0.213→0.600 / 0.249→0.187 / 0.229→0.389）。
学习法没有这个问题，但也没有明显时序增益（因为它主要靠 2D 局部几何 + 尺度先验）。

### 5.6 延迟 / 显存 / ONNX

| 运行方式 | 单次窗口（16 帧） | 备注 |
|---|---|---|
| torch CPU | 2.36 ms | 小 batch 的框架开销占比大 |
| torch GPU（5070） | 0.97 ms | kernel launch 主导 |
| **ORT CPU（默认线程）** | **≈0.2 ms**（实测 0.13–0.21） | 推荐运行时 |
| ORT CUDA | 0.71 ms | — |

**端到端**（真实关键点，特征化 + 推理 + 落地对齐，各测 5×200 次取中位）：

| 链路 | ms/人/帧 |
|---|---|
| 纯 ORT + numpy（推荐） | **0.52** |
| 纯 torch + numpy | 2.95 |
| 现有几何法（对照） | 0.24 |

参数 576K、ONNX 2.3 MB、推理峰值显存 **2.9 MB**。→ 几何 + 学习合计 ≈ **0.75 ms/人/帧**，10 人 ≈ 7.5 ms/帧（仍在 1.5 ms/人 预算内、且只需 8GB 档开启）。
注意：因果流式下每人每帧都要跑一次整窗（不是摊薄到 16 帧）。

### 5.7 真实帧（`harden-55`，20–24 帧，RTMO 2D）

```bash
.venv/Scripts/python.exe docs/research/pose_2d_to_3d_bench.py real
#   geo      n=96 骨长变异 0.633 身高 1.50m 脚离地 -0.448m 骨盆高 -0.48m 越界 0%
#   learned  n=96 骨长变异 0.385 身高 1.52m 脚离地 0.093m 骨盆高 0.12m  越界 0%
```

**怎么读这组数**：几何法给出"骨盆在地面下 0.48 m"，说明当前 `calibration.json` 的 H 是坏的（与既定结论一致）；
学习法的"脚离地 0.09 m"是它**按构造**贴地的结果，**不是精度证据**。
→ **真实米制精度仍然无法评估**；等另一条线的标定修好后，这个脚本可以直接复用做 A/B。

### 5.8 MediaPipe BlazePose（唯一许可可用的第三方权重，实测）

```bash
# mediapipe 装在隔离目录（不动 .venv；jax/opencv-contrib 会污染主环境）
.venv/Scripts/python.exe -m pip install --no-deps --target .tools/mp     mediapipe==0.10.21 protobuf==4.25.3 absl-py attrs
.venv/Scripts/python.exe docs/research/pose_mp_bench.py --frames 20 --model lite
.venv/Scripts/python.exe docs/research/pose_mp_bench.py --frames 20 --model full
```

| 指标 | lite | full |
|---|---|---|
| 成功检出 | 72/80 | 72/80 |
| 延迟（CPU，单人裁剪） | **28.0 ms**（p90 34.8） | **38.3 ms**（p90 45.6） |
| world 肩宽 / 大腿 / 小腿 | 0.300 / 0.317 / 0.350 m | 0.332 / 0.361 / 0.343 m |
| \|骨盆→踝\|（世界坐标） | 0.62 m | 0.66 m |
| 与 RTMO 同名 2D 点偏差 | 15.5% 框高 | 19.8% 框高 |

真人参考值：肩宽≈0.40、大腿≈0.42、小腿≈0.43、骨盆→踝≈0.85 m。
→ BlazePose 的"米"比真人**系统性小 15–25%**（GHUM 关节定义 + >4 m 距离外推），且 2D 与 RTMO 分歧 15–20% 框高。
→ **不能作为深度/尺度来源**；单人 28 ms 也撑不起 10 人转播。**结论：许可可用，但技术上不用。**

**能否对齐球场**：BlazePose 只给"骨盆原点、根相对"的世界坐标，落地仍要我们的 H 做脚锚/平移（§4）。
本次没有做它的米制落地，原因是两重：(a) 当前 H 不可信；(b) 它自带 ~20% 的尺度偏差，
直接用会让"脚锚 + 17 cm 腿长误差"这类系统偏差进入米制链路。这条**不建议继续投入**。

### 5.9 训练成本（自训方案的现实数字）

| 环节 | 实测 |
|---|---|
| 合成数据生成（24k 段 × 16 帧，numpy） | 105 s（单进程） |
| 训练 250 epoch（576K 参数，bs=256，RTX 5070） | 156 s |
| 一次完整"数据+训练+评估"迭代 | **< 5 分钟** |
| 域内 rel MPJPE | 0.111 m（手腕 0.185 m） |
| 跨生成器闭环（世界坐标） | 0.224 m（混合）/ 0.281 m（纯学习） |

---

## 6. 人体模型许可：SMPL 系能不能用？

| 模型 | 许可 | 与 Apache-2.0 兼容 | 证据 |
|---|---|---|---|
| **SMPL** | 非商业科研（注册下载） | 否 | 与 SMPL-X 同族条款；原文需登录，**未逐字核实** |
| **SMPL-X** | "Software Copyright License for **non-commercial scientific research** purposes"，单用户、不可转让 | **否** | 已核实 `vchoutas/smplx/LICENSE` 原文 |
| **STAR** | "License … for **non-commercial** scientific research purposes" | **否** | 已核实 `ahmedosman/STAR/LICENSE` 原文 |
| **MHR**（Meta Momentum Human Rig，2025） | **Apache-2.0** | **是** | 已核实 `facebookresearch/MHR/LICENSE` + README "MHR is licensed under the Apache Software License 2.0"；资产在 GitHub Release（`assets.zip` 199 MB / 分 LOD 24 MB…） |
| GHUM（BlazePose 内置） | 模型卡 `LICENSED UNDER Apache License, Version 2.0` | 是 | 已核实 PDF |

**许可干净的人体表示方案（按推荐度）**：

1. **程序化胶囊/关节骨架（现状）**：零许可风险、零资产、可视化足够表达战术/起跳/朝向；
   保真度差距：没有体表/衣服/手部细节，近景回放看着"塑料"。
2. **MHR（Apache-2.0）**：真实网格 + 骨骼 + FACS 表情，可导出 glTF/TorchScript；
   差距：官方推荐 PyMomentum（`pymomentum-cpu` 需 PyTorch 2.8+），v1.2.0 起也提供"只依赖 PyTorch"的转换资产；
   资产 25–200 MB（按 LOD）、要写 COCO17→MHR 关节的重定向（1–2 天）；
   注意：README 声明 Apache-2.0，但**资产包内是否有逐文件声明未核实**，落地前要把 `assets.zip` 里的 LICENSE 也读一遍。
3. **CC0 网格 + 自建蒙皮**（Blender Studio 的 CC0 base mesh；MakeHuman 的**资产**是 CC0 但**代码是 AGPL**，只能取资产）：法律最干净，但要自己做绑定与蒙皮权重，
   工程成本高于 MHR；适合"只要一个能穿球衣的人形"的展示场景。

**不建议**：为了 SMPL 的外观去接 SMPL/SMPL-X/STAR——那会把整个仓库拖进非商业许可，与 Apache-2.0 再分发直接冲突。

---

## 7. 学习法 vs 几何法：能提升多少，代价是什么

| 维度 | 几何法（现状） | 学习法（自训 TCN A/B） | 混合（推荐，B） |
|---|---|---|---|
| 上半身前后深度 | 0.258 m | 0.232 / 0.235 m | **0.175 m（−32%）** |
| 手腕 | 0.369 m | 0.311 / 0.318 m | **0.277 m（−25%）** |
| 踝 | **0.206 m** | 0.162 / 0.161 m | 0.210 m（+2%） |
| 遮挡（整肢缺失） | **0.280 m** | 0.508（A）/ 0.356（B） | **0.292 m** |
| 总体 MPJPE | 0.279 m | 0.281 / 0.282 m | **0.227 m（−19%）** |
| 延迟 | 0.237 ms/人/帧 | +0.52 ms（ORT 端到端） | +0.52 ms |
| 显存 | 0 | 2.9 MB | 2.9 MB |
| 许可 | 自有 | **自训（自有数据）** | 同左 |
| 需要什么 | H + 当前帧 | H + **过去 16 帧** + 每人状态 | 两者 |

**代价清单**：16 帧因果历史（每 track 一个 ring buffer，~1 KB）、每帧一次小推理、需要与几何法共用同一份 H 与 track 状态、
训练数据必须自造（不能引权重）、遮挡增强训练（否则遮挡下比几何法差 80%）。

---

## 8. 自训小模型方案（规避第三方权重许可）

### 8.1 数据需求

| 项 | 我们的实测取值 |
|---|---|
| 生成器 | 本项目骨长比例表 + 连续关节角（髋/膝/踝/肩/肘/外展/体前倾/扭转）+ 骨长个体差异 ±6% |
| 相机 | 与 `eval_pose_3d.random_camera` 同分布（10–32 m、6–24 m 高、f 500–1500 px@960） |
| 噪声/退化 | 像素 σ 0.5–3 px；25% 概率整肢/单点丢失（可提到 55%） |
| 规模 | 24k 段 × 16 帧（≈38 万帧），生成 105 s |
| 关键：**域差** | 合成数据里"镜头/模糊/球衣/遮挡/变焦"都是缺的 → 精度会被系统性高估；必须补真实素材微调或用真实帧做人审 |

### 8.2 模型与训练

- 结构：逐帧 1×1 Conv(52→192) + 4 层因果膨胀 TCN（dilation 1,2,4 循环，GELU + BN）+ 1×1 Conv(192→51)；
- 输入 52 维 = 17×2 相机射线斜率 + 17 维可见性掩码 + 1 维 `log(角尺度)`；
  *为什么这样：射线斜率让网络对相机姿态不变；角尺度是单目唯一可观测的尺度线索；掩码让网络知道哪些点不可信。*
- 输出 51 维 = 17×3 相机系、根相对的**米制** 3D；
- 损失：MSE + 0.5×骨长一致性 L1；
- 训练：AdamW、OneCycle、bs=256、250 epoch、156 s。

### 8.3 预期精度（诚实区间）

- 合成域内（相机系根相对）：**0.11 m**（变体 A）/ **0.12 m**（变体 B，遮挡增强的代价）；
- 跨生成器（闭环、世界坐标、混合）：**0.22 m**；
- 遮挡（整肢缺失）：**0.29 m**（需 55% 遮挡增强；否则 0.42 m）；
- 真实转播：**未知**。按经验，合成→真实的域差通常让 MPJPE 再涨 30–80%；
  我们的判断是"真实域 0.30–0.45 m 量级"，只有等标定修好、有 20–50 个真实标注样本才能定论。

### 8.4 最小验证方案（建议按此验收）

1. 标定修好后，挑 3 段真实片段 × 每段 20 帧，人工标注 6 个关节（踝/膝/髋/肩/肘/腕）米制位置；
2. 同一批帧上跑：几何 / 混合 / 混合+门控，比 MPJPE、踝、腕、前后深度；
3. 验收门槛：**混合方案在"腕"和"上半身前向"上比几何法好 ≥20%，且踝不劣化 >5%**，否则不上线；
4. 延时验收：ORT CPU 端到端（特征+推理+对齐）≤0.8 ms/人/帧，10 人 ≤8 ms/帧；
5. 回滚开关：`BSK_POSE_LIFT=geo|hybrid`，默认 `geo`。

---

## 9. 最小适配器草案（不改生产代码，接口对齐 `Pose3DLifter.lift_result`）

```python
# 拟新增：bsk/perceive/pose_lift_net.py（本次不写，只给草案）
# 目标：作为 Pose3DLifter 的"形状提供者"，对外仍由 lift_result 返回 LiftResult。

class LearnedShape:
    """COCO17 + H → 相机系根相对米制 3D（因果窗口）。许可：自训权重，Apache-2.0。"""
    def __init__(self, onnx_path="data/models/pose_lift_tcn.onnx", k_win=16):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.k_win, self._hist = k_win, {}          # {track_id: deque(maxlen=k_win)}

    def featurize(self, uv, conf, cam):             # → (52,)  见 §8.2
        ...

    def predict(self, uv, conf, cam, tid):          # → (17,3) 相机系、根相对、米
        f = self.featurize(uv, conf, cam)
        self._hist.setdefault(tid, deque(maxlen=self.k_win)).append(f)
        win = list(self._hist[tid])
        win = [win[0]] * (self.k_win - len(win)) + win          # 冷启动=重复首帧（与几何法首帧等价）
        return self.sess.run(None, {"x": np.asarray(win, np.float32)[None]})[0][0, -1].reshape(17, 3)


class HybridPose3DLifter(Pose3DLifter):
    """几何法给米制锚（脚点/根/身高），学习法给关节形状；缺失关节回退几何。"""

    def lift_result(self, kpts_px, bbox_px, H, track_id=-1, frame=0):
        geo = super().lift_result(kpts_px, bbox_px, H, track_id, frame)   # 现有几何解
        if geo is None or not self.use_net:
            return geo
        cam = self.camera(H)
        rel = self.net.predict(kpts_px[:, :2], kpts_px[:, 2], cam, track_id)
        p_w = rel @ cam.R                                     # 相机系 → 世界轴
        shift = geo.root - 0.5 * (p_w[11] + p_w[12])          # 只借几何的"根节点"
        xyz = p_w + shift
        if self.gate_missing:                                 # 默认 False（§5.4 实测：训练充分后门控反而变差）
            miss = kpts_px[:, 2] < VIS_CONF
            xyz[miss] = geo.kpts3d[miss]
        geo.kpts3d = xyz                                      # 其余字段（conf/visible/root/ground）沿用
        geo.reproj_err_px = float(np.median(
            np.linalg.norm(cam.project(xyz)[0] - kpts_px[:, :2], axis=1)))
        return geo
```

**替换"几何逐关节深度"的具体位置**：`Pose3DLifter._solve()` 的第 5 步（躯干/上肢/头骨长二次方程推进）
与第 3/4 步（腿链）都保留——它们负责根节点与落地；学习法只覆盖"骨盆以上的相对几何"，
即把 `_solve` 返回的 `xyz` 中 `[0..10]`（头/肩/肘/腕/鼻眼耳）替换成学习值，髋/膝/踝保持几何解。
这样"踝不劣化"这条验收就天然满足。

---

## 10. 风险清单

1. **许可**：任何第三方 H36M/3DPW/3DHP 权重（含 HF 上"自标 Apache/MIT"的转档）都**不可用**；
   `MotionBERT-3d-ONNX` 这类社区转档的 license 标签**不能洗白上游训练数据**。
2. **合成→真实域差**：闭环 0.22~0.23 m 是"同分布自证"，真实域预计明显更差；**不要把这个数字写进对外材料**。
3. **遮挡反噬**：不训练遮挡时误差扩散全身（实测踝 0.225→0.510）；**优先做遮挡增强训练**（0.418→0.292），门控是否开按 A/B 决定（训练充分后开了反而变差）。
4. **自洽的错误**：H 坏了时几何/学习都"重投影≈0"却全错 → 仍需标定侧的可信度门控（现有 `is_plausible` 系）。
5. **依赖**：ONNX Runtime 已在依赖里（Apache-2.0）；MediaPipe 若引入需装 `mediapipe`（Apache-2.0，但依赖 jax/opencv-contrib，**会污染主环境**，本次是装在隔离目录 `.tools/mp`）。
6. **尾延迟**：TCN 是"每人每帧一次整窗"，实测端到端 0.52 ms/人 → 10 人 ≈5.2 ms/帧，可接受；若换成 MotionAGFormer-B 级（48.3 GFLOPs/窗口）会直接打爆预算。
7. **模型文件管理**：`data/models/` 已 gitignore；**任何受限权重都不得放入仓库**，包括本次实测的 `pose_landmarker_*.task`（虽许可宽松，也按"不入库"处理）。

---

## 11. 最小集成计划（建议顺序）

| 步骤 | 内容 | 产出/验收 |
|---|---|---|
| P0（0.5 天） | 把本文件 §9 的适配器落到研究分支；不改 `pose_3d.py` 主路径，开关默认关 | A/B 可跑、可回滚 |
| P1（1 天） | 合成数据加强：变焦序列、运动模糊、字幕条遮挡、真实白线背景（复用 `calib_seg_real.py` 的合成管线） | 跨域精度不再掉 30% |
| P2（1 天） | 遮挡增强：整肢缺失概率 0.55、连续 5–10 帧缺失、只露上半身（§5.4 已验证 0.55 单帧丢失从 0.418→0.292） | 遮挡下踝误差回到 ≤0.25 m |
| P3（0.5 天） | 标定修好后，按 §8.4 做真实 60 帧标注验收 | 决策：是否把 3D 档默认切到 hybrid |
| P4（可选，2–3 天） | 若需要"人形"展示：接 MHR（Apache-2.0）做 COCO17→MHR 关节重定向 + glTF 导出 | 视觉验收，不影响米制链路 |

**明确不建议做的**：
- 接 MotionBERT/MotionAGFormer 权重（许可）；
- 接 SMPL/SMPL-X/STAR（许可 + 需要注册令牌 + 再分发冲突）；
- 用 MediaPipe 做深度来源（场景 out-of-scope + 28–38 ms/人）。

---

## 12. 参考链接

**2D→3D 提升**
- VideoPose3D（CC BY-NC 4.0）：https://github.com/facebookresearch/VideoPose3D
- MotionBERT（Apache-2.0 代码）：https://github.com/Walter0807/MotionBERT ｜权重 https://huggingface.co/walterzhu/MotionBERT
- MixSTE（无 LICENSE）：https://github.com/JinluZhang1126/MixSTE
- PoseFormerV2（MIT）：https://github.com/QitaoZhao/PoseFormerV2
- MotionAGFormer（Apache-2.0）：https://github.com/TaatiTeam/MotionAGFormer
- MeTRAbs（MIT 代码 / 非商业权重）：https://github.com/isarandi/metrabs
- mmpose 3D lifter 配置（Apache-2.0 框架，H36M 权重）：https://github.com/open-mmlab/mmpose/tree/main/configs/body_3d_keypoint

**网格 / 人体模型**
- OSX（MIT 代码 + SMPL-X）：https://github.com/IDEA-Research/OSX
- SMPLer-X（S-Lab 非商业）：https://github.com/MotrixLab/SMPLer-X
- SMPL-X（非商业）：https://github.com/vchoutas/smplx
- STAR（非商业）：https://github.com/ahmedosman/STAR
- **MHR（Apache-2.0）**：https://github.com/facebookresearch/MHR
- Sapiens（CC BY-NC 4.0）：https://github.com/facebookresearch/sapiens

**其他**
- Pose2Sim（BSD-3，多机位）：https://github.com/perfanalytics/pose2sim
- SportsPose（仓库无 LICENSE）：https://github.com/ChristianIngwersen/SportsPose
- MediaPipe Pose Landmarker（Apache-2.0，模型卡 `Model Card BlazePose GHUM 3D.pdf`）：https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- Human3.6M EULA（仅学术）：http://vision.imar.ro/human3.6m/eula.php
- 3DPW 数据许可（非商业、禁止再分发）：https://virtualhumans.mpi-inf.mpg.de/3DPW/

**本仓库产出**
- 研究脚本：`docs/research/pose_2d_to_3d_bench.py`（合成数据生成 + 自训 TCN + 闭环/遮挡/时序/延迟基准）
- 研究脚本：`docs/research/pose_mp_bench.py`（MediaPipe BlazePose 真实帧基准）
- 权重（不入库，`data/models/`）：`pose_lift_tcn.pt`（2.3 MB，自训 A）、`pose_lift_tcn_occ.pt`（2.3 MB，自训 B，推荐）、`pose_lift_tcn.onnx`（2.3 MB）、`pose_landmarker_{lite,full}.task`（MediaPipe，仅用于许可对照实验）

---

## 附录 A：许可证证据等级

### A.1 已核实（直接读文件/元数据/模型卡，附原文）

| 对象 | 证据 |
|---|---|
| VideoPose3D 代码 | `LICENSE` 首行 = `Attribution-NonCommercial 4.0 International`；README "This work is licensed under CC BY-NC." |
| MotionBERT 代码 | `LICENSE` = `Apache License Version 2.0`；GitHub API `license.spdx_id = Apache-2.0` |
| MixSTE | 仓库根目录只有 `.DS_Store/.gitignore/README.md/assets/common/requirements.yaml/run.py`，**无 LICENSE**（GitHub API `license = null`）；README 自述"based on VideoPose3D" |
| PoseFormerV2 | `LICENSE` = `MIT License Copyright (c) 2024 Qitao Zhao` |
| MotionAGFormer | `LICENSE` = `Apache License Version 2.0`；README 给出 H36M / MPI-INF-3DHP 权重下载 |
| MeTRAbs | `LICENSE` = MIT；README 原文："**The models can only be used for non-commercial purposes due to the licensing of the used training datasets.**" |
| OSX | `LICENSE` = MIT；README 要求从 smpl-x.is.tue.mpg.de 下载 SMPL-X 资产 |
| SMPLer-X | `LICENSE` = `S-Lab License 1.0 … use for non-commercial purpose`；第 4 条要求商业用途联系作者 |
| SMPL-X | `LICENSE` = `Software Copyright License for non-commercial scientific research purposes`（单用户、不可转让） |
| STAR | `LICENSE` 同款非商业科研条款 |
| MHR | `LICENSE` = Apache-2.0；README "MHR is licensed under the Apache Software License 2.0"；Release v1.2.0 资产 `assets.zip` 199 MB |
| Sapiens | `LICENSE` 首行 = `Creative Commons Attribution-NonCommercial 4.0 International` |
| Pose2Sim | `LICENSE` = `BSD 3-Clause License Copyright (c) 2022, perfanalytics` |
| mmpose | `LICENSE` = Apache-2.0；`configs/body_3d_keypoint/{image_pose_lift,video_pose_lift,motionbert}` 下**只有 `h36m/`** |
| MediaPipe BlazePose | 模型卡 PDF `Model Card BlazePose GHUM 3D.pdf` 字段 `LICENSED UNDER: Apache License, Version 2.0`；out-of-scope 明写"Multiple people in an image"、">4 m"、"Applications requiring metric accurate depth" |
| Human3.6M | `eula.php` 原文："GRANT OF LICENSE FREE OF CHARGE FOR **ACADEMIC USE ONLY**"、"non-exclusive, non-assignable and **non-transferable**"、"you may not … **sub-license or transfer**" |
| 3DPW | 许可页原文："For **non-commercial** and scientific research purposes"、"The Data may not be reproduced, modified and/or made available in any form to any third party without Max-Planck's prior written permission." |

### A.2 未核实（不得作为引入依据，仅线索）

| 对象 | 原因 |
|---|---|
| MPI-INF-3DHP 数据集条款 | 页面需注册/申请，未取到原文；按同族惯例视为研究-only。因 H36M 已足以否决，不影响结论 |
| SMPL（原始 v1.0/1.1）条款原文 | 下载需登录；与 SMPL-X 同族，保守按非商业处理 |
| SportsPose 数据集许可 | 官网只给"到网站下载"，Zenodo 检索接口未返回许可字段 |
| OpenVINO `human-pose-estimation-3d-0001` 训练数据 | OMZ 模型页未取到；HF 转档（`ukyotachibana/...-INT4`）自标 MIT 属再分发者单方声明 |
| HF 社区转档 `bukuroo/MotionBERT-3d-ONNX` | 卡片自标 `apache-2.0`，但上游是 H36M 训练 → 标签无效 |
| MHR `assets.zip` 内逐文件许可 | 仅核实仓库 LICENSE 与 README 声明；落地前需解压核对 |
| `pose_3d.py` 之外的几何解算细节 | 由另一条线负责，本文件只做对照 |