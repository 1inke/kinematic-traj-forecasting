# 面向固定候选预算（K=20）的解码端优化与场景先验：文献核验、T1 实证复盘与单因素实验设计（v2.0 修订版）

> **版本**：v2.0（依据 [REVIEW.md](../qoder_work/DECODER_SURVEY_REVIEW_20261023/REVIEW.md) 审计意见与 [T1_execution_20261023](../qoder_work/F16_CONTEXT_TAIL_RETENTION_DESIGN_20261023/T1_execution_20261023/RESULT_ANALYSIS.md) 实测结果全面大修）  
> **日期**：2026-10-23  
> **定位说明**：本文为**选择性相关工作梳理与下一步单因素实验设计启发**（非穷尽式系统文献综述）。文中所有机制均明确区分**“原论文原始设定”**、**“本项目任务适配”**与**“待验证科学假设”**，撤回一切未经实测的数学保证性承诺。

---

## 一、 核心修订说明与当前最新证据起点

### 1. 本次 v2.0 修订改正的关键边界问题
在 v1.0 版本中，存在三类跨越科学证据边界的表述，已在本文全面订正：
1. **文献书目与输入条件核实**：
   * 订正 **SmartRefine (CVPR 2024)** 第一作者为 **Yang Zhou**（非 Haosheng Zhao），明确其核心为基于预测质量分数（Quality Score）的迭代细化次数自适应控制，而非简单 $\sigma$ 门控残差；
   * 订正 **SHENet (NeurIPS 2022)** 正式标题为 *Forecasting Human Trajectory from Scene History*（Mancheng Meng 等），明确其完整模型**同时使用了场景 RGB 图像与预训练 Swin 编码器**，我们仅借鉴其 *Group Trajectory Bank* 历史轨迹检索思想，而非宣称原论文为免图像方法；
   * 明确 **SingularTrajectory (CVPR 2024)** 的环境自适应锚点依赖**图像语义分割生成的可通行图（Traversability Map）**，并非免图像历史轨迹统计；
   * 补齐 **MemoNet (CVPR 2022)** 完整引文信息。
2. **数学与代码语义严谨化**：
   * 修正 **aWTA (ICRA 2025)** 与本地 `src/f16/hybrid.py` 的关键差异：补充**对玻尔兹曼责任权重停止梯度（`.detach()`）**的导数推导，避免未停止梯度时对高误差候选产生反向推离梯度；
   * 删除“分段线性基/位置正交天然保证 `accel_mean` 达标”的错误推论：位置空间正交（$\mathbf{B}\mathbf{\Phi}^\top = 0$）不推出二阶差分空间正交，且锚点幅度放大或速度方向往复同样会推高 `accel_mean`；
   * 修正参数量账本：明确冻结编码器（$267,159$）+ 现学生头（$82,008$）整模为 **$349,167$ 参数**，已超出 G5 的 $305,000$ 预算，并给出逐层精确参数核算。
3. **更新实验起点（纳入今日已完成的 `T1` 结果）**：
   * `T1`（`A_replay` vs `P_relative`）已于 2026-10-23 14:25 完成全部 6 个 fresh runs 与独立审计，结论为 **`TRANSFER_UNRESOLVED` / `MECHANISM_SUPPORTED=false`**。

---

### 2. 最新实证基线复盘：`S1` 与 `T1` 揭示了什么？（OBSERVATION vs INTERPRETATION）

结合 `10.22 S1 报告`、`D0/D1 补充分析` 与今日刚完成的 `10.23 T1 报告`：

| 指标（`inner change` 组，497 窗 / 35 tracks） | `native20` 参考 | `A_replay` (seed 42 / 43 / 44) | `P_relative` (seed 42 / 43 / 44) | 关键实证观察 (OBSERVATION) |
| :--- | :---: | :---: | :---: | :--- |
| **完整好候选覆盖窗口数** | 166 | 186 / 191 / 192 | 167 / 175 / 197 | `P_relative` 在 seed 42/43 仅保住 `A` 正净增的 $5\% / 36\%$（未达 $90\%$ 门） |
| **相对 `native` 救回 / 丢失** | 0 / 0 | 53/33, 62/37, 63/37 | 51/50, 46/37, 61/30 | `P_relative` 的原生成功丢失率仍达 $30.1\% / 22.3\% / 18.1\%$ |
| **正相对退化 $R = \operatorname{mean}[J^*_{\text{stu}} - J^*_{\text{nat}}]_+$** | 0 | 0.4201 / 0.3130 / 0.3433 | 0.3653 / 0.3177 / 0.3268 | 三 seed 平均仅降 $6.20\%$（低于 $20\%$ 目标），seed 43 反向（$+1.48\%$），CI 跨 0 |
| **`minADE` (m)** | 0.2597 | 0.3066 / 0.2814 / 0.2911 | 0.2965 / 0.2889 / 0.2869 | 相对 `native` 仍恶化 $+0.0312\text{ m}$；共同失败子集仍是主要正退化贡献源 |
| **`accel_mean` 相对 `native` 变化** | $0.0898\text{ m/s}^2$ | $+41.7\% / +32.4\% / +30.0\%$ | $+33.5\% / +36.3\% / +28.5\%$ | 24/24 组五项联合运动门依然 0/24 通过，全部卡在 `accel_mean` |

* **OBSERVATION（确凿事实）**：
  1. 在事先按旧 `A` / `native` 冻结的四类子集上，`P_relative` 虽然降低了旧 `A` 丢失子集（$-0.410 / -0.242 / -0.171$）和共同失败子集（$-0.193 / -0.005 / -0.051$）的平均 $J$，但**同时显著劣化了旧共同成功子集（$+0.205 / +0.041 / +0.047$）和旧 `A` 救回子集（$+0.336 / +0.338 / +0.253$）**。
  2. 所有 20 个槽位在最终 epoch 均承担了 WTA 责任（distinct 均值 $19.45 \sim 20$），并非静态意义上的“槽位完全死掉”。
  3. 运动学上，`native` 在 11 个加速度步中仅第 4 步（$v_5-v_4$）和第 8 步（$v_9-v_8$）显著非零，而 `A` 与 `P` 在全部 11 个时间步上均有非零加速度（且第 1 步 $v_2-v_1$ 幅值达 $0.258 \sim 0.323\text{ m/s}^2$），呈现“峰值 `accel_p95` 和 `jerk` 大幅下降，但算术均值 `accel_mean` 升高 $28\% \sim 41\%$”的结构特征。
* **INTERPRETATION（受限解释）**：
  * 固定权重的全样本平均单侧相对惩罚 $L_{\text{rel}}$ 通过共享网络参数更新产生了**跨样本子集的折中交换（Trade-off）**：为了拉回未达标样本的 $J^*$，牺牲了已达标/刚救回样本在阈值边界附近的精度，不能充当逐窗口硬保留机制。
* **NOT PROVEN（未证明项）**：
  * `D0` 中共同失败子集对 `minADE/FDE` 退化的算术贡献最大，仅定位了“退化发生在哪类窗口”，**并未证明“非赢家外飘或排斥项是导致赢家恶化的唯一因果机制”**（因为固定预测下改坏非最优槽位不会直接改变 $\min_k \text{ADE}_k$，且 $0.3\text{ m}$ 线性 hinge 在距离 $>0.3\text{ m}$ 时梯度为 0）。
  * 同样，上游 F16 编码器在历史训练中已见过 `FIT` 和 `inner`，当前实验均属于**固定编码表征下的解码器机制探索**，不构成独立全模型未见泛化证据。

---

## 二、 相关文献核验与可借鉴技术边界（2022–2025）

| 文献（作者、年份、会议） | 论文正式标题与原始任务设定 | 原论文核心机制 | 本项目可借鉴的合理启发 | 严格边界（原论文不保证什么） |
| :--- | :--- | :--- | :--- | :--- |
| **DeMo**<br/>Bozhou Zhang 等<br/>**NeurIPS 2024** | *DeMo: Decoupling Motion Forecasting into Directional Intentions and Dynamic States*<br/>（自动驾驶 Argoverse 2 / nuScenes，长时域 K=6） | 将查询拆为 **Mode Queries**（跨注意力捕获多模态方向意图 + 模式间自注意力）与 **State Queries**（利用 Attention/Bi-Mamba 显式建模时间步动态状态序列），最后通过 Hybrid Coupling 融合 | 启发我们将“模式分叉选择”与“未来 12 步时间演化结构”分开建模，而非直接用 `128->24` 无结构线性层输出自由坐标 | 原文的 State Queries 依赖序列状态空间模型（Mamba/Attn），并非固定低秩曲率基底；**不能借 DeMo 之名宣称天然保证 ETH/UCY 五项运动学过门** |
| **SmartRefine**<br/>Yang Zhou 等<br/>**CVPR 2024** | *SmartRefine: A Scenario-Adaptive Refinement Framework for Efficient Motion Prediction*<br/>（多阶段自适应轨迹细化） | 在每个细化步预测质量分数 $q_i$（用 $\ell_1$ 损失监督离 GT 最近模式的准确度与剩余潜力），当初始质量足够高或改善饱和时停止细化，避免**过度修正（Over-refinement）** | 提供了“无差别修正会改坏已有好预测”的文献依据；启发我们探讨相对于基准路径的有条件/受限残差更新 | 原文是针对多轮迭代细化的推理控制（且原文亦指出并非每条轨迹均受益），**不等于单步 $\sigma$ 门控残差，更不提供逐窗口零 Lost 的数学定理** |
| **aWTA**<br/>Yihong Xu 等<br/>**ICRA 2025** | *Annealed Winner-Takes-All for Motion Forecasting*<br/>（多模态运动预测中的 WTA 训练策略） | 采用温度退火玻尔兹曼分布分配多假设责任权重，且**对权重显式停止梯度（`.detach()`）**：$w_k = \operatorname{stopgrad}(\operatorname{softmax}_k(-\ell_k / \tau(t)))$，随训练将 $\tau(t)$ 退火至趋近硬 WTA | 可作为单因素实验检验：**正确的 stop-gradient 退火责任分配**相比硬 WTA 是否能改善训练早期的槽位分工与尾部样本收敛 | 共享主干参数的梯度更新仍可能在窗口间发生折中；**不能承诺彻底消除尾部发散或解决原生好窗口丢失**，且温度区间需在本地预注册验证 |
| **TUTR**<br/>Liushuai Shi 等<br/>**ICCV 2023** | *Trajectory Unified Transformer for Pedestrian Trajectory Prediction*<br/>（ETH/UCY & SDD 行人轨迹预测，$K=20$） | 用显式 Mode-level Transformer 解码器替代事后聚类后处理，通过模式查询间的交互并行输出多模态轨迹与概率 | 支持在固定 $K=20$ 解码内部引入**跨槽位通信（Cross-mode Interaction）**以协调候选分工 | 原文使用标准 Transformer 注意力而非 `mean/max` 拼接 MLP；原文也未证明“一切 GT-free 选择器在理论上必然失败” |
| **EigenTrajectory**<br/>Inhwan Bae 等<br/>**ICCV 2023** | *EigenTrajectory: Low-Rank Descriptors for Multi-Modal Trajectory Forecasting*<br/>（ETH/UCY & SDD，$K=20$） | 用训练集 SVD 提取前 $k$ 个奇异向量作为低秩轨迹描述符（Descriptor），在紧凑低秩空间预测锚点与细化偏移 | 支持用**数据驱动或结构化的低维时间基底**替代 24 维逐点自由回归，改变解码器的时间归纳偏置 | 低秩奇异向量仅保证训练集二阶范数重构误差最小，**不自动等于低加速度均值，也不等于与 `native` 三段折线基底正交** |
| **SingularTrajectory**<br/>Inhwan Bae 等<br/>**CVPR 2024** | *SingularTrajectory: Universal Trajectory Predictor Using Diffusion Model*<br/>（五类行人预测任务通用框架） | 将不同任务的运动投影至统一 Singular Space，利用**由输入图像经语义分割所得的可通行图（Traversability Map）**修正不合理的初始原型锚点，再经级联扩散去噪 | 启发“在低维运动子空间内对先验原型锚点进行环境条件修正”的几何思路 | **原文的自适应锚点依赖视觉图像分割可通行图**，不能将其引述为“免图像历史轨迹统计先验”的直接证明 |
| **SHENet**<br/>Mancheng Meng 等<br/>**NeurIPS 2022** | *Forecasting Human Trajectory from Scene History*<br/>（ETH/UCY & SDD） | 构建 **Group Trajectory Bank** 检索场景历史代表性轨迹作为参考候选；同时使用预训练 **Swin Transformer 提取场景 RGB 图像特征**并进行跨模态交互 | 其 **Group Trajectory Bank 子模块**直接支持“挖掘场景内过往其他行人轨迹以提供未来路径先验”的思想 | **完整 SHENet 含重型视觉骨干（非免 RGB）**；引用时必须明确说明本项目的 `HistogramPrior` 是仅借鉴其历史轨迹库思想的纯轨迹轻量化变体 |
| **MemoNet**<br/>Chenxin Xu 等<br/>**CVPR 2022** | *Remember Intentions: Retrospective-Memory-Based Trajectory Prediction*<br/>（ETH/UCY & SDD） | 构建训练实例的过去运动 Key 与未来意图终点 Value 记忆库，通过可学习检索与聚类得到代表性目的地，条件化解码路径 | 支持“先由历史记忆检索少量代表性终点模式，再作为解码器条件输入”的两步生成逻辑 | 原文采用基于特征相似度的神经记忆库与实例级检索，不等同于固定空间网格 2D 直方图，亦无未知场景冷启动无损定理 |
| **NPSN**<br/>Inhwan Bae 等<br/>**CVPR 2022** | *Non-Probability Sampling Network for Stochastic Human Trajectory Prediction*<br/>（ETH/UCY & SDD，$K=20$） | 在预训练随机预测器上附加小型可学习采样网络，利用 Box-Cox 准蒙特卡洛（QMC）序列差异度改善 $K=20$ 潜变量采样分布 | 支持“固定 $K=20$ 下通过优化候选集合分布减少抽样冗余”的动机 | 潜空间均匀覆盖不等于输出空间满足本项目同路径 $\text{ADE}\le 0.2\text{ m} \land \text{FDE}\le 0.4\text{ m}$、原生保留或运动学护栏 |

---

## 三、 六项关键技术问题的严谨数学与工程澄清

### 1. `aWTA` 的停止梯度（Stop-Gradient）数学条件
在本地 `src/f16/hybrid.py` 的现有 `set_loss` 中，`weights = torch.softmax(-q / temperature, dim=1)` 未做 `.detach()`。
若直接对未停止梯度的 $L = \sum_{j=1}^K w_j Q_j$ 求关于第 $j$ 条候选误差 $Q_j$ 的偏导：
$$\frac{\partial L}{\partial Q_j} = w_j \left(1 + \frac{\sum_{k=1}^K w_k Q_k - Q_j}{\tau}\right)$$
**反例**：当某非最优候选误差较大，满足 $Q_j > \sum_k w_k Q_k + \tau$ 时（例如 $Q=[0, 1],\ \tau=0.25$，则 $w_2 \approx 0.0180$，但 $\frac{\partial L}{\partial Q_2} \approx -0.0527 < 0$），梯度下降反而会**增大** $Q_2$（将劣质候选进一步推离真值）！
因此，若测试 **aWTA (ICRA 2025)**，必须严格按照作者官方实现，**对责任权重执行 `.detach()`**：
```python
# 严格停止梯度的 aWTA 任务项（待验证假设，非现成已验证代码）
w = torch.softmax(-q / tau, dim=1).detach()
loss_awta = (w * q).sum(dim=1).mean()
```
此时 $\frac{\partial \mathcal{L}_{\text{aWTA}}}{\partial Q_j} = w_j \ge 0$ 对全部 $K=20$ 槽位恒成立。

### 2. 时间结构化基底与 `accel_mean` 的真实数学关系
在采样间隔 $\Delta t = 0.4\text{ s}$ 下，单条轨迹的平均加速度定义为：
$$\text{accel\_mean} = \frac{1}{11 \Delta t} \sum_{t=1}^{11} \|v_{t} - v_{t-1}\|_2$$
* **为什么不能写“分段线性基天然保证 `accel_mean` 达标”**：
  即使完全采用与 `native` 相同的 $t \in \{4, 8, 12\}$ 三节点分段线性插值矩阵 $\mathbf{B}_{\text{pw}}$（仅在第 1、4、8 步改变速度），只要预测的锚点转折角或加减速幅度大于 `native`，$\|v_5 - v_4\|_2$ 就会增大，`accel_mean` 依然会超过 $1.01 \times \text{native}$。
* **为什么 S1 学生 `A` 的 `accel_mean` 暴涨 $30\% \sim 41\%$**：
  不仅是因为速度变化分摊到了 11 个时间步，更关键的是：(1) 第 1 步（观测末尾速度 $v_{\text{obs}}$ 到第 1 步预测速度 $v_1$ 的衔接差 $\|v_1 - v_{\text{obs}}\|_2 / \Delta t$）达到了 $0.258 \sim 0.323\text{ m/s}^2$；(2) 24 维自由坐标输出在相邻时间步之间存在非单调的速度方向摆动（使得路径速度总变差 $\sum_t \|v_{t} - v_{t-1}\|_2 > \|v_{12} - v_{\text{obs}}\|_2$）。
* **正确的研究定位**：结构化时间参数化（如速度增量基底或受限节点插值）是通过**消除首步速度跳变与相邻步非必要摆动**来降低加速度总变差的**待检验归纳偏置**，其能否在不损伤 `change` 覆盖率的前提下通过 G1 护栏，必须通过实验实测，不能预先宣称数学保证。

### 3. 整模参数预算（$349,167$ vs $305,000$）与逐层精确账本
* **当前实测状态（`execution_20261022/core.py`）**：
  * 冻结上游 F16 编码器：$267,159$ 参数；
  * `ContextK20` 学生解码器：$82,008$ 参数：
    * `trunk`: `Linear(210, 128)` ($27,008$) + `Linear(128, 128)` ($16,512$) = **$43,520$**
    * `queries`: `Parameter(20, 16)` = **$320$**
    * `fuse`: `Linear(144, 128)` ($18,560$) + `Linear(128, 128)` ($16,512$) = **$35,072$**
    * `head`: `Linear(128, 24)` ($3,096$) = **$3,096$**
    * **学生合计**：$43,520 + 320 + 35,072 + 3,096 = \mathbf{82,008}$ 参数。
  * **整模部署合计**：$267,159 + 82,008 = \mathbf{349,167}$ 参数（**已超出 `PROTOCOL.md` G5 的 $305,000$ 上限 $44,167$ 参数**）。
* **工程核算约束**：
  1. 在当前固定冻结编码器（$267,159$）的局部探索阶段，若要保持与 `ContextK20`（$82,008$）严格同容量对比，任何新增模块必须通过**等参数替换**实现，并精确断言参数量。
  2. 若要满足整模 $\le 305,000$ 的正式 G5 协议门，则解码器总参数必须控制在 $\le 37,841$ 以内，或者在端到端重训阶段同步压缩编码器。

### 4. `src/f16/scene_prior.py` 现有接口的实际行为边界
1. `Fragment` 数据类虽然定义了 `midpoint` 字段，但 `HistogramPrior.query()` **仅对 `r.endpoint` 构建了 2D 直方图**，`modes()` 也只返回 $m$ 个二维终点格心 `[m, 2]`。
2. 当局部 `(cell, heading)` 样本数 $\le 5$ 时，`query()` 依次回退到 `cell` $\to$ `scene_heading` $\to$ `global`；此时返回的 `answer['support']` 是**回退层级的样本数**，判断局部支持不足必须检查 `answer['level'] != 'cell_heading'`。
3. 评测 `A0` 时，必须完整执行 `DESIGN_AB.md` 第 2.3 节的全套对照（`M0`、`M1`、`PRIOR`、`SHUF`、`XSCENE`）及三道证伪闸门。

---

## 四、 修订后的研究优先级与严格单因素实验设计

我们将后续探索拆解为**三个相互独立、严格单因素对照**的最小实验方向，**绝不打包混跑**：

### 优先方向 A：观测速度衔接的结构化时间输出参数化（针对 `accel_mean` 与时间结构）
1. **固定时间基底矩阵 $\mathbf{\Psi} \in \mathbb{R}^{M \times 12}$（先做离线零训练可达性核验）**：
   $$\hat{Y}_{k,t} = t \Delta t \cdot v_{\text{obs}} + \sum_{m=1}^{M} c_{k,m} \Psi_{m,t}, \qquad t=1,\dots,12$$
2. **离线零训练可达性门（Pre-training Reachability Gate，必跑）**：
   在启动任何训练前，先用纯 NumPy 将 `FIT` 和 `inner` 的真实 GT 轨迹以及 `S1` 学生 `A` 的 20 条轨迹投影到基底 $\mathbf{\Psi}$ 上，实测可达率与 `accel_mean` 变化。若离线投影本身就会丢失 `change` 覆盖或无法降低 `accel_mean`，则直接终止该基底。

### 优先方向 B：场景历史轨迹先验零训练探针 `A0`（借鉴 SHENet / MemoNet 的纯轨迹子集）
* 完整运行 `DESIGN_AB.md` 第 2.3 节的五组零训练对照（`M0`、`M1`、`PRIOR-static`、`PRIOR-causal`、`SHUF`/`XSCENE`），若在 `inner change` 上未同时通过三道证伪条件，则直接关闭主线 A 并归档为 `SUCCESS_SCIENTIFIC_NEGATIVE`。

### 次优先方向 C：责任分配（`stopgrad-aWTA`）与集合通信（`SetMix`）的独立单因素验证
* **实验 C1（仅改损失责任分配）**：固定 `ContextK20`（$82,008$ 参数）不变，仅将 `q.min(1)` 替换为带 `.detach()` 的 `stopgrad-aWTA`。
* **实验 C2（仅改跨槽通信结构）**：固定 `L_base` 不变，对比跨槽位 `SetMix` 与严格同参数的单槽位 MLP（`IsoParamMLP`）。

---

## 五、 参考文献（经核验订正版）

1. **[DeMo, NeurIPS 2024]** Zhang, B., Song, N., & Zhang, L. *DeMo: Decoupling Motion Forecasting into Directional Intentions and Dynamic States*. NeurIPS 2024. https://arxiv.org/abs/2410.05982
2. **[SmartRefine, CVPR 2024]** Zhou, Y., Shao, H., Wang, L., Waslander, S. L., Li, H., & Liu, Y. *SmartRefine: A Scenario-Adaptive Refinement Framework for Efficient Motion Prediction*. CVPR 2024, pp. 15281–15290. https://arxiv.org/abs/2403.11492
3. **[SingularTrajectory, CVPR 2024]** Bae, I., Park, Y.-J., & Jeon, H.-G. *SingularTrajectory: Universal Trajectory Predictor Using Diffusion Model*. CVPR 2024, pp. 17890–17901. https://arxiv.org/abs/2403.18452
4. **[aWTA, ICRA 2025]** Xu, Y., Letzelter, V., Chen, M., Zablocki, É., & Cord, M. *Annealed Winner-Takes-All for Motion Forecasting*. ICRA 2025. https://arxiv.org/abs/2409.10369
5. **[TUTR, ICCV 2023]** Shi, L., Wang, L., Zhou, S., & Hua, G. *Trajectory Unified Transformer for Pedestrian Trajectory Prediction*. ICCV 2023, pp. 9675–9684.
6. **[EigenTrajectory, ICCV 2023]** Bae, I., Oh, J., & Jeon, H.-G. *EigenTrajectory: Low-Rank Descriptors for Multi-Modal Trajectory Forecasting*. ICCV 2023, pp. 10017–10029.
7. **[QCNet, CVPR 2023]** Zhou, Z., Wang, J., Li, Y.-H., & Huang, Y.-K. *Query-Centric Trajectory Prediction*. CVPR 2023, pp. 17863–17873.
8. **[SHENet, NeurIPS 2022]** Meng, M., Wu, Z., Chen, T., Cai, X., Zhou, X., Yang, F., & Shen, D. *Forecasting Human Trajectory from Scene History*. NeurIPS 2022, 35, pp. 24920–24933. https://arxiv.org/abs/2210.08732
9. **[MemoNet, CVPR 2022]** Xu, C., Mao, W., Zhang, W., & Chen, S. *Remember Intentions: Retrospective-Memory-Based Trajectory Prediction*. CVPR 2022, pp. 6488–6497.
10. **[NPSN, CVPR 2022]** Bae, I., Park, J.-H., & Jeon, H.-G. *Non-Probability Sampling Network for Stochastic Human Trajectory Prediction*. CVPR 2022, pp. 6477–6487. https://arxiv.org/abs/2203.13471
