# kinematic-traj-forecasting 主线 A / B 实验设计方案 v0.1

> **状态**：草案（待 S0 预注册冻结）｜**上位规范**：《kinematic-traj-forecasting 统一评价协议与实验治理规范 v1.0》（下称 PROTOCOL）｜**适用基准**：内部协议（FIT / inner / outer / original_CAL）
>
> 本文所有判定、统计、阶段门都直接引用 PROTOCOL，不另立标准。凡与 PROTOCOL 冲突处，以 PROTOCOL 为准。

---

## 0. 总览

| 项 | 主线 A：轨迹统计场景先验 → 生成 | 主线 B：单阶段互补生成（绕开选择器） |
| --- | --- | --- |
| 要解决的问题 | change 组“去哪儿”的信息缺失：编码器只看 8 帧自身与邻居，看不到“在这里朝这个方向走的人后来去了哪” | 原生 ∪ KIN 的 K40 互补性很强（+9.4pp），但压回 K20 一直靠选择器，而选择器正是瓶颈 |
| 核心假设 | H-A：由**其他行人的历史轨迹**统计出的条件流向/终点先验，携带 F16 编码器没有的意图信息；把它注入**生成**（而非重排），能增加 change 组完整好候选 | H-B：只要训练目标是**集合级覆盖**、并让两种解码方式在槽位间**软竞争**，一个 K=20 生成器能在训练期内化 K40 的互补性，推理时不需要任何 scorer / selector |
| 第一道闸 | A0 零训练信息探针（便宜，先证伪） | B-S1 冒烟 + 槽位坍缩诊断 |
| PRIMARY | outer change ΔCov（vs 同批对照） | 同左 |
| 关键对照 | 置换先验 A-shuf（同容量、错信息） | 同预算重训 native B0；固定配额 B3 作为与 EXP-168A 的连接 |
| 最大风险 | 跨划分 track 重叠导致先验泄漏；ETH/UCY 场景少 | 槽位坍缩到单一解码头；覆盖↑但 ADE/FDE↑（重演 cond_boost_11_9） |

**共同前置依赖（未完成前不得进入 S2）：**

1. **D-0 实测**：给出 outer change 在 3 seeds 下的 ΔCov 噪声与 MDE。本文中的 +2pp / +4pp 等目标值在 D-0 后复核，若 MDE 大于目标值，先按 PROTOCOL 讨论扩样（加 seeds 或并入 inner 复评）再开跑。
2. **PROTOCOL §2.4 track 重叠检查（C1–C6）**：对主线 A 是**硬阻塞**（先验直接由 FIT 轨迹构建，重叠即泄漏）；对主线 B 是常规阻塞。

---

## 1. 证据基础与去重

### 1.1 已有结论（本设计的出发点）

| 证据 | 结论 | 对 A / B 的含义 |
| --- | --- | --- |
| K0-C oracle：change_good 126/133/127 vs native 110，平均约 0.4 次替换 | 好候选“差一点就有”，缺口小而集中 | 两条主线的增益空间都在 +5～+9pp 量级，不要预期大幅跃升 |
| K1-S 误拒约 82% 有益动作；GTF 几何 top1 命中 0.65% | 选择器几乎无判别力 | B 的动机：不再依赖选择器 |
| Q0：TRAINING_EXPOSURE_OVERFIT（仅 C_seen 有信号） | 基于训练暴露的信号不迁移 | A 必须做严格的 track 级隔离与置换对照 |
| KIN（显式运动学积分 + 速度/加速度软限幅）单用覆盖仍比 native 低约 4pp；native ∪ KIN K40 +9.4pp | 两者互补而非替代 | B 的“老师”与双头结构来源 |
| cond_boost_11_9（GT-free K40→K20）+4.4pp，但 minADE +11%、minFDE +19% | 覆盖—误差折中，未达部署门 | B 必须过 G3（ADE/FDE 恶化 ≤3%） |

### 1.2 与既往实验的去重表

| 既往实验 | 做了什么 | 结果 | 本设计的差异 |
| --- | --- | --- | --- |
| EXP-168A 联合混合 K20 | 固定 10 structured（H3+V2）+ 10 free kinematic，共享 F16 encoder 与 coverage objective | 五折 macro J +1.22%（门槛 2%），ETH 退化 2.11%，FAIL | B：不固定配额；用 **KIN 头**（非 H3/V2）；每槽位软混合，配额随样本变化；集合级损失 + 可选 union40 老师 |
| EXP-168B 功能互补分析 | structured 负责宽/远尾部，free 负责常见；Univ 中 structured winner 随速度 13.3%→39.8% | 互补随运动状态变化 | B 的软配额正是为此设计；B 的诊断要复现“按速度分桶的头胜出率” |
| EXP-168C family gate | observation-only 分类 family，AUC 0.708 | 质量收益失败 | B 不做硬 gating；不训练独立分类器 |
| EXP-172A K40→K20 零训练 K-medoids 蒸馏 | GT-free 距离（ADE+0.25FDE）压缩 union | **complete，FAIL**：macro J −3.1%（比 V2 更差），union 增益保留率 −0.16，仅 2/5 折更好，最差折 −9.7% | 证明**事后、无监督**地压缩 union 会丢掉互补增益。B 改为**训练期、有 GT 监督**的集合学习；172B 未实施，本设计不复用 coreset |
| EXP-178A 场景拓扑可达性（SDD） | 训练视频轨迹弱标签训练 9.7K walkability CNN，沿骨架生成 proposal | **FAIL**：minADE +0.18%、minFDE +1.30%（门槛 5%），7/17 视频正向；新候选仅在 1.49% 样本胜出；拓扑特征与候选质量 Spearman 约 0.03～0.05 | 失败原因是“几何可达但缺意图”。A 的先验是**条件化流向**（在此位置、此朝向的人后来去哪），不是可通行性 |
| EXP-183A 场景意图探针（SDD，DINOv2） | P3_rich vs motion-only P0 预测终点 | **STOP**：终点增益 0.38%（门槛 3%），场景置换优势 0.37%；rich 特征可识别视频 ID（0.447 vs 0.107），有捷径风险 | A0 照搬其**置换对照**思路，但信息源换成轨迹统计；同时把“识别场景 ID”列为泄漏诊断 |
| EXP-190 路径示能重排 | 场景作为重排特征 | regret 仅降 0.68%，CI 跨 0，MARGINAL | A 作用于生成而非重排 |
| EXP-191 动态残差（SDD） | 非行人动态障碍避碰惩罚作重排残差 | NEUTRAL：增益 0.00%，CI [0,0] | 同上；A 不做惩罚型残差 |

**一句话边界**：既往场景工作全部在 SDD、基于图像/拓扑、作用于重排或稀疏 proposal，均失败；既往 K20 互补工作要么固定配额（168A），要么事后压缩（172A / cond_boost），均未达门。A、B 都只在这些失败点之外开新口子，并把“为何这次不同”写成可证伪条件。

---

## 2. 主线 A：轨迹统计场景先验注入生成（ScenePrior-Gen）

### 2.1 假设与可证伪陈述

- **H-A**：对 change 窗口，条件先验 $p(\Delta x_{t+12} \mid \text{cell}, \text{heading})$（由其他行人轨迹统计）比运动学外推更能预测终点方向；把先验模式注入 Mode Query 生成，会产生 native 缺失的完整好候选。
- **证伪条件（任一成立即停 A）**：
  1. A0 中，先验相对 motion-only 基线在 inner change 上的终点 NLL 改善 ≤3%（与 183A 门槛同级）；
  2. A0 中，正确先验相对置换先验的优势 ≤ 总改善的 1/2（说明增益来自“有个分布”而非“对的分布”）；
  3. A0 中，native 漏掉的 inner change 窗口里，先验 top-M（M=4）终点模式能救回的比例 ≤10%；
  4. S2 中，A-prior 相对 A-shuf 的 inner change ΔCov 点估计 ≤0。

### 2.2 先验构建（不使用图像）

**坐标与离散化**

- 世界坐标（ETH/UCY 原生米制），网格 0.5 m；朝向分 8 个方向 bin。
- 每个 (cell, heading) 存一个**局部坐标系**下 12 帧终点位移的 2D 直方图（例如 16×16，覆盖 ±6 m），外加一个低分辨率中间点（t+6）直方图。
- 支持数 n ≤ 5 的格子用**分层平滑**回退：(cell, heading) → (cell) → (scene, heading) → 全局，Dirichlet 伪计数，伪计数 α 只在 inner 上选一次并冻结。

**两种来源（对应两种泄漏风险模型）**

| 版本 | 构建数据 | 适用 | 泄漏控制 |
| --- | --- | --- | --- |
| P-static | 仅 FIT 窗口的轨迹（含 future） | 内部协议 S2–S3 | 依赖 §2.4 结果：若 C 检查发现 FIT 与 inner/outer 共享 track，则对每个评估窗口**按 track 剔除**（leave-track-out 构建），并额外剔除同场景 ±T 秒内的同 track 片段 |
| P-causal | 仅当前时刻 t 之前**已经结束**的轨迹片段（任何 agent，片段终点帧 ≤ t） | 内部协议 + S4 ETH/UCY 五折（留一场景时测试场景无训练轨迹） | 天然因果；冷启动期（片段数不足）回退全局先验，并单独报告冷启动窗口 |

> P-causal 是主线 A 能进入 S4 五折的唯一途径。P-static 只用于确认“信息是否存在”，不作为最终方法。

**泄漏诊断（必做，写入 S1 测试）**

- T-A1：构建器单测——对任一评估窗口，断言其自身 track 的任何帧都未进入该窗口所用先验。
- T-A2：时间因果单测（P-causal）——断言所用片段终点帧全部 ≤ t。
- T-A3：场景 ID 可识别性——用先验特征做最近质心分类场景/视频 ID，报告准确率（参照 183A 的 0.447）；若远高于运动特征，在结果中标注捷径风险。

### 2.3 A0：零训练信息探针（第一道闸，最便宜）

**目的**：在花任何训练预算之前，确认先验“有新信息”且“对的先验才有用”。

| 比较对象 | 定义 |
| --- | --- |
| M0 motion-only | 常速度外推终点 + 由 FIT 统计的各向同性残差分布（同直方图分辨率） |
| M1 motion + heading 全局 | 只按 heading 统计、不分位置（检验“位置”是否必要） |
| PRIOR | 2.2 的 (cell, heading) 先验，P-static 与 P-causal 各一份 |
| SHUF | 置换对照：同一场景内随机置换 cell 对应关系（保留边缘分布与支持数分布） |
| XSCENE | 跨场景对照：用另一场景同 heading 的先验 |

**指标（inner，按 stable / transition / change 分组报告；只看 change 做闸门）**

1. 终点 NLL（在 GT 终点处的平滑密度）；
2. 角度误差（先验众数方向 vs GT 终点方向）；
3. **救回潜力**：native K20 中不存在完整好候选的窗口里，先验 top-4 终点模式中至少一个距 GT 终点 ≤0.4 m 的比例（只说明上限，不代表可实现）；
4. 冗余度：先验 top-4 模式与 native K20 终点的最小距离分布（若先验模式都已被 native 覆盖，则注入无意义）。

**闸门**：同时满足 2.1 的证伪条件 1–3 均不成立，才进入 A1。A0 不消耗 outer，不消耗 CAL。

### 2.4 A1：注入生成（需训练）

**结构（参数增量 ≤10k，总参数受 G5 约束 ≤305k）**

- **先验 token**：在目标当前位置取以朝向对齐的局部先验补丁（例如 8×8 cell × 8 heading 的计数特征，log 化归一），过一个小卷积 + MLP 得 1–2 个 token，作为 Mode Query cross-attention 的额外 key/value。
- **先验初始化的查询**：K20 中保留原有 learned queries，另有 m 个查询由先验 top-m 终点模式（局部坐标）经线性层初始化，m ∈ {2, 4} 只在 inner 上选。查询仍走原解码器与原损失，不新增 scorer。
- **不改**：F16 编码器、稀疏运动学有向图、RBF 偏置、训练 epochs、优化器、K=20。

**实验臂**

| 臂 | 内容 | 作用 |
| --- | --- | --- |
| CTRL | native F16，同批同 seed 重训 | 基线 |
| A-shuf | 结构与 A-prior 完全相同，但输入置换先验（SHUF） | 同容量、错信息对照；区分“多了参数/多了 token”与“多了信息” |
| A-prior | 结构如上，输入 P-static（S2–S3）；S4 改为 P-causal | 主臂 |

S3（outer）只跑 A-prior 与 A-shuf 两臂（符合 PROTOCOL ≤2 臂），CTRL 为同批对照。

### 2.5 判定

- **PRIMARY**：outer change ΔCov（A-prior − CTRL），按 PROTOCOL track 级两级配对 bootstrap 判 PASS / NON-INFERIOR / INCONCLUSIVE / FAIL。
- **机制归因（必须同时报告，不作为 PASS 条件但决定能否写“场景先验有效”）**：ΔCov（A-prior − A-shuf）的 CI 下界 > 0。若 PRIMARY PASS 但归因不成立，结论只能写“额外容量/查询有效”，主线 A 转为容量问题，不得宣称场景先验有效。
- **护栏**：G1–G5 全部适用；额外报告 stable 组 ΔCov，防止先验把直行样本拉偏。
- **S4 迁移**：P-causal 版本在 ETH/UCY 五折上报告；冷启动窗口单列。

### 2.6 风险与预案

| 风险 | 信号 | 预案 |
| --- | --- | --- |
| track 重叠泄漏 | §2.4 检查发现共享 track；或 A-prior 在 C_seen 显著好于 C_unseen | 强制 leave-track-out；以 C_unseen 结果为准 |
| 场景少（5 场景）且 Univ 密集主导 | 增益集中在单一场景 | 按场景分层报告；S3 结论需至少 2 个场景方向一致 |
| 先验只是“多数派方向”，伤害 change | change 中 GT 偏离先验众数的窗口变差 | 报告按“GT 与先验众数夹角”分桶的 ΔCov |
| 冷启动（P-causal） | 早期帧覆盖低 | 回退全局先验；报告冷启动比例与该子集结果 |

---

## 3. 主线 B：单阶段互补生成（绕开选择器）

### 3.1 假设与可证伪陈述

- **H-B**：native 解码与 KIN 解码的互补性来自“不同运动状态下谁更擅长”，这种分工可以由**每槽位软混合**在训练期学到；配合集合级覆盖损失，一个 K=20 生成器能直接输出覆盖接近 union40 的集合，而 ADE/FDE 不劣化超过 3%。
- **证伪条件（任一成立即停 B）**：
  1. S2（inner）中 B1、B2 的 change ΔCov 点估计都 ≤ +2pp（待 D-0 复核）；
  2. S2 中 B1、B2 都违反 G3（Q-winner ADE/FDE 恶化 >3%）——即重演 cond_boost_11_9 的折中；
  3. 槽位坍缩：≥90% 槽位的混合系数 α 收敛到同一端（全 native 或全 KIN），且 B1 与 B0 的 ΔCov CI 覆盖 0；
  4. B3（固定 10+10）与 B1 在 inner 上无差异（ΔCov 差 CI 覆盖 0 且点估计 ≤0.5pp）——说明软配额没带来超出 168A 的东西。

### 3.2 结构

- **共享**：F16 编码器 + 稀疏运动学有向图 + RBF 偏置 + K20 Mode Query（与 native 一致）。
- **每个槽位 k 两个解码头**：
  - native 头：直接输出位置（与现有一致）；
  - KIN 头：输出加速度/角速度序列，显式积分得到轨迹，速度/加速度软限幅（沿用已验证的 KIN 配方，参数不重新调）。
- **软混合**：$\hat{Y}_k = \alpha_k \hat{Y}^{\text{nat}}_k + (1-\alpha_k)\hat{Y}^{\text{kin}}_k$，$\alpha_k = \sigma(g(q_k, h_{\text{obs}}))$。$g$ 为单层线性，输入为槽位查询与观测编码，**不看未来**。
  - 变体 B1-hard（仅诊断）：推理时 α 取 0/1（选择更好的头）检查是否为混合插值造成的“平均轨迹”。
- **参数**：KIN 头 + 门控增量计入 G5（总参数 ≤305k）；若超限，KIN 头与 native 头共享前两层。

### 3.3 损失

$$
\mathcal{L} = \mathcal{L}_{\text{softWTA}} + \lambda_c\, \mathcal{L}_{\text{cov}} + \lambda_t\, \mathcal{L}_{\text{teach}} + \lambda_e\, \mathcal{L}_{\alpha\text{-ent}}
$$

- $\mathcal{L}_{\text{softWTA}}$：对 20 条按 GT 误差做温度 τ 的 softmin 加权（τ 退火到接近硬 WTA），避免只有一个槽位得到梯度。
- $\mathcal{L}_{\text{cov}}$：直接针对评价定义的平滑覆盖损失。令 $J_k = \max(\text{ADE}_k/0.2,\ \text{FDE}_k/0.4)$，用 $\text{softmin}_k J_k$ 的 hinge：$\max(0,\ \text{softmin}_k J_k - 1 + m)$，m 为小 margin。与 PROTOCOL 的“同一条 ADE≤0.2 且 FDE≤0.4”严格一致，不拼接不同候选。
- $\mathcal{L}_{\text{teach}}$（仅 B2）：离线冻结 native 与 KIN 两个已训模型，在 FIT 上生成 union40；取其中 $J \le 1$ 的“好候选”集合 G（训练期用 FIT 的 GT，合法），以单向 Chamfer 距离让学生集合覆盖 G（每个 G 中元素到学生最近一条的距离）。若 G 为空则该样本不计此项。**inner / outer / CAL 不生成老师集合。**
- $\mathcal{L}_{\alpha\text{-ent}}$：对 batch 内 α 的分布加小的熵正则，防止全部坍缩到一端；λ_e 只在 inner 上选一次。

> 与 172A 的根本区别：172A 是在推理期用 GT-free 几何距离压缩 union，丢失了“哪一条是好的”这一监督信息；B2 是在训练期用 FIT GT 指明“哪些是好的”，再让学生覆盖它们。

### 3.4 实验臂

| 臂 | 内容 | 作用 |
| --- | --- | --- |
| B0 | native 结构，同预算（同 epochs、同 batch）重训，加 $\mathcal{L}_{\text{cov}}$ | 同批对照；同时隔离“覆盖损失本身”的作用 |
| B1 | 双头软混合 + softWTA + cov | 主臂候选 1（无老师） |
| B2 | B1 + teach | 主臂候选 2（集合蒸馏） |
| B3 | 双头但固定 10 native + 10 KIN（α 固定） | 仅 inner；与 EXP-168A 的连接点，检验软配额的增量 |

S2 跑全部四臂（inner，seed 42）；S3 只带 B1/B2 中较优者 + 一个备选（≤2 臂），对照为 B0。**注意：原始 native checkpoint 也要在同批 outer 上报告**，以区分 B0 中覆盖损失的效果。

### 3.5 判定

- **PRIMARY**：outer change ΔCov（B-best − B0），PROTOCOL 两级配对 bootstrap。
- **目标值**（D-0 后复核）：点估计 ≥ +4pp，即追回 union40 增益（+9.4pp）的约 40%；
- **union 保留率**（辅助）：在同一 outer 窗口上，$(\text{Cov}_{B} - \text{Cov}_{B0}) / (\text{Cov}_{\text{union40}} - \text{Cov}_{\text{native}})$，与 172A 的 retention 同口径可比；
- **护栏**：G1–G5 全部适用，G3 是本主线最容易失败的一条，结果表中单独置顶。

### 3.6 诊断（S2 起每次都出）

1. α 直方图（按槽位、按样本）；槽位使用熵；
2. 胜出槽位的“头类型”占比，按速度分桶（复现 168B：高速段 KIN/structured 是否更常胜出）；
3. 有效好候选聚类数（PROTOCOL 去重口径）；
4. B1-hard vs B1：若 hard 明显更好，说明软混合在制造平均轨迹，改为 Gumbel 直通或推理期取 argmax；
5. 按 change / stable 分组的 minADE、minFDE，防止覆盖提升以 stable 退化为代价。

### 3.7 风险与预案

| 风险 | 信号 | 预案 |
| --- | --- | --- |
| 槽位坍缩 | α 集中一端、使用熵低 | 提高 λ_e；或按槽位初始化 α 偏置（一半偏 native、一半偏 KIN）后放开 |
| 覆盖—误差折中 | Cov↑ 但 G3 失败 | 降 λ_c、加 FDE 项权重；若仍失败按证伪条件 2 停 |
| 老师偏差（B2） | B2 只在 C_seen 上好 | 以 C_unseen 为准；老师只用 FIT |
| 训练不稳（KIN 积分） | loss 尖峰、速度越限 | 沿用已验证限幅；梯度裁剪；不新调 KIN 超参 |

---

## 4. 执行计划（对齐 PROTOCOL S0–S4）

| 阶段 | 主线 A | 主线 B | 消耗的数据 |
| --- | --- | --- | --- |
| S0 预注册 | 冻结：网格 0.5 m / 8 heading / 直方图规格、A0 闸门、A-prior/A-shuf 定义 | 冻结：四臂定义、损失项、λ 搜索范围（只在 inner 选一次）、目标值 | 无 |
| S1 冒烟 + 测试 | T1–T8 + T-A1～T-A3；小样本前向/反向 | T1–T8 + 头积分单测、α 范围单测、老师集合只来自 FIT 的单测 | FIT 子集 |
| A0 探针 | 零训练，inner | — | inner |
| S2 inner seed 42 | CTRL / A-shuf / A-prior | B0 / B1 / B2 / B3 | inner |
| S3 outer seeds 42–44 | A-prior、A-shuf（vs CTRL） | B-best + 备选（vs B0） | outer（每配置只评一次） |
| S4 晋级 | P-causal：CAL + seeds 42–46 + ETH/UCY 五折 | 同左 | CAL（每方向最多解封一次） |

**并行与顺序**

- A0 与 B 的 S1 可并行；A0 是零训练，应最先出结果。
- §2.4 track 重叠检查未完成前：B 可以做 S1；A 的 P-static 构建暂停，可先完成 P-causal 构建器与单测（P-causal 不依赖 FIT 轨迹）。
- 两条主线**不在 S3 之前合并**。若两者都在 S3 PASS，再单独预注册 A+B 组合实验，不在本文范围内。

**训练运行数估计（不含失败重跑）**

| 阶段 | A | B |
| --- | --- | --- |
| S2 | 3 臂 × 1 seed = 3 次（+ m 选择 2 次） | 4 臂 × 1 seed = 4 次（+ λ 选择 ≤4 次）+ 老师集合离线生成 1 次 |
| S3 | 2 臂 × 3 seeds = 6 次（CTRL 3 次可与 B 的原生对照共享） | 2 臂 × 3 seeds = 6 次 + B0 3 次 |
| S4 | 1 臂 × 5 seeds + 五折 | 同左 |

---

## 5. 预注册表（S0 填写后冻结）

| 字段 | 主线 A | 主线 B |
| --- | --- | --- |
| 实验 ID | 待分配（建议新系列 SCENE_PRIOR_GEN_SERIES） | 待分配（建议新系列 SINGLE_STAGE_SET_GEN_SERIES） |
| 假设 | H-A | H-B |
| PRIMARY | outer change ΔCov vs CTRL | outer change ΔCov vs B0 |
| 目标值 / MDE | 待 D-0 | +4pp（待 D-0） |
| 护栏 | G1–G5 | G1–G5（G3 置顶） |
| 证伪条件 | 2.1 的 1–4 | 3.1 的 1–4 |
| inner 上可调项 | α（伪计数）、m | λ_c、λ_t、λ_e、τ 退火终点 |
| 禁止调整 | 编码器、epochs、K、评价代码 | 同左 + KIN 限幅参数 |
| 代码基线 | GEMINI_EXP/core/evaluation.py、data_loader.py（SHA 记录） | 同左 |
| 判定方式 | 由协议脚本按数值输出判定，禁止硬写 PASS；人工只复核 | 同左 |

---

## 6. 决策树

1. **A0**：证伪条件 1–3 任一成立 → 关闭主线 A，记录为“轨迹统计先验在 ETH/UCY 内部协议上无新信息”的负结果；否则进入 A1。
2. **S2**：
   - A：A-prior ≤ A-shuf → 停；否则带 A-prior / A-shuf 进 S3。
   - B：证伪条件 1–4 任一成立 → 停；B3 ≈ B1 时只保留结论“软配额无增量”，不进 S3。
3. **S3**：PASS 或 NON-INFERIOR 且护栏全过 → S4；INCONCLUSIVE → 按 PROTOCOL 处理（不得追加 seeds 换结论，只能预注册扩样）；FAIL → 停。
4. **S4**：A 只以 P-causal 版本晋级；B 以 B-best 晋级。两者都 PASS 时再预注册 A+B 组合。

---

## 7. 待确认问题

1. B 的目标值 +4pp 是否接受，还是以“追回 union40 增益的 50%”为准（约 +4.7pp）？
2. A 是否允许 S2 同时跑 P-static 与 P-causal（多 1 臂），还是严格先 P-static 确认信息存在？
3. B2 的老师集合需要 native 与 KIN 两个已冻结 checkpoint：是否沿用 10 月锁定的互补池版本（需确认其训练数据确为 FIT，而非包含 inner）？
4. 参数预算 G5（≤305k）下，当前 native 的实际参数量是多少？这决定 A 的先验编码器与 B 的 KIN 头是否需要共享层。
