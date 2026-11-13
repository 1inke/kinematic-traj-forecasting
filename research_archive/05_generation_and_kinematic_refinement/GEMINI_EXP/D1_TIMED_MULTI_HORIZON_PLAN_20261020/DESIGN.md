# D1: 多时间点计划监督（Timed Multi-Horizon Plan Supervision）实验设计

日期：2026-10-20
状态：PREFLIGHT_PENDING -> PRODUCTION_LAUNCH
标准遵循：`RULE[user_global]`, `MOTION_UNDERSTANDING_THREE_BRANCHES_DESIGN_20261019/00_共同数据评价与实施合同.md`, `01_D1_多时间点计划监督.md`

---

## 1. 科学问题与研究背景

在之前的长终点实验（H1）中，长目标引导（$z_{20}$）在部分窗口上展现出宏观趋势引导作用，但短期轨迹过程缺乏中间约束，导致多模态生成在 `outer change` 困难分叉场景下出现轨迹松散和好窗口破坏（Lost）。

D1 提出核心问题：
> **在不改变物理约束阶数、不增加历史特征通道的前提下，显式的多时间进度监督（4/8/12/20步联合计划）能否提供更紧凑的过程约束，显著改善短期路径预测质量，同时抑制好窗口丢失？**

---

## 2. 严格配对的两臂对照设计 (Control vs Treatment)

为了严格隔离“多时间点计划监督”这单一科学因素，本实验设立完全同构的两臂对照：

| 实验臂 | 角色 | 四个计划变量的监督 Target（均折算为 12 步等效米） | 科学意义 |
|---|---|---|---|
| **Arm A (FarRepeated)** | **匹配对照 (Control)** | $[0.6 \cdot y_{20}, 0.6 \cdot y_{20}, 0.6 \cdot y_{20}, 0.6 \cdot y_{20}]$ | 固定网络容量、相同 4 点计划头、相同辅助损失权重，但仅提供单一远目标信息 |
| **Arm B (TimedPlan)** | **治疗组 (Treatment)** | $[3.0 \cdot y_4, 1.5 \cdot y_8, 1.0 \cdot y_{12}, 0.6 \cdot y_{20}]$ | 引入 4、8、12、20 步的显式时序推进计划监督 |

### 共同固定合同：
1. **输入与上下文**：
   - 上下文：$p_2 \in \mathbb{R}^{96}$（来自冻结编码器），末速度 $v_0 \in \mathbb{R}^2$。
   - 20 个候选 query $\in \mathbb{R}^{20 \times 16}$，按 $\mathcal{N}(0, 0.02^2)$ 配对初始化。
2. **网络结构**：
   - `context`: $\text{Linear}(96, 128) \to \text{GELU} \to \text{Linear}(128, 64) \to \text{GELU}$ (20,672 参数)
   - `query`: 20 $\times$ 16 (320 参数)
   - `plan_head`: $\text{Linear}(80, 64) \to \text{GELU} \to \text{Linear}(64, 8)$ (5,704 参数)，末层权重偏置零初始化。
   - `process_head`: $\text{Linear}(90, 64) \to \text{GELU} \to \text{Linear}(64, 6)$ (6,214 参数)，末层权重偏置零初始化。
   - 总可训练参数：**32,910** 个，两臂完全一致。
3. **初始几何基线**：
   - 4 个计划变量均初始化为 $4.8 \cdot v_0 + \text{centers12}_k$。
   - 过程系数 $a, b, c$ 初始为 0。
   - 轨迹算子：$p(s) = 4.8 \cdot s \cdot v_0 + s^2 (z_4 - 4.8 \cdot v_0 + a) + s^2 (1 - s) b + s^2 (1 - s) (2s - 1) c$，其中 $s = t/12$，$z_4 = \text{plan}[:, :, 3]$。
   - 在未训练时，两臂初始轨迹均严格退化为 $4.8 \cdot s \cdot v_0 + s^2 \text{centers12}_k$。
4. **损失函数**：
   - $j = \arg\min_k (\text{ADE}_k + 0.5 \text{FDE}_k)$（基于预测 12 步路径的真实最优候选）。
   - $L_{\text{path}} = \frac{1}{B} \sum_{i=1}^B Q(p_{i, j}, y_i)$
   - $L_{\text{plan}} = \frac{1}{B} \sum_{i=1}^B m_{20}[i] \cdot \frac{1}{4} \sum_{r=0}^3 \|\text{plan}[i, j, r] - \text{target}[i, r]\|_2$
   - $L_{\text{div}} = \frac{1}{190 B} \sum_{i=1}^B \sum_{j < k} \text{relu}(0.30 - \text{dist}(p_j, p_k))^2$
   - $L = L_{\text{path}} + 0.5 \cdot L_{\text{plan}} + 0.05 \cdot L_{\text{div}}$

---

## 3. 科学假设与证伪准则 (Hypothesis & Falsification)

### 科学假设 (HYPOTHESIS)：
多时间点计划监督（Arm B）通过显式施加 4/8/12 步的阶段性推进约束，能够有效防止轨迹在弯曲和突变阶段的过度松弛，在保持远端引导的同时显著改善短期路径预测质量，在 `outer change` 测试集上获得超越 Arm A 的覆盖率净增益，并降低原生好窗口的丢失率。

### 预注册证伪准则 (FALSIFICATION)：
若出现以下任一情况，则假设被证伪：
1. **多目标梯度冲突**：4/8/12 步的中间监督与 12 步路径总体误差损失发生负面冲突，导致 Arm B 在 `outer change` 上的完整好候选覆盖率低于或不显著优于 Arm A（配对 Bootstrap 95% 置信区间跨零）；
2. **计划与路径脱节**：Arm B 的计划点误差 $L_{\text{plan}}$ 显著下降，但实际生成的连续路径并没有贴近计划点（中间路径与计划点残差 $\Delta_r$ 过大），表明多点监督未能有效传递给过程生成器；
3. **运动学超标**：Arm B 生成轨迹的五项运动学指标中任意一项超过 $1.01 \times \text{native}$。

---

## 4. 评价标准与交付清单

1. **评价人群**：
   - Primary 目标：`outer change`（253 个窗口，68 个目标 tracks）。
   - 全人群验证：`FIT` (18,909), `outer` (1,035), `inner` (1,856), `original_CAL` (2,879)。
   - 细分组别：`overall`, `stable`, `transition`, `change`。
2. **种子设置**：
   - 3 个随机种子：`42`, `43`, `44`。
   - 每臂 3 种子 $\times$ 25 epoch，共 6 个完整生成模型训练。
3. **统计检验**：
   - 20,000 次配对目标轨迹抽样（Clustered Paired Bootstrap, PCG64 seed=20261019）。
   - 报告观测差异均值、95% 置信区间与 p 值。
4. **必存诊断**：
   - 4 个计划点预测误差（4, 8, 12, 20步）。
   - 路径与计划点的一致性偏差 $\Delta_r$。
   - 计划变量冻结消融诊断（将 B 的前三计划点强制替换为 $z_4$，观察路径响应）。
