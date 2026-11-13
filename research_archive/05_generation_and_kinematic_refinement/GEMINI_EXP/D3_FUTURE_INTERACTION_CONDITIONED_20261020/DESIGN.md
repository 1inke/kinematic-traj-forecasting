# D3: 未来交互条件生成与两轮路径协调（Future Interaction Conditioned Generation）实验设计

日期：2026-10-20
状态：PREFLIGHT_PENDING -> PRODUCTION_LAUNCH
标准遵循：`RULE[user_global]`, `GEMINI.md`, `MOTION_UNDERSTANDING_THREE_BRANCHES_DESIGN_20261019`

---

## 1. 科学问题与现象根因剖析

在前序完成的四大实验中，无论是多时步计划监督（D1）、历史自监督预训练（D2），还是非负责软锚定（EXP01）与运动学阻尼（EXP03），其共同特征均为**单智能体自洽建模**：网络仅基于目标行人自身的过去 8 步观测 $p2$ 对候选轨迹进行修正。
然而在主测试集 `outer change`（253 个经历剧烈航向或速度改变的窗口）中，行人发生转弯、减速或避让的核心物理根因往往并非自身的内部动力学习惯，而是**周围其他行人的空间占用与未来运动意图**。

若模型只编码过去静态邻居（$t \le 0$），无法感知自身候选轨迹在未来（$t = 1 \dots 12$）是否会与迎面走来的邻居发生空间碰撞或路径重叠。因此，本实验正式提出 D3 核心科学问题：
> **在同一候选世界中，将目标行人的未来预测路径与周围邻居的未来外推路径进行显式时空交互对齐（两轮消息更新），能否打破单智能体泛化瓶颈，在 `outer change` 上显著提升完整好候选覆盖率（Rescue 净收益）？**

---

## 2. 严格配对对照设计 (Control vs Treatment)

两臂采用完全相同的输入管线、完全相同的过去 8 步邻居规范化坐标、相同优化配置与随机种子：

| 实验臂 | 输入与表示 | 架构机制 | 阶段交互 |
|---|---|---|---|
| **Arm A (Control)** | $p2 \in \mathbb{R}^{96}$ + $s_{\text{past}} \in \mathbb{R}^{32}$ + $\text{native}_k \in \mathbb{R}^{24}$ | 单轮多层前馈网络，仅基于过去邻居上下文生成候选 | 无未来空间交互反馈 |
| **Arm B (Treatment)** | $p2 \in \mathbb{R}^{96}$ + $s_{\text{past}} \in \mathbb{R}^{32}$ + $\text{native}_k \in \mathbb{R}^{24}$ | **两轮消息更新机制**：第一轮生成初选路径 $P^{(0)}$；计算其与邻居未来外推路径的时空冲突距离 $m_k$；第二轮条件化细化输出 $P^{(1)}$ | 显式未来路径对齐反馈 |

### 核心模块数学定义：
1. **邻居过去上下文编码（A/B 共享，严格零泄漏）**：
   - 提取 8 个邻居的规范化过去轨迹与有效掩码 $M \in \{0, 1\}^8$。
   - 提取每个邻居的当前位置 $\mathbf{p}_j(0)$ 与速度 $\mathbf{v}_j(0)$。
   - 过去邻居特征经线性投影与有效均值池化，输出 $s_{\text{past}} \in \mathbb{R}^{32}$。
2. **邻居未来无标签外推（Past-only extrapolation）**：
   - 基于过去无泄漏观测，邻居未来路径按等速模型外推：
     $$\hat{\mathbf{y}}_j(t) = \mathbf{p}_j(0) + (t \times 0.4) \cdot \mathbf{v}_j(0), \quad t=1 \dots 12$$
   - 绝不使用任何邻居的真实未来 GT（避免未来信息泄露）。
3. **未来时空冲突特征计算 (Arm B Future Interaction Message)**：
   - 对每个候选轨迹 $p_k^{(0)} \in \mathbb{R}^{12 \times 2}$，计算其与所有有效邻居未来路径的欧氏距离：
     $$d_{k, j}(t) = \|p_k^{(0)}(t) - \hat{\mathbf{y}}_j(t)\|_2$$
   - 计算高斯碰撞冲突强度：
     $$c_{k, j}(t) = \exp(-d_{k, j}(t)^2 / (2 \sigma^2)), \quad \sigma = 1.0\,\text{m}$$
   - 聚合冲突消息 $m_k \in \mathbb{R}^{32}$：输入包括最小空间间距、最大时空冲突强度及跨时步平均冲突度。
4. **两轮路径更新 (Two-Round Refinement)**：
   - 第一轮生成：$P^{(0)} = \text{Refiner}_1([p2, s_{\text{past}}, \text{native}])$
   - 第二轮更新：$P^{(1)} = P^{(0)} + \text{Refiner}_2([z^{(0)}, m_k])$，其中末层零初始化。

---

## 3. 科学假设与预注册证伪准则 (Hypothesis & Falsification)

### 科学假设 (HYPOTHESIS)：
通过对候选路径与周围邻居未来潜在轨迹进行两轮冲突对齐，模型能够动态识别并压制穿入邻居空间的劣质候选，同时拉取绕开冲突的安全达标候选，在 `outer change` 数据集上实现 Rescue 大于 Lost 的显著正向净收益（Bootstrap 95% CI 完全大于零）。

### 预注册证伪准则 (FALSIFICATION)：
1. **交互增益失效**：Arm B 相对于 Arm A 的覆盖率差值置信区间跨零，或 $p > 0.05$；
2. **运动学生理超标**：两轮修正累加导致加速度超过 $1.01 \times \text{native}$；
3. **好模式破坏反弹**：第二轮修正打乱了原生好候选，导致 Lost 窗口增加超过 15 个。

---

## 4. 实验规模与评价规范

- 3 独立随机种子：`42`, `43`, `44`。
- 训练轮数：每臂 25 epoch，batch 256，AdamW (lr=1e-3, wd=1e-4)。
- Primary 目标：`outer change` (253 窗口，68 tracks)。
- 四人群全量评测：FIT (18909), outer (1035), inner (1856), CAL (2879)。
- 20,000 次 Clustered Paired Bootstrap 检验。
