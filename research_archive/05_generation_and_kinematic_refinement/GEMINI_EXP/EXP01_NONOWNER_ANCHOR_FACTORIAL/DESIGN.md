# EXP01: 非负责候选几何锚定（Non-Owner Anchor）析因实验设计

日期：2026-10-18
状态：PREFLIGHT_PASSED -> RUNNING

---

## 1. 科学背景与核心问题

在先前的生成端细化实验（如 T1, T2 系列）中，模型在训练集（FIT change）上能够取得显著拟合（覆盖率提升 +5~6 百分点），但在外层测试集（outer change）上泛化收益完全停滞（原生 110/253，细化后仅 107~113/253）。

深入诊断表明，其根本障碍并非模型没有救回能力（每次平均救回约 15 个失败窗口），而是：
**模型在救回的同时，摧毁了几乎等量甚至更多的原生已达标窗口（Lost 达到 15~19 个，丢失率高达 13.6%~17.3%）。**

由于所有 20 个候选槽位（slots）通过同一个共享 MLP 进行细化，仅由赢家槽位（owner slot）监督 GT，导致非赢家槽位受到不可控的参数漂移影响，进而破坏了原本优良的原生模式。

---

## 2. 实验对照设计

本实验采用严格配对的两臂对照（Control vs Treatment），在完全相同的特征、网络架构、初始化、数据洗牌和优化预算下进行对比：

| 臂标识 | 责任分配规则（Owner） | 非负责候选锚定（Non-Owner Anchor） | 目标损失定义 |
|---|---|---|---|
| **Arm A (Control)** | 动态最优：$a = \arg\min_j Q(p_j, y)$ | 关闭（$\lambda = 0$） | $L = L_{\text{task}} + 0.05 L_{\text{div}}$ |
| **Arm C (Treatment)** | 动态最优：$a = \arg\min_j Q(p_j, y)$ | 开启（$\lambda = 1.0$） | $L = L_{\text{task}} + 0.05 L_{\text{div}} + 1.0 L_{\text{anchor}}$ |

### 目标定义：
- 候选质量：$Q(p_j, y) = \text{ADE}(p_j, y) + 0.5 \text{FDE}(p_j, y)$
- 任务损失：$L_{\text{task}} = \frac{1}{B} \sum_{i=1}^B Q(p_{i, a[i]}, y_i)$
- 锚定损失（仅对非 owner 施加）：
  $$d_{\text{native}}(p_{i,j}, b_{i,j}) = \text{mean}_t \|p_{i,j,t} - b_{i,j,t}\|_2 + 0.5 \|p_{i,j,12} - b_{i,j,12}\|_2$$
  $$L_{\text{anchor}} = \frac{1}{19 B} \sum_{i=1}^B \sum_{j \ne a[i]} d_{\text{native}}(p_{i,j}, b_{i,j})$$
- 多样性损失：
  $$L_{\text{div}} = \frac{1}{190 B} \sum_{i=1}^B \sum_{j < k} \text{relu}(0.30 - \text{mean}_t \|p_{i,j,t} - p_{i,k,t}\|_2)^2$$

---

## 3. 科学假设与证伪准则

### 假设（HYPOTHESIS）：
非负责候选的几何锚定损失能够有效约束 19 个非监督槽位的自由漂移，显著压制原生好窗口的破坏（Lost 显著下降），同时允许负责槽位继续向真实未来修正，从而实现 **净覆盖率（Coverage = Native + Rescue - Lost）在 outer change 上的显著提升**。

### 证伪准则（FALSIFICATION）：
若观察到以下任一现象，则假设被证伪：
1. **救回连带抑制**：锚定损失在抑制 Lost 的同时，等比例甚至更大幅度地抑制了 Rescue，导致净增益并未改善或甚至倒退；
2. **迁移无增益**：C 与 A 在 outer change 上的覆盖率差异 C - A 的 95% 置信区间跨零；
3. **退化为恒等映射**：C 的位移修正范数趋近于零，模型失去任何改写能力。

---

## 4. 评价指标与重大成功门

- **完整好候选判定**：同一候选必须同时满足 $\text{ADE} \le 0.20\,\text{m}$ 且 $\text{FDE} \le 0.40\,\text{m}$。
- **Primary 评价人群**：`outer change`（253 窗口，68 tracks）。
- **运动学要求**：五项指标（speed p95, accel mean/p95, jerk mean/p95）必须严格 $\le 1.01 \times \text{native}$。
- **随机种子**：42, 43, 44，每 seed 独立训练 25 个 epoch，batch_size=256，AdamW (lr=1e-3, wd=1e-4)。
