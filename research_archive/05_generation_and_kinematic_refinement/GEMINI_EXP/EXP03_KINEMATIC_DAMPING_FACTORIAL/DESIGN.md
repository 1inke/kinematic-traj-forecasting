# EXP03: 运动学阻尼与控制平滑正则化（Kinematic Damping & Smoothness Regularization）实验设计

日期：2026-10-20
状态：PREFLIGHT_PENDING -> PRODUCTION_LAUNCH
标准遵循：`RULE[user_global]`, `GEMINI.md`

---

## 1. 科学问题与现象根因剖析

在本项目前述的所有生成与细化实验中，始终存在两大长期悬而未决的顽疾：
1. **运动学严重超标 (Kinematics Inflation)**：
   每次微调后，平均加速度均从原生的 $0.0837\,\text{m/s}^2$ 剧烈攀升至 $0.094\sim0.107\,\text{m/s}^2$（增幅达到 +12% ~ +28%），全部未能通过五项运动学生理检验（严格 $\le 1.01 \times \text{native}$）。
2. **好窗口过度破坏 (Collateral Destruction of Good Modes)**：
   由于控制信号 $u \in \mathbb{R}^{B \times 20 \times 12 \times 2}$ 未受能量约束，网络自由采用极值加速度或高频振颤来拉取胜者，导致所有 20 个槽位均产生高频几何抖动，把原本已经达标的优良路径震出阈值之外（Lost 达到 15~35 个）。

因此，本实验提出核心科学问题：
> **引入显式的控制能量阻尼与二阶平滑惩罚，能否在不扼杀有效救回（Rescue）的前提下，从物理机制上彻底消除高频抖动，将运动学严格控制在原生 $1.01\times$ 之内，并大幅遏制好窗口丢失？**

---

## 2. 严格单因素配对对照设计 (Control vs Treatment)

两臂采用完全相同的 120 维输入（96 维上下文 + 24 维原生候选）、完全相同的 25,304 参数网络、完全相同的零初始化、相同数据与优化配置：

| 实验臂 | 损失函数定义 | 物理机制 |
|---|---|---|
| **Arm A (Control)** | $L = L_{\text{task}} + 0.05 L_{\text{div}}$ | 无控制能量与平滑惩罚的标准基线 |
| **Arm B (Treatment)** | $L = L_{\text{task}} + 0.05 L_{\text{div}} + 0.05 L_{\text{ctrl}} + 0.10 L_{\text{smooth}}$ | 引入控制力阻尼与相邻时间差分平滑约束 |

### 正则化项数学定义：
- 控制能量损失：
  $$L_{\text{ctrl}} = \frac{1}{B \times 20 \times 12} \sum_{b=1}^B \sum_{k=1}^{20} \sum_{t=1}^{12} \|u[b, k, t]\|_2^2$$
- 二阶控制平滑损失（相邻时步控制率变化）：
  $$L_{\text{smooth}} = \frac{1}{B \times 20 \times 11} \sum_{b=1}^B \sum_{k=1}^{20} \sum_{t=2}^{12} \|u[b, k, t] - u[b, k, t-1]\|_2^2$$

---

## 3. 科学假设与证伪准则 (Hypothesis & Falsification)

### 科学假设 (HYPOTHESIS)：
通过对加速度控制量施加二次能量阻尼与一阶差分平滑约束，能够有效滤除高阶高频扰动，使生成的轨迹严格恢复生理平滑（五项运动学通过率达到 100%），同时由于消除了不必要的高频剧烈抖动，原生好窗口的破坏数将显著下降（Lost 明显减少），从而在 `outer change` 测试集上获得正向净收益。

### 预注册证伪准则 (FALSIFICATION)：
1. **运动学依然超标**：Arm B 的加速度均值依然超过 $1.01 \times \text{native}$；
2. **救回连带冻结**：控制阻尼使得模型位移修正幅度衰减至几乎为 0，导致 Rescue 率降低至 0；
3. **净覆盖率无改善**：配对 Bootstrap 检验中，B - A 的覆盖率置信区间跨零且均值无提升。

---

## 4. 实验规模与评价规范

- 3 随机种子：`42`, `43`, `44`。
- 每模型 25 epoch，batch 256，AdamW (lr=1e-3, wd=1e-4)。
- Primary 目标：`outer change` (253 窗口，68 tracks)。
- 四人群全量评测：FIT, outer, inner, CAL。
- 20,000 次 Clustered Paired Bootstrap 检验。
