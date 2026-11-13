# D2: 运动预测预训练表示（Masked Reconstruction vs Causal Motion Forecasting Pretraining）实验设计

日期：2026-10-20
状态：PREFLIGHT_PENDING -> PRODUCTION_LAUNCH
标准遵循：`RULE[user_global]`, `MOTION_UNDERSTANDING_THREE_BRANCHES_DESIGN_20261019/00_共同数据评价与实施合同.md`, `02_D2_运动预测预训练表示.md`

---

## 1. 科学问题与研究背景

在先前的一系列研究（M1 动态历史、I1 显式速度/加速度读出）中，直接将观测速度/加速度拼接入多模态生成网络并未建立可靠的额外迁移收益（CI 均跨零，均值 MSE 下降并未转化为候选覆盖率改善）。

D2 提出根本性机制转变：
> **与其在生成阶段直接拼接局部物理量，不如先在历史运动片段上执行自监督预训练任务，提取表征运动内在连续性与因果趋势的高维特征，再将其作为条件输入给生成器。**
> 进一步，**插值重建（Interpolation Infilling）与因果运动预测（Causal Forecasting）这两种自监督任务，哪一种能为下游多候选轨迹生成提供更具备泛化能力的条件表示？**

---

## 2. 严格配对的两臂实验设计 (Control vs Treatment)

两个实验臂具有完全相同的两阶段架构、完全相同的参数量、完全相同的两阶段优化预算与随机种子：

| 实验阶段 | Arm A (Interpolation Control) | Arm B (Causal Forecast Treatment) | 对照控制项 |
|---|---|---|---|
| **第一阶段：自监督预训练** | 可见点: 1, 2, 3, 4, 7, 8<br>隐藏预测: **5, 6** (缺失插值重建) | 可见点: 1, 2, 3, 4, 5, 6<br>隐藏预测: **7, 8** (因果前向预测) | 相同 2 层 TransformerEncoder，同 4 维输入，同 20 epoch，同仅用 FIT 8步历史 |
| **中间衔接：特征提取** | 冻结 Encoder，送入完整 8 步，提取第 8 步 $h_{64}$ | 冻结 Encoder，送入完整 8 步，提取第 8 步 $h_{64}$ | 统一固定取出第 8 个 token，不加任何可训练 pooling 混杂 |
| **第二阶段：下游生成训练** | 输入 $[P_{2, 96}, h_{64}]$ (160 维)<br>同 25 epoch，AdamW | 输入 $[P_{2, 96}, h_{64}]$ (160 维)<br>同 25 epoch，AdamW | 相同 40,328 参数生成器，相同初始几何基线，相同损失函数 |

### 共同固定合同与零泄漏准则：
1. **预训练坐标系与零泄漏**：
   - 每段 8 步轨迹以**第 1 个观测点为局部原点**：$p_{\text{rel}}[t] = \text{past}[t] - \text{past}[0]$。
   - 严禁用可能被遮挡的第 8 个点居中，严禁按全窗口末速度旋转坐标轴。
   - 每个 token 输入 4 维：$[x_{\text{rel}}, y_{\text{rel}}, \text{mask}, t/8]$。
   - 隐藏 token 坐标置 $(0, 0)$，mask 置 $0$。
   - **预训练严格只使用 FIT 18,909 样本**，严禁使用 outer/inner/CAL。
2. **预训练损失**：
   - 2 个隐藏点的平均欧氏距离（米）：$L_{\text{pre}} = \frac{1}{2 B} \sum_{i=1}^B \sum_{k=1}^2 \|\hat{y}_{i, k} - y_{i, k}\|_2$。
3. **下游多模态生成网络**：
   - 编码器全部 67,264 参数及所有 buffers 严格冻结（`requires_grad = False`）。
   - 上下文编码：$160 \to 128 \to 64$ GELU。
   - Query：$20 \times 16$，$\mathcal{N}(0, 0.02)$ 配对初始化。
   - 目标头：$80 \to 64 \to 2$ GELU，末层零初始化。
   - 过程头：$84 \to 64 \to 6$ GELU，末层零初始化。
   - 下游损失：$L = L_{\text{path}} + 0.5 \cdot L_{\text{goal}} + 0.05 \cdot L_{\text{div}}$。

---

## 3. 科学假设与证伪准则 (Hypothesis & Falsification)

### 科学假设 (HYPOTHESIS)：
相比于无方向性的缺失插值重建（Arm A），从较早历史预测未来走势的因果前向自监督任务（Arm B）能够促使 Transformer 编码器提取包含速度阻尼、意图延伸与曲率变化的动力学特征，在冻结后能为下游轨迹生成器提供更有效的初始引导，在 `outer change` 测试集上获得超越 Arm A 的覆盖率净增益，并降低原生好窗口的丢失率。

### 预注册证伪准则 (FALSIFICATION)：
若观察到以下任一现象，则假设被证伪：
1. **预训练任务与下游生成脱节**：Arm B 在预训练阶段损失大幅下降，但在下游生成中与 Arm A 相比无显著收益（`outer change` 完整好候选覆盖率 Bootstrap 95% 置信区间跨零，p > 0.05）；
2. **特征退化或冗余**：生成的候选轨迹对 $h_{64}$ 不敏感（置零消融前后路径漂移 $< 0.005$m），表明生成器主要依赖 $P_2$，预训练特征被下游网络完全旁路；
3. **运动学生理恶化**：五项运动学指标中任意一项超过 $1.01 \times \text{native}$。

---

## 4. 评价标准与交付清单

1. **评价人群**：
   - Primary 目标：`outer change`（253 个窗口，68 个目标 tracks）。
   - 全人群验证：`FIT` (18,909), `outer` (1,035), `inner` (1,856), `original_CAL` (2,879)。
   - 细分组别：`overall`, `stable`, `transition`, `change`。
2. **种子设置**：
   - 3 个随机种子：`42`, `43`, `44`。
   - 预训练 6 个模型（2 臂 $\times$ 3 种子 $\times$ 20 epoch） + 生成 6 个模型（2 臂 $\times$ 3 种子 $\times$ 25 epoch）。
3. **统计检验**：
   - 20,000 次配对目标轨迹抽样（Clustered Paired Bootstrap, PCG64 seed=20261019）。
4. **必存专项诊断**：
   - FIT 与 Outer 上的预训练自监督误差（衡量表征学习本身的跨域泛化能力）。
   - 特征消融响应：置零 $h_{64} = 0$ 时的路径漂移。
   - 特征错配响应：跨 track 乱序 shuffle $h_{64}$ 时的生成扰动。
