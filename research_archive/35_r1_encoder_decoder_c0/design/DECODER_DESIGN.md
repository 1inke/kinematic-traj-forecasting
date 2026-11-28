# 逐层差异化模式解码：级联分配与覆盖序判别（设计思路与方案）

Nov 26, 2026 · @akaza

## 1. 概述

本方案把同构堆叠的多层模式解码器改成三种分工不同的层：撒网层（宽松分配、保覆盖）、收网层（逐层收紧分配、提精度）、一层轻量判别层（按边际覆盖效用排队、去重并输出置信度）。推理输出沿用判别层的覆盖序，一个顺序同时给出前 1、3、5、20 条。

**要解决的三个问题（均来自仓库已完成实验）：**

- 固定 0.5 m 的 EMTA 匹配阈值下，ETH 观测速度大于 1.3 m/s 的窗口匹配率只有 3.6%，快速行人几乎拿不到正样本。
- 多层解码在不带 EMTA 时，Top-1 FDE 比 B0 差 8.4%–10.8%；候选越细化越聚集，置信度分不开。
- 层与层之间作用本来不同（第一层移动最大，末层在高速后减速窗口变差 28%–52%），同构堆叠让每层做同一件事。

**定位：** 研究一（纯轨迹、固定推理预算）的解码端拟创新点。三个部件分别借鉴 Cascade R-CNN、MDS-DETR/DAC-DETR 与设施选址（k-median），本文的不同在于把它们组合到行人模式解码中，并用边际覆盖效用定义判别层中的先后与冗余。

**状态：** 设计阶段。第 2 节的现象和数字已在仓库核实；第 4 节的方法尚未训练验证，验证计划见第 6 节。

## 2. 设计思路与想法来源

三个部件各自回应仓库里的一组现象，每个部件的做法都有一个外部来源。先有现象，再找来源，最后做适配。

### 2.1 出发点：仓库实验中的现象

表中数字均为 C0 开发集或五折结果，正值表示误差变大。

| 编号 | 现象 | 数据 | 来源 |
| --- | --- | --- | --- |
| F1 | 固定阈值让快速行人拿不到正样本 | 0.5 m 时匹配率：C0 整体 0.667，ETH 整体 0.294，ETH 且 v>1.3 m/s 仅 0.036；1.0 m 时分别为 0.941、0.618、0.259（只查 t6、t12 两步，是真实匹配率的上界） | archive 27 `DESIGN_QRANK.md` 附录 A.4 |
| F2 | 多层堆叠在 WTA 训练下伤 Top-1 | 不带 EMTA 的多层臂 Top-1 FDE +8.4% 至 +10.8%；加 EMTA 后 −0.9% 至 −1.4%；minFDE 均为 −5.3% 至 −5.9% | archive 27 `results/stage1/runs` |
| F3 | 排序依据与顺序依赖不影响落点 | 置信度、风险、质量、随机排序与全注意力 D3-full 的 minFDE 全在 0.2967–0.2986 m | 同上 |
| F4 | 各层作用不同 | 第 0 层到末层：匀速窗口五折都改善 10.8%–18.9%；高速后减速窗口 ETH 变差 28.4%（QFL）/ 51.5%（E0），ZARA01 变差 39.9% / 43.9% | archive 29 `REPORT.md`；archive 28 `README.md` |
| F5 | 跨层候选互补，但无真值时难以利用 | 初层加末层 40 条的上限 minFDE 低 16.5%–23.7%（候选数翻倍，含数量效应）；按覆盖选 16+4 条为 −1.3% 至 +2.4% | archive 30 `layer_pool/results/REPORT.md` |
| F6 | 输出端覆盖排序稳定有效 | B0（C0）@3 −3.5%、@5 −4.9%；B0（ETH）@3 −13.5%、@5 −16.3%；@1 在 ±2% 内；54 个模型中 @3 全部改善 | archive 27 `dev_metrics.npz` 中 `riskp_*` 与 `conf_*` |
| F7 | 置信度越差，覆盖排序补得越多 | 多层不加 EMTA 时 @3 约 −11%，加 EMTA 后约 −6%，B0 为 −3.5% | 同上 |
| F8 | 用 MBR 选 Top-1 会被拉向簇中心 | Weighted-MBR 使 Top-1 变差 0.83% | archive 04 EXP-186A |

### 2.2 想法来源：每个部件借鉴了什么

**级联分配（撒网到收网）← Cascade R-CNN。** 检测中强行提高 IoU 阈值会让正样本指数级减少，它改用逐级递增的阈值，每一级用上一级的输出训练。F1 正是轨迹版的“正样本消失”，所以把 EMTA 的单一距离阈值改为逐层收紧。

**判别层（职责分离）← MDS-DETR、DAC-DETR、EDA。** MDS-DETR 用若干一对多层加一层一对一去重层，去重层不含交叉注意力，按置信度排序后用因果掩码；理由是解码器自注意力对排列等变，去重需要不对称。DAC-DETR 发现交叉注意力把查询聚向目标、自注意力把它们推开，两者作用相反。EDA 报告锚点更新越多、预测越聚集、打分越差。这三点对应 F2、F3：落点交给细化层，判别单独一层，顺序只在判别层使用。

**覆盖序（带权 k-median）← 设施选址、MBR、Metric-agnostic、MBRM、Wayformer 聚合。** 这些工作都在模型外部做后处理，要么固定 K，要么输出不排序。F6、F7 说明覆盖序在输出端有效，F8 说明第一位不能交给 medoid。本方案把覆盖序当作判别层的排队规则和冗余定义，并保留嵌套前缀。

**EMTA ← ModeSeq。** ModeSeq 用模式顺序定义最早匹配的正样本。本方案保留 EMTA，但只用在判别层，顺序改为覆盖序。

**细化层数受控 ← SmartRefine。** 它观察到过度细化会把预测推离真值，与 F4 一致；因此判别层不再移动几何。

### 2.3 与已有工作的区别

| 已有工作 | 它们的做法 | 本方案 |
| --- | --- | --- |
| [ModeSeq](https://arxiv.org/abs/2411.11911) / [Mode-as-Sequence](https://arxiv.org/abs/2605.24037)（轨迹） | 6 层同构，每层按置信度重排，推理仍带因果掩码；未比较排序依据，DETR 基线只在 WTA 下 | 各层分工不同；只有判别层有顺序；顺序为覆盖序 |
| [EDA](https://arxiv.org/abs/2312.09501)（轨迹） | 各层分配规则相同，只换锚点来源；训练时按置信度 NMS 后再匹配 | 各层匹配阈值不同（由宽到严） |
| [Cascade R-CNN](https://arxiv.org/abs/1712.00726)（检测） | 逐级提高 IoU 阈值 | 搬到模式解码：逐层收紧线性距离阈值，动机是 F1 |
| [MDS-DETR](https://arxiv.org/abs/2605.23507)（检测） | 末层按最大类别概率排序去重 | 按边际覆盖效用排序，冗余由与前序候选的终点距离定义，直接对齐 minFDE@k |
| [Metric-agnostic](https://arxiv.org/abs/2607.01133) / [MBRM](https://arxiv.org/abs/2409.10585)（轨迹） | 模型外后处理；固定 K=6 或输出不排序；连续重优化终点 | 放进模型判别层；从已有候选中选；第一位由置信度决定，其余按覆盖序；嵌套前缀 |

在轨迹预测中暂未检索到三者的组合，但只做过一次针对性扩展检索，写入开题前需再查两次（见第 7 节）。

## 3. 符号与基础定义

所有定义沿用现有 QRANK 代码的口径，只有阈值序列 θℓ 和覆盖序的用法是新的。

**候选与锚点。** 每个窗口有 K = 20 条候选，预测 T = 12 步，步长 Δt = 0.4 s。候选 k 由 3 个锚点 Aₖ ∈ ℝ^(3×2) 经固定插值矩阵 W ∈ ℝ^(12×3) 得到，p₀ 为最后一个观测位置：

```latex
Y_k(t) = p_0 + \sum_{i=1}^{3} W_{t,i}\, A_{k,i}, \qquad t = 1,\dots,12
```

层编号 ℓ = 0, 1, …, L\_g 为几何层（ℓ = 0 是 B0 的 H3 提议），D 为判别层。第 ℓ 层的候选记为 Yₖ^(ℓ)，终点 eₖ^(ℓ) = Yₖ^(ℓ)(12)，置信度 logit 为 zₖ^(ℓ)。

**误差与质量。** 真值为 Y\*，逐步误差 dₖ(t) = ‖Yₖ(t) − Y\*(t)‖₂。候选质量沿用代码中的 q，用于无匹配时的回退：

```latex
q_k = \frac{1}{T}\sum_{t=1}^{T} d_k(t) + 0.25\, d_k(T), \qquad \hat{k} = \arg\min_k q_k
```

**匹配函数。** 阈值 θ 指最后一步的允许误差，前面各步线性缩放（与 `emta_indices` 一致）：

```latex
\mathrm{match}_{\theta}(k) = \mathbb{1}\left[\, d_k(t) \le \theta \cdot \tfrac{t}{T} \;\; \forall t \in \{1,\dots,T\} \right]
```

阈值序列 θ₀ ≥ θ₁ ≥ … ≥ θ\_D 由宽到严，默认值见第 5 节。

**覆盖权重与覆盖目标。** 以最后一个几何层的候选为对象。权重 wₖ = softmax(zₖ / τ)，Dᵢⱼ = ‖eᵢ − eⱼ‖₂。候选子集 S 的覆盖风险是“按权重加权的、每条候选到 S 中最近一条的终点距离”：

```latex
F(S) = \sum_{j=1}^{K} w_j \min_{s \in S} D_{sj}
```

F(S) 是在模型自身预测分布下 minFDE@|S| 的期望值（用 K 条候选近似分布）。

**覆盖序。** 给定起始集合 S₁，贪心地逐个加入让 F 下降最多的候选，得到排列 π 与边际增益 Δ：

```latex
\pi_r = \arg\min_{s \notin S_{r-1}} F\left(S_{r-1} \cup \{s\}\right), \qquad S_r = S_{r-1} \cup \{\pi_r\}, \qquad \Delta_r = F(S_{r-1}) - F(S_r)
```

现有 `risk_order` 从空集开始，第一位即加权 medoid。本方案规定第一位取置信度最高者：π₁ = argmaxₖ zₖ，S₁ = {π₁}，之后按上式贪心（理由见 F8）。rₖ 表示候选 k 在 π 中的位次（0 起算）。

**前序距离与重叠度。** 候选 k 到排在它前面的候选的最近终点距离，以及由此得到的重叠度（σₒ = 0.35 m，沿用 MCUS）：

```latex
d^{\mathrm{prev}}_k = \min_{j:\, r_j < r_k} D_{jk}, \qquad o_k = \exp\left(-\frac{(d^{\mathrm{prev}}_k)^2}{2\sigma_o^2}\right)
```

排第一位的候选没有前序，取 d^prev = 2.0 m、o = 0。

## 4. 具体设计

默认配置是两个几何层加一个判别层，匹配阈值依次为 1.0 m、0.7 m、0.5 m，推理参数量目标不超过 40 万（三层 QRANK 为 492,519）。

### 4.1 总体结构

| 层 | 角色 | 结构 | 训练分配 | 是否移动候选 |
| --- | --- | --- | --- | --- |
| 第 0 层 | 撒网（覆盖） | B0 的 H3 提议头，不改结构 | 一对多，θ₀ = 1.0 m，最多 3 个正样本 | 生成初始锚点 |
| 第 1 层 | 收网（精修） | 交叉注意力 + FFN，不含模式间自注意力 | 一对多，θ₁ = 0.7 m，最多 2 个正样本 | 有界锚点偏移 |
| 判别层 D | 排座次（判别） | 覆盖序因果自注意力 + FFN，不含交叉注意力 | 一对一 EMTA，θ\_D = 0.5 m，按覆盖序取最早匹配 | 不移动，只改置信度 |

第 1 层去掉模式间自注意力，依据是 F3 与 NO\_SET\_GAIN（archive 17）：在你的设定下它对落点没有可测作用，去掉可省约 1.7 万参数。如果消融显示有必要，可改回全注意力（不加因果掩码）。

### 4.2 级联分配（几何层的训练）

每个几何层用自己的阈值定义正样本集合。匹配上的候选按 q 从小到大取前 mℓ 个；一个都没匹配上时回退到质量最好的候选：

```latex
P^{(\ell)} = \mathrm{Top}_{m_\ell}\left\{ k : \mathrm{match}_{\theta_\ell}\big(Y^{(\ell)}_k\big) = 1 \right\} \text{ by } q_k, \qquad P^{(\ell)} = \{\hat{k}\} \text{ if empty}
```

回归损失对正样本加权，权重在每个窗口内归一化为 1。这样正样本多的窗口不会得到更大的总损失权重（排除“只是损失变大”的混杂）：

```latex
\mathcal{L}^{(\ell)}_{\mathrm{reg}} = \sum_{k \in P^{(\ell)}} \alpha_k \, \ell_{\mathrm{reg}}\big(Y^{(\ell)}_k, Y^*\big), \qquad \alpha_k = \frac{\exp(-q_k / 0.10)}{\sum_{j \in P^{(\ell)}} \exp(-q_j / 0.10)}
```

ℓ\_reg 为现有配方里的三项之和：`criterion.regression`、终点 smooth L1（乘 endpoint\_weight）、前若干步 smooth L1（乘 short\_horizon\_weight）。温度 0.10 与现有 cover 项相同。

几何层也输出置信度，用于下一层的覆盖权重。它用多标签 focal 损失训练，目标 yₖ = 1\[k ∈ P^(ℓ)\]，α = 0.25 / 0.75、γ = 2（与现有 `confidence_loss` 相同）：

```latex
\mathcal{L}^{(\ell)}_{\mathrm{conf}} = \frac{1}{K}\sum_{k=1}^{K} \left(1 - e^{-b_k}\right)^2 b_k \left(0.25\, y_k + 0.75\,(1 - y_k)\right), \qquad b_k = \mathrm{BCE}\big(z^{(\ell)}_k, y_k\big)
```

其余项（diversity hinge、cover 项）沿用 `emta_layer_loss`，只把单一赢家换成 P^(ℓ)。

### 4.3 覆盖序判别层

判别层只决定“谁排前面、置信度多少”，不移动任何候选，也不重新读场景信息。

1. **排队（不传梯度）。** 用最后一个几何层的终点和置信度计算覆盖序 π：π₁ = argmax zₖ，其余按第 3 节的贪心式。同时得到 rₖ、dₖ^prev、oₖ 和归一化边际增益 δₖ = Δ\_(rₖ) / max\_(r≥2) Δ\_r（π₁ 取 1）。
2. **输入令牌。** 每条候选一个 64 维令牌，hₖ 为最后一个几何层的模式表征，Āₖ = Aₖ / scale 展平为 6 维，E\_rank 为按位次学习的嵌入：

```latex
x_k = W_{\mathrm{in}} \left[\, h_k ;\; \bar{A}_k ;\; z_k ;\; o_k ;\; \log(d^{\mathrm{prev}}_k + 0.05) ;\; \tfrac{r_k}{K-1} ;\; \delta_k \right] + E_{\mathrm{rank}}[r_k]
```

3. **覆盖序因果注意力。** 候选 k 只能看排在它前面的候选，再加 n\_s = 4 个始终可见的可学习“汇聚令牌”（借鉴 MDS-DETR，保证排第一的候选也有可看的对象）。每个头 h 加一个由终点距离得到的偏置：

```latex
a^{h}_{kj} = \frac{(W^h_Q x_k)^\top (W^h_K x_j)}{\sqrt{16}} + g^h\!\left(D_{kj}\right), \qquad j \in \{ j : r_j < r_k \} \cup \mathcal{S}_{\mathrm{sink}}
```

g^h 是一个 1→16→4 的小 MLP（对汇聚令牌取可学习常数）。因为 D 对称，偏置也对称，不引入方向信息。

4. **输出。** 注意力加残差后过 FFN（64→128→64），线性头输出 Δzₖ，最终分数为：

```latex
s_k = z^{(L_g)}_k + 1.5 \tanh(\Delta z_k), \qquad Y^{D}_k = Y^{(L_g)}_k
```

幅度 1.5 与 MCUS 的 `calib_head` 相同。hₖ 不切断梯度：archive 22 显示切断打分对几何的梯度后，转向难组覆盖下降 1.5 个百分点。π、o、d^prev、δ 在无梯度下计算。

5. **训练（EMTA，按覆盖序）。** 在 π 的顺序里取第一条以 θ\_D 匹配的候选为正样本，其余全为负样本（包括也匹配上的重复候选）：

```latex
k^\star = \pi_{r^\star}, \quad r^\star = \min\{ r : \mathrm{match}_{\theta_D}(\pi_r) = 1 \}, \qquad k^\star = \hat{k} \text{ if none matches}
```

损失 L\_D 为对 s 的二元 focal（目标 one-hot(k\*)，参数同 4.2）。与 ModeSeq 的区别只在顺序来源：ModeSeq 用置信度顺序，这里用覆盖序。

### 4.4 推理输出

输出顺序与判别层的排队规则相同，但权重改用判别层的分数：σ₁ = argmax sₖ，其余按覆盖序贪心，权重 w′ₖ = softmax(sₖ / τ\_out)。输出列表 (Y\_σ₁, …, Y\_σ₂₀)，下游要 k 条就取前 k 条。

- minFDE@20 与 minADE@20 不受输出顺序影响，按定义不变。
- 第一位交给置信度，是为了避免加权 medoid 把 Top-1 拉向簇中心（F8）。
- τ\_out 从 {0.5, 1, 2} 中选，目标是 minFDE@3 与 minFDE@5 的均值最小，只在训练划分内部留出的部分上选，不在 dev 或测试折上选。

### 4.5 损失汇总

```latex
\mathcal{L} = \sum_{\ell=0}^{L_g} \left( \mathcal{L}^{(\ell)}_{\mathrm{reg}} + c\, \mathcal{L}^{(\ell)}_{\mathrm{conf}} + \mathcal{L}^{(\ell)}_{\mathrm{div}} + 0.20\, \mathcal{L}^{(\ell)}_{\mathrm{cover}} \right) + c\, \mathcal{L}_{D}
```

c 为现有 `criterion.classification_weight`，其余权重沿用现有配方；各几何层权重均为 1（与现有深监督一致）。

### 4.6 参数与计算预算

| 组成 | 参数量 | 依据 |
| --- | --- | --- |
| B0（编码器 + H3 提议） | 295,474 | 已核实：archive 27 `RESULT.json` |
| 第 1 层几何层（不含自注意力） | 约 49,000 | 推测：QRANK 每层均摊约 65,700，减去自注意力 16,640 |
| 判别层 D | 约 40,000 | 推测：输入投影约 4,900，自注意力 16,640，FFN 16,576，位次嵌入与汇聚令牌 1,536，其余不到 500 |
| **合计** | **约 385,000** | 推测，实现后以实测为准；参照：MCUS 单层 359,334，三层 QRANK 492,519 |

判别层没有交叉注意力，只在 20 个候选加 4 个汇聚令牌之间计算；覆盖序贪心每窗口约 K³ = 8,000 次加法。此外存在一个三个几何层的变体（阈值 1.0 / 0.8 / 0.6 m，判别层 0.5 m），参数约 43.4 万（推测），仅在默认配置有效时作为扩展。

## 5. 实现说明

实现建议以 archive 27 的 `MLDecoder`（`qrank_model.py`）为基础改写，编码器与 H3 提议头保持不变。

### 5.1 前向伪代码

```python
def forward(encoded, past):                        # 尺寸：B 窗口, K=20, T=12
    c = context(encoded, past); q = mode_mlp(c + mode_embedding)
    scale = softplus(proposal_scale(c)) + 0.25
    a = proposal(q, scale)                         # 第 0 层：(B,K,3,2)
    z = score_head(cat(q, a.flatten(2)))           # (B,K)
    outs = [dict(paths=traj(a), logits=z)]
    h = q
    for l in range(1, L_g + 1):                    # 第 1..L_g 层：只有交叉注意力
        x = h + geometric[l](a / scale)
        h = cross_layer[l](x, memory, padding)     # CrossLayer(self_attention=None)
        a = a + 0.30 * scale * tanh(offset[l](h))  # 有界锚点偏移，沿用 SHIFT
        z = z + score[l](cat(h, a.flatten(2)))
        outs.append(dict(paths=traj(a), logits=z))
    with no_grad():                                # 判别层：排队与冗余特征
        e = traj(a)[:, :, -1]                      # 终点 (B,K,2)
        pi, delta = cover_order(e, softmax(z / tau_u), first=argmax(z))
        d_prev, o = predecessor_overlap(e, pi, sigma_o=0.35, first_d=2.0)
    tok = W_in(cat(h, a / scale, z, o, log(d_prev + 0.05), rank_norm(pi), delta)) + E_rank[rank(pi)]
    u = masked_attention(sort(tok, pi), sinks=4, bias=g(dist(e)), mask=strict_lower)  # 只看前序 + 汇聚令牌
    s = z + 1.5 * tanh(head(ffn(unsort(u, pi))))
    return outs, traj(a), s, pi
```

### 5.2 训练与推理的差别

- **只在训练时存在：** 各层的匹配与正样本集合 P^(ℓ)、判别层的 EMTA 正样本 k\*、全部损失项。
- **训练与推理都有：** 几何层、判别层、覆盖序 π（它是前向计算的一部分，不依赖真值）。
- **只在推理输出时：** 用 s 重新计算一次覆盖序 σ（τ\_out），作为最终输出顺序。

### 5.3 与现有代码的对应

| 本方案部件 | 现有代码 | 需要的改动 |
| --- | --- | --- |
| 匹配函数 match\_θ | `qrank_model.emta_indices` | 返回全部匹配掩码，允许每层传入不同 θ |
| 级联分配损失 | `qrank_model.emta_layer_loss` | 单一赢家改为 P^(ℓ) 加权；focal 目标改为多标签 |
| 几何层 | `advisor_model.CrossLayer(self_attention=None)` | 无改动 |
| 覆盖序 | `order_numpy.risk_order`、`qrank_model.risk_order` | 增加参数 `first`，第一位固定为 argmax z；返回边际增益 |
| 前序距离与重叠度 | `mcus/run_mcus_probe.py` 第 135–165 行 | 直接复用 |
| 判别层注意力 | 无（MCUS 的 `MCUSCrossLayer` 含交叉注意力） | 新写：去掉交叉注意力，加汇聚令牌和距离偏置 |
| 输出评估 | `qrank_eval.py` 第 66 行的 `riskp` | 增加“第一位取 argmax s”的顺序 |

### 5.4 超参数

| 符号 | 默认值 | 含义 | 来源 |
| --- | --- | --- | --- |
| K / T / Δt | 20 / 12 / 0.4 s | 候选数 / 预测步数 / 步长 | 现有协议 |
| L\_g | 1 | 第 0 层之后的几何层数 | 本方案 |
| θ₀ / θ₁ / θ\_D | 1.0 / 0.7 / 0.5 m | 各层最后一步的匹配阈值 | 1.0 与 0.5 取自附录 A.4；0.7 为中间值 |
| m₀ / m₁ | 3 / 2 | 各层最多正样本数 | 本方案 |
| τ\_u | 1.0 | 判别层内覆盖权重温度 | 与 `riskp` 一致 |
| τ\_out | {0.5, 1, 2} 选一 | 输出覆盖序温度 | 在训练划分内部选 |
| σₒ | 0.35 m | 重叠度尺度 | MCUS |
| 首位 d^prev | 2.0 m | 无前序时的填充值 | MCUS |
| n\_s | 4 | 汇聚令牌数 | 借鉴 MDS-DETR（其用 10 个，对应 300 个查询） |
| SHIFT | 0.30 | 锚点偏移上限系数 | `qrank_model.SHIFT` |
| 判别层分数幅度 | 1.5 | s = z + 1.5 tanh(Δz) | MCUS `calib_head` |
| 模型宽度 / 头数 / FFN | 64 / 4 / 128 | 与 `CrossLayer` 一致 | 现有代码 |

## 6. 验证计划

先做一个零训练预演，再在 C0 上跑七臂消融；只有 C0 的判据通过，才做五折确认。两项都需要你本地的数据和训练环境。

### 6.1 【已完成】形成方案所依据的实验

QRANK 的 C0 筛选与五折（archive 27）、EXP-REFINE-PRED（archive 29）、QRANK 冻结重放（archive 28）、LAYER-POOL 与 MCUS 消融（archive 30）、EXP-186A（archive 04）。结论见第 2.1 节。

### 6.2 【待执行】V0：输出顺序零训练预演

在已保存的 B0 与 QRANK 预测上（需要本地完整候选终点和 logit），比较三种输出顺序：置信度顺序、现有 `riskp`（首位为加权 medoid）、本方案的顺序（首位为 argmax）。

- **判据：** 本方案顺序的 minFDE@3 与 @5 不比 `riskp` 差超过 0.5%，且每个折都比置信度顺序好。
- **不作为证据的量：** 本方案顺序的 @1 按构造等于置信度顺序的 @1，@20 按构造不变。

### 6.3 【待执行】V1：C0 七臂消融

统一设置：C0 的 21,719 个训练窗和 4,753 个开发窗，种子 42–44，批量 64，64 轮，最终 EMA 权重，各臂样本顺序和初始化配对。每个臂的输出都按置信度顺序和本方案顺序各评估一次。

| 臂 | 内容 | 回答的问题 |
| --- | --- | --- |
| M0 | B0（复用已有结果） | 参照 |
| M1 | 第 0 层 + 第 1 层，各层均 θ = 0.5 m、单一正样本，无判别层 | 级联分配的对照 |
| M2 | M1 + 级联分配（θ = 1.0 / 0.7 m，m = 3 / 2） | 逐层收紧有没有用 |
| M3 | M2 + 判别层，按置信度排队（MDS-DETR 式） | 单独的判别层有没有用 |
| M4 | M2 + 判别层，按覆盖序排队（完整方案） | 覆盖序是否优于置信度序 |
| M5 | M4 去掉 o、d^prev、δ 三个冗余特征，保留顺序与掩码 | 冗余特征是否必要 |
| M6 | M1 加宽，参数量与 M4 相差不超过 2% | 容量对照 |

**判据**（相对变化以误差降低为正，区间为轨迹聚类 bootstrap）：

| 编号 | 比较 | 通过条件 |
| --- | --- | --- |
| G1 | M2 对 M1 | minFDE@20 改善 ≥ 1.0% 且 95% 区间下限 > 0；v > 1.3 m/s 组 minFDE 不变差 |
| G2 | M3 对 M2 | Top-1 FDE 改善 ≥ 3%；minFDE@20 变化在 ±1% 内 |
| G3 | M4 对 M3 | Top-1 FDE 变差不超过 1%；且 minFDE@3 或 @5 改善 ≥ 1.5%（区间下限 > 0）或 brier-minFDE 改善 ≥ 2% |
| G4 | M4 对 M5 | minFDE@3 改善 ≥ 1% |
| G5 | M4 对 M6 | minFDE@20 改善 ≥ 1%；否则收益归为容量 |
| G6 | M4 | 实测推理参数量 ≤ 400,000 |

G1–G3 与 G6 全部通过才进入五折；五折单独报告 ETH（唯一与 C0 选臂窗口无重叠的折）。

**操纵检查**（训练前预检 + 训练后复核）：

- MC-a：M2 第 0 层每窗平均正样本数 ≥ 1.5（M1 为 1）；C0 匹配率 ≥ 0.90；ETH 且 v > 1.3 m/s 匹配率 ≥ 0.20。
- MC-b：判别层不移动候选，即 max |Y^D − Y^(L\_g)| = 0。
- MC-c：M4 与 M3 的顺序确实不同：第 2–5 位至少一位改变的窗口比例 ≥ 15%，前 5 重合度 ≤ 0.80（首位按构造相同）。
- MC-d：M6 与 M4 参数量差不超过 2%，训练步数相同。

### 6.4 设计自检

| 项 | 结果 |
| --- | --- |
| 1. 干预隔离 | M2 除阈值外还改变正样本数和置信度目标：用归一化权重 α 控制损失规模，置信度目标变化属于干预本身；M3、M4 增加约 4 万参数，由 M6 对照；M4 相对 M3 同时改变顺序和由顺序算出的特征，由 M5 拆开 |
| 2. 同义反复 | @1（首位相同）与 @20（顺序无关）按构造必然成立，不当证据；不用模型自身候选的分散度证明“去重生效”，只用对真值的指标；τ\_out 只在训练划分内部选 |
| 3. 依赖结构 | C0 开发集 4,753 窗 / 248 个轨迹簇；报告前 10 簇占比；轨迹聚类 bootstrap 4,000 次；逐录制留一与去掉 ZARA02 的敏感性 |
| 4. 相关≠因果 | 各臂是干预实验，结论在 C0 内按因果解读；注意力图、置信度分布只作描述 |
| 5. 参数完整 | 全部阈值、上限、分组见 5.4 节；高速组定义为 7 步位移均值 / 0.4 s > 1.3 m/s（`qrank_eval.py` 第 59 行）；引用数字均在第 2.1 节标注了文件 |
| 6. 预演 | 匹配率已有零训练预演（附录 A.4）；V0 与参数实测需要本地文件，由你执行 |
| 7. 非预期原因 | G1：收益可能来自多标签置信度而非阈值，可加一臂只对回归用级联。G2：Top-1 收益可能来自额外参数，看 M6。G3：覆盖序改变了 EMTA 的负样本集合，这是预期机制；若只在 ZARA02 成立，归为场景效应 |
| 8. 混杂 | 参数量不同由 M6 处理；判别层与覆盖序增加每步耗时，但各臂轮数与步数相同；输入与数据各臂一致 |

## 7. 风险与备选

| 风险 | 早期信号 | 备选 |
| --- | --- | --- |
| R1 级联分配无效：宽阈值的多正样本使候选聚到同一模态（EDA 指出的预测聚集） | G1 未过；M2 前 20 条候选终点的平均两两距离比 M1 缩小 > 10% | 只对回归使用级联分配，置信度仍用单一正样本；或把第 0 层改成软分配（按距离加权，不用硬阈值） |
| R2 判别层只带来参数收益 | G2 通过但 G5 未过 | 研究一的主张收缩为“级联分配 + 输出覆盖序”，判别层降为附录消融；参数预算回到约 34 万 |
| R3 覆盖序不优于置信度序 | G3 未过；或 MC-c 显示两种顺序几乎一样 | 保留判别层但按置信度排队（即 M3），把覆盖序只用在推理输出（σ），创新点改写为“训练排队与输出顺序分离” |
| R4 收益只来自 ZARA02 或个别录制 | 逐录制留一后区间跨过 0；前 10 簇占比 > 30% | 论文中如实报告为场景相关的效果，并在第 6 节判据上加注；不扩大主张 |
| R5 参数量超出 40 万预算 | G6 未过 | 判别层 FFN 由 64→128→64 改为 64→64；汇聚令牌由 4 个减为 2 个；或第 1 层与判别层共享注意力权重 |
| R6 τ\_out 或覆盖序对 ETH 这种快速、稀疏场景失效（EMTA 0.5 m 匹配率只有 0.036） | ETH 折 @3/@5 劣于置信度序 | 按速度缩放判别层阈值 θ\_D = 0.5·max(1, v/1.3) m；该改动需要单独一臂验证，不能直接并入主方法 |
| R7 开题时被质疑“与 MDS-DETR / ModeSeq 雷同” | — | 回答要点见第 2.3 节区别表：分配阈值逐层收紧（MDS-DETR 是一对多 + 一对一两级、阈值不随层变）；排队依据是带权 k-median 覆盖增益（MDS-DETR 与 ModeSeq 都按置信度）；输出是嵌套前缀，@k 对任意 k 都有意义 |

**开题时的表述边界：** 在 V1 完成前，只把第 2.1 节的现象作为动机，把第 4 节作为拟采用的方法；不报告预期的提升数字。

## 8. 参考文献与证据索引

### 8.1 文献（本次讨论中实际打开过的页面）

- ModeSeq: Taming Sparse Multimodal Motion Prediction with Sequential Mode Modeling（CVPR 2025）— [arXiv:2411.11911](https://arxiv.org/abs/2411.11911)
- Metric-agnostic 轨迹输出选择（2026）— [arXiv:2605.24037](https://arxiv.org/abs/2605.24037)
- EDA: Evolving and Distinct Anchors for Multimodal Motion Prediction — [arXiv:2312.09501](https://arxiv.org/abs/2312.09501)
- Cascade R-CNN: Delving into High Quality Object Detection — [arXiv:1712.00726](https://arxiv.org/abs/1712.00726)
- MDS-DETR（2026）— [arXiv:2605.23507](https://arxiv.org/abs/2605.23507)
- 集合蒸馏相关工作（2026）— [arXiv:2607.01133](https://arxiv.org/abs/2607.01133)
- MBRM / 最小贝叶斯风险轨迹选择（2024）— [arXiv:2409.10585](https://arxiv.org/abs/2409.10585)
- DETA: NMS Strikes Back — [arXiv:2207.05844](https://arxiv.org/abs/2207.05844)
- H-DETR: DETRs with Hybrid Matching — [arXiv:2207.13080](https://arxiv.org/abs/2207.13080)
- DAC-DETR: Divide the Attention Layers and Conquer（NeurIPS 2023）
- Rank-DETR for High Quality Object Detection — [arXiv:2310.08854](https://arxiv.org/abs/2310.08854)
- SmartRefine: A Scenario-Adaptive Refinement Framework for Efficient Motion Prediction — [arXiv:2403.11492](https://arxiv.org/abs/2403.11492)

Wayformer、MultiPath++、TNT、DenseTNT、MTR 在本次讨论中按其已知机制作对比，未重新打开原文；写论文前需逐一核对原文。

### 8.2 仓库证据索引（1inke/kinematic-traj-forecasting）

| 证据 | 文件 |
| --- | --- |
| EMTA 实现、置信度 focal 损失、逐层损失、MLDecoder | `research_archive/27_qrank_mldec/EXP_QRANK_MLDEC_L4_20261122/code/sealed/qrank_model.py` |
| CrossLayer、memory()、proposal()、trajectory() | 同目录 `advisor_model.py` |
| 带权 k-median 贪心 `risk_order` | 同目录 `order_numpy.py` |
| 输出端风险排序的评估、高速组定义 | 同目录 `qrank_eval.py`（第 59、66 行） |
| 多层 / EMTA / 匹配率的结论与附录 A.1–A.4 | `research_archive/27_qrank_mldec/.../DESIGN_QRANK.md` |
| 冻结重放与 EA | `research_archive/28_qrank_followups_and_ea/README.md` |
| 细化预测 | `research_archive/29_refine_pred/results/REPORT.md` |
| MCUS 流程与消融臂 | `research_archive/30_local_decoder_followups/mcus/run_mcus_probe.py`、`mcus_ablation/run.py` |
| LAYER-POOL | `research_archive/30_local_decoder_followups/layer_pool/results/REPORT.md` |
| 编码器 τ 敏感性 | `research_archive/34_tau_encoder_sensitivity/README.md` 与 results；`32/results/R_TAU2.json` |
| 参数量与效率审计 | `research_archive/05/UNIFIED_EFFICIENCY_AUDIT_SERIES/.../efficiency_results.csv` |
| 顺序依赖对 minFDE 的影响 | `research_archive/04/.../EXP_186A/RESULT_ANALYSIS.md` |
| 打分耦合 | `research_archive/22_score_coupling/README.md` |
