# 研究一·编码端设计：运动学分工式稀疏交互注意力

Nov 26, 2026 · @akaza

## 1. 概述

本文档是研究一编码端的设计说明，与解码端文档《由宽到严的级联分配 + 覆盖序判别层》配套。编码端的主张可以用一句话概括：**运动学量负责描述关系，稀疏图负责限定预算，注意力权重由学习决定。** 现有工作常把这三种角色混在一起，用运动学先验决定“看谁”，或用学习分数先算稠密图再剪枝。本仓库的实验显示，混在一起时“用运动学决定看谁”这一半无效，而“用运动学描述关系”这一半稳定有效。

**两个设计点：**

- **C1（主，已有实现）分工式稀疏交互注意力。** 稀疏图只给每个行人一个推理前确定、宽度固定（含自身不超过 7）的支撑集；TTC、DCA、closing 等预期性运动学量只作为关系编码进入注意力的打分偏置与消息，不参与选择性的重要度判断。对应 B0 现有的 F16 编码器，本文给出它的定位、定义和需要补的证据。
- **C2（拟改进，待验证）逐时刻运动学偏置。** F16 在“邻居 × 时刻”上做联合 softmax，但现有关系偏置在时间维上是常数，跨时刻结构没有物理线索指出该看邻居历史的哪一刻。C2 把偏置改为逐时刻的运动学函数，只增加 80 个参数，零初始化，初始时与 B0 完全等价。

**不作为创新点的内容：** 因子化时序/社会结构（只作为骨架）、按碰撞时间选邻与玻尔兹曼能量解释（实验已否定，只保留 Karamouzas 作为“关系通道中包含 TTC”的依据）、学习式路由与代理注意力（已验证退化）。

**证据状态：** C1 的“描述有效、选择无效”由已完成实验支持；“预算可预知”的计算收益只在大 N 下成立，ETH/UCY 上不能声称加速。C2 只有旧底座上的弱正信号。开题时只陈述动机和拟采用的方法，不报告预期提升数字。

## 2. 设计思路与想法来源

### 2.1 仓库中的现象

下表数字均当场查阅文件核实。相对变化以误差上升为正，除非另注。

| 编号 | 现象 | 数值 | 来源 | 对设计的含义 |
| --- | --- | --- | --- | --- |
| K1 | 运动学关系 token（相对位置/速度、距离、closing、TTC、DCA）进入注意力偏置与消息 | 五折 seed 1、参数量相同，J 降低 2.09%（E1→E2） | EXP-172 | 运动学作为“描述”有效，归因干净 |
| K2 | 稀疏交互编码器整体 vs 纯轨迹 | 五折 J 降低 3.16%（混入 robust state）；C0 三种子 minFDE 降低 4.22% | EXP-172；archive 34 E-3 | 交互编码有效，但不能归因给“稀疏”本身 |
| K3 | F16 跨时刻交叉 vs E2（摘要式社交注意力） | J 上升 0.99%（seed 1，未做多种子） | EXP-172 | 跨时刻结构目前没有证据支持 |
| K4 | 按碰撞时间选邻 | T1 +1.60%，T2 +1.76%（区间 +0.07% 至 +3.46%），C0 判 STOP | archive 32 | 运动学作为“选择”有害 |
| K5 | 选邻规则横向对照 | 全连接比 T2 低 1.37%；T2 比同邻居数距离 top-k 低 0.81%；区间均跨零 | archive 33 E-1 | 在 ETH/UCY 上选法对精度影响很小 |
| K6 | 少数据下的选邻对照 | 25% 数据时 T2 相对 top-p 的收益反而下降 2.13 个百分点 | archive 33 E-2 | “先验在少数据时有用”不成立 |
| K7 | 手工交互势 vs 逐样本社交收益 | Spearman 约 0（−0.002 至 0.059）；社交信息本身五折宏平均 FDE 优势 5.99% | CSNA；EXP-169B | 社交信息要用，但手工分数不知道该看谁 |
| K8 | 稀疏的计算收益 | 核心注意力约 −67%，端到端延迟不变；N=128 峰值显存 151.7 MB → 8.3 MB（快速结果） | ATTENTION\_COMPUTE；SPARSE\_EXECUTION | 稀疏的价值是大 N 时的成本上界 |
| K9 | 逐时刻物理偏置 PB1（80 参数） | General minFDE 相对同等微调基线 −0.751%，相对原始 146B 仅 −0.04% | PHYSICAL\_ATTENTION\_BIAS | C2 只有弱正信号 |
| K10 | 学习“关键时刻”的 KDTA | Top-1 FDE 改善 4.66%，minFDE 退化 0.75% | PAIR\_AWARE\_ATTENTION | “看哪一刻”的信号影响排序 |

### 2.2 为什么“用运动学选邻”失败：三种解释

- **(a) 邻居太少，硬筛选得不偿失。** ETH/UCY 全连接平均 5.26 个非自身邻居，T2 保留 1.85 个（archive 33）。省下的计算很少，却可能丢掉并行同伴这类 TTC 无定义的有用邻居。【推测】；T1 只改选邻就变差，部分支持。
- **(b) 单帧估计噪声大，硬决策出错不可逆。** TTC/DCA 按常速外推估计，低相对速度时不稳定。做选择时错了无法挽回；做软特征时网络可以学着打折。【推测】；与 T3（只加偏置）中性、K1 有效一致。
- **(c) 注意力本身会压低无关邻居。** 全连接不差于任何稀疏规则；ART 式学习 top-p 在本骨干里也没赢全连接（K5）。

**排除：** K6 显示运动学选邻在少数据下也不更有利，排除了“先验在数据少时才有用”。(a) 与 (b) 用现有数据分不开，需要在 SDD 等高密度场景按邻居数分层（见 6.3 节 V3）。三种解释都指向同一设计结论：运动学不应决定看谁。

### 2.3 想法来源

- **Karamouzas 等（PRL 2014）：** 行人交互能量取决于预计碰撞时间而非距离，是预期性的。这是在关系通道里包含 TTC、DCA、closing 的依据。但 K4 显示，直接把它当选邻或能量先验不成立。
- **TPK（IV 2025）：** 比较了把交互先验乘入或门控接入注意力的两种办法，精度都没有提升，并指出先验会掩盖或被网络忽略。这支持“不把先验当重要度”。
- **SGCN、ART、ViTE：** 稀疏由学习分数决定，先算稠密分数再剪枝或 top-p。稀疏不降低打分成本，且 k 随样本变化。这引出“支撑集应在学习计算之前、以固定宽度确定”。
- **ART、QCNet、AgentFormer：** ART 的时间分辨只来自坐标学到的特征；QCNet 的社交注意力在同一时刻内进行；AgentFormer 允许任意时刻互看但全连接、无运动学。三者都没有用逐时刻的运动学关系告诉注意力该看邻居历史的哪一刻，这引出 C2。
- **解码端的“输出预算 k”：** 编码端的“交互预算 K\_max”与之同构，并在研究二中自然延伸：加入场景 token 后支撑集变大，需要线性复杂度。

### 2.4 与已有工作的区别

| 方法 | 稀疏由什么决定 | 是否先算稠密分数 | 每行宽度 | 运动学的角色 | 时间分辨的关系 |
| --- | --- | --- | --- | --- | --- |
| SGCN | 学习分数 + 固定阈值 0.5 | 是 | 可变 | 无（只用坐标） | 无 |
| GST | Gumbel 采样 | 是 | 可变 | 无 | 每步采样一图 |
| ART | 学习边权 + top-p | 是 | 可变 | 无 | 有，但来自学习特征 |
| ViTE | 学习门控 top-p（专家层面） | 是 | 可变 | 无 | 无 |
| Polar Collision Grids | TTC + 接近角启发式选择 | 否 | 可变 | 决定看谁 | 无 |
| TPK | 全部邻居 | — | 全部 | 作为注意力重要度先验 | 无 |
| QCNet | 半径内全部 | — | 可变 | 相对位姿位置编码 | 同时刻社交 |
| **本方案** | **闭式标量填满固定预算** | **否** | **≤ 7，推理前可知** | **只描述关系（偏置 + 消息）** | **C2：逐时刻运动学偏置** |

STP（TPAMI 2025）用动态交互位置构造稀疏交互以达到实时，是“稀疏 + 实时”定位上的直接竞争者。本次只看到摘要，未列入上表；写论文前必须读全文并补入。

### 2.5 设计原则

1. **运动学不决定看谁。** 它只以关系编码的形式进入打分和消息，由网络决定信任程度。
2. **稀疏只限定预算。** 支撑集宽度固定、在任何学习计算之前确定；不以精度为目的调选邻规则。
3. **时间分辨的关系告诉注意力“何时”。** 跨时刻注意力只有在关系线索随时间变化时才有存在的理由（C2）。

## 3. 符号与定义

### 3.1 基本符号

| 符号 | 含义 | 取值 |
| --- | --- | --- |
| N | 场景内行人数（含填充） | 随样本 |
| T | 观测步数 | 8（3.2 s × 2.5 Hz） |
| f | 采样频率 | 2.5 Hz |
| d | token 维度 | 100（spatial 64 + temp 32 + walk 4） |
| H, d\_h | 注意力头数、每头维度 | 4, 25 |
| p\_i(t), v\_i(t) | 行人 i 在第 t 步的位置与速度 | v 为差分 × f，v\_i(1) = v\_i(2) |

### 3.2 逐时刻相对运动学量（关系通道）

对每对 (i, j) 和每个观测步 t：

```latex
\Delta p_{ij}(t) = p_j(t) - p_i(t), \qquad \Delta v_{ij}(t) = v_j(t) - v_i(t)

d_{ij}(t) = \lVert \Delta p_{ij}(t) \rVert, \qquad
c_{ij}(t) = -\frac{\Delta p_{ij}\cdot \Delta v_{ij}}{\max(d_{ij},\,10^{-3})}

\tau_{ij}(t) = \mathrm{clip}\!\left(-\frac{\Delta p_{ij}\cdot \Delta v_{ij}}{\max(\lVert\Delta v_{ij}\rVert^2,\,10^{-4})},\ 0,\ 10\right), \qquad
\delta_{ij}(t) = \lVert \Delta p_{ij} + \tau_{ij}\,\Delta v_{ij} \rVert

r_{ij}(t) = \big[\Delta p_{ij},\ \Delta v_{ij},\ d_{ij},\ c_{ij},\ \tau_{ij}/10,\ \delta_{ij}\big] \in \mathbb{R}^{8}
```

c > 0 表示正在接近。代码中把 τ 叫作 ttc，它实际是常速假设下的最近接近时间；δ 是最近接近距离（DCA）。

### 3.3 交互支撑集 𝒩\_i（预算）

支撑集由 `Stage1GraphBuilder` 的 `adaptive_kinematic` 模式在任何学习层之前算出，只用标量。它使用一套独立的风险分数：

```latex
\tau^{c}_{ij}(t) = \min\!\left(\frac{d_{ij}}{c^{+}_{ij} + \epsilon},\ 8\right), \quad c^{+} = \max(c, 0), \quad
a_{ij}(t) = \max\!\big(0,\ \hat u_i \cdot \hat b_{ij}\big)

\rho_{ij}(t) = 0.25\,e^{-d_{ij}/2} + 0.20\,\min(c^{+}_{ij}/1.5,\,1) + 0.20\,e^{-\tau^{c}_{ij}/2} + 0.15\,a_{ij} + 0.20\,e^{-\delta^{h}_{ij}/1}

\bar\rho_{ij} = \sum_{t=1}^{T} w_t\,\rho_{ij}(t), \qquad w = \mathrm{softmax}(0.25\,t)

\mathcal{C}_i = \{\, j \ne i :\ \lVert p_j(T) - p_i(T) \rVert < 6\ \mathrm{m} \,\}, \qquad
n_i = \min\!\Big(\mathrm{clip}\big(|\{ j\in\mathcal{C}_i : \bar\rho_{ij} > 0.35 \}|,\ 2,\ 6\big),\ |\mathcal{C}_i|\Big)

\mathcal{N}_i = \mathrm{TopK}_{n_i}\big(\bar\rho_{i\cdot};\ \mathcal{C}_i\big) \cup \{ i \}, \qquad |\mathcal{N}_i| \le K_{\max} + 1 = 7
```

û\_i 是 i 的速度方向，b̂\_ij 是从 i 指向 j 的方位单位向量；δ^h 是把最近接近时间截断到 4.8 s 后的 DCA。图不对称化。

**在 C1 中的定位：** ρ̄ 只用来“填满预算”，不被解释为重要度，也不以任何形式进入注意力打分。支撑集的关键性质是宽度有上界且在推理前可知。

### 3.4 关系 token（描述）

```latex
u_{ij}(t) = \mathrm{MLP}\big(\mathrm{LN}(r_{ij}(t))\big), \quad \mathrm{MLP}: 8 \to 50 \to 100\ (\mathrm{GELU})

s^{(1)}_{ij} = \tfrac12\textstyle\sum_{t\in\{T-1,T\}} u_{ij}(t), \quad
s^{(2)}_{ij} = \tfrac14\sum_{t=T-3}^{T} u_{ij}(t), \quad
s^{(3)}_{ij} = \tfrac1T\sum_{t=1}^{T} u_{ij}(t)

\omega_{ij} = \mathrm{softmax}_k\big(w_s^\top s^{(k)}_{ij} + b_s\big), \qquad
\bar e_{ij} = \textstyle\sum_{k=1}^{3} \omega_{ij,k}\, s^{(k)}_{ij} \in \mathbb{R}^{d}
```

### 3.5 跨时刻交叉注意力

h\_j(t) 是逐行人时序编码（1 层 Transformer）后的时刻 token，a\_i 是动态时序查询汇聚出的行人摘要。接收者用当前摘要，发送者暴露全部历史时刻：

```latex
q_i^{h} = W_q^{h} a_i, \quad k_j^{h}(t) = W_k^{h} h_j(t), \quad v_j^{h}(t) = W_v^{h} h_j(t)

\ell^{h}_{ij}(t) = \frac{q_i^{h}\cdot k_j^{h}(t)}{\sqrt{d_h}} + \beta^{h}_{ij} + \underbrace{b^{h}_{ij}(t)}_{\text{C2}}, \qquad \beta_{ij} = W_\beta\,\bar e_{ij} \in \mathbb{R}^{H}

\alpha^{h}_{ij}(t) = \frac{\exp \ell^{h}_{ij}(t)}{\sum_{j'\in\mathcal{N}_i}\sum_{t'=1}^{T} \exp \ell^{h}_{ij'}(t')} \quad (j \in \mathcal{N}_i)

m^{h}_{ij}(t) = v_j^{h}(t) + [W_e \bar e_{ij}]^{h}, \qquad
o_i = W_o \big\Vert_h \textstyle\sum_{j\in\mathcal{N}_i}\sum_{t} \alpha^{h}_{ij}(t)\, m^{h}_{ij}(t)
```

softmax 在“邻居 × 时刻”的联合集合上归一化。在 B0 中没有 b，β\_ij 对所有 t 相同（代码中 `edge_bias[..., None]` 沿时间维广播）。

### 3.6 门控回注

```latex
g_i = \sigma\big(\mathrm{MLP}([a_i;\, o_i])\big)\ (\text{输出偏置初始化为 } -1), \qquad
\hat a_i = \mathrm{LN}(a_i + g_i \odot o_i), \qquad
\tilde h_i(t) = \mathrm{LN}\big(h_i(t) + W_{fb}\,\hat a_i\big)
```

### 3.7 逐时刻运动学偏置 b（C2）

```latex
\pi_{ij}(t) = \sigma\big(c_{ij}(t)/0.2\big), \qquad L(x) = \mathrm{clip}\big(\log(1+x)/\log 7,\ 0,\ 1.5\big)

\tilde d = L(d), \quad \tilde c = \tanh(c/1.5), \quad
\tilde\tau = \pi\cdot\mathrm{clip}(\tau/10,\,0,\,1) + (1-\pi), \quad
\tilde\delta = \pi\cdot L(\delta) + (1-\pi)

\phi_m(x) = \exp\!\Big(-\tfrac12\big(\tfrac{x-\mu_m}{\sigma}\big)^2\Big),\ m=1..5;\quad
\mu = \mathrm{linspace}(0,1,5),\ \sigma=0.25\ (\tilde d,\tilde\tau,\tilde\delta);\quad
\mu = \mathrm{linspace}(-1,1,5),\ \sigma=0.5\ (\tilde c)

b^{h}_{ij}(t) = B\,\tanh\!\Big(\tfrac1B \textstyle\sum_{c\in\{\tilde d,\tilde c,\tilde\tau,\tilde\delta\}}\sum_{m=1}^{5} W^{h}_{c,m}\,\phi_m\big(x_{c,ij}(t)\big)\Big), \quad B = 2,\quad b^{h}_{ii}(t) = 0
```

W ∈ R^{H×4×5} 零初始化，共 80 个参数。接近门 π 的作用：两人远离时，τ̃ 与 δ̃ 被推到 1（“无冲突”），避免常速外推在背离时给出无意义的接近时间。

**变体 C2-lag：** 把相对量改为邻居第 t 步状态相对接收者当前状态，即 Δp = p\_j(t) − p\_i(T)、Δv = v\_j(t) − v\_i(T)，其余不变。它对应“我现在会对邻居早先的哪个状态做出反应”。默认采用同时刻版本，因为它的物理含义更清楚，并且已有 PB1 实现。

## 4. 具体设计

### 4.1 结构

数据流为：轨迹 → 支撑集（闭式）与关系通道（闭式）→ 时序编码 → 关系 token → 跨时刻交叉注意力 → 门控回注 → 解码端。前两步不含任何可学习参数。

| 模块 | 输入 → 输出 | 性质 | 对应角色 |
| --- | --- | --- | --- |
| 支撑集构造 | p \[B,T,N,2\] → 邻接 \[B,N,N\]（每行 ≤ 7 个真值） | 闭式，无参数 | 预算 |
| 关系通道 | p → r \[B,N,N,T,8\] | 闭式，无参数 | 描述的原料 |
| 时序编码 + 动态时序查询 | token \[B,N,T,d\] → h \[B,N,T,d\]，a \[B,N,d\] | 可学习 | 个体状态 |
| 关系 token | r → ē \[B,N,N,d\] | 可学习 | 描述 |
| 跨时刻交叉注意力 | a, h, ē, 支撑集 \[, b\] → o \[B,N,d\] | 可学习 | 学习权重 |
| C2 逐时刻偏置 | r → b \[B,H,N,N,T\] | 80 参数，零初始化 | 告诉“何时” |
| 门控回注 | a, o, h → h̃ \[B,N,T,d\] | 可学习 | 按需注入 |

### 4.2 支撑集：可预知的交互预算

定义见 3.3 节。它提供三条性质：

1. **宽度上界。** 每行至多 K\_max + 1 = 7 个有效槽，与场景人数 N 无关。
2. **先于学习计算确定。** 支撑集只依赖标量运算，不需要先算 d 维投影或稠密注意力分数。这是与 SGCN、ART、ViTE 的核心区别。
3. **成本可预知。** 注意力分数元素数上界为 N·7·T·H，稠密版本为 N²·T·H。

```latex
\text{支撑集构造}:\ O(N^2 T)\ \text{标量运算}; \qquad
\text{交叉注意力}:\ O(N \cdot K \cdot T \cdot d)\ \ \text{vs}\ \ O(N^2 \cdot T \cdot d)\ (\text{稠密})
```

**表述边界：** 当前实现是带掩码的稠密计算，实际的稀疏执行在小 N 下没有加速（K8）。论文只报告理论规模、注意力元素数和大 N 显存缩放，不写“加速”。支撑集规则本身不作为精度来源主张。

### 4.3 关系编码：运动学只描述，不选择

关系 token ē\_ij 以两种方式进入注意力：作为每头的打分偏置 β\_ij 和作为消息的加性项 W\_eē\_ij。与 TPK 式先验的区别是：β 由可学习的映射得到，不被约束为重要度（非负、归一、与风险单调），也没有乘性掩码。所以网络可以把高风险的邻居压低，把 TTC 无定义的并行同伴抬高。

不采用的做法：把 ρ̄ 作为注意力偏置（`use_soft_risk_bias`，F16 中未启用）；把 τ 当作能量先验（archive 32 的 T2）。

### 4.4 跨时刻交叉注意力与 C2

定义见 3.5 节和 3.7 节。设计要点：

- **不对称：** 查询是接收者的摘要 a\_i，键是发送者的每个历史时刻 h\_j(t)。与同时刻社交（QCNet）和任意时刻全连接（AgentFormer）都不同，分数元素为 N·|𝒩|·T，而不是 (N·T)²。
- **联合归一化：** softmax 在“邻居 × 时刻”上归一化，邻居之间、时刻之间相互竞争。
- **B0 的缺口：** β\_ij 对 t 是常数，选择“哪一刻”完全只靠内容点积 q·k。这可能是 K3（E3 不如 E2）的原因，【推测】。
- **C2 的修补：** 加入随 t 变化的 b^h\_ij(t)，让每个历史时刻都带上“当时两人的距离、接近速度、接近时间、最近距离”的物理线索。它保留 β（PB1 的叠加形式）；替换 β 的 PB2 和只用 b 的 PB3 在旧底座上都更差。
- **初始等价：** W 零初始化，训练第 0 步的输出与 B0 逐位相同，所有差异都来自学到的 80 个系数。
- **与文献的区别：** ART 的时间分辨来自同时刻坐标特征的学习打分；Reverberation 学的是逐行人的潜伏期核。C2 的时间线索来自成对的、逐时刻的运动学量。

### 4.5 门控回注

社交消息经 sigmoid 门按维度注入摘要，门的输出偏置初始化为 −1（初始约 0.27）。再经线性层广播回每个时刻 token。K7 显示社交收益是稀疏、不均匀的，门控给网络按样本决定注入强度的自由度；本方案不对它施加手工先验。

### 4.6 与解码端的接口

编码端输出形状与 B0 完全相同（\[B, T·N, d\]）。解码端的级联分配与覆盖序判别层不依赖编码端内部结构，两端可以独立消融。

### 4.7 参数预算

| 配置 | 参数量 | 说明 |
| --- | --- | --- |
| B0（F16 + H3） | 295,474 | 效率审计记录 |
| B0 + C2 | 295,554 | +80（H×4×5） |
| B0 + C2 + 解码端方案 | 约 385k | 解码端文档 4.6 节的估计，【推测】 |

## 5. 实现说明

### 5.1 前向伪代码

```python
def encode(past, tokens, num_valid, use_c2=False, c2_lag=False):
    # past [B,T,N,2]; tokens [B,T*N,d]
    allowed = graph.build(past, num_valid)        # 3.3: adaptive_kinematic, 每行 <= 7, 无参数
    r = relation_features(past)                   # 3.2: [B,N,N,T,8], 无参数
    h = temporal_encoder(tokens)                  # [B,N,T,d]
    h, a = temporal_queries(h)                    # a: [B,N,d]
    e = relation_tokens(r)                        # 3.4: [B,N,N,d], 多尺度时间汇聚

    q = W_q(a); k = W_k(h); v = W_v(h)            # 分成 H=4 个头
    logit = einsum('bhid,bhjtd->bhijt', q, k) / sqrt(d_h)
    logit += W_beta(e).permute(0,3,1,2)[..., None]     # β: 时间广播
    if use_c2:
        r_t = lagged_relation(past) if c2_lag else r
        logit += physical_time_bias(r_t)               # 3.7: [B,H,N,N,T], 80 参数
    logit = logit.masked_fill(~allowed[:, None, :, :, None], -1e4)
    alpha = softmax(logit.flatten(-2), -1).reshape_as(logit)   # 邻居×时刻联合归一
    msg = v[:, :, None] + W_e(e)[..., None, :]               # 消息 = 内容 + 关系
    o = W_o((alpha[..., None] * msg).sum(dim=(3, 4)))

    g = sigmoid(social_gate(cat(a, o)))
    a_hat = LN(a + g * o)
    return LN(h + token_feedback(a_hat)[:, :, None])  # [B,N,T,d] → [B,T*N,d]
```

### 5.2 训练与推理

- 编码端不引入新损失，沿用 B0 的训练目标。不加先验对齐损失（TPK 的 KL 项），因为本方案不要求注意力与运动学风险一致。
- 训练与推理的支撑集、关系通道计算完全相同，没有采样或随机剪枝。
- C2 的 W 零初始化；从 B0 检查点热启动时，第 0 步的输出与 B0 逐位相同。正式对照一律从头训练，不用热启动，避免“额外微调”混杂（PB1 报告的第 3 节已经观察到这种漂移）。

### 5.3 与现有代码的对应

| 定义 | 文件 | 位置 |
| --- | --- | --- |
| 支撑集 𝒩\_i | `ASTRA_ACCURACY_Q20/models/stage1_graph.py` | `Stage1GraphBuilder.build`、`_adaptive`、`_risk_history` |
| 关系通道 r | `FRONT_ENCODER_SERIES/front_encoder_models.py` | `relation_features` |
| 关系 token ē | 同上 | `relation_tokens`（`use_multiscale_relation`） |
| 跨时刻注意力 | 同上 | `temporal_social_attention`（`use_temporal_social_cross`） |
| 门控回注 | 同上 | `forward` 中 `social_gate`、`token_feedback` |
| C2 偏置 | `01_early_encoders_and_f16/PHYSICAL_ATTENTION_BIAS_SERIES/physical_bias_models.py` | `FactorizedPhysicalTimeBias`、`pb1_time_bias_add` |
| C2-lag | 新增 | 在 `relation_features` 旁加一个以 p\_i(T)、v\_i(T) 为参考的版本 |

前两列的路径基于 `research_archive/26_advisor_3dir/EXP_ADVISOR_3DIR_20261121/code/sealed/`，最后两项除外。PB1 挂在旧版 EXP-146B 底座上，移植到 B0 时需要确认 `_validate_locked_f16` 的检查条件仍成立。

### 5.4 超参数

| 参数 | 取值 | 来源 |
| --- | --- | --- |
| 半径 R | 6.0 m | `motif_rbf_official_4p8.yaml` `GRAPH.RADIUS` |
| K\_min / K\_max | 2 / 6 | `GRAPH.K_MIN` / `GRAPH.K_MAX` |
| 自适应阈值 | 0.35 | `GRAPH.SCORE_THRESHOLD` |
| 风险权重 λ\_d, λ\_c, λ\_t, λ\_a, λ\_dca | 0.25, 0.20, 0.20, 0.15, 0.20 | `stage1_graph.py` 默认值 |
| 尺度 σ\_d, σ\_t, σ\_dca, closing | 2.0, 2.0, 1.0, 1.5 | 同上 |
| 历史权重 γ / TTC 上限 / DCA 时域 | 0.25 / 8 s / 4.8 s | 同上 |
| d / H / FFN | 100 / 4 / 192 | `astra_model.py` 第 136 行；yaml `NHEAD`、`DIM_FEEDFORWARD` |
| C2：RBF 个数 / σ / 偏置上限 B / 接近门温度 | 5 / 0.25、0.5 / 2.0 / 0.2 m/s | `physical_bias_models.py` |

## 6. 验证计划

顺序为：零训练预检 → C0 七臂消融 → 效率缩放 → SDD 分层。只有判据通过的臂进入五折。各项都需要你本地的检查点、数据和训练环境。

### 6.1 【已完成】形成方案所依据的实验

EXP-172 五折编码器消融；archive 32 EXP-TAU C0；archive 33 E-1/E-2；archive 34 E-3/E-5；CSNA 与 EXP-169B 社交反事实审计；PHYSICAL\_ATTENTION\_BIAS（PB1–PB3）；PAIR\_AWARE\_ATTENTION（KDTA 等）；ATTENTION\_COMPUTE 与 SPARSE\_EXECUTION。结论见 2.1 节。

### 6.2 【待执行】V0：零训练预检

- **V0-a 支撑集统计。** 在 C0 开发集和 SDD 上报告每行宽度分布、达到上限 6 的行占比、半径内候选数 |𝒞\_i| 的分布。用来确定“预算”在哪个数据集上真正起作用。
- **V0-b 描述性诊断。** 在 B0 检查点上计算每个邻居的注意力总质量 Σ\_t α\_ij(t) 与 ρ̄\_ij 的 Spearman 相关。只用于提出假设，不作因果结论。
- **V0-c 等价性。** C2 零初始化时，在 64 个窗口上与 B0 的输出最大绝对差 ≤ 1e-6。

### 6.3 【待执行】V1：C0 七臂消融

统一设置：C0 的 21,719 个训练窗和 4,753 个开发窗，种子 42–44，批量 64，64 轮，最终 EMA 权重，解码端统一为 H3，全部从头训练。

| 臂 | 内容 | 回答的问题 |
| --- | --- | --- |
| E0 | B0（F16） | 参照 |
| E1 | E0 去掉关系 token（β 与 W\_eē 置零） | “运动学描述”在 C0 上是否成立 |
| E2 | 摘要式社交注意力 + 关系 token（f4，无跨时刻） | K3 的多种子复核 |
| E3 | E0 + C2 | 逐时刻偏置是否有效 |
| E4 | E0 + C2-lag | 同时刻与滞后版本哪个更好 |
| E5 | E0 的支撑集换成半径内全连接 | 预算的精度代价 |
| E6 | E0 的支撑集换成同宽度距离 top-k | 填充规则是否重要 |

**判据**（收益以误差降低为正，区间为轨迹聚类 bootstrap）：

| 编号 | 比较 | 通过条件 |
| --- | --- | --- |
| G1 | E0 对 E1 | minFDE 收益 ≥ 1.5%，95% 区间下限 > 0 |
| G2 | E3 对 E0 | minFDE 收益 ≥ 0.75% 且区间下限 > 0；或 Top-1 FDE 收益 ≥ 2% 且 minFDE 变差不超过 0.3% |
| G3 | E0、E3 对 E2 | 若 E2 比 E0 和 E3 都好超过 0.5%（区间不含 0），则不保留跨时刻结构，C2 一并放弃 |
| G4 | E5 对 E0 |  |
| G5 | E6 对 E0 | 只报告；若 |
| G6 | E4 对 E3 | 收益 ≥ 0.5% 才改用 C2-lag |

G1 与 G4 支撑 C1 的主张；G2 与 G3 决定 C2 是否写入方法。通过的臂与 E0、E2 一起进入五折和 SDD，五折单独报告 ETH。

**操纵检查：**

- MC-1：E1 中 β 与 W\_eē 恒为 0，其余参数与 E0 同名同形。
- MC-2：E3 训练后平均 |b| ≥ 0.05；对每对 (i, j)，std\_t(b) / (mean\_t|b| + 1e-6) 的均值 ≥ 0.2；注意力在时间上的 argmax 相对 E0 改变的行占比 ≥ 10%。
- MC-3：E5 每行宽度等于 |𝒞\_i| + 1；E6 每行宽度与 E0 逐行相同。
- MC-4：V0-c 通过。

### 6.4 【待执行】V2：效率缩放

合成场景 N ∈ {8, 16, 32, 64, 128, 256}，批量 1 和 64。比较带掩码稠密、边列表稀疏执行、E5 稠密三种实现的注意力元素数、峰值显存和延迟。只有延迟出现交叉点时，才在论文中提延迟。

### 6.5 【待执行】V3：SDD 分层

按 |𝒞\_i| 分为 ≤ 3、4–6、7–12、> 12 四档，比较 E0、E5、E6。若只有在 |𝒞\_i| > K\_max 时填充规则才拉开差距，则支持解释 (a)；若各档差距相近，则更倾向 (b) 或 (c)。

### 6.6 设计自检

| 项 | 结果 |
| --- | --- |
| 1. 干预隔离 | E1 同时去掉偏置和消息两条通路，参数少约 1.6 万，容量差异在结论中注明；E5 增加了计算但不增参数；E6 只改填充规则、宽度逐行一致；E3 只增加 80 个参数且零初始化。各项由 MC-1 至 MC-4 检查 |
| 2. 同义反复 | 不在训练集上评估；不用模型自己的注意力分箱去证明注意力有效（V0-b 只作描述）；C2 的超参数直接沿用 PB1，不在 C0 开发集上调 |
| 3. 依赖结构 | C0 开发集 4,753 窗 / 248 个轨迹簇；报告前 10 簇占比；轨迹聚类 bootstrap 4,000 次；逐录制留一与去掉 ZARA02 的敏感性 |
| 4. 相关≠因果 | V1 各臂是干预实验，结论在 C0 内按因果解读；V0-b 与注意力可视化只作描述；K7 的相关系数只用于提出“运动学不应决定看谁”的假设 |
| 5. 参数完整 | 全部阈值、上限、分档见 3.3 节与 5.4 节，并对应到文件；2.1 节的数字均当场查阅 |
| 6. 预演 | V0 需要本地检查点，由你执行；本文未用替代数据预演 |
| 7. 非预期原因 | G1：收益可能部分来自参数量差，可加一臂用等参数的随机常数替换 r。G2：收益可能来自时间维上的任意位置偏置而非运动学，用“只依赖 t 的可学习偏置”（80 参数以内）排除。G4：ETH/UCY 候选本来就少，全连接与稀疏接近不等于预算在高密度下也无损，由 V3 补充。G3：E2 比 E0 好可能只是优化更稳定，应同时报告各种子的训练曲线 |
| 8. 混杂 | E1 参数更少（见第 1 项）；E5 计算更多、每步耗时更长，但各臂轮数与步数相同；E2 与 E0 参数量相同（EXP-172 已记录）；数据与解码端各臂一致 |

## 7. 风险与备选

| 风险 | 早期信号 | 备选 |
| --- | --- | --- |
| R1 评审认为 C1 只是“边特征 + kNN 图”，很常见 | — | 回应落在角色划分上：支撑集在学习计算之前、以固定宽度确定（对比 SGCN/ART/ViTE 先算稠密分数）；TTC、DCA 等预期性量只描述、不选择（对比 Polar Collision Grids 与 TPK）；并给出 K4–K7 的反证 |
| R2 C2 无效 | G2 未过；MC-2 显示 b 几乎不随 t 变化 | C2 降为附录消融，编码端主张只保留 C1；参数不变 |
| R3 跨时刻结构不如摘要式（K3 复现） | G3 触发 | 改用 E2 作为编码器，C2 放弃。C1 不受影响，因为它只依赖支撑集和关系编码 |
| R4 预算在高密度场景下明显损失精度 | V3 中 | 𝒞\_i |
| R5 “描述”的收益在 C0 上复现不了 | G1 未过 | K1 只是五折 seed 1；先补 EXP-172 的多种子。若仍不成立，C1 的表述收缩为“预算可预知的稀疏交互”，不再强调运动学 |
| R6 与 STP（TPAMI 2025）过于接近 | 读全文后发现其交互位置也由闭式量确定 | 把区别放在“运动学只描述”与 C2 上；在 2.4 节补一行对比 |
| R7 开题时被问“稀疏到底快了多少” | — | 如实回答：ETH/UCY 上无延迟优势，价值在于大 N 时的成本上界（K8），并为研究二引入场景 token 预留预算接口 |

**开题时的表述建议：** “现有稀疏交互方法多在稠密打分之后剪枝，运动学先验则多被用作邻居重要性。本研究将二者解耦：由闭式运动学量确定固定宽度的交互支撑集以约束推理成本，同时把 TTC、DCA 等预期性运动学量作为逐时刻的关系编码注入注意力，由学习决定权重。” V1 完成前不报告预期提升数字。

## 8. 参考文献与证据索引

### 8.1 文献（本次调研中实际打开或检索到的页面）

**稀疏交互**

- SGCN: Sparse Graph Convolution Network for Pedestrian Trajectory Prediction（CVPR 2021）— [arXiv:2104.01528](https://arxiv.org/abs/2104.01528)（已读方法与消融）
- Learning Sparse Interaction Graphs of Partially Detected Pedestrians for Trajectory Prediction（GST，RA-L 2022）— [arXiv:2107.07056](https://arxiv.org/abs/2107.07056)
- Multi-Stream Representation Learning for Pedestrian Trajectory Prediction（MSRL，AAAI 2023）— [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/25389)
- ART: Adaptive Relational Transformer for Pedestrian Trajectory Prediction with Temporal-Aware Relations（2026）— [arXiv:2604.03649](https://arxiv.org/abs/2604.03649)（已读方法与消融）
- ViTE: Virtual Graph Trajectory Expert Router for Pedestrian Trajectory Prediction（AAAI 2026）— [arXiv:2511.12214](https://arxiv.org/abs/2511.12214)（已读路由与消融）
- Sparse Trajectory Prediction（STP，TPAMI 2025）— [PubMed 摘要](https://pubmed.ncbi.nlm.nih.gov/41171662/)（**仅摘要，全文待读**）

**运动学与物理先验**

- Universal Power Law Governing Pedestrian Interactions（Karamouzas 等，PRL 2014）— [arXiv:1412.1082](https://arxiv.org/abs/1412.1082)
- Polar Collision Grids（ITSC 2023）— [arXiv:2308.06654](https://arxiv.org/abs/2308.06654)
- TTC-SLSTM: Human trajectory prediction using time-to-collision interaction energy（2023）— [HAL](https://archivesic.ccsd.cnrs.fr/IRIT-ICI/hal-04251961v1)
- Human Trajectory Prediction via Neural Social Physics（NSP，ECCV 2022）— [arXiv:2207.10435](https://arxiv.org/abs/2207.10435)
- SocialCVAE（AAAI 2024）— [arXiv:2402.17339](https://arxiv.org/abs/2402.17339)
- TPK: Trustworthy Trajectory Prediction Integrating Prior Knowledge（IV 2025）— [arXiv:2505.06743](https://arxiv.org/abs/2505.06743)（已读先验接入方式与结论）
- AVGCN（视野约束的注意力）— [arXiv:2101.05682](https://arxiv.org/abs/2101.05682)
- SocialCircle（CVPR 2024）— [arXiv:2310.05370](https://arxiv.org/abs/2310.05370)

**时空交互结构**

- AgentFormer（ICCV 2021）— [arXiv:2103.14023](https://arxiv.org/abs/2103.14023)
- Query-Centric Trajectory Prediction（QCNet，CVPR 2023）— [CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf)
- Reverberation: Learning the Latencies Before Forecasting Trajectories（2025）— [arXiv:2511.11164](https://arxiv.org/abs/2511.11164)（只读了前半部分）

### 8.2 仓库证据索引（1inke/kinematic-traj-forecasting）

| 证据 | 文件 |
| --- | --- |
| K1–K3 五折编码器消融 | `research_archive/01_early_encoders_and_f16/OFFICIAL_FIVEFOLD_ENCODER_ABLATION_SERIES/EXP_172_OFFICIAL_FIVEFOLD_ENCODER_ABLATION/RESULT_ANALYSIS.md` |
| K2 C0 编码器敏感性 | `research_archive/34_tau_encoder_sensitivity/README.md` |
| K4 按碰撞时间选邻 | `research_archive/32_tau_sparse_attention/README.md` |
| K5、K6 选邻对照与少数据 | `research_archive/33_tau_selection_followups/README.md` |
| 图阈值调整 | `research_archive/02_cross_dataset_and_efficiency/MULTI_DATASET_GRAPH_NEIGHBOR_BENCHMARK_20261019/SCHEME1_AND_3_COMPREHENSIVE_BENCHMARK_REPORT.md` |
| K7 交互势与社交收益 | `research_archive/09_misc_studies/COUNTERFACTUAL_SOCIAL_NECESSITY_AUDIT/RESULT_ANALYSIS.md`；`research_archive/02_cross_dataset_and_efficiency/OFFICIAL_SPARSE_SOCIAL_COUNTERFACTUAL_AUDIT/EXP_169B_SPARSE_SOCIAL_COUNTERFACTUAL_AUDIT/RESULT_ANALYSIS.md` |
| K8 计算与显存 | `research_archive/01_early_encoders_and_f16/ATTENTION_COMPUTE_SERIES/FINAL_REPORT.md`；`.../SPARSE_EXECUTION_SERIES/EXPERIMENT_ASSESSMENT.md` |
| 已验证退化的注意力变体 | `.../ATTENTION_COMPUTE_SERIES/LITERATURE_AND_EXPERIMENT_MAP.md` |
| K9 PB1 | `.../PHYSICAL_ATTENTION_BIAS_SERIES/RESULT_ANALYSIS.md`、`physical_bias_models.py` |
| K10 KDTA | `.../PAIR_AWARE_ATTENTION_SERIES/FINAL_REPORT.md` |
| 支撑集实现 | `research_archive/26_advisor_3dir/EXP_ADVISOR_3DIR_20261121/code/sealed/ASTRA_ACCURACY_Q20/models/stage1_graph.py` |
| F16 实现 | 同上目录 `FRONT_ENCODER_SERIES/front_encoder_models.py` |
| 超参数 | 同上目录 `OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/motif_rbf_official_4p8.yaml`；`ASTRA_ACCURACY_Q20/models/astra_model.py` |
