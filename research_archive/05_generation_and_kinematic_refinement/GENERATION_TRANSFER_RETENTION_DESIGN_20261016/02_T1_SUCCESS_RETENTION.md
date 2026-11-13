# T1：原生成功保留监督能否迁移？

状态：DESIGN_ONLY。6个fresh runs：A/B×seed42/43/44；其他设置见共同协议。

## 1. 假设与唯一因素

U1支持原生成功保留有跨样例缺口，但不证明一定是缺少loss项。本实验直接干预这一假设：

> 在不改变输入、head、输出算子、候选预算和基础任务的条件下，给训练内原生成功增加集合保留监督，是否减少outer原生成功丢失，而且不只是让输出退回native？

A：L0。B：L0+Lkeep。固定系数1，固定一次，不在outer/CAL扫权重。系数1是按下述米制量纲作出的设计选择，不代表理论最优。阴性只能否定本次实现/预算下的有效性，不否定所有保留目标。

## 2. 目标精确定义

对FIT每个窗口，以immutable native20和FIT GT预先计算：

\[
J^0_{ik}=\max(ADE(p^0_{ik},y_i)/0.2,FDE(p^0_{ik},y_i)/0.4),\quad
G_i^0=\{k:J^0_{ik}\le1\},\quad m_i^0=\min_k J^0_{ik}.
\]

在新输出上计算同一J：

\[
\ell_i^{keep}=\begin{cases}
0.20\,[\min_{k\in G_i^0}J(\hat p_{ik},y_i)-m_i^0]_+, &G_i^0\ne\varnothing,\\
0,&G_i^0=\varnothing,
\end{cases}\qquad L_{keep}=B^{-1}\sum_i\ell_i^{keep}.
\]

- 0.20把无量纲J换成米制误差；L0也保持原米制。不把它误写成额外的可调系数。
- 分母是整个batch B，不是成功窗口数，也不是G0候选数。不同batch成功比例自然进入期望，不改变采样。
- native mask、m0均detach；只取原good集合中当前最好一条作为保留代表，不强迫全部20条或全部G0都不能改变。
- 比较native原有质量而非仅J=1边界，避免把原先有余量的成功推到临界点。保护余量也可能压缩可修复空间，属于本实验要测的代价。
- J组件精确tie按ADE-first；候选min精确tie按physical slot升序；ReLU在0导数0。近tie只标记数值敏感，不改科学阈值。
- GT只用于FIT训练loss及posthoc。G0、j0、m0、group/scene/ID/logit均不得成为forward输入；不能部署时依据GT回退native。

保留的是“至少一个旧成功来源的质量”，不是新建selector，也不是R1–R5的可修复性标签/GT修正teacher，更不是对未来误差做预测训练。G0条件可能导致偏向native容易样本，所以必须把救回率作为共同终点。

## 3. 强制记录

每epoch保存原L0、quality、diversity、keep、batch-size加权总目标；成功窗口占比、keep激活率、G0大小、keep负责slot、Qwinner与keep slot交叉表、J的ADE/FDE活动组件/ties。

保存梯度总范数及少量预注册固定FIT诊断batch的L0/keep输出坐标梯度夹角与范数；它们是诊断，不用于自适应权重或optimizer改写。不要把output-coordinate gradient当参数梯度或局部角度当有限AdamW因果证明。

原生失败窗口上的keep必须严格为0；替换posthoc GT不得改变已经生成的预测；零初始化全部native identity。对空G0不要执行inf−inf后乘0，必须先过滤有效窗口。

## 4. 预注册判定

主对比B−A，outer change：

**RETENTION_SIGNAL**要求同时满足：

1. lost rate平均下降至少3pp，三个seed均下降，family24差区间上界<0；
2. rescue rate平均不下降，且family24差区间下界≥−2pp；这是局部非劣界，不是救回已证明相同；
3. coverage三seed不下降；不是只凭loss下降；
4. 不产生NaN/源码/身份异常；所有运动学与protected group结果完整，不隐去新伤害。

3pp和2pp是事前机制筛选阈值，不是旧重大成功阈值；若CI太宽，标记INCONCLUSIVE而非降低门。

结果解释：

| 结果 | 结论与下一步 |
|---|---|
| FIT/outer均改善保留，救回保住 | 目标干预有迁移信号；可在T2有信号后进入D |
| FIT改善、outer无改善 | 该监督的泛化仍不足；不扫系数，继续独立T2 |
| lost减少、rescue显著下降/输出近identity | 稳定性—可塑性折中，不是有效新生成；不进入D |
| lost仍高但coverage上升 | 不能满足保护目标，报告交换而非成功 |
| motion继续失败 | 可记录机制信号，绝不升级模型；不得新增物理loss掩盖本因素 |

“近identity”不另设可调门：完整报告残差均值/分位数、与native逐数组相等比例、救回数，按上述救回条件决定。不得事后挑选最好seed或最低loss epoch。

## 5. 最强反驳

Lkeep可能只是一般正则，而不是学到“应当保护谁”。本实验只能支持这个具体监督整体的作用，不能区分样本选择、梯度大小、margin目标的全部机制。当前不再添加一整套正则强度扫描。若保留阳性却救回下降，应接受反证，不以“以后调好系数就行”延续。
