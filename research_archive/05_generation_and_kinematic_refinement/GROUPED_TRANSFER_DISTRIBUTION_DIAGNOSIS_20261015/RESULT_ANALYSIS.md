# 分组迁移后的输入分布与共同支持诊断

2026-10-15，状态 ANALYZED。本轮人类明确回复“继续”后恢复同一协议。计算176.669秒完成，资源保护、源与输出校验、保存算术复核均通过；旧资源失败来源保持原样。

## 这一步改变了什么判断

当前证据不足以用“CAL整体离开FIT输入分布”单独解释迁移失败。按已固定的历史运动、native候选和token摘要特征，原FIT→CAL的支持外比例约5%–7%，接近源校准集预定的5%。Zara留出人群的候选几何偏移更明显，但双方都支持的人群里，grouped相对random的完整候选覆盖与J差距仍然保留。

研究分支因此细化为：**分布/暴露差异存在，但下一优先级是条件映射的迁移稳定性和成功样例保护。** 这不证明共享head是唯一原因，也未给loss或优化过程免责。单纯增加归一化、重新评分或增加路由专家缺乏新的决定性依据。

## 原FIT与CAL是否处于大幅分离的区域

支持范围的定义完全固定：源训练track以SHA256 tag分成约80%参考、20%校准；统计只由源训练拟合；查询近邻排除同(sequence,pedestrian)目标；源校准距离95分位作阈值。下面数字是越过该几何阈值的窗口比例，既非不可预测率，也非可学性上界。

|源→目标|历史运动10维|native候选10维|frozen token时间均值96维|合并116维|
|---|---:|---:|---:|---:|
|原FIT25912→CAL2879|5.245%|5.280%|6.704%|6.009%|
|只用university FIT→CAL|6.877%|7.259%|6.252%|6.322%|
|场景实验random→zara outer|2.883%|6.278%|2.883%|2.627%|
|场景实验grouped→zara outer|3.395%|11.531%|4.292%|3.908%|

原FIT→CAL的22个标量描述中，最大绝对pooled-SMD约.167：observed displacement1.634→1.432m，mean speed .594→.521m/s，last speed .588→.516m/s。仍存在可量化差异，但这套特征没有呈现早期另一General/Challenge DSA中的大幅分离；那份审计的人群和输入定义不同，不能搬用其AUROC或SMD来解释本CAL。

只在university内部比较，native候选平均acceleration .08413→.08970m/s²、jerk .46274→.49332m/s³，pooled-SMD约.276；说明同场景族内部仍有候选几何差异。支持外比例接近源基准不能证明条件分布一致，也不能排除密度、邻居、时序或模型未充分利用的信息。

## Zara差异更集中在哪个输入族

Grouped训练完全不含zara，native候选10维的支持外比例11.531%，random为6.278%。Grouped源→zara的若干标量pooled-SMD：endpoint-speed离散度+.862、candidate mean acceleration/jerk各约+.672、candidate-CV endpoint gap的95分位+.631、candidate mean speed+.511。Random也有同方向偏移，但多数较小。

因此候选状态分布是值得保留的信号。它不能被解读成“模型没有观察到转向”或“新的encoder已经失败”；输入中含native预测的确定函数，token只使用96维时间均值，并没有重放完整head表示或验证全部时序信息。

合并116维的欧氏距离受96维token摘要影响，不能以合并结果覆盖native单族的偏移。四种预先固定的定义均完整保存，未挑选阈值或只展示最有利的一族。

## 去掉支持外窗口，差距消失了吗

没有消失。以下均为同一outer ID、固定epoch80、三seed均值，grouped减random。负覆盖表示grouped较少好窗口。条件子集是描述性分析，没有新置信区间或显著性检验；其支持率也由不同训练源估计，存在选择偏差。

|实验与人群|N / tracks|整体覆盖差pp|变化组覆盖差pp|整体J差|
|---|---:|---:|---:|---:|
|目标隔离，全体|1035 / 191|-0.870|-2.240|+.03568|
|目标隔离，合并输入双方支持|984 / 190|-0.813|-1.541|+.03083|
|时间隔离，全体|566 / 154|-3.357|-5.634|+.06291|
|时间隔离，合并输入双方支持|531 / 150|-3.139|-4.580|+.06233|
|场景隔离，全体|1561 / 309|-2.178|-4.107|+.06292|
|场景隔离，合并输入双方支持|1495 / 307|-2.074|-4.529|+.05876|
|场景隔离，native候选双方支持|1366 / 280|-1.977|-3.013|+.05881|

最后一行变化子集177windows/41tracks：native完整覆盖42.373%，random40.301%，grouped37.288%。即使这些候选状态处于双方训练参考范围内，两个适应head也未稳定超过native，grouped比random更弱。这个描述与“全部损害只来自支持外的少数窗口”不一致。

Native候选支持外的场景变化子集仅31windows/19tracks，差距约-10.753pp，但触发预定THIN标记，不能用它认定主要因果来源。完整240条support-stratum结果包含所有小样本和反例，未删除它们。

## 必须保留的CAL反例

场景实验CAL变化700windows/73tracks，全体random31.381%、grouped33.095%、native34.714%；grouped较random好1.714pp，却仍低于native。合并输入双方支持的638windows/70tracks中，random31.766%、grouped33.804%、native35.266%，grouped-random为+2.038pp。故zara上的random优势并非跨所有目标集普遍成立。

近期固定motion责任T1在FIT本身也比dynamic对照弱，且两臂五项motion未全部通过。这进一步限制“只剩CAL分布shift”的解释。其STOP仍生效，本轮未触发T2/T3。

## 三项研究的完整复盘

详见 [FULL_THREE_STUDY_REVIEW.md](FULL_THREE_STUDY_REVIEW.md)。本次重新核对171个保存科学job的samecandidate ADE/FDE/Q/J/coverage，最大误差0；核对648条组记录、720条原探索CI、18条完整80epoch历史、54模型与342输出的原SHA。共1051个读入输入/科学source/结果/模型/历史/文档文件在初末核验不变。

原SciPy GT-free complete-link聚簇和五项pooled motion使用各实验已有ARITHMETIC审计及其保存输出SHA。本轮不声称重新聚簇全部输出、重算全部motion、重新采样720CI或重放神经模型/梯度。另有 SUMMARY_AUDIT.json 独立复核保存标量、子集分区、rescue-lost恒等式、motion布尔及生产verify分支对坏SHA的真实拒绝；它不是外部独立评审。

本轮峰private1.02388GiB，RSS1.04446GiB；运行期间physical余量最低20.6811GiB、commit26.9007GiB，原process6GiB及各reserve6GiB限制保持。旧attempt1的physical1.6095GiB资源失败仅是历史来源，不代表当前阻碍，也没有被删除或伪称科学结果。

## 后续方向与决策约束

当前应优先研究“相似可观测条件下，head如何把训练信号稳定转成完整正确路径”，包括跨样例共享更新、candidate条件映射以及成功窗口的保护；这属于用户既定的共享映射/迁移分支。结果尚未唯一指定gating、MoE或某种新loss。既有candidate路由、中心化/标准化、fixed responsibility等失败必须作为下一精确对照的反证，避免重命名重试。

完成本冻结诊断及三项总复盘后，本轮计算结束；未自动启动第四个分组训练、失败T1的条件阶段或旧暂停自动化。Coverage-first、GT-free有效good次之、scorer最后；samecandidate .2/.4及全部major/五motion1%保持。

## ARS inline反方2（同助手，非外审）

最强攻击：source95%支持范围几何很宽、仅10/10/96维摘要；“落在支持内”并不等于拥有相同P(Y|X)、同样训练密度或相同可学映射。本报告只排除一种过强解释：当前损害全部由这套几何定义的明显输入新颖窗口造成。没有宣称消除所有domain shift或证明shared参数因果。子集只描述、不增加CI，保留native原生比较与CAL反例。

## ARS inline反方3（同助手，非外审）

完整性检查：三项变化覆盖原主区间仍全部跨0；原720区间没有多重确认性校正，不能把新support子集当确认性显著结果。三项outer不同不池化成人群；3seed不是3dataset，frozen F16已见数据、窗口与上下文相关、场景极不均、时间token均值丢信息等局限保留。没有新预测器训练或baseline升级。本文由AI辅助本地证据整理和计算，引用均为保存的项目结果文件。
