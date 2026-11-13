# 分组迁移后的分布诊断（事前固定分析，2026-10-15）

人类本轮授权：按前面确定的场景/运动状态分布差异方向继续。先完成既有三项分组实验的证据总复盘，再执行本冻结缓存诊断。原有静态科学文件、模型与结果保持原值；本目录保存全部新代码、结果、失败来源与报告。

## 问题与范围

检验目标/时间/场景隔离后的表现差距，是否与可观测历史、immutable native candidate 或既有 encoder token 的分布距离有关，以及差距在固定共同支持范围内是否仍然存在。只读取既有 FIT25912、原CAL2879及三项已锁定split/保存输出。标签只用于事后完整质量核验；任何描述输入均无GT、group、logits、身份或场景标签。身份只用于排除同track近邻和汇总分母。单纯分布差异不能认定因果，也不能断言encoder根因。

历史 General/Challenge DSA、official-fold shift审计的样本人群不同，不替代本诊断。2026-10-15固定motion责任T1已失败，不重启其条件T2/T3；它在FIT本身也较弱，故不将所有失败归于CAL迁移。

## 固定分析

1. 三项完整复盘：全部648个arm/seed/policy/split/group行、全部已保存bootstrap（180+360+180=720）、18条80epoch历史、54模型/342输出SHA；核验各实验双SUCCESS和原ARITHMETIC/FROZEN/PRESERVATION。逐保存科学job重新从轨迹和原GT算同candidate ADE/FDE/Q/J/coverage，核对stored字段与组汇总。原聚簇/五项motion算术使用已有审计及SHA，本轮不声称重新运行全部聚簇、motion或模型。
2. 固定三个输入族：原past8的7个速度/6个加速度统计、最后观测运动方向；native K20完整12点相对last-v的CV偏差、速度/endpoint/path离散度及加速度/jerk；原frozen8tokens的96维时间均值。dt=.4，坐标来源为原prepared past及native预测；feature计算不读未来标签。量化各arm训练→同outer、训练→原CAL、原FIT→CAL及university-only FIT→CAL的mean/std/分位/Wasserstein、source/pooled标准差SMD。各源统计独立且只用源训练ID，不以目标表现选feature。
3. 多变量支持：每输入族及合并输入，source track用固定SHA256tag划80%参考/20%校准；标准化均值/std只源训练，std floor=.001。KD-tree欧氏距离除sqrt(D)，目标近邻强制排除同(sequence,pedestrian)track；source校准track也完全离开参考。source校准距离95分位为固定支持阈值，报告目标支持率、source校准率、排除身份后的实际最近距离、track数量。源分布本身95%只是一种描述范围，不是真实可学性或数据独立保证。
4. 主arm比较仍在同outer ID。逐source族报告双方都支持、仅grouped不支持等固定子集里的三seed coverage/J差、native→救回/损失和组别分母；子集无新CI或显著性筛选。少于100windows或20tracks标记THIN；change也报告但不凑独立样本。所有target保留，总结同时包含完整人群。支持范围因arm训练样本而变，条件分析有选择偏差；不能称去混杂因果效应。

无predictor训练、无模型前向/反向/optimizer、无超参/特征/分位搜索、无DEV/heldtest/fold/external data。原samecandidate .2/.4、全部major和五motion1%标准保留。

## 验证和资源

运行前固定源与本代码SHA；继承并核对三项注册source、输出、模型digest以及所读原input的旧digest；运行后再次核验所读全部文件。代码fixture覆盖CV零偏差、constantmotion、同track近邻排除、disjoint ADE/FDE不拼路径、badSHA拒绝。IDs显式对齐；不import旧module，不旧输出目录写入。内存process6GiB/systemphysical+commit各reserve6GiB，约17GB持久规划；CPU有界数组/单job，内部resourceguard。失败保留trace并STOP，不调OS/pagefile/kill用户进程。

## 预定分支

- 大幅输入距离且对应更差迁移：支持下一项单因素表示/条件分布控制设计，但不直接启动新神经训练。
- 共同支持人群仍持续恶化：分布距离不足以解释，优先精确学习/共享映射与函数形式对照；不自动改scorer。
- 所有距离或子集证据不足：明确未知，不搜索新feature直到出现想要的结论。

ARS反方1（本助手inline，非外审）：本研究的强攻击是近邻指标、单zara族、F16已见数据和有限优化耦合。本设计用固定输入族/阈值、排同target近邻、全分母及反例报告约束解释；仍保留上下文相关与选择偏差。当前授权已明确，直接完成诊断，不再要求通用研究许可。
