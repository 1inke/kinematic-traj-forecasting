# 可观测信息诊断：步骤1/2完整结果

2026-10-27。状态：`ANALYZED_STEPS12_THEN_IMPLEMENT34`。这是有限诊断结果，不是部署成功、baseline晋升或输入信息论上限。

## 执行与审核

生产worker和exit双SUCCESS：00:52:08.927061至01:04:21.280973，实际732.354秒（12分12秒）。唯一约定半程核验实际01:27:36.0221905执行时已经完成；count=1，不再检查该已完成进程。原68.23分钟是保守吞吐规划，不是实耗。

步骤1完成I/S×三初始化共6个512-update拟合，3072 updates；步骤2完成3个M_full×1024 updates及6个固定64-iteration树。T_perm三个seed按原分层cyclic shift规则均BLOCKED，没有合并层、删窗或改负对照规则。

24个tiny完整读数和27个probe人口（87993个重复model-window，不是87993独立样本）原保存审核全部完成。原worker独立重算完整K20同candidate质量、complete-link与pdist聚簇、五pooled运动学、固定19/STOP和20k bootstrap。新增[completion_review/REVIEW.json](completion_review/REVIEW.json)又核95个done合同、全部保存选择/槽级质量/计数、实际20000个track与seed抽样及数值；1660继承源与所有已登记输出初末SHA一致，并显式锁根级结果/资源/CI/FINALLOCK等。CPU元数据补充核9个神经checkpoint的全部参数有限、统计、嵌入合同、optimizer/RNG/history存在且匹配，以及全部9个probe终态模型SHA。没有模型前向/反向、optimizer更新或外审；新增补充没有再次计算全部SciPy聚簇/pooled-motion，而是验证原完整双实现审核的SHA合同。

实际生产保留峰private=2619637760 bytes（2.440 GiB）、RSS=1681530880（1.566 GiB），全阶段最小available physical=15460147200（14.399 GiB）、commit=18072596480（16.832 GiB）。全部1.3动态remaining及Windows retained-peak守卫PASS。不是固定6GiB门，也未改变OS/pagefile或关闭用户程序。预检峰与本生产峰分开记录。

## 第一步：同一固定tiny128，排序目标可拟合，BCE未通过

所有初始化使用同pool42和同128窗；B非空74、B空54，原native成功82，存在安全rescue机会10。三个初始化是模型重复，不是三个独立数据集。

|方法/seed|最终hit T|hit B（分母74）|B空STOP（分母54）|成功窗|rescue/lost|
|---|---:|---:|---:|---:|---:|
|I/42|58/128|4/74|54/54|82|0/0|
|I/43|58/128|4/74|54/54|82|0/0|
|I/44|56/128|2/74|54/54|82|0/0|
|S/42|123/128|70/74|53/54|90|8/0|
|S/43|126/128|72/74|54/54|90|8/0|
|S/44|126/128|72/74|54/54|90|8/0|

原门hitT≥.95、hitB≥.90、B空STOP≥.90：S三个初始化全部PASS，I全部FAIL。S最终各利用8/10真实rescue机会，tiny覆盖82→90（+6.25pp），不是全体FIT/hold改善。0/32/128/512所有读数均保留：I在32步B空STOP曾仅.815/.796/.759；S在128步成功83/83/85，最终90，不把最终结果倒推为每步都好。所有24读数overall五motion通过；完整分组明细见ALL_TINY_READINGS。

输入字节碰撞审计32661个eligible归一化动作输入均唯一，未见逐行精确冲突。这与S tiny拟合一起反驳“当前输入根本无法拟合这128窗”的强断言，但不能证明整个4096窗可拟合、训练预算充分或能迁移。S与I目标/梯度不同；本轮不据此更换后续共同BCE目标或延长epoch。

## 第二步：异构同输入探针没有带来覆盖增益

三种方法在所有seed、所有split的窗口级good均与native完全相同，rescue=lost=0。

|人口|overall native→所有方法|change native→所有方法|含change tracks|
|---|---:|---:|---:|
|opt|2415/4096→2415/4096|396/977→396/977|原opt固定人口|
|hold（主）|1971/3825→1971/3825|340/1023→340/1023|74|
|inner（次）|932/1856→932/1856|166/497→166/497|35|

**hold全部9个方法seed共34425次决策都为STOP**。因此主结果不是救回与丢失抵消，也不是新路径覆盖与native偶然等计数；原K20路径原样保留。M_full三seed在opt/hold/inner全部STOP。T_full的opt执行0/72/16次，B内0/68/14次；T_geom的opt执行0/51/0次，B内0/45/0次。T_full44在inner另执行1次稳定组B内连续改善。它们都未救回新窗口，保留这些非全STOP反例，不写成所有树从未调整轨迹。

机会并非消失：hold overall三个池有392/401/391个真实rescue机会，change有78/86/93；inner change41/46/38。它们均未利用。树opt43/44极小连续ADE/FDE/J改善不等于覆盖跨阈值；少量有害/无效执行仍在完整ASSIGNMENT中，不隐藏。

主hold-change T_full−M_full及T_full−T_geom三个seed差值全0pp，20000有效draw、0无效，95%/family6校正/固定seed敏感性区间均[0,0]；相对native同样0。learning_scheme_signal=false。family6没有因为T_perm不可运行或步骤3未完成而缩小。T_full−T_perm是BLOCKED而非阴性；不能据此声称观测特征没有增量，或geometry-only已足够。

27人口×4组共108个pooled五motion组门均PASS，native-success lost=0且protected误差门通过。大量PASS来自STOP不变，不是具备安全改善能力的证据；所有major覆盖/ADE/FDE增益门未获得支持，baseline不变。

## ARS反方审核（同助手inline，非独立外审）

**DA2—证据解释：PASS，有重大限制。** 最强反证是S已经在固定tiny救回8窗，故第二步BCE/树阴性不能推广成“输入没有信息”。另一个解释是共同IPW有限动作训练与原全动作目标差异、固定训练预算/树容量、目标中大量连续微增益和STOP的排序优势，而非唯一观测不足。未增加variant或偷偷换rescue-only标签；这些解释保持未识别。

**DA3—最终压力测试：PASS，结论严格限本合同。** held全STOP确实解释覆盖0，但不揭示其唯一优化成因；负对照不可用不能当负结果。tiny可记忆不保证迁移，3个seed/相关窗口/F16可能已见hold与inner/反复开发限制均保留。没有因统计零区间声称模型等价或未来不可预测。完整token与显式邻居下一步仍须独立来源和预检；不会把历史局部worker的NOT_AUTHORIZED字段当成人类未授权。

## 下一步

按最新人类授权实际实施第三步：共同42208参数、完整token/合法过去邻居mask消融，原T/动作表/IPW/训练协议不变。若邻居无法从观测8时刻唯一合法重建，neighbor/both整arm BLOCKED，不删样本；base/token继续。第三步完整审核分析后执行第四步固定近邻条件歧义CPU审计。所有内容保留，不删除文件、失败或自动记录，不开启额外局部变体。

本报告为AI辅助代码与研究证据整理；原科学输出和失败来源未改。
