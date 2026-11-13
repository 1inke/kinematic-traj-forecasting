# 第四步：计算审核通过，但条件歧义结论支持不足

2026-10-27｜`ANALYZED_STEP4_UNDERPOWERED_FINITE_SERIES_COMPLETE`。这是原固定距离与人口下的保存数据审计，不是新模型训练、覆盖率改进实验、Bayes 下界或信息论不可预测证明。前三步见[四步总报告](../RESULT_ANALYSIS.md)。

## 完成与证据边界

worker 与 exit 双 SUCCESS：02:38:01.322529 至 02:40:28.874505，实际147.551976秒（2分28秒）。唯一预计半程核验 actual02:49:52.9054597（scheduled02:48:35.837）已确认 taskReady、processnone、XML0、stderr0，count1；原02:59:11预计完成节点作废，不再查询该已完成进程/任务/日志。

全部三个池各8117个 calibration/query 短名单 pair，共24351个pair、48702个科学20×20分配问题，未超过49152上限。另有48702个反序匹配及974040个禁边替代问题作为额外证书审核，不把它们混为新增科学人口。原第二种 Hungarian primal-dual 实现独立核全部最优代价，误差0；全部映射双射、来源、完整20 source-index时间隔离及无GT短名单审核通过。固定 SciPy 版本和全部映射保存在每seed的 SELECTION/MATCHES；没有扰动代价挑解。

02:40:04.206068已锁定全部三池的GT-free选择/τ/输出SHA，再开始held B/quality/GT事后读取。[ALL_SELECTIONS_LOCK.json](ALL_SELECTIONS_LOCK.json)和posthoc done合同绑定这一顺序。所有7921 opt+hold IDs保留；未知时间不配对，但不是删除样本或宣称其不可预测。

原保存审核全通过；新增[completion_review/REVIEW.json](completion_review/REVIEW.json)又核全部7个科学done、全部登记输出和3536继承source的初末SHA、11个根级结果/资源/LOCK/统计文件SHA；重新显式构造每个被用窗口的401个最终K20（115/117/115个不同质量窗口，按池计），quality最大误差0、B与literal原标准EXACT；498条primary+reverse事后pair全部共同域、全B集合、映射、J regret独立复算。16项统计的每项20000实际draw均再次以track重数复算，最大误差6.67e-16以内，无效分母不补抽。

本补充没有重新运行匹配求解器、旧模型、Torch、前后向或optimizer；最优性证书引用已核SHA的原完整第二算法审核，不冒称又作第三次求解。新补充是显式新增CPU证据，不更换3536原锁源或旧digest，也不是外部审稿。

## 支持门：校准合格，query不足

每池128条calibration reference均有合法短名单，τ为各自最近距离的linear25%分位数。固定128条query track各只取预登记哈希1窗，其中稳定78、过渡26、变化24，不按GT改选。

|池seed|τ|支持query/128|支持变化query/24|最终pair|两边共同B非空pair|对应query/变化query|
|---|---:|---:|---:|---:|---:|---:|
|42|1.060994334|34|10|82|14|10 / 4|
|43|1.063844833|34|10|84|30|12 / 2|
|44|1.065444206|34|10|83|24|12 / 2|

每池均有92条query因为没有pair满足固定τ而无支持，另外2条query时间未知。不是94条“未来歧义”；也不是94条没有可改进动作。全人口时间未知为opt17/hold32，本次哈希query只命中其中2条。

原门要求支持overall≥50、change≥20。当前34/10，故**所有seed的overall与change均UNDERPOWERED**。没有扩大τ、更换query/窗口、合并人口、放宽时间排除或用inner补样本。稳定支持21/21/20，过渡3/3/4，变化10/10/10；全部seed×mode×group分母和事件见[ALL_GROUPS.json](completion_review/ALL_GROUPS.json)。反序匹配的阈值、支持和选择集合与primary相同，未择有利版本。

## 事后冲突及正反例：只能保留为薄层描述

|seed|两边B非空：冲突 / 有共同正动作|两边原B均空|仅一边原B为空|两边原B非空但共同域清空|
|---|---:|---:|---:|---:|
|42|9 / 5（共14）|24|44|0|
|43|20 / 10（共30）|15|39|0|
|44|17 / 7（共24）|24|35|0|

以上是pair计数，不是独立样本比例。按每query先平均、再按query track汇总，overall条件冲突点读数为.650/.6875/.743056；固定三池均值.693519，描述95%CI [.481270,.887503]，有效19999/20000（1次无效）。**即使区间看似不含0，也因整体支持不足而不得升级为稳健总体结论。** 条件分母只有10/12/12条query，并非全部128或全部34条支持query。

变化组冲突pair为6/8、2/3、3/4，两边B非空仅对应4/2/2条query；组合bootstrap仅17433次有效、2567次无效（12.835%），不补抽，更不据此判断变化组未来不可辨识。所有普通描述CI保留于[statistics/SUMMARY.json](statistics/SUMMARY.json)，不扩充或替代步骤2/3固定family6主检验。

whole-query平均J双样本折中regret为.006161/.011321/.019402，固定三池均值.012295；变化组.005381/0/.002047。它只是在有限共同eligible动作域、已知两个未来下的事后折中，不是模型性能、Bayes风险或理论下界。

更重要的反证见[CONFLICT_STRATA.json](completion_review/CONFLICT_STRATA.json)：

- 冲突pair中，至少一边native本来成功的有9/9、18/20、15/17；原B包含保留成功后的连续微改善，不等同于跨覆盖阈值rescue。
- 所有选定pair中，**两边同时拥有真实rescue机会的pair为0**，不能用本轮冲突直接证明两边救回目标互不兼容。
- 冲突pair里J regret literal恰好0的有5/9、6/20、10/17；反过来，有共同B的pair也可能无法同时达到两边各自最小J。B集合交集与J最优折中是不同问题。
- 整体共同域没有把两边原B均非空的pair全部清空，但会删去部分正动作，不能把“清空事件0”说成“B从未被准入限制”。STOP始终在共同域且安全；B不交不意味着不存在共同安全动作。

固定哈希示例来自[HASH_EXAMPLES.json](completion_review/HASH_EXAMPLES.json)，不是挑最戏剧化样本：seed42/43 query11092与library18107，dH=.561881、dP=.825750/.812653，query/library共同B分别12/33和9/31，交集0、regret0；两边并无同时rescue机会。相反seed43 query15077与11166，共同B各20、交集20、regret0，未来平均距离约.01053m。seed44 query25657与20822有1个共同正动作，但regret约.29633。完整正反例、ID和原生成功状态保留，不用单个oracle槽替代全集。

正反序在所有primary支持pair的事件、overlap、regret上均未发现ALIGNMENT_SENSITIVE，所选pair对称差0；选定pair的数值最优歧义标记0。此结果不把保守数值证书升级为数学唯一性，也不解决距离不是充分统计量的问题。相似过去/候选下未来可不同，但本次query并非完全相同输入，平均未来距离的描述不能当条件熵估计。

## 资源与执行

生产记录的保留private峰227356672bytes（约.212GiB）、RSS247439360（.230GiB），最小available physical17584250880、commit20602404864bytes。全阶段原E=2GiB×1.3动态remaining与Windows保留峰门PASS；新补充自身资源另存，不代替生产峰。没有改OS/pagefile、关闭用户程序、安装依赖或删除任何东西。

## ARS反方审核——同助手inline，不是外审

DA2：PASS，但科学解释必须降为支持不足。最强反证是34/10低于注册门，条件冲突只有10–12/2–4条query，且没有两边同时rescuable的pair。多数冲突涉及原成功窗的连续微改善，B不交与覆盖不可兼容不是同义；高条件比例不能拿来支持“历史没有信息”。已将所有条件分母、零regret反例、空机会及无支持分别报告。

DA3：PASS，有限审计可以结案，强信息结论不通过。反序一致与独立匹配审核增强的是计算可靠性，不是距离语义充分性。每个过去通常只有一个真实未来，library固定、三个池相关、数据反复开发/F16可能已见hold；CI不覆盖library抽样或整个生成器的不确定性。没有为了克服UNDERPOWERED而扫τ/换人口，未改major、lost1%、protected1%或五motion1%门。

## 结束分支

第四步完整执行且保存审核PASS，科学结论`UNDERPOWERED_CONDITIONAL_AMBIGUITY_UNRESOLVED`。连同前三步完成本有限四步系列，汇总报告后停止仅本次自动跟进的后续调度并保留记录。无新loss/头/teacher/scorer/解冻/新数据/局部扫描的自动后续授权；baseline、historycycle11和长期未完成目标不变。所有原科学文件、失败与日志均保留。

本报告为AI辅助本地代码与证据整理；ARS审核为同助手反方检查。
