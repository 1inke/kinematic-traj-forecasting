# A0 分析：固定 bottom4 的理想修复机会不足，停止 A1/B

**执行成功，科学分支停止：STOP_FIXED_BOTTOM4_OPPORTUNITY。** 已按指南先完成 A0，不再启动本设计的 A1 安全见证、B 教师路径训练或条件性 V。最新“继续执行”用于完成当前协议的核验，不被理解为绕过预注册停止门。[原始决策](<<WORKSPACE_ROOT>/GENERATION_FEASIBILITY_CONDITIONING_DESIGN_20261015/A0_execution_20261015/DECISION.json>)

## 1. 最关键的发现

FIT 变化组4725窗口/337tracks中，原生有1937个完整好候选窗口（40.9947%）。在同一位置界 b=.20(t/12)^2 内，即使逐点知道GT、取消时间basis和物理约束：

- 只修原固定bottom4：最多2024个（42.8360%），净救回87个，**+1.8413个百分点**。
- 允许全部20槽修正：最多3108个（65.7778%），净救回1171个，**+24.7831个百分点**。
- 固定bottom4只覆盖这组all20放松新增机会的7.43%；另1084个窗口的此类几何机会不在固定bottom4可提供的范围中。

5pp源侧筛选需要至少237个净救回；87个明显不足。精确整数判断是100×87=8700 < 5×4725=23625；不是把近边界结果四舍五入成失败。all20的117100≥23625通过，但不能替代bottom4门。**因此，在这个固定四槽+位置界下继续训练P2新head，不足以实现本设计要求的源侧覆盖增量；加安全约束只会缩小这一本来就偏低的上界。** 这只是受限空间的覆盖上界结论，不是所有生成方法无效的结论。[完整结构化结果](<<WORKSPACE_ROOT>/GENERATION_FEASIBILITY_CONDITIONING_DESIGN_20261015/A0_execution_20261015/RESULT_ANALYSIS.json>)；[保存数据汇总审核](<<WORKSPACE_ROOT>/GENERATION_FEASIBILITY_CONDITIONING_DESIGN_20261015/A0_execution_20261015/completion_review/REVIEW.json>)

这里的5pp是研究空间筛选，既不是相对原生覆盖的5%增幅，也不替代原major的change ADE/FDE各≥5%及全部保护门。

## 2. 所有人口与组别：不只挑变化组

|split/group|n|native good|bottom4 UB good|all20 UB good|bottom4增量pp|all20增量pp|
|---|---:|---:|---:|---:|---:|---:|
|FIT/overall|18909|10858|11000|14996|0.7510|21.8838|
|FIT/stable|10767|7414|7433|9473|0.1765|19.1232|
|FIT/transition|3417|1507|1543|2415|1.0536|26.5730|
|FIT/change|4725|1937|2024|3108|1.8413|24.7831|
|outer/overall|1035|625|631|834|0.5797|20.1932|
|outer/stable|604|434|435|538|0.1656|17.2185|
|outer/transition|178|81|83|130|1.1236|27.5281|
|outer/change|253|110|113|166|1.1858|22.1344|
|inner/overall|1856|932|936|1403|0.2155|25.3772|
|inner/stable|978|601|602|833|0.1022|23.7219|
|inner/transition|381|165|165|282|0.0000|30.7087|
|inner/change|497|166|169|288|0.6036|24.5473|
|original_CAL/overall|2879|1490|1511|2149|0.7294|22.8899|
|original_CAL/stable|1535|994|996|1308|0.1303|20.4560|
|original_CAL/transition|644|253|264|423|1.7081|26.3975|
|original_CAL/change|700|243|251|418|1.1429|25.0000|

仅FIT/change决定分支；outer/inner/CAL是固定规则后的描述性反例核查，没有用于选半径、槽位、阈值或训练臂。其变化组的bottom4增量也只有1.1858/.6036/1.1429pp；all20则为22.1344/24.5473/25.0000pp。这支持机会受固定bottom4限制的解释，但不把重复窗口或单CAL场景当独立泛化证明。每个split/group/port的C、J、ADE、FDE、Q、rescue/lost、Q/J各自同slot早中晚、聚簇与全部motion都在原始和结构化报告中保存。[原始48组指标](<<WORKSPACE_ROOT>/GENERATION_FEASIBILITY_CONDITIONING_DESIGN_20261015/A0_execution_20261015/RESULT.json>)

## 3. 为什么这个结果不是“all20方法有24.78点实际收益”

A0的解析误差下界 dLB=max(||native−GT||−b,0) 可由同一条逐点朝GT的球投影路径同时达到，因此不拼接不同slot的ADE与FDE。对12个独立二维球，它是精确质量最优；对带时间basis、模型参数化或安全约束的路径，仅是乐观上界。

fullK20预算始终不变，UBbottom4另16路径EXACT；native≤UBbottom4≤UBall20逐窗口成立，所有48组native-good lost=0。单窗口完整候选仍要求同一slot ADE≤.2且FDE≤.4。[构造和下界代码](<<WORKSPACE_ROOT>/GENERATION_FEASIBILITY_CONDITIONING_DESIGN_20261015/A0_execution_20261015/analytic_envelope.py>)；[独立保存几何审核](<<WORKSPACE_ROOT>/GENERATION_FEASIBILITY_CONDITIONING_DESIGN_20261015/A0_execution_20261015/ARITHMETIC_AUDIT.json>)

但这些具体GT球路径**不是安全正见证**：native16/16个split×group通过原五motion门，而UBbottom4和UBall20各0/16组同时通过全部五项literal≤1.01native。这说明这批上界达到路径不能直接部署或充当已通过A1的教师，不能证明其他安全路径不存在。A1没有运行，不存在安全见证质量、alpha或回退率结果；B没有训练，不存在P2增量价值的实验证据。[全部48组物理指标](<<WORKSPACE_ROOT>/GENERATION_FEASIBILITY_CONDITIONING_DESIGN_20261015/A0_execution_20261015/RESULT.json>)；[补充完整组标量核验](<<WORKSPACE_ROOT>/GENERATION_FEASIBILITY_CONDITIONING_DESIGN_20261015/A0_execution_20261015/completion_review/REVIEW.json>)

## 4. 执行、审计和来源

独立任务Trajectory-GenerationFeasibility-A0-20261015，唯一Python22360/serviceparent2688，14:02:31.459632HK开始，14:04:32.038421HK worker/exit双SUCCESS，总耗时120.578789秒，约2分00.579秒。实际结束早于原14:05:14半程节点；完成是在启动后的控制文件整理中发现，后续“继续执行”完成报告核验，没有虚构按时做了半程OS检查，也没有再查已完成PID或重启。

全24679个唯一ID，4个科学split jobs、12个port人口、48组指标、8个主要NPZ/metric输出和4个done契约完整。8个预检128窗口块不混入正式结果，也没有伪造三个模型seed：A0只使用一份冻结native缓存。

生产独立CPU审计在保存轨迹上重构全slot下界/逐点达到路径/完整质量/原SciPy GT-free ALLK20 complete-link .10meanpath/.20endpoint/五pooled motion（12v/11a/10j，dt.4）/固定16/实际literal分类，最大误差3.5527137e−15≤原2e−12；来源与输出SHA初末通过。此误差容差只核验浮点算术，**未加到科学覆盖或motion门**。[全量独立审计](<<WORKSPACE_ROOT>/GENERATION_FEASIBILITY_CONDITIONING_DESIGN_20261015/A0_execution_20261015/ARITHMETIC_AUDIT.json>)

补充CPU保存标量审核读取全部48组，独立重算分母、coverage、rescue/lost、嵌套与FIT整数门，并初末验证1823来源、8个主要输出和4个done文件SHA/契约，PASS，耗时11.210686秒；没有神经模型前向、optimizer、新CI，未冒称再做一次全聚簇/物理/模型/梯度回放。[补充审核](<<WORKSPACE_ROOT>/GENERATION_FEASIBILITY_CONDITIONING_DESIGN_20261015/A0_execution_20261015/completion_review/REVIEW.json>)；[审核脚本](<<WORKSPACE_ROOT>/GENERATION_FEASIBILITY_CONDITIONING_DESIGN_20261015/A0_execution_20261015/completion_review/summarize_saved.py>)

1823来源完整继承此前1578个digest，扩入D0完成输出/报告和本次新源；不覆盖旧hash，不改旧科学源码、权重、报告或失败证据。原D0 STOP_NO_U1、T1/T2T3和旧暂停跟进保持原状。source digest为78ce4f9e4c9d8c9d7ea9fa5441a386e636cf1552139ee15cd519d42947100887。[来源保存证据](<<WORKSPACE_ROOT>/GENERATION_FEASIBILITY_CONDITIONING_DESIGN_20261015/A0_execution_20261015/PRESERVATION.json>)

实际全流程峰值private1.554741GiB/RSS1.569702GiB；最小可用physical11.500935GiB/commit18.410896GiB，process6GiB和各reserve6GiB资源门满足。没有OS/pagefile调整或终止用户程序。生产独立审计61.139042秒已包含在主耗时内，不能另加两遍。[实际资源](<<WORKSPACE_ROOT>/GENERATION_FEASIBILITY_CONDITIONING_DESIGN_20261015/A0_execution_20261015/RESOURCE_USAGE.json>)

## 5. 当前分支和下一步边界

现在走到指南的第二种停止分支：**all20有较大几何空间，但固定bottom4机会不足。** 不是“全部0.2m局部球空间都太小”的第一种分支，也不是“有足够安全教师但P2学不会”。

按指南停止A1、B及V，不因“继续”就改变槽位、半径、5pp门或先做六run训练。模型基线、原major/全部五motion1%保护、historycycle11和未完成的长期目标均不重置。

若继续更大方向，应另立“原生候选支撑如何分配 / 如何生成不受低机会anchor束缚的新轨迹”的具体设计；当前数据没有授权GT选top4部署，也没有证明重启R1–R5排序标签、增加MoE、调优化器或修改scorer有效。本轮不启动这些范围不同的实验。

## ARS 同助手反方检查

**DA1—范围：PASS。** 解析界回答的是独立球域机会，不是安全路径可预测性；不因为触发率高就宣称结构根因。当前执行沿用既定设计，不补造事前结果。

**DA2—反例：PASS，保留限制。** 最强反例是all20上界确实很大；因此不能说局部修正整体无价值。相反，两种oracle路径的五motion均失败，说明大上界不等于可实现安全收益。两类证据同时保留。

**DA3—结论：PASS，仅支持注册STOP。** 本轮没有训练P2/教师学生，没有A1最优安全求解，也没有独立新场景；不下历史信息无用、不可预测、唯一shared-head/decoder根因、SOTA结论。A是确定性指定人口算术，无新显著性CI或三seed伪重复。AI辅助整理与反方检查均由同一助手完成，非外部独立评审；原worker简报另存completion_review/INITIAL_WORKER_REPORT.md。
