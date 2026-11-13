# D0 结果分析：未触发冲突投影正式训练

结论：**D0 正常完整完成，但预注册触发条件未通过，停止该冲突投影子分支，不启动 U1、V-time 或 V-scene。** 这不是运行失败，也不证明整个共享映射问题已被排除。[原始决策](<<WORKSPACE_ROOT>/SHARED_MAPPING_GENERALIZATION_DESIGN_20261015/D0_recovery_resource_20261015/DECISION.json>)

## 1. 完成与证据范围

独立 worker 于日本标准时间 2026-10-15 12:34:57.776604 成功结束，实际总耗时约 8 分 20.512 秒。唯一完成核验于 12:35:54.7614958 确认 task Ready、无活跃进程、0 trigger、worker/exit 双 SUCCESS、stderr 0。24 个 seed×数据包、48 个临时干预副本、192 次 optimizer step、219 个保存任务全部完成。三个 seed 和所有八包均纳入，没有挑包或挑 seed。[运行结果](<<WORKSPACE_ROOT>/SHARED_MAPPING_GENERALIZATION_DESIGN_20261015/D0_recovery_resource_20261015/RESULT.json>)

保存后的独立算术审计覆盖全部 219 份轨迹：同一完整候选质量、原 SciPy GT-free complete-link .10meanpath/.20endpoint、五项 pooled motion、固定16路径和位置界，最大重算误差 0；1578 个锁定来源和输出 SHA 初末核验通过。补充 CPU 汇总读取全部876个保存组记录，独立核对96个组/时点/臂/split汇总和192步trace，重算触发门与逐包差值误差0，并再次验证1578来源/219输出初末SHA。补充没有模型前向、梯度重放、optimizer或新CI，也不是外部独立评审。[完整保存算术审计](<<WORKSPACE_ROOT>/SHARED_MAPPING_GENERALIZATION_DESIGN_20261015/D0_recovery_resource_20261015/ARITHMETIC_AUDIT.json>)；[补充汇总及全部组](<<WORKSPACE_ROOT>/SHARED_MAPPING_GENERALIZATION_DESIGN_20261015/D0_recovery_resource_20261015/completion_review/REVIEW.json>)

生产全流程峰值 private 3.123 GiB / RSS 2.245 GiB；系统最小可用 physical18.440 / commit22.921 GiB，原 process6GiB及两项各reserve6GiB均满足。第一次资源失败目录、代码和日志完整保留，不与本次成功混淆。[实际资源](<<WORKSPACE_ROOT>/SHARED_MAPPING_GENERALIZATION_DESIGN_20261015/D0_recovery_resource_20261015/RESOURCE_USAGE.json>)

## 2. 普通更新是否稳定损伤其他轨迹？

每个 seed 从同一旧 position3 末轮checkpoint分别重置八包；source每包256，probe固定4111窗口/138tracks。两者整track无交集，但均是原先训练见过的 FIT；同scene/time/context仍可能相关。比较step4减step0，八包等权。K4原loss和K20完整候选J不是同一个指标；J=min_k max(ADE/.2,FDE/.4)，越低越好，J≤1才有同一条完整好候选。

|seed|普通 source ΔL|普通 probe ΔJ|source降且probeJ升 /8|普通干扰门|
|---|---:|---:|---:|---|
|42|−0.00164285|+0.00000914|5|不通过：不足6包|
|43|−0.00159276|+0.00004209|6|通过|
|44|−0.00169114|−0.00001438|4|不通过：平均probeJ反而改善|

三个seed的普通source loss均下降，幅度约0.078–0.083%；但只有43达到预定干扰门，不能说跨seed稳定复现“学一批就破坏另一批”。probe原K4 loss三个seed均上升（约0.040/0.061/0.051%），说明局部训练误差迁移存在权衡，却不能用K4 loss上升替换事前指定的K20 J损害触发门。step1的source平均loss两臂三seed都先上升、step4再下降，也不能声称每一步都单调改善。[逐包结果](<<WORKSPACE_ROOT>/SHARED_MAPPING_GENERALIZATION_DESIGN_20261015/D0_recovery_resource_20261015/PACKET_SUMMARY.json>)；[所有时点/组汇总](<<WORKSPACE_ROOT>/SHARED_MAPPING_GENERALIZATION_DESIGN_20261015/D0_recovery_resource_20261015/completion_review/REVIEW.json>)

## 3. 投影是否缓解损害且保持学习？

|seed|保留普通source改善幅度|投影probe ΔJ|相对普通J损害减少|原good平均lost：普通→投影|最终门|
|---|---:|---:|---:|---:|---|
|42|102.05%|+0.00000977|−6.81%，略更差|1.25→1.25|不通过|
|43|96.66%|+0.00003291|21.82%，未达50%|0→0|不通过|
|44|98.22%|−0.00001633|不适用：普通无平均J害|0.50→0.375|不通过普通干扰门|

投影不是简单让模型不动：保留96.66–102.05%的source改善，实际每步参数位移均正；但没有跨seed达到损害减少50%的门。seed44原始数值projection_gate=True仅表示其公式成立，不能当成“缓解已有干扰”，因为ordinary_gate=False且普通J已改善。所有seed合取门失败。[预定门结果](<<WORKSPACE_ROOT>/SHARED_MAPPING_GENERALIZATION_DESIGN_20261015/D0_recovery_resource_20261015/DECISION.json>)；[完整更新trace](<<WORKSPACE_ROOT>/SHARED_MAPPING_GENERALIZATION_DESIGN_20261015/D0_recovery_resource_20261015/TRACE.json>)

## 4. 完整覆盖、组别和运动学反例

probe中原完整good数为2510/2508/2509（每seed分母4111）。下面是step4相对原checkpoint的八包平均变化，单位为**百分点**；不是八包合成一个部署模型：

|seed|整体覆盖Δ：普通/投影|变化组覆盖Δ：普通/投影（n=1008）|
|---|---:|---:|
|42|−0.00304 / −0.00304|+0.07440 / +0.06200|
|43|+0.01824 / +0.01824|+0.04960 / +0.04960|
|44|0 / +0.00304|−0.02480 / −0.01240|

seed43 的J恶化同时覆盖微增，seed42变化组覆盖微增而过渡组下降；因此平均J、阈值覆盖和lost不应混用。稳定组2330窗口在三seed两臂均无净覆盖变化；过渡组773窗口seed42约−0.11320/−0.09702pp，seed43/44约+0.03234pp。这些很小的变化不构成稳定覆盖提升，也不是大规模collapse。有效distinct均无平均改变；effective-good的微变不能替代覆盖。

对每seed×包×probe组，step4五项motion相对step0父checkpoint的最大上升分别约 speedp95 .0705%、accelmean .1130%、accelp95 .1073%、jerkmean .1130%、jerkp95 .2906%，没有超过父step0的1%。**这只是局部父checkpoint对照，不是正式native-major全部运动学门通过，更不是新基线。** [全部分母/组/运动学数据](<<WORKSPACE_ROOT>/SHARED_MAPPING_GENERALIZATION_DESIGN_20261015/D0_recovery_resource_20261015/completion_review/REVIEW.json>)

## 5. 梯度冲突存在，但不能据此进入训练

普通臂96步中32步出现至少一个负组梯度对，53/288个组对机会为负。投影臂96步中32步实施投影，共98次投影；它确实介入了计算，却未达到有限步收益门。记录中lambda<1约95–96%的受修候选，体现这个position3/bottom4/0.2m受限平台常触发缩放；只能作为局限，不能把阴性推断成容量根因。实际梯度、幅度及AdamW轨迹耦合，也不是纯方向因果隔离。[trace汇总](<<WORKSPACE_ROOT>/SHARED_MAPPING_GENERALIZATION_DESIGN_20261015/D0_recovery_resource_20261015/completion_review/REVIEW.json>)

## 6. 下一步与停止边界

按原路线，现在停在“**D0局部干扰未跨seed稳定复现，且投影缓解不足**”这一分支。U1及后续迁移训练不满足启动门，不增加epoch、改投影顺序、换group/学习率或放宽50%/6-of-8标准。失败T1的T2/T3不复活；F16 baseline、原major5%/五motion1%保护与未完成的总体研究目标不变。

更广义的共享映射、成功保护、条件分布差异和可达集合限制仍未被排除。若要继续，需要另一个事前明确、可被否定的新问题与设计，不能把当前阴性直接变成“应该上更大head/MoE/新loss/scorer”的许可。

## ARS 同助手反方检查

**DA2：PASS，保留局限。** 不能只挑seed43干扰或seed44 lost改善。最强反例是seed44普通probeJ改善、seed43完整覆盖仍微增。主结论限定为既定触发失败；不把阴性变成不存在共享干扰的证据。

**DA3：PASS，仅支持预定STOP。** 本实验是已见FIT、末轮受限head附近的fresh零moment短步干预；8包共享probe和模型，不独立，不造显著性CI。可达界、lambda饱和、有限步和候选选择均限制外推。没有新未见场景泛化、唯一根因、运动理解、SOTA或不可预测率结论。报告由AI辅助整理，反方检查在同一助手内完成，不是独立外审。

完整可追溯数据：[结构化分析](<<WORKSPACE_ROOT>/SHARED_MAPPING_GENERALIZATION_DESIGN_20261015/D0_recovery_resource_20261015/RESULT_ANALYSIS.json>)、[补充审核代码](<<WORKSPACE_ROOT>/SHARED_MAPPING_GENERALIZATION_DESIGN_20261015/D0_recovery_resource_20261015/completion_review/summarize_saved.py>)。原worker简报另存completion_review/INITIAL_WORKER_REPORT.md；科学结果、代码与失败来源未改。
