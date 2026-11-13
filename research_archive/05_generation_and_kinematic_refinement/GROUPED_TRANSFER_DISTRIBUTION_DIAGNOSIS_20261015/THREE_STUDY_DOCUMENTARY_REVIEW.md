# 三项分组迁移的总复盘与当前执行边界

2026-10-15。复盘依据为三项既有 RESULT.json、原分析及审计描述；本次读取648行汇总和720条原探索区间的计数及主结果。尚未执行本轮计划的171保存job重新算术、342输出/54模型再次SHA或特征诊断。不能把本文称为新完整重放审计。

## 三项证据共同说明什么

|隔离因素|outer N / change N|整体好窗口 random 3seed / grouped 3seed|变化好窗口 random / grouped|grouped-random整体覆盖pp及原95%探索CI|变化覆盖pp及原CI|
|---|---:|---|---|---|---|
|目标行人|1035 / 253|642,649,632 / 642,622,632|113,115,113 / 107,108,109|-0.870 [-3.198,+1.177]|-2.240 [-6.241,+1.660]|
|录制时间|566 / 142|323,320,316 / 286,304,312|59,58,62 / 41,53,61|-3.357 [-7.843,+0.885]|-5.634 [-13.732,+5.556]|
|zara场景族|1561 / 211|1069,1074,1077 / 1042,1035,1041|84,77,80 / 73,72,70|-2.178 [-3.671,-0.691]|-4.107 [-7.985,+0.511]|

完整合格候选始终为同candidate ADE<=.2 AND FDE<=.4。三项每项216行指标、18模型、114输出、6条80epoch历史；总计648行/54模型/342输出/18条历史/720原探索区间。三项的outer IDs及组成不同，不能将其百分点作严格梯度或排序，不将54checkpoint当54独立训练。

连续误差方向较一致：变化组grouped-random ADE分别+.014755/+.013908/+.026371m，FDE+.029233/+.047737/+.073420m。目标实验的变化组原区间支持误差差异；时间实验仅6个变化time-block，变化主区间跨0；场景实验46个变化track，连续误差区间支持差异但覆盖差区间跨0。场景overall覆盖区间不跨0，也只支持当前adaptive单族与track模型seed条件下的探索差异。

原五项motion保护失败行数分别5/216、2/216、18/216；均不能升级baseline。变化覆盖没有满足原major/三seed/原生与fresh control全部约束。

## 限制如何改变结论

随机训练本身的变化好窗口相对native分别110→113/115/113、60→59/58/62、82→84/77/80，并非持续明显提升。第一项grouped在inner变化166→165/172/179，第二项236→237/240/244，第三项177→178/179/174，均有反例。第三项原CAL变化grouped222/235/238高于random221/214/224，但两臂都低于native243；场景暴露优势没有无条件传到原CAL。

场景实验同时减少同track、同时间与同族暴露，也改变有限训练组成。当前只能说相关暴露/分布变化与迁移弱化相伴，尚未把真正场景shift与相关窗口依赖分离，更不能唯一归因共享head。原F16已见该数据；只有3个场景族、3个model seed、重叠windows及重复探索，不是独立wholemodel未见泛化结论。

最近2026-10-15固定motion责任T1在FIT本身也较dynamic对照弱、两臂运动学均失败；这构成“所有问题都只是CAL分布shift”的反证。它的STOP仍有效，本诊断不启动T2/T3或重复标签变体。引用：[T1分析](../MOTION_PATTERN_TRANSFER_DESIGN_20261015/T1_execution_20261015/RESULT_ANALYSIS.md)。

## 本轮已执行与阻碍

已经创建固定 DESIGN/protocol、完整CPU诊断代码和CLAIM_INTENT。本轮首次执行在入口第一条resource guard停止，尚未fixtures、未读取旧数据/模型、未source核验、未产生特征或质量新结果。失败快照：可用物理1.6095047GiB，commit17.2195282GiB；诊断进程peak private .0510864GiB。物理余量低于原6GiB。snapshot为失败当时，不是OS当前状态或正式训练生命周期峰值。

保留 failures/20261015_104657/{traceback.txt,source.py}。不自动轮询资源/重试或启动worker；人类释放资源明确继续后，按同协议完成fixtures、源锁和冻结分析。当前只有文档复盘完成，完整三项再算术审核及分布实算均为BLOCKED_RESOURCE，不计formal完成或新科学成功。

## ARS inline反方2（同助手，非外审）

主要问题：将zara整体2.18pp差描述成已经证明CAL分布根因，会过度外推；CAL不是zara、子群覆盖CI宽，随机对照也不稳定。本文保留全3seed、native和inner/CAL反例。固定输入距离分析只能检验关联及共同支持，仍不能消除条件选择和未观测因素。

## ARS inline反方3（同助手，非外审）

执行边界是本轮最重要的完整性检查：不得把已写代码当预检PASS、不得把原审计PASS当新171job重算、不得依据resource失败推科学结论。下一步继续本已固定诊断，资源恢复前不做大缓存读取。原major及五motion1%全部保持。本文为AI辅助的本地证据复盘，不是独立外部评审。

## 原始来源与当前RESULT内容SHA

- [目标实验](../GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment1_target_track/results/RESULT.json)：d419b7d77050cd159e409285597a473d783cf81ae59415f6af16de1ee4c1c7f9
- [时间实验](../GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment2_recording_time/results/RESULT.json)：5cd3ad999adeab8765a4f033099cd77f792f8e173ed9af0b8b54b13f33de13d4
- [场景实验](../GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment3_scene_family/results/RESULT.json)：363af53d70252b927d0b0a67fdc4e0c6b7f55778c92ff58302b33019e8ce06b2
