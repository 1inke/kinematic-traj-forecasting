# F16上下文候选生成：误差退化与成功保留实验系列

日期：2026-10-23（Asia/Tokyo）  
状态：**DESIGN_ONLY；未启动诊断、训练、预检或自动任务。**

## 这次研究什么

不是再证明“能生成不同轨迹”，而是检验：**能否在固定K20和相同轻量解码器下，保留已经出现的覆盖增益，同时减少原生成功丢失和仍未覆盖样例的连续质量退化？**

最近S1在inner变化组由166/497覆盖提高到186/191/192，但平均误差和全部组的加速度均值门失败。新增算术分解发现，平均误差退化主要由“新旧共同失败”样例贡献。故本系列不直接恢复B/E_bal蒸馏，也不先改scorer。

## 执行顺序与数量上限

|阶段|问题|数据/运行|是否训练|
|---|---|---|---|
|D0 退化定位|哪些误差、哪些人口造成退化？|旧A三seed×FIT/inner共6份输出及2份native|否|
|D1 运动时序核对|新增加速度是否对应GT运动变化？|与D0共用同一保存数据扫描|否|
|T1 条件单因素对照|原生相对质量惩罚是否有效？|A_replay与P_relative×3seed，最多6个fresh runs，各8轮|有条件|
|R 总结与阶段门|改善机制还是达到可用基线？|完整保存审核、结果与反例|否|

最多6个新训练、48epochs、12份模型×人口输出。D0/D1不是6次模型前向，也不能算新正式训练。T1不满足入口条件则停止，不能临时换结构或扫描系数。

## 文件

- [DESIGN.md](DESIGN.md)：假设、诊断、唯一损失因素、统计与结论分支。
- [IMPLEMENTATION_CONTRACT.md](IMPLEMENTATION_CONTRACT.md)：输入、核心伪代码、保存字段、预检与执行验收。
- [protocol.draft.json](protocol.draft.json)：机器可读设计草案；空实现字段不得冒充预检通过。

本次用户只要求设计文档。执行人员应在收到后续执行请求后，新建execution子目录并完成实际预检；本目录的存在不是启动许可。不得覆盖旧报告、源、权重、失败记录；不得恢复旧暂停任务。

## 主要证据

1. `../F16_CONTEXT_COMPLEMENTARITY_DESIGN_20261022/execution_20261022/RESULT_ANALYSIS_FINAL.md`
2. 同目录`completion_review/ALL_GROUP_RESULTS.csv`、`ALL_MOTION.csv`、`REVIEW.json`。
3. 同目录`posthoc_interpretation/ANALYSIS.md`及`recompute.py`。
4. 同目录`protocol.json`、`core.py`、`training.py`、`input_contract.json`。

这是本地探索性设计，不是公开注册、外部同行评审或新颖性检索。仍保留上游F16已见FIT/inner、反复使用inner、仅3seed与相关窗口等限制。
