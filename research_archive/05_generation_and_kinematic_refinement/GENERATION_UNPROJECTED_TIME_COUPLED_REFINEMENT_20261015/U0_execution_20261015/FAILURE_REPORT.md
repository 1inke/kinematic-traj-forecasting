# U0资源保护停止：训练已完成，独立审核未完成

状态：STOP_RESOURCE_WAIT_HUMAN。当前不能给出已审核的科学结果或判定准确性/运动学分支。

## 已确认
- 唯一后台Python8528从2026-10-15T16:33:05.04348+09:00运行到2026-10-15T16:47:20.871703+09:00，worker与exit为FAILURE。
- 首次实际检查2026-10-15T20:25:21.8393464+09:00发现任务Ready/进程已退出/XML触发器0/本轮stderr1888字节。原定16:41:51.356半程检查实际迟到，不伪称按时执行；检查计数1，不再重复。
- 两臂×三seed的6份history均实际25轮，最后epoch25；6个run完成合同登记了checkpoint SHA。
- 28个评估job全部保存（24科学+4native参考），84个主要paths/aux/metrics文件摘要已登记，BOOTSTRAP.json也已保存。
- 尚无ARITHMETIC_AUDIT.json、RESULT.json、PRESERVATION.json或最终RESOURCE.json；模型/输出SHA本次未全部重新计算，不称全审核PASS。

## 停止原因
原始RESOURCE_FAILURE.json记录2026-10-15T16:47:20.582854HK：
process private2.964710GiB/RSS0.798023GiB，系统physical可用4.892677GiB<6GiB保留门；commit可用24.081173GiB。
触发的是系统物理内存余量门，不是process6GiB上限，也不是GPU OOM或科学指标失败。
这些是**失败瞬间保存读数**，不是现在的资源快照。
生命周期peak private4.283550/RSS3.281307GiB，minphysical4.892677/mincommit20.574085GiB，不拿预检峰值替代生产。

最后落盘审核进度为U0_INDEPENDENT_SAVED_AUDIT，job free_position/seed44/FIT、done20/28。
异常发生于后续stage入口guard；没有完整终审核文件，不根据该进度推断精确已通过job数量。

## 保留与下一步
失败源码和trace保留failures/1792054040732607900，科学源码、模型、history、所有结果/合同/旧来源不改不删。
原先预检通过与两轮子集EXACT恢复不能替代本次生产终审核。

按原资源协议立即停止自动重试/资源轮询，不降低6GiB余量、不调OS/pagefile、不关闭用户程序。
请释放内存后明确继续；之后先一次当前资源核验，只恢复保存结果的完整独立质量/聚簇/运动学/CI/来源审核及收尾，**不重跑6次训练**。
恢复须新目录保留父失败/来源；无新loss/scorer/参数扫描或科学阈值更改。

原一次half节点已消耗，原16:50:37.670完成节点不再适用，runtime/账本STOP_RESOURCE_WAIT_HUMAN。
u0跟进保持原状态但无新human决断直接安静，不每日自动重试；旧S0及其它暂停任务不变。
总目标尚未完成，baseline/历史cycle11不变。

备注：本轮检查包装器有一条未使用的Join-Path引用空PSScriptRoot导致六条只读控制警告；check_once与6history仍返回，原始输出保留control_check_provenance.txt。
该控制警告与worker独立stderr内的MemoryError严格分开；未因此重复查询PID或重启worker。
