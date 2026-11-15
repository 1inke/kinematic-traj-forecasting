# 用户指定的 seed45 完成后暂停

2026-11-14 用户明确要求 seed45 跑完后暂停，准备关闭电脑。控制器只暂挂等待中的调度父进程，训练子进程继续完成64轮；取得真实退出码0、验证FINAL_LOCK及完整保存状态后停止原Windows任务，seed46不会启动。

GATE_ARMED.json 表示暂停点已经设置，仍不可据此认定训练完成。只有 PAUSE_COMPLETE.json 中 safe_to_shutdown=true，且 CURRENT 为 PAUSED_BY_USER，才表示后台已停且可关机。

恢复时双击上级目录 START.cmd，已完成阶段按成功收据和SHA跳过，从seed46继续。控制器没有周期触发，不会自动恢复实验，也不会关机。原源代码、合同、种子和旧记录均保留。
