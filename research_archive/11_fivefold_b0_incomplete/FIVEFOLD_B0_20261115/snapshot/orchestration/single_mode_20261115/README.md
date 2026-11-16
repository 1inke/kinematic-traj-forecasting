# 单任务执行模式

用户于2026-11-15明确要求“切换成单个执行，我需要额外的gpu资源”。后续预算与最终训练都只允许运行一个训练任务，直到用户明确要求改回并行。原模型、训练轮数、数据、seed、指标和全部已完成结果不变。

worker_single.py通过外层适配将原封存控制器所有pool调用（含显式limit=2）限制为1，同时将状态元数据标为SINGLE/concurrency_limit=1。旧科学源码、原233源清单和并行314源清单均不修改，新适配单独封存。保留原Job Object子进程清理、断点SHA验证、DONE跳过、预算封存和最终汇总。

单路优先继续Zara01 E128（暂停时54/128），完成后继续Zara02 E128（暂停时22/128），随后15个最终run也顺序执行。不是丢弃第二个任务；等待任务只保留磁盘断点，不保持GPU上下文。中断当轮按原机制重算。

本目录START.cmd为新的启动/恢复入口，STATUS.cmd看原parallel根standalone聚合状态，PAUSE.cmd暂停。新按需Windows任务Trajectory-Fivefold-B0-Single-20261115；原Parallel/Standalone两任务停用，避免误开双路。所有控制器共用同一mutex，零trigger、无限执行时限、IgnoreNew；保持登录与开机可退出Codex，自动睡眠临时抑制、屏幕可关闭。手动关机后用本目录START恢复。

训练仍会占用部分GPU，单路不保证固定利用率或专属GPU配额；释放第二份训练占用，代价是全程耗时增加。单路策略不会自动恢复双路。实际状态以parallel根standalone/CURRENT.json、CHILD_STATE、WORKER_EXIT及Single任务/真实进程为准；之前双路ETA已不适用。
