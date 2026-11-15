# 独立后台运行与恢复

本轮停止的实际原因：首模型 E1A_s43 已完成8轮/2720更新并保存 final.pt；旧 execution.py 在写 FINAL_LOCK 时因局部变量 seal 遮蔽函数而失败。原始失败、日志、代码和模型保持原样。恢复入口直接审核并复用已完成模型，不重新训练它。

双击 START.cmd 即可通过 Windows 任务计划程序启动/继续有限实验；无需打开 GPT 或 Codex，无需网络或 OpenAI API。双击 STATUS.cmd 查看状态。重复启动不会产生并行副本。任务名称：Trajectory-NextPhase-E1E4-Standalone-20261114；没有周期触发器，运行时限不限。

运行期间由 worker 进程临时阻止系统自动睡眠，退出时解除；未改系统电源方案，屏幕可以关闭。关闭 GPT/Codex 不影响任务。关机、重启、注销和手动睡眠仍会中断运行；之后可用 START.cmd 从已保存的完整轮次继续。

每轮保存完整模型、EMA、优化器、调度器、随机状态、batch顺序和历史。已完成阶段经SHA校验后跳过。每次独立启动的日志和成功/失败收据在 attempts/；CURRENT.json 是最新状态，WORKER_EXIT.json 是最后退出状态。实验成功必须有 RESULT.json status=SUCCESS 和 WORKER_EXIT.json exit_code=0。检测到科学/来源/资源错误时保存现场并停止，不盲目重试。

科学结构、数据、seed、预算选择、损失、判定门、E1→E2/E3/E4有限顺序沿用原封存内容。原训练代码文件没有修改；adapter调用原 Trainer.step，只修复记录与恢复基础设施。E1A_s43 的完整权重按原合同审核后复用，所有新增修复源码另有 RECOVERY_MANIFEST.json。
