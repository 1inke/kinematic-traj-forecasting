# 前瞻执行修订：运动学口径协调

2026-10-25。人类对“沿原项目12/11/10口径、保留原设计其余规则”的明确答复为“同意”。本修订在E0枚举和任何新模型训练之前生效。

唯一科学定义修订：轨迹先转float64，以最后观测原点补首段，dt=0.4，速度12项、加速度11项、jerk10项；L2后池化全部候选和时间，p95用NumPy linear quantile。五项仍为speed_p95、accel_mean、accel_p95、jerk_mean、jerk_p95。逐窗口及全人口对应五项literal <= native×1.01；不加floor、epsilon或四舍五入。

该定义替代原实施合同第2节的11/10/9、不补观测点表述。三份原设计、PROTOCOL_MOTION_AUDIT.json和协调提案均保留不改；旧BLOCKED报告是修订前的历史状态，不是E0科学结果。

其余全部保持：固定native20/KIN20；STOP+400一次替换；GT-free物理门；B/T多正例；固定hash整track分区和4096样本；固定oracle排序；324特征；I/S唯一监督差异；六模型8epoch；所有覆盖、保留、motion、成本和统计门。不得跑两种口径后选择有利版本。

先完成输入身份、源和当前E0完整数值/保存/资源预检，再运行E0。E0全部seed门通过才允许E1自己的真实GPU及恢复预检。未创建新goal，不修改旧baseline/historycycle11或PAUSED任务。
