# 方向一：当前冻结生成器的ADE/FDE+DPP

入口`run.py`默认只打印计划。两臂accuracy_only/accuracy_dpp×三seed42/43/44，均从原nativeidentity的新DSF开始，旧F16+h3完全冻结。仅目标DPP开关不同，GT只供FIT loss。

完整方案、系数、held全12final锁、资源与禁止启动规则见上级README.md及protocol.json。当前allow_training=false，未启动；不复用旧失败DSF权重。
