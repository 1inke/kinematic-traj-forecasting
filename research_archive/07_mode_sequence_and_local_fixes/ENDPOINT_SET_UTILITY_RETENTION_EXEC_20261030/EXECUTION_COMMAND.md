# 执行与范围

当前预检实际入口为preflight_v3_adapter.py：原audit接口callback改keyword、预检/资源收据另命名空间；原数学、模型、优化器未改，所有失败保留。完整GPU8、独立全4770参数梯度/固定双eps、三arm hash513+12两epoch精确恢复、真实128保存审核已重新PASS。

当前tiny实际入口run_tiny_resource_adapter.py，显式读取MEMORY_ESTIMATE_v3.json，原资源收据不覆盖。PowerShell用治理Python -B运行；外层仅在异常时将完整traceback独占写FAILURE_PRODUCTION_时间.json。它不改变模型、数据、RNG或训练步骤。标准输出由工具保留，不将持续变动日志写入冻结source清单。

新科学源码在run_tiny.main启动前封存；先9个fresh tiny，每个固定256更新，未过效用门则停止正式分支。并非旧tiny延长或G/H恢复。
