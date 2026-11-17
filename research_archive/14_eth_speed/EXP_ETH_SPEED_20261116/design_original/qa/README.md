# 验证证据说明

此目录仅保存编写期验证，**不是 EXP-ETH-DIAG 或 EXP-SPEED-AUG 的正式结果**。

- `CPU_TESTS.log`：合成数据单元与诊断流程测试。日志中的建议、误差与 gain 来自虚构轨迹和测试替身模型，不能用于判断 ETH。
- `CPU_RUNTIME_PREFLIGHT.json`：实际 B0 引擎在 65 窗合成数据上的 CPU 预检，验证 4 次更新、配对初始化、全部保存状态的精确恢复和推理标签隔离；未读取基准轨迹。
- `DEPENDENCIES_CHECK.json`：在临时目录创建输入合同并重读，核验旧封存依赖和本机引用文件；不开展新的数据诊断或模型推理。
- `VALIDATION_SUMMARY.json`：编写期交付汇总。

目标 GPU、实际预处理审核、新的 ETH/HOTEL 推理和正式增强训练在此轮均未执行。PowerShell 脚本只做解析检查，尚未注册和实跑新的 Windows 后台任务。

`create_local_config.py` 与 `check_delivery.py` 是工作区内编写辅助工具。前者按工作区相对位置生成配置，复制到 Downloads 后不要再次运行它覆盖已填好的配置。
