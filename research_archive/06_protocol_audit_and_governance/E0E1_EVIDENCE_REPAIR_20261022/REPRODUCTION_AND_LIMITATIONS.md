# 复现入口与不可补造的材料

## 本轮入口

使用现有环境，不需要安装新依赖。`F16_ROOT`可覆盖项目根路径；所有输出只写本目录。冻结补评保留完整现有FIT/inner人口，不选最好epoch或seed。

实际入口顺序：prepare.py → preflight_v3.py → run_frozen_v3.py → completion_review.py → repair_report.py → final_verification.py。P0只运行一次，已有清单时拒绝覆盖。v1/v2预检失败、原源码、修复依据、数值差异和新源锁保留在failures与IMPLEMENTATION_TRANSITION_V2/V3，不默认它们代表原科学训练失败。

完整当前代码和原输入SHA见source_manifest_before、两份实施transition与最终ENVIRONMENT_AND_CODE。新增审计/报告脚本是在后续阶段写入，不冒充原模型训练源码。部分补评时的CPU审计缓存标PARTIAL，不能当22份全部完成；只以FINAL_VERIFICATION及最终REVIEW闭合为完成。

不要对本轮已完成入口反复启动。若需要复制复现，应在单独版本目录保留原清单并重新登记；完整源码、输入、输出SHA必须兼容才能复用，禁止只看文件存在就跳过或重跑旧训练。

## 数值修复范围

原检查发现CUDA float32 GEMM在batch128/32的max差4.2915344e-6。固定计算batch256、仅补零且不保存补零行，使caller分块32/128结果EXACT；每个窗口的同一MLP函数与权重不变。无BatchNorm/跨batch通信。真实inner旧JSON所有离散计数仍需EXACT，不因此放宽good/kinematics边界。

seed42旧JSON没有native_good_count：adapter标记推导good−rescue+lost，并与冻结native计数核对；后两个seed直接比较所存字段。不能声称早期文件拥有未存的字段。

## 永久缺口

- 旧student checkpoint只含state_dict/arm/seed；没有optimizer、所有RNG或训练时完整源码/ID绑定，不能做exact training resume认证。
- 旧8epoch总loss记录不足以重建逐epochcoverage、历史slot责任、逐batchteacher权重；新输出仅最终权重描述。
- 当前数据与source映射可复核，不代表native/KIN原训练时的cache来源链已经全部追溯闭合。
- v1.1是CAL划分批准，不是整个协议/新模型泛化/S4授权。原native训练split仍接触inner及cal_v11；不称完全未见whole-model验证。
- CPU独立算术是同一助手的不同实现，不是外部AI/人工外审。未重新训练F16、KIN或任何学生。
- E0本轮复核范围是保存索引的coverage及共享track统计；没有新增全量E0 W/聚簇/物理审核。学生P2的完整22份几何/聚簇/五motion审计另列，不混范围。

指南误带的固定30分钟/6GiB要求已明确登记为被用户更晚规则覆盖：使用任务估计1.3倍资源规则；实测耗时长不触发重训或删减审核。
