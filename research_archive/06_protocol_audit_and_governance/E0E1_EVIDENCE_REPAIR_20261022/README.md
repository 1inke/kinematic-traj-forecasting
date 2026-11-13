# E0/E1 证据修复与冻结补评

执行授权：2026-10-22 用户要求按 `E0E1_REVIEW_20261020` 内容执行。范围为指南 P0–P3：不训练、不改旧报告/权重、不新增outer/CAL学生评价，不自动启动P4。

现有11个学生 × FIT/inner =22份最终权重补评。由当前已核对实现严格加载，不能冒称重建原训练时源码；旧checkpoint没有optimizer/RNG，不补造续训能力。所有新产物只在本目录。

资源按用户后续明确规则：启动可用physical/commit均≥1.3×预计任务峰值，运行按剩余预计需求与Windows保留峰值判定。指南中误带的固定6GiB/30分钟不是现行约束，差异在protocol登记。预算0新训练、0GPU训练小时；可用CUDA仅冻结前向。

实际顺序：`prepare.py`保全清单 → `preflight_v3.py`完整预检 → `run_frozen_v3.py`22份保存与独立审核 → `completion_review.py`额外独立核验 → `repair_report.py`及`final_verification.py`收尾。v1/v2失败来源和实现修复均保留，不运行旧实验main。旧目录只读；每份done绑定输入、代码与输出SHA，禁止靠文件存在就跳过。

## 已完成的结果

22份冻结补评完整结束，0次新训练。入口：[修订报告](REPORT_REVISED.md)、[一页决策](DECISION.md)、[逐条纠错](CORRECTIONS.csv)、[独立审核](completion_review/REVIEW.json)、[最终验收](FINAL_VERIFICATION.json)。原始运行约20分39秒；本轮不启动P4，不更改baseline。
