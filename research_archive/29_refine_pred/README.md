# EXP-REFINE-PRED

2026 年 11 月 23 日完成。使用五折中 34 个已训练的 QRANK 检查点做冻结重放，涵盖 34,161 个测试窗口和 206,422 条模型窗口记录，训练次数为零。E0 为主臂，QFL 检查变化方向。ETH、ZARA01 用于探索，HOTEL、ZARA02、UNIV 用于确认。

## 判定

按预设规则判为 **STOP**。E0 的匀速组在五折都改善。三个确认折的减速组 minFDE 从初层到末层分别变化 +0.96%、+2.13%、−0.22%，按行人轨迹聚类的 95% 区间均跨零，C1 未通过。

观测转向角在 HOTEL 和 ZARA02 对末层 minFDE 增加超过 0.10 m 的窗口达到 C2，AUROC 分别为 0.752 和 0.697。三个确认折的纵向散布比均显示匀速组收缩更多，C3 未通过。完整结果包括按种子和录制的敏感性分析。

[预注册设计](DESIGN.md) · [报告](results/REPORT.md) · [完整汇总](results/SUMMARY.json) · [预检](results/PREFLIGHT.json)

## CPU 复算

在本目录运行 `python -I -B tools/reanalyze.py`，需要 NumPy。工具使用 34 份逐窗派生指标、匿名行人聚类编号和机动标签复算折内统计、区间和 C1–C3。它会用同仓库的正式 QRANK 回执以及两折较早的冻结重放校验运行来源。

本目录保存原始实验设计、科学代码、结果回执及可供复算的窗口数据。从头重放需要 ETH/UCY 原始数据、模型权重及本地实验依赖。

[运行状态](STATUS.json) · [文件散列](FILES.sha256) · [正式 QRANK 实验](../27_qrank_mldec/EXP_QRANK_MLDEC_L4_20261122/) · [早期冻结重放](../28_qrank_followups_and_ea/frozen_replay/results/SUMMARY.json)
