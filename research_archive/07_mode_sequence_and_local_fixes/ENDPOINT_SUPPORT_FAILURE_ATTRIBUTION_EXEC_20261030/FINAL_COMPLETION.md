# 最终完成索引

状态：PASS_OFFLINE_ATTRIBUTION；建议PRIORITIZE_SET_UTILITY；0训练/0模型前向。

- [完整结论报告](RESULT_ANALYSIS.md) / [机器可读分析](RESULT_ANALYSIS.json)
- [五端口每候选参数误差、quality与失败位](five_port_candidate_parameters.csv)：12800行，注明actual float32教师与exact END49参数两个口径。
- [全部49动作已有安全/效用数据](all49_saved_action_utility.npz)：125440既有J/safe表单元及距离；无新的全动作几何枚举。
- [最新完整保存审核](completion_review_v2/REVIEW.json) / [全部20组及48收益单元补充审核](completion_review_v2/GROUP_SUPPLEMENT.json)
- [全部5120反事实当前版本](counterfactual_metrics_v2.npz)：junction先转float64；v1及失败证据均保留，见报告第9节。
- [全部7丢失+8救回的15页PDF](cases/ALL_15_CASES.pdf) / [案例索引](CASE_INDEX.md)
- [资源记录](RESOURCE.json)：v3预警制；绘图阶段出现提交内存1.3规划余量预警，未出现紧急/持续压力STOP；未锁内存或改OS。阶段峰值是各进程生命周期读数，不能相加冒充同时总峰值。
- 当前最终封存是FINAL_PRESERVATION_V2.json与FINAL_SEAL_CHECK_V2.json，包含首轮130文件封存和后补完整参数表等；所有旧封存文件保持原样。

同助手第二套实现与反方审核，不是外部审稿。旧tiny90%STOP、TIME停止、P3不触发、正式G/H未测均不变；不自动启动任何后续训练。
