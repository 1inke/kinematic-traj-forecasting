# 步骤3：固定合同的信息增量诊断

2026-10-27。人类已授权四步顺序执行；本目录仅在前两步完成审核分析后进行自身准备/预检。原设计不改，旧模型/失败/结果/源保留，禁止删除。

共同1284→32→32→1 GELU/42208参数，四mask分别base/token/neighbor/both。只使用原opt4096、共享动作表与IPW原T监督。每合法arm三fresh8epoch/1024updates，完整final锁后一次hold/inner最终评价。不扫描结构、损失、学习率、epoch或邻居规则。

完整token按原8×96顺序展平；对原last/float32mean作逐窗精确一致性检验。新增邻居必须从原label逐时记录重建，不使用旧未来存活num sidecar。末两观测存在/最近8/stableID排序，过去缺失为0且mask0，只改内存未来副本做不变性试验。不能唯一确定过去时序的样本保留原ID与原因；任何必须输入无法合法构造则neighbor/both整arm BLOCKED_INPUT_PROVENANCE，不删窗凑门。

执行状态以runtime_control和真正生产worker为准；此README是实施前快照，不代表已启动或预检通过。自身实际GPU8/双epsFD/513opt两epoch全部状态恢复/保存128全质量聚簇五motion/资源与吞吐审核全部PASS后才能启动唯一独立OSworker。第四步必须在本步完整审核分析后实施。

来源与结论边界：完整tokens只是固定缓存读取诊断；旧F16可能使用未来可用性掩码选择邻居，不能称新增传感信息或完全past-only部署证书。相同名义结构不等相同有效容量。family6和原所有效果/保护/五motion门保持，saved算术不冒称全模型梯度重放/外审。
