# 第四步：跨轨迹条件歧义，有限CPU保存数据审计

2026-10-27，人类四步顺序执行授权；第三步完整分析之后实施。零模型/forward/backward/optimizer/训练，原候选/标签/运动学门不改，不读outer/CAL/inner新效果，不删除任何内容。

原设计未指定的“窗口另一个固定hash”在读取本步标签前登记：query为`ambiguity-query-window-v1|track_key|decimal_ID`，calibration为`ambiguity-calibration-window-v1|track_key|decimal_ID`；track主tag仍是原ambiguity-query-v1与ambiguity-calibration-v1。哈希用SHA256字节升序，同hash按track字典序/整数ID。人口不按group或结果筛选。

过去轨迹按每条target最后观测点中心化；RMS定义为`sqrt(mean_t(sum_xy(delta**2)))`，.10m归一。每合法library track近1窗（同distance按ID），全短名单按dH/track_key/ID取32；计算已固定D的原始Hungarian解，按d/track_key/ID取最多5且literal d<=tau。tau原opt nearest-d linear25%，不加epsilon、不扫阈值。

源metadata由既有world-XY full20来源指纹核对得到，仅用于明确sequence/target/source-index时间隔离；不将其当模型特征/可部署原始定位。unknown/ambiguous time在全IDregistry留原因，不配对、不删原样本；不以未来质量修时间。source-index不是视频frame或秒。

primary匹配严格升序physicalslot，reverse行列敏感性固定同时计算。数值最优非唯一审计：逐禁原解一条边，记录最小可替代总代价gap；gap<=2e-12*max(1,abs(cost))仅标“数值最优歧义/无法认证唯一”，不声称数学精确tie。该2e-12只核matching objective算术，不改D、tau、quality或1.01门。额外禁边问题是唯一性证书审核，不新增科学候选搜索；不按B/GT挑解。reverse的独立calibration阈值与选择是明确敏感性，不替代primary。

所有3池的GT-free shortlist、mapping、tau、query support先保存且锁定SHA，再读取原B/quality/GT做posthoc共同域、conflict/Jaccard/regret/未来差异。原eligible含STOP为GT-free物理准入。无支持不是无机会或未来歧义；正动作集合不交也不表示无共同安全动作。

支持overall<50或change<20/校准reference<50按原UNDERPOWERED，不改人口或扩大tau。20000次对固定query track的bootstrap只作描述，不加入六主比较家族。统计每query先均值再track，窗口/配对数不当独立复制；library不确定性不在该CI内。

每项必须实际来源/匹配/动作映射/质量/算术/资源预检PASS才生产。README是实施前snapshot，实际状态以runtime_control/worker为准。完成后报告四步有限结论并停止调度、保留全部记录，不自动新loss/新模型/扫描。
