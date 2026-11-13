# E0/E1 修订报告与冻结补评（2026-10-22）

本报告纠正原总报告，不覆盖历史文件。原文SHA256：`8105bc0f8269d30663fb22a10ef24e6fe216e9c17f80968c35684d3b26374954`。逐条纠错与来源见CORRECTIONS.csv、CACHE_LINEAGE_VERIFIED.json、EXPOSURE_AUDIT.csv。

## 事实一：E0仍是机会见证，不是数学上界或无污染证明

cal_v11 change443窗native146→182/180/186、lost0；dropped257窗native97→117/117/116、lost0。均值+8.2769pp。共享50track固定3seed95%CI[5.8411,10.9568]pp；共同track+paired-seed敏感性[5.5556,11.2710]pp。20k原RNG约定保留，均为事后探索。

当前划分v1.1确获作者批准，不再混同旧“尚无v1.1”；但h3/v2实际split均见cal_v11全部1620身份/时间窗。新划分不会重训或抹去底座暴露。E0实际CPV文件与原输入清单的另处同名文件分列；本轮完整9套KIN缓存数值行重放最大2.86102295e-06，当前映射可用，不替代同期训练/缓存锁源。E0没有本轮全W/聚簇审核，原W=True保留为原报告层。

## 事实二：11已有学生×FIT/inner的最终冻结结果

|arm/seed|人口|overall good/N|change good/N|change rescue/lost|change五motion|
|---|---|---:|---:|---:|---|
|A/42|inner|693/1856|140/497|37/63|FAIL|
|A/42|FIT|8078/18909|1633/4725|406/710|FAIL|
|B/42|inner|785/1856|145/497|34/55|FAIL|
|B/42|FIT|9858/18909|1742/4725|428/623|FAIL|
|C/42|inner|672/1856|138/497|30/58|FAIL|
|C/42|FIT|8218/18909|1678/4725|371/630|FAIL|
|D/42|inner|811/1856|142/497|38/62|FAIL|
|D/42|FIT|9262/18909|1443/4725|352/846|FAIL|
|E/42|inner|700/1856|161/497|45/50|FAIL|
|E/42|FIT|8837/18909|1866/4725|519/590|FAIL|
|A/43|inner|644/1856|137/497|26/55|FAIL|
|A/43|FIT|8014/18909|1655/4725|369/651|FAIL|
|B/43|inner|659/1856|133/497|31/64|FAIL|
|B/43|FIT|8031/18909|1625/4725|369/681|FAIL|
|E/43|inner|641/1856|129/497|30/67|FAIL|
|E/43|FIT|7899/18909|1532/4725|382/787|PASS|
|A/44|inner|664/1856|148/497|42/60|FAIL|
|A/44|FIT|8121/18909|1738/4725|428/627|FAIL|
|B/44|inner|665/1856|137/497|32/61|FAIL|
|B/44|FIT|7921/18909|1621/4725|395/711|FAIL|
|E/44|inner|646/1856|138/497|38/66|FAIL|
|E/44|FIT|7842/18909|1600/4725|388/725|FAIL|

所有22个checkpoint×split都严格加载57432参数，全部参数与buffer前后EXACT，eval/inference-mode，无optimizer或参数更新。inner计数全部对齐原JSON；seed42未存native_good_count，明确用good−rescue+lost恒等式交叉核对，不伪称旧字段存在。差异细目见REPLAY_RECONCILIATION.json。

保存的同candidate质量、完整K20 GT-free聚簇、五pooledmotion已由独立NumPy/SciPy入口核验；额外全部逐窗标量、教师/槽责任、原生成功/可教失败/不可教失败子集和最终人口loss分项另有CPU第二实现审核。当前审核max_abs_diff=0，旧输入前后SHA相同。不是历史训练源追溯认证，也不是外部审稿。

## 口径与边界

WTA与教师最近匹配分开，责任是最终模型后验，不是历epoch使用率。每slot成功数与distinct不等于有效互补。minADE/minFDE可来自不同slot，good/J和分段要求同完整candidate。运动学12v/11a/10j、dt.4、linear pooled p95、literal1.01；未对学生做事后安全过滤。原总loss history不可倒推出逐epoch coverage，最终全人口teacher比值也不等于训练batch均值。

B/E原教师集合与归一化均不同，因此不能纯归因“是否分解”。CPU合成全参数fixture证明同teacher同权重union/decomposed标量及全部参数梯度等价；这不是值得再花训练预算的独立机制。新的R/N组间重加权会改变优化几何，不是恒等式。

## 推断（不是唯一根因）

当前history-only学生未使用F16交互表征；训练8epoch也没有收敛证明。将“57k容量不足×监督稀疏”写成已证根因是不成立的。须同时检查FIT拟合、inner迁移、保留/补全权衡及运动代价；完整描述性contrast和子集分母已保存，不只引seed42change最好结果。

## 未证明与后续

未证明未知场景/whole-model未见泛化、运动理解、数学最优上界、学生能力上限或新baseline。此阶段结束于P3决策，不自动训练P4。若后续选择组间权重实验，必须另冻结同信息/同teacher/同总lambda/同预算与CI家族；不得同时加宽、过采样、换表征或加物理项。
