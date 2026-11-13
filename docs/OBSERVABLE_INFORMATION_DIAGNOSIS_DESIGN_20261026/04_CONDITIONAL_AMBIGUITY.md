# 第四步：跨轨迹条件歧义审计

2026-10-26｜DESIGN_ONLY；冻结CPU保存数据分析，零训练、零新模型前向。

## 1. 要回答什么

在本次定义的相近过去与相近候选集合下，不同行人的可接受动作是否经常互不兼容？

重点不是比较单个oracle slot，而是比较**全部可接受动作集合**。如果相似输入对应不同正动作，说明逐例模仿未来知情oracle可能困难；但有限距离近邻不等于相同输入，本审计不能证明Bayes不可约误差或“历史没有信息”。

旧R4研究的是另一套锚点P/N/U及邻域富集，不能直接回答当前401动作的条件冲突。本步仅在前面未形成充分答案时启动，不为负结果无限搜索距离。

## 2. 固定人口与无GT选邻居

- Library：FIT_opt全部4096窗；每池seed独立原候选。
- Query：FIT_hold按`ambiguity-query-v1|track_key`哈希取最多128个track，每track按另一个固定hash选1窗，不按变化标签、成功或失败挑样本。
- Calibration：FIT_opt以同法选128个track各1窗，用`ambiguity-calibration-v1`独立tag。
- 同recording sequence匹配，排除同target track；已知完整20个source-index区间不能相交（任何行人）。source index不是未经验证的视频帧或秒。
- 时间未知者不用于时间隔离pair，保留全ID及未配对原因；不得宣称其不可预测或删除原样本。

目标track隔离且区间不重叠，仍不证明scene/context独立。查询人口固定，不为了凑足变化组改选window。

## 3. 固定距离、门槛与动作对齐

先按8步目标中心化过去轨迹RMS，在每个合法library track只保留最近1窗，再取最多32个不同track作为短名单。距离与shortlist均不用未来GT。

对每个短名单pair，分别在native20之间、KIN20之间执行20×20最小总代价二分图匹配；不能native/KIN跨族混配。

```
D(p,q) = max(mean_t ||p_t-q_t|| / .10,
             ||p_last-q_last|| / .20)
dH = RMS(past_query - past_library) / .10
dP = (mean matched native D + mean matched KIN D) / 2
d  = max(dH, dP)
```

按实际保存轨迹float64计算。匹配使用本机固定版本线性分配实现、physical slot升序输入，记录版本与映射。不要加任意扰动到代价求“唯一”匹配。对非唯一最优匹配记录歧义；实施前固定反向行列次序重算敏感性并映回原槽，若结论随等价最优解变化，标记ALIGNMENT_SENSITIVE，不能称稳健动作冲突。

每池seed在128个calibration reference上得到其短名单最近d；合法参考不足50时该seed UNDERPOWERED。门槛τ固定为这些最近d的25%分位数，linear方法，只用opt无GT拟合。

Query从其32个短名单中按d取最多5个且d≤τ的不同track邻居。没有近邻就无支持；不扩大τ、换权重、加k或删负例求支持。保留dH/dP分量，不能只报总距离。

## 4. 多正例集合与共同动作域

令query槽到library槽的匹配为πN、πL。动作(r,l)映射到(πN(r),πL(l))，STOP映射STOP。

```python
mapping = make_action_map(native_assignment, kin_assignment)
E = eligible_q & eligible_p[mapping]   # includes STOP
Bq = B_q & E
Bp = B_p[mapping] & E

if Bq.any() and Bp.any():
    conflict = not (Bq & Bp).any()
    overlap = (Bq & Bp).sum() / (Bq | Bp).sum()
else:
    conflict = None                  # opportunity/admission mismatch, not conflict

best_q = J_q[E].min()
best_p = J_p[mapping][E].min()
regret = maximum(J_q - best_q, J_p[mapping] - best_p)[E].min()
```

必须分别统计：原B是否为空、共同eligible后是否为空、两边都有机会但无共同正动作、存在共同正动作。**STOP始终安全，正动作集合不交不意味着不存在共同安全动作。**

`regret`是这对样本在有限共同动作域上的事后双样本折中损失；J使用每动作完整K20的samecandidate定义。它需要GT，不是可部署打分、未来训练标签或严格信息论下界。也不能把pair-wise折中当全数据最优风险。

同时报告匹配输入距离、两未来路径/端点距离、query/library各自rescue机会与成功保留。近邻未来差异只作辅助，不把未来距离参与邻居选择。

## 5. 统计与可解释性门

先在每query内对最多5邻居求均值，再按query track汇总；不可把所有pair视为独立样本。报告每池seed及合并描述，固定library条件下对query tracks进行20000次配对bootstrap，rng20261026，95%描述CI。它不覆盖library抽样不确定性，也不是第2/3步的新增显著性比较。

- overall至少50个有支持query tracks；变化组至少20个有支持track，才报告该组稳定描述。
- pair正动作两边非空分母、映射敏感pair分母单列；不能只报告容易产生冲突的子群。
- 不足则UNDERPOWERED；不将没有近邻当成未来歧义。
- 未发现冲突也不证明输入充分，距离和有限样本可能漏掉真正歧义。

示例窗按预先hash在各已报告事件类选图，仅作说明，不能据展示图片建立新的总体结论。

## 6. 输出及独立复核

建议文件：`query_registry.json`、`calibration_thresholds.json`、`neighbor_pairs.parquet`、`slot_mappings.npz`、`positive_set_overlap.json`、`pair_regret.json`、`SUMMARY.md`、`REVIEW.json`。若本机无parquet依赖，预注册改用CSV/NPZ容器，不安装包、不改数值。

独立CPU复核：source/IDs/time区间、hash短名单、实际匹配代价、映射双射、action400+STOP完整性、共同域、B集合、J重构、bootstrap。重排物理槽的夹具应保持最小匹配距离和集合结论；非唯一匹配影响明确留痕。不得读取旧已完成进程或重训练。

运算规模上限：每seed最多256 reference/query×32短名单×2族匹配，三seed最多49152个20×20分配问题；实际合法pair可能更少。只报规模，不凭此冒报实际ETA。

## 7. 四步合并后的决策

|组合证据|建议的大分支|
|---|---|
|tiny通过、异构同输入hold阳性|研究更合适的学习器/目标与压缩迁移；不是立刻增加新观测|
|同输入弱、完整token阳性|优先改时序表示读取，保留轻量约束|
|只有合法邻居增量阳性|优先显式交互条件，不再只用局部候选几何猜答案|
|模型阴性且稳健近邻冲突较多|避免逐例复制GT oracle；考虑在固定K下学习条件覆盖与保留，但本审计不直接授权新loss|
|邻域不足或匹配敏感|承认现有数据无法判清，不继续局部扫描凑结论|

“有信息没学会”和“部分未来不可辨识”可以同时存在，不要求二选一。最终报告应区分可利用的平均条件规律与无法逐例确定的未来分支。任何新模型方案要另行精确设计，不在审计中偷偷训练。

反方：距离不是充分统计量；动作匹配可能非唯一；每个过去通常只有一条真实未来。上述三点使本步只能提供条件冲突提示，不能宣称已估计真实条件熵或证明预测上限。
