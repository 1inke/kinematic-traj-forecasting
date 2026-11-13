# D0：已有分数的排序—STOP分解审计

2026-10-27｜仅设计。零训练、零模型前向；不修改任何checkpoint或决策阈值。

## 1. 固定来源

按run manifest与显式ID登记下列完整终态，不根据D0效果挑模型：

|来源|模型|split|人口数|
|---|---|---|---:|
|原K20 E1_execution_20261026|I/S×42/43/44|opt/hold/inner|18|
|四步execution_peak_measured_20261026|M_full/T_full/T_geom×三seed|opt/hold/inner|27|
|四步step3_execution_20261027|C_base/C_token×三seed|opt/hold/inner|18|

共63个已保存模型人口；重复窗口不是独立样本。tiny最终读数只作既有背景，不混进hold统计。原T_perm和neighbor/both的BLOCKED不能作为模型或零分数输入。

必须先核每个来源的完整分数是否实际保存、all401动作索引/eligible/B/质量/ID是否兼容。缺全分数时单列BLOCKED_SAVED_SCORES，不为填表重训或偷偷做冻结forward。旧无损档案需要恢复时先核空间与原SHA，另写新恢复位置，不覆盖不同内容。

## 2. 要区分的四种事件

E为eligible非STOP动作，B为原全部安全正动作。E空则直接标记NO_ELIGIBLE_EDIT，不计算虚假排名或负无穷分数差。

令a*为E内分数最高的动作，tie按原action_id升序；m=s(a*)−s(STOP)。

1. **正确编辑被STOP挡住**：a*∈B且m≤0。再拆成真rescue与连续误差改善。
2. **编辑内部已排错**：B非空但a*不在B。不管STOP是否放行，都不能叫“只有门控问题”。
3. **有机会且成功放行**：a*∈B且m>0，核实际选择与保存路径。
4. **没有正机会但放行**：B空且m>0，按质量分有害/无效，不能只看编辑率。

同时单列原生已成功与原生失败窗口；B非空不是rescue存在，二者分母不得混用。

## 3. 固定指标

每source/seed/split/group都报告：

- 非STOP第一名hit_B；在B非空窗口条件比例、在全部窗口无条件比例都报。
- 非STOP第一名真实rescue命中；rescue机会窗分母和全部原失败窗分母都报。
- 固定k=1/5/10的Recall@k(B)与Recall@k(rescue)，最优正动作rank、MRR。
- m的min、5/25/50/75/95分位与max，按四类事件拆分；跨模型不直接比较分数量纲。
- 原始实际执行的coverage、rescue、lost、无效、有害，与“第一名编辑是好动作却被挡住”的计数并列。
- 分数完全相等/tie数量，避免把STOP优先的零分初始化式现象解释成已学会保守判断。

随机排序参照采用解析式，不拟合也不随机挑一次有利排名。若某窗口有m个eligible编辑、b个正动作，随机top-k至少命中一个的概率为：

`1 - C(m-b, min(k,m)) / C(m, min(k,m))`。

组合数越界的分子视0，m=0视NA。B与rescue分别计算，再在相同窗口分母汇总。physical action_id排序可作固定无训练参照，不能把它挑成部署政策。

## 4. 明确禁止的推论与操作

- 不把删除STOP后的第一名强制用于实际部署，不据此修改hold输出。
- 不按hold寻找STOP bias、概率阈值、top-k或温度。
- top-10中有好动作不是top-1部署成功；大量坏动作也排在前面时要保留。
- 条件于B/GT的诊断是事后解释，不是无GT可调用的gate。
- 若画score分布，不将sigmoid称为未来概率；被加权训练的分数也不自动校准。
- 权重比仅证明目标存在不对称，不能从中推断每个实际模型的梯度轨迹或唯一根因。

## 5. 权重与梯度证据

从旧动作表独立核STOP正负总权重、非STOP正负总权重、窗口数量及每窗总权重。对完整原I目标另外按标签计算，不只读摘要。

在零score处：B空时STOP正梯度为−.25（有负动作窗）；B非空时STOP负梯度为+.25/Nneg。单窗标量与解析梯度用双精度小夹具重算，不加载模型、不更新参数。无负动作窗按原I规则loss0，须另列，不能套用上述非退化公式。

对只按STOP身份给两个常数的简化模型，分别求p*=Wpos/(Wpos+Wneg)，完整展示该解的假设；它不是实际预测概率或质量收益。

## 6. 审核、统计与交付

只读来源与输出初末SHA。逐窗重构stable argmax、B/rescue条件、rank、margin、原实际选择；全部63人口做流式统计，不复制全量模型/轨迹。

描述CI按整target track配对重采样20000次、三个池seed等权，rng20261027；固定seed只抽track敏感性；条件分母无效draw保留，>5%标记UNDERPOWERED。D0是多来源解释性审计，不宣称其中挑出的显著性为新确认性发现，也不用于更改D1合同。

建议新增输出：`SOURCE_REUSE_MATRIX.json`、`STOP_WEIGHT_AUDIT.json`、`PER_WINDOW_RANK_MARGIN.npz`、`ALL_GROUPS.csv`、`D0_RESULT_ANALYSIS.md`、`REVIEW.json`。source模型、原报告和分数不改。

交付只回答：问题主要表现为编辑排序弱、STOP压制、两者兼有，还是保存信息不足。**D0不作为挑训练人口或换目标的搜索器。** 来源/标签不兼容是硬阻碍；科学阴性则如实保留，再按未来授权范围决定是否运行已固定D1。
