# 04｜P2：时间隔离复核与有限分支

状态：DESIGN_ONLY / CONDITIONAL_NOT_STARTED。只在03的P1门全部成立后执行，不能先训练再决定要不要报告。

## 1. 为什么不是再换一个head

P1在target-disjoint但非context-disjoint数据上训练。若成功，需要检验收益能否脱离同一时间附近其他行人的上下文暴露，而不是立刻加入更大的模型、MoE或scorer。

P2只检验已指定D对A的可重复性，**不是**把P1与P2不同outer IDs的数值相减后归因于泄漏。P1/P2训练人数、场景比例、窗口数量都不同；只能在各自固定population内作A/D配对，再描述跨population方向是否一致。

## 2. 数据固定

来源：[时间隔离身份审核](../GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment2_recording_time/identity_audit/RESULT.json)、[精确split archive](../GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment2_recording_time/identity_audit/splits.npz)。

- 只取旧 **grouped_training=16104**；两臂都是这16104，不再新建random臂。
- outer566、其中change142；inner2134；original CAL2879，只posthoc。
- 固定50 SOURCE indices的保留块，完整20索引窗口；这是source索引，不假称50秒或未核实视频帧。
- 原设计grouped训练清除任何行人窗口与outer/inner保留块的完整时间区间相交；同时间暴露为0，但outer仍有227个窗口在远时间有同target暴露。
- outer共14timeblocks、154targettracks，不能当566独立样本，也不是target/time/context全独立。
- 123未知/歧义时间记录沿用原registry和资格标记，不新删难例，不改变窗口资格、block宽度或hash。

本次只复用IDs、元数据及其审计事实。**不复用旧80epoch/91596参数/physical loss训练协议、模型或结果作为A/D对照。**

共同统计改为本次16104实际训练IDs；不照搬旧random/grouped交集12544的统计。P2两臂同一FIT-only算法/数值统计，P1统计不带入。标量fixture、缓存online对齐及精确ID审核必须重新通过。

## 3. 训练与输出

沿用01/03的120维25304参数U0、相同输出域、A/D目标及λ；三seed42/43/44、fresh paired init、25epoch、batch256和同优化器。没有warmstart、没有CAL选择。

6模型×25=150history；每模型FIT/outer/inner/CAL四份完整保存，共24科学job；四native reference单列。每模型固定epoch25为primary。

Primary统计按sequence×timeblock及paired模型seed重采样；同target-track重采样只作敏感性，不取更有利者当primary。14blocks很少，区间不稳必须披露；仅3seed不提供可靠的总体训练随机性分布估计。

## 4. 复核结论

`REPLICATION_SIGNAL=true` 必须同时满足：

1. D−A outer change coverage每seed不下降，均值≥2pp，family4区间下界>0；minJ每seed不恶化且family4上界<0。
2. D−A rescue率均值增加、lost率均值不增加；完整分母和CI报告。
3. D相对native outer change每seedcoverage不下降，按保留family3口径区间下界>0；outer每组旧成功lost率≤1%；overall/stable/transition ADE/FDE/J≤1%伤害。
4. D全部4split×3seed×4group五motion合格；完整审计通过。

未满足重大成功的每seed+5pp覆盖、每seedADE/FDE各≥5%及所有原门，即使REPLICATION_SIGNAL成立，也只称“值得进一步验证的模式保持修正机制”，不称大突破。

## 5. 整个系列的预定解释表

| 观察 | 可以说什么 | 不能据此做什么 |
|---|---|---|
| W0有严格见证，P1无迁移 | 有限样本有机会，当前共享映射/目标未获得稳定收益 | 不直接说历史信息不足；不自动扩模型 |
| W0无见证，P1也失败 | 当前两条有限方法均无正证据 | 不宣称安全域空或生成不可能 |
| B/D比A/C更差 | 固定native责任在此设置不利 | 不以“标签不够精确”为由无限更换oracle |
| C/D更保守但rescue减少 | 保护—可塑性折中，可能退向identity | 不以lost降低代替生成成功 |
| D在P1好、P2弱 | 对时间隔离population的稳健性不足 | 不唯一归因窗口泄漏，不能偷换primary为track敏感性 |
| D在P1/P2均好，但五motion失败 | 候选质量有信号但不可按当前标准采用 | 不调物理阈值，不宣布安全 |
| P1/P2全部原门成立 | 支持此输入/候选域下的可迁移模式保持修正 | 不证明心理意图、全模型未见泛化或SOTA |

系列结束必须综合F0/I1/T1/T2及全部新阴阳性，而非仅展示D的最好seed。P2没触发也是完整的有限研究结果，不通过凑第三项或第四变体延续。

## 6. 后续大分支只作为建议

若P1/P2阴性，下一轮应重新评估上游表征的条件信息、native支撑选择或更广输出域；这些需要独立设计，不是本系列附带实施。特别是旧F16/decoder/scorer解冻、换数据或新概率目标，不在此文件授权范围内。

本系列不运行额外动态路由、teacher蒸馏、簇阈值搜索、专家数/宽度/LR/epoch扫描，也不恢复旧暂停worker或自动化。
