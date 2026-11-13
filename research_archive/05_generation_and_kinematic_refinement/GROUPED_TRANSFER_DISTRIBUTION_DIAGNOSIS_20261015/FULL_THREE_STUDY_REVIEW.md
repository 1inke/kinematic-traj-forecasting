# 三项分组迁移研究总复盘（保存结果复核完成）

2026-10-15。此前 THREE_STUDY_DOCUMENTARY_REVIEW.md 是资源阻碍时的文档快照；本报告由人类“继续”后实际计算结果补全。三项正式对照均早已完成；本次冻结审计不增正式训练数量。

## 全部假设和对照

|研究|原假设/曝光因素|random/grouped训练数|共享统计交集|相同outer / inner|原主bootstrap|
|---|---|---:|---:|---:|---|
|目标轨迹|同一目标相关窗口曝光|18909 / 18909|15572|1035 / 1856|目标track×3seed|
|录制时间|同sequence完整20源索引区间曝光，固定50索引块|16104 / 16104|12544|566 / 2134|timeblock×3seed；track敏感性|
|zara场景族|是否有任意zara训练窗口|16337 / 16337|12183|1561 / 1773|目标track×3seed，3recording仅描述|

前两项sequence×原lossgroup匹配；第三项只有global lossgroup匹配，不能维持scene组成。三个实验只训练同一91596参数uniform双rawpersistent专家、同一训练内endpoint保持acceleration投影与原total+.1P、80epoch×3pairedseed；旧F16/decoder/scorer固定。原CAL2879事后评估，不选择checkpoint；primary为固定epoch80，constrained/legacy为secondary。跨项训练人群与统计不同，因此不能作等预算单因素跨项排名。

## 全部主结果和区间

|隔离因素|变化native好窗口|random 3seed|grouped 3seed|grouped-random变化覆盖pp及原95%探索CI|整体覆盖pp及原CI|
|---|---:|---|---|---|---|
|目标轨迹|110 /253|113,115,113|107,108,109|-2.240 [-6.241,+1.660]|-0.870 [-3.198,+1.177]|
|时间|60 /142|59,58,62|41,53,61|-5.634 [-13.732,+5.556]|-3.357 [-7.843,+0.885]|
|场景|82 /211|84,77,80|73,72,70|-4.107 [-7.985,+0.511]|-2.178 [-3.671,-0.691]|

三项grouped-random变化ADE差+.014755/+.013908/+.026371m，FDE+.029233/+.047737/+.073420m。整体连续误差区间方向更一致，但严格覆盖跨阈值的证据较弱；场景overall例外。第一项变化连续误差原探索区间不跨0，第二项变化timeblock主区间均跨0，第三项变化误差/J区间为正而coverage仍跨0。

所有648个seed/policy/split/group行和720条原探索区间完整保存在 FULL_THREE_RECORDS.json / FULL_THREE_EXPLORATORY_CIS.json，包含ALLpolicy/ALLgroup。非只选final好结果；没有重算或重新挑选区间。独立组聚合核对保存质量/救回/损失。三个研究不同outer不能相加为一个paired试验，多指标适应性探索不升级确认性推断。

## Coverage、有效候选与运动学

三项均使用同candidate ADE<=.2 AND FDE<=.4/J<=1，不拼独立minADE/FDE或分段oracle。Effective-good使用原ALLK20、GT-free完整聚簇；第一项grouped整体有效good簇平均多于random但coverage较弱，第三项grouped某seed簇多也没有稳定的完整窗口收益。数量不能替代覆盖，更不能用复制K20补数量。Inheritednativeargmax只是沿用槽位，原scorer未对新轨迹重新估概率。

五项pooled motion原严格1%失败行数：5/216、2/216、18/216，均有speedp95反例。投影保护每window acceleration mean不保证speedp95或全部group/motion；不放宽1%或用均值抵消尾部。三项全部原major FAIL、baseline及未完成长期目标不变。

## FIT/CAL、容量、重复与历史反证

训练loss各项可下降，grouped在主outer连续误差却更差。不同训练集的batch-loss不能当同ID目标下降因果证据；三项此处没有额外完整FIT模型输出，不能伪称全部训练质量重放。共享inner在目标/时间研究的变化覆盖可改善，zara原CAL中grouped在部分seed优于random；这些反例约束“完全不能泛化”。原CAL本身只university，native F16已见旧数据；outer目标隔离不是neighbor/context独立。

当前结构承接曾有raw candidate-state微小探索增益、discrepancy中心化/自身scaler未恢复、物理P降低加速度代价但CAL和major失败、candidate routing及train projection未解决全部标准。GTwitness/freeincrements/timequery/velocitybasis/conditioning/mode/social/soft-energy-mixture/tinyFITCAL等失败与可达性限制仍保留，不能把本研究当第一次发现泛化问题。最近固定motion责任T1在FIT已弱且运动学FAIL，否定唯一CALshift解释。

每臂名义同参数并不消除有限优化、样本梯度/组成的差异；shared control重复运行不是新独立数据泛化。18fresh trainings×80=1440历史行、54policy checkpoint不等于54独立训练。

## 实际审计范围

- 三项worker/exit保存双SUCCESS、全部6×80历史和原ARITHMETIC/FROZEN/PRESERVATION PASS。
- 本次逐171科学savedjob从保存轨迹/GT独立重算ADE/FDE/Q/J/coverage，误差最大0；核对组分母及rescue-lost恒等式。
- 54模型、342输出及合并589相关科学source、读入缓存/metadata/splits/报告/history等共1051文件在初末SHA均一致；无历史全34k再次核验声明。
- 原complete-link和五项motion使用既有原审计与SHA；本次不是全部模型前向、所有梯度、全部聚簇/motion或720CI重新采样。SUMMARY_AUDIT另核保存标量分区、motion布尔和坏SHA真实拒绝。

## 总体研究决定

完整三项复盘已完成3/3；不启动第四个局部split/块宽/族比例/路由/专家数/权重/LR/epoch变体。新固定分布诊断发现：CAL在所测输入摘要的支持外率接近源基准，zara候选状态shift更大，但三项共同支持人群内仍有迁移差距。因此继续按照条件映射稳定性与成功保护方向准备精确对照，具体共享head干扰、objective或信息不足根因尚未唯一可辨认。

研究方法/反方2和3已列 RESULT_ANALYSIS.md：同助手inline、AI辅助、非外部独立审稿。3scene、3seed、重叠window、F16seen和适应性探索局限保留；无SOTA、心理意图、不可预测比例或wholemodel未见泛化声明。
