# 方向B：场景直方图先验探针——最终结果

结论：SCENE_HISTOGRAM_PRIOR_FALSIFIED（仅该预注册257桶/层次回退配置）。没有触发条件生成器设计或训练。

FIT库18909，仅历史endpoint作库；inner1856查询，全层排除targettrack，同序列causal只取end<=t0。inner-change497/35tracks全部时间可核验，其中native失败331窗；全inner另1个未知时间查询保留注册，不进入两个causal臂有效人口。SOURCE索引不是未经核实的真实视频帧或秒。M1虽命名heading_global，实际匹配scene+heading，避免夸称无场景对照。

| 固定臂 | change NLL↓ | native失败Opp4↑ |
|---|---:|---:|
| M0_motion_iso | 4.0580 | 11.48% |
| M1_heading_global | 4.1840 | 12.08% |
| PRIOR_static | 5.1256 | 10.57% |
| SHUF_static | 5.1233 | 12.39% |
| XSCENE_static | 4.9360 | 7.55% |
| PRIOR_causal | 5.0497 | 10.57% |
| SHUF_causal | 4.9736 | 11.18% |
| XSCENE_causal | 4.9360 | 7.55% |

真实static先验NLL5.1256对M0 4.0580，相对恶化26.31%，差+1.06762nats，track bootstrap95%CI[0.663, 1.370]。causal先验NLL5.04975，相对恶化24.44%，差+0.99177，CI[0.456, 1.318]。两个配置B1/B2/B3均false。

static和causal的Opp4都35/331=10.574%，但分别低于打乱先验41/331和37/331；M0为38/331。Opp4是4个桶中心距真终点<=.4的端点机会，不是完整轨迹coverage，不可与生成器good拼同一指标。没有正M0增益，advantage_ratio按合同为null，不对负分母制造优势。

所有1856查询八臂的257桶和完整FIT source bitset已保存；生产只独立审核概率算术，补充另外实现了全部资格/层次/打乱/因果回退/source成员重算，全一致。causal使用相同497 change人口比较，无删难例。各group的cold-start/层次支持及32行人口结果见RESULT.json的all_groups；不能把回退人口当全部可靠同场景已知信息。

区间为20000次default_rng(20261023)的同ID差值、共同target-track与配对model-seed交叉bootstrap：每draw先抽3个配对seed，再抽35条track，按draw内窗口数加权，最后均seed。固定seed敏感性不抽seed，按其原始RNG顺序单独复核；没有因跨0而替换主区间。所列95%CI未做多比较校正，是预定局部门的探索区间，不能包装成跨全部方向的确认性显著结论。native参考不是3个独立模型。B只有单个确定性探针人口，不伪造3模型seed。

仅FIT18909训练、inner1856评价；inner change497窗/35条target tracks，三个配对seed42/43/44。native change166/497，原A为186/191/192。本轮不读取outer/CAL/test，不更改旧F16、decoder/scorer、数据或评价阈值。F16上游已经见过FIT和inner、inner反复用于开发，target-disjoint不等于context-disjoint；结果是探索性机制证据，不是独立整模泛化、SOTA、心理意图或唯一根因证明。

DA2（同助手非外审）：overall与stable的某些条件NLL/Opp4可改善，说明不能将change阴性外推所有人口。DA3：当前表征以worldcell/heading筛选，但库endpoint在各自local坐标，分箱分辨率/稀疏/相关窗口和回退均限制解释；阴性不是场景信息无用或未来不可预测。只终止本固定直方图分支，不扫桶宽/半径或自动生成器。

证据：probe_execution_20261024/RESULT.json、probe_arrays.npz与总completion_review/B_PROBE_REVIEW.json。

