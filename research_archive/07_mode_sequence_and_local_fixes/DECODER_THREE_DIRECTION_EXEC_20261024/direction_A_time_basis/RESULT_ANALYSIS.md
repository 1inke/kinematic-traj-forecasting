# 方向A：固定时间基底——最终结果

结论：TIME_BASIS_NOT_SUPPORTED。低维表示有平滑效果，但未同时保留原有覆盖收益；不晋升baseline。

## A-Screen与实际训练不是同一结论

两固定基底均通过预注册筛查：knot3的inner-change GT重构100%、旧A净增保留90/100/88.46%、平均加速度比0.54147；svd4相应100%、100/100/96.15%、0.74035。按原最高投影覆盖规则选svd4（186/191/191对knot3的184/191/189），不是事后选更平滑的knot3。GT投影是特权L2表示诊断，不是部署收益或安全可达证明。基底只由FIT非中心SVD拟合并在inner结果前锁定，补充已重算。

A-Train用共同CV初始状态和82008参数，free与固定svd4投影单因素，8轮/3seed；原A仅辅助参考，不能把CV-free结果预设与原A相同。

| 臂 | seed | good/497 | rescue | lost/166 | minADE(m) | minFDE(m) | 有效good簇/窗 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_cv_free | 42 | 163 | 50 | 53 | 0.3109 | 0.5858 | 0.479 |
| A_cv_free | 43 | 176 | 57 | 47 | 0.2973 | 0.5578 | 0.445 |
| A_cv_free | 44 | 175 | 55 | 46 | 0.3030 | 0.5601 | 0.449 |
| A_cv_basis | 42 | 165 | 49 | 50 | 0.3054 | 0.5781 | 0.463 |
| A_cv_basis | 43 | 165 | 54 | 55 | 0.2852 | 0.5190 | 0.378 |
| A_cv_basis | 44 | 182 | 56 | 40 | 0.2997 | 0.5575 | 0.509 |

加速度均值basis相对free逐seed降低约17.1%、21.2%、16.3%；平均差-0.02175m/s²，配对95%CI[-0.029, -0.017]，支持真实平滑效应。但42/44未达到每seed至少20%下降，且basis相对native的accelmean仍+7.00/+14.38/+1.97%，严格1%全失败。仅seed43 inner-stable五运动项全过是保留反例，不是整体过门。

覆盖basis-free均值-0.134pp，95%CI[-4.790, 3.506]pp；不证明等效。相对原A的正净增保留−5%、−4%、61.54%，均未达90%。basis-inner整体1041/1013/1025高于free1018/979/1020，是正例；不能用整体替代预注册change门。ADE/FDE平均略降但配对seed主CI跨0，且逐seed保护不齐，coverage/acceleration/errors三个局部门皆false。

## 运动与保留

全部24组G_base：free0/24、basis0/24；五运动联合：free0/24、basis1/24。basis change原正确候选丢失50/55/40（分母166），远高1%门；rescue49/54/56不能只报净增。所有FIT/inner×4group及全部五运动详见总completion_review/ALL_GROUPS.csv、ALL_MOTION.csv，救回丢失ID留在各metrics与SHA绑定NPZ。

## 统计、边界与反证

区间为20000次default_rng(20261023)的同ID差值、共同target-track与配对model-seed交叉bootstrap：每draw先抽3个配对seed，再抽35条track，按draw内窗口数加权，最后均seed。固定seed敏感性不抽seed，按其原始RNG顺序单独复核；没有因跨0而替换主区间。所列95%CI未做多比较校正，是预定局部门的探索区间，不能包装成跨全部方向的确认性显著结论。native参考不是3个独立模型。B只有单个确定性探针人口，不伪造3模型seed。

仅FIT18909训练、inner1856评价；inner change497窗/35条target tracks，三个配对seed42/43/44。native change166/497，原A为186/191/192。本轮不读取outer/CAL/test，不更改旧F16、decoder/scorer、数据或评价阈值。F16上游已经见过FIT和inner、inner反复用于开发，target-disjoint不等于context-disjoint；结果是探索性机制证据，不是独立整模泛化、SOTA、心理意图或唯一根因证明。

DA2（同助手非外审）：最强反例是加速度确实改善、整体覆盖改善、seed44 change有正收益；因此不能说时间低维完全无用。DA3：screen oracle成功并不保证学习成功；目前只否定此固定svd4+共同CV+8轮配方已达到联合目标，不证明全部低秩不可行或未来不可预测。保留固定合同停止，不扫描rank/knot/epoch。

证据：screen_execution_20261024/RESULT.json、train_execution_20261024/DECISION.json及总completion_review/，无额外训练。

