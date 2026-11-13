# 解码端三方向有限系列——最终报告

状态：ANALYZED_FINITE_SERIES_COMPLETE；计算执行成功，三个方向未达到各自完整科学门，无baseline晋升或后续自动训练。

## 一页结论

| 方向 | 实际发现 | 科学判定 |
|---|---|---|
| A 时间基底 | oracle/L2筛查两基底均通过；实际训练加速度有明确下降，但change覆盖收益未保留，native运动保护仍失败 | TIME_BASIS_NOT_SUPPORTED |
| B 场景先验 | static/causal NLL较M0分别恶化26.31%/24.44%，Opp4不胜打乱，均无信息门支持 | 本固定直方图分支终止；未训练B生成器 |
| C1 aWTA | change覆盖平均−7.445pp，ADE/FDE均变坏，原成功丢失增加 | AWTA_SCHEDULE_NOT_SUPPORTED |
| C2 集合交互 | 平均+1.341pp，逐seed−5/−4/+29窗，CI跨0且保护失败 | SETMIX_RESIDUAL_NOT_SUPPORTED |

不存在“没有训练”的pending状态：A screen通过后实际做A-Train，C1与C2也全部完成。18fresh runs各8轮=144epochrows/10656optimizersteps，三个正式实验；A-Screen与B-Probe CPU诊断不增formal。原baseline/historycycle11/长期目标保持，不为有限阴性结果宣称总体目标完成。

## 执行来源与时间

Windows唯一按需0trigger任务Trajectory-DecoderThreeDirections-20261024，worker2026-10-24 00:19:46.133524启动→00:52:31.178153双SUCCESS，实际32分45.045秒；阶段A_SCREEN→B_PROBE→A_TRAIN_CONDITIONAL→C1→C2。原计划70min/45–105min不是生产保证。唯一50%预计elapsed节点原00:54:46.133524，实际00:55:26.3296911一次核验，已完整；不是“训练恰好50%”。没有重启或重复startup/旧PID轮询，stderr0。当前补充是保存审核与成本修复，原18模型未重训。

## 数据、判据与完整性

仅FIT18909训练、inner1856评价；inner change497窗/35条target tracks，三个配对seed42/43/44。native change166/497，原A为186/191/192。本轮不读取outer/CAL/test，不更改旧F16、decoder/scorer、数据或评价阈值。F16上游已经见过FIT和inner、inner反复用于开发，target-disjoint不等于context-disjoint；结果是探索性机制证据，不是独立整模泛化、SOTA、心理意图或唯一根因证明。

每臂K20完整12步，good要求同candidate ADE<=.2 AND FDE<=.4，J=max(ADE/.2,FDE/.4)；Q=ADE+.5FDE独立单槽评价，不拼不同候选。GT-free complete-link .10meanpath/.20endpoint统计distinct/effectivegood，不能用copies充数。速度p95、加速度mean/p95、jerkmean/p95按dt=.4、全部20×12/11/10时间轴汇总；五项分别不得超过native1.01，首步velocity_join另列。没有因本轮允许运动代价分析而放宽科学成功阈值。

区间为20000次default_rng(20261023)的同ID差值、共同target-track与配对model-seed交叉bootstrap：每draw先抽3个配对seed，再抽35条track，按draw内窗口数加权，最后均seed。固定seed敏感性不抽seed，按其原始RNG顺序单独复核；没有因跨0而替换主区间。所列95%CI未做多比较校正，是预定局部门的探索区间，不能包装成跨全部方向的确认性显著结论。native参考不是3个独立模型。B只有单个确定性探针人口，不伪造3模型seed。

## A：确有平滑作用，不等于联合目标成功

筛查knot3/svd4都满足GT重构>=98%、旧A净增保留>=80%、平均acc下降>=15%；按投影覆盖选svd4。实际CV-basis inner change good165/165/182，CV-free163/176/175，旧A186/191/192、native166。basis-free差−.134pp，95%CI[-4.790,+3.506]pp。accmean差−.021748m/s²，CI[-.028869,−.016630]，逐seed下降约17.1/21.2/16.3%，只有一个seed达到20%门。保留“平滑有效”的局部证据，但native accelmean仍+7.00/+14.38/+1.97%，基线0/24。详见[方向A报告](direction_A_time_basis/RESULT_ANALYSIS.md)。

## B：所选场景统计不能支持生成器投资

主change497全可核验时间，native-failure331；M0 NLL4.05797，static5.12560（差+1.06762，95%CI[+.66317,+1.37041]），causal5.04975（差+.99177，CI[+.45580,+1.31769]）。两先验Opp4均35/331，低于各自打乱41/331、37/331，M0为38/331。三项B门全false。不能由此推断所有场景信息无用；当前只是固定桶/支持层次/回退规则的负结果，Opp4不是完整路径coverage。详见[方向B报告](direction_B_scene_prior/RESULT_ANALYSIS.md)。

## C：软分配退化，集合交互证据不足

C1 hard原A复现186/191/192，awta146/146/166；平均−7.445pp，95%CI[-12.250,−3.512]pp；ADE+.07696m、FDE+.20623m，主CI均偏坏。C2 iso170/178/164→mix165/174/193，只有seed44正；平均+1.341pp，CI[-3.798,+6.571]pp，未达联合+2pp/三seed正/保护门。mix有效good确增加，为正例但不取代覆盖。C1/C2每臂均0/24 G_base，所有原成功保留/五运动保护未达标。详见[方向C报告](direction_C_set_mechanisms/RESULT_ANALYSIS.md)。

## 成本统计的发现与修复

C2整模267159冻结encoder+98552新decoder=365711参数，超过305000预算，不能称整模更轻。复核发现原profile的unsupported算子筛选遗漏linear/matmul/融合Transformer，标准counter未完整计入融合层；原11,846,966 MAC/窗是低估，原锁定代码和报告原样保留。新独立顶层算子counter补齐QKV/O/FFN及注意力矩阵乘，无嵌套双计数，实际整模两臂均17,265,974 MAC/窗（FMA=2，即34,531,948FLOPs；不含非MAC逐点/非线性）。分解linear11,738,912、matmul4,374、融合Transformer5,419,008、bmm41,472、einsum62,208。新测warmGPU batch1 iso9.39ms/mix10.66ms，batch256 iso15.03ms/mix14.31ms/批；GPU allocated峰约242.7MiB。短warm测量不含磁盘/数据准备，快慢小差不能据此声称稳健加速。原前检以不完整成本记录放行的偏差永久披露，补充测量修正当前交付，不倒改原生产前置记录。

补测V1的独立CPU缩并回归又暴露einsum漏计62,208 MAC，V1源码与结果未覆盖，新增V2计数器和PROFILE_COST_SUPPLEMENT_V2.json；双操作数显式einsum逐索引维度计数，2×3×4与2×4×5固定fixture由0纠正为120 MAC。最终17,265,974恰为原11,846,966加完整融合Transformer5,419,008；这进一步交叉验证遗漏来源。当前报告采用V2，不把V1冒称完整。

## 保存审核与资源

全部18个checkpoint及resume的保存模型/optimizer张量/RNG/assignment/history合同逐项核对，8轮×592步/模型，共144epochrows、10656更新。36个科学人口NPZ、180个done绑定输出SHA、144组几何/运动、720单项运动门、576个固定旧A/native子集已复核；36个生产独立完整SciPy聚簇审核全部PASS，补充审核不第三次声称重跑聚簇。另A12投影人口/4oracle保存路径重构和12完整聚簇审核均通过。B八臂全部1856查询（一个未知时间查询的两个causal臂无效单列）用第二套向量化资格实现逐项核对source membership、257桶、NLL/Opp4/回退，主change497全部时间可核验。全部主/敏感性20k区间由原始窗口乘重第二实现重算；143134个补充比较最大绝对差1.3323e-15。源锁根652保留615继承SHA；各训练合同扩展合计655项源/输入SHA前后不变。保存算术审核不是所有模型/梯度再训练重放；仅成本修复额外做冻结encoder→新decoder的FIT有限前向，无backward或optimizer实例化。

所有生产阶段1.3动态available/remaining+Windows retainedpeak原记录均通过；CPU E3GiB/GPU E6GiB来自inventory而非固定硬阈值，无OS/pagefile调整或用户程序关闭。各阶段真实生产峰和最低物理/commit见completion_review/PRODUCTION_RESOURCE_REVIEW.json，不用预检或补充审核峰替代。原模型、代码、失败、source manifest及不完整profile均保留。

## 完整表与可复核输出

- completion_review/ALL_GROUPS.csv：144组n/track/good/rescue/lost/ADE/FDE/J/effectivegood/G_base。
- completion_review/ALL_MOTION.csv：720项逐seed×split×group五运动及严格判定。
- completion_review/FROZEN_OLD_SUBSETS.csv：576固定旧A/native成功、救回、丢失、皆失败子集；原JSON保留IDs和有符号误差贡献。
- completion_review/A_SCREEN_ALL_GROUPS.csv、B_PROBE_REVIEW.json、BOOTSTRAP_REVIEW.json与各draw NPZ：保留筛查、八臂人口和完整20k复算。
- CHECKPOINT_CONTRACTS.json、SOURCE_OUTPUT_SHA.json、CONTRACT_REVIEW.json、PROFILE_COST_SUPPLEMENT_V2.json：来源、模型、RNG/合同、成本修复。
- 各run metrics_*.json保留完整救回/丢失ID、J/Q槽位与分段、motion和correction；done.json绑定全部输出SHA。无只报净增的筛选。

## 同助手反方审核与边界

使用academic-research-suite的证据/反证结构与nature-statistics的重复单位、分母和CI审查；同一助手执行DA2/DA3，不是外部或独立审稿人。

DA2：A加速度有明确改善、A整体覆盖改善、C2 seed44及effectivegood上升、B非change人口有局部正例，禁止删除这些反例以支持“全无效果”。C1当前固定日程反而有一致退化证据。DA3：oracle表示能力≠可学习安全增益；CI跨0≠等价无效；同参数≠同优化几何；相关窗口及反复inner开发限制迁移结论。最强反驳是更充分优化/其他条件表示可能改变结果，本轮并未排除它；但不能以此改原门或无限追加变体。

最终行动：按合同结束本有限系列，不自动扫基底秩/温度/layer/lambda/epoch，不扩旧F16/scorer/data权限。仅删除本decoder-three-directions一次性跟进，旧PAUSED监控不动。后续新路线须另有具体设计与人类请求。
