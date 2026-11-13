# 方向C：集合训练与跨候选交互——最终结果

C1与C2分别独立单因素，未叠加；两项正式训练均完成，但均未通过科学机制与基线门。

## C1：hard vs detached-aWTA

固定592step温度.20→.02，旧hard任务/190pair endpoint线性hinge不变，82008参数。三hard最终FIT/inner路径全部与原A逐字节相同，是控制复现，不新增独立泛化。

| 臂 | seed | good/497 | rescue | lost/166 | minADE(m) | minFDE(m) | 有效good簇/窗 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1_hard | 42 | 186 | 53 | 33 | 0.3066 | 0.5781 | 0.618 |
| C1_hard | 43 | 191 | 62 | 37 | 0.2814 | 0.5190 | 0.517 |
| C1_hard | 44 | 192 | 63 | 37 | 0.2911 | 0.5118 | 0.539 |
| C1_awta | 42 | 146 | 41 | 61 | 0.3841 | 0.7718 | 0.505 |
| C1_awta | 43 | 146 | 37 | 57 | 0.3762 | 0.7570 | 0.495 |
| C1_awta | 44 | 166 | 47 | 47 | 0.3498 | 0.6988 | 0.586 |

awta-hard change覆盖均值-7.445pp，95%CI[-12.250, -3.512]pp；逐seed−40/−45/−26窗。ADE差+0.07696m，CI[0.052, 0.108]；FDE差+0.20623m，CI[0.142, 0.281]。这组配对探索区间支持当前固定日程造成实际退化，不仅是区间宽。awta change lost61/57/47高于hard33/37/37，rescue41/37/47低于53/62/63。FIT change也从2156/2083/2135降至1730/1516/1746，不是仅inner迁移问题。

原hard和awta最后一轮hard-active均20，不可只由活跃槽数量认定“消除塌缩”。awta末轮权重熵约.593/.924/.592，winner-switch约.79高于hard .476/.378/.382；这是过程描述，不足证明不稳定唯一根因。coverage/retention/errors三个局部门false，AWTA_SCHEDULE_NOT_SUPPORTED；不扫温度或追加epoch。

## C2：相同残差块self/self/self vs self/mean/amax

同98552新参数、同配对起点/顺序，仅跨候选统计输入不同；相同参数不等于相同有效容量或优化几何。

| 臂 | seed | good/497 | rescue | lost/166 | minADE(m) | minFDE(m) | 有效good簇/窗 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C2_isomlp | 42 | 170 | 49 | 45 | 0.3114 | 0.5613 | 0.517 |
| C2_isomlp | 43 | 178 | 51 | 39 | 0.2984 | 0.5545 | 0.561 |
| C2_isomlp | 44 | 164 | 48 | 50 | 0.3010 | 0.5648 | 0.531 |
| C2_setmix | 42 | 165 | 47 | 48 | 0.3676 | 0.7097 | 0.712 |
| C2_setmix | 43 | 174 | 59 | 51 | 0.2965 | 0.5450 | 0.573 |
| C2_setmix | 44 | 193 | 64 | 37 | 0.2969 | 0.5477 | 0.636 |

mix-iso变化为−5/−4/+29窗，均值1.341pp，未达+2pp且不是三seed均正；配对95%CI[-3.798, 6.571]pp，固定seed敏感性[-0.964, 3.421]pp亦跨0。相对旧A，前两seed仍低于186/191，44的193略高于192是反例。effectivegood均值确从约.536增至.640簇/窗，这一局部门true，但不能最大化簇数替代覆盖与误差。

mix change ADE均值较iso+0.01676m、FDE+0.04059m；配对主CI跨0，固定seed敏感性反而明确偏坏，不能择有利区间。mix lost48/51/37 vs iso45/39/50，前两seed变坏；errors_retention不通过。SETMIX_RESIDUAL_NOT_SUPPORTED仅限该残差接口+8轮，不否定所有跨候选通信。

## 运动、基线与真实成本

C1 hard/awta、C2 iso/mix全部各0/24 G_base，五运动联合也各0/24。awta内层change accelmean较native+66.06/+71.65/+25.92%；mix相应+48.40/+27.71/+20.06%，均不满足1%门，即使jerk尾部部分更低。逐组720项运动表保留所有正反例。

C2整模267159冻结encoder+98552新decoder=365711参数，超过305000预算，不能称整模更轻。复核发现原profile的unsupported算子筛选遗漏linear/matmul/融合Transformer，标准counter未完整计入融合层；原11,846,966 MAC/窗是低估，原锁定代码和报告原样保留。新独立顶层算子counter补齐QKV/O/FFN及注意力矩阵乘，无嵌套双计数，实际整模两臂均17,265,974 MAC/窗（FMA=2，即34,531,948FLOPs；不含非MAC逐点/非线性）。分解linear11,738,912、matmul4,374、融合Transformer5,419,008、bmm41,472、einsum62,208。新测warmGPU batch1 iso9.39ms/mix10.66ms，batch256 iso15.03ms/mix14.31ms/批；GPU allocated峰约242.7MiB。短warm测量不含磁盘/数据准备，快慢小差不能据此声称稳健加速。原前检以不完整成本记录放行的偏差永久披露，补充测量修正当前交付，不倒改原生产前置记录。

补测V1的独立CPU缩并回归又暴露einsum漏计62,208 MAC，V1源码与结果未覆盖，新增V2计数器和PROFILE_COST_SUPPLEMENT_V2.json；双操作数显式einsum逐索引维度计数，2×3×4与2×4×5固定fixture由0纠正为120 MAC。最终17,265,974恰为原11,846,966加完整融合Transformer5,419,008；这进一步交叉验证遗漏来源。当前报告采用V2，不把V1冒称完整。

## 统计、反证与终止

区间为20000次default_rng(20261023)的同ID差值、共同target-track与配对model-seed交叉bootstrap：每draw先抽3个配对seed，再抽35条track，按draw内窗口数加权，最后均seed。固定seed敏感性不抽seed，按其原始RNG顺序单独复核；没有因跨0而替换主区间。所列95%CI未做多比较校正，是预定局部门的探索区间，不能包装成跨全部方向的确认性显著结论。native参考不是3个独立模型。B只有单个确定性探针人口，不伪造3模型seed。

仅FIT18909训练、inner1856评价；inner change497窗/35条target tracks，三个配对seed42/43/44。native change166/497，原A为186/191/192。本轮不读取outer/CAL/test，不更改旧F16、decoder/scorer、数据或评价阈值。F16上游已经见过FIT和inner、inner反复用于开发，target-disjoint不等于context-disjoint；结果是探索性机制证据，不是独立整模泛化、SOTA、心理意图或唯一根因证明。

DA2（同助手非外审）：C2 seed44覆盖和有效good增加是正例，禁止用另外两seed阴性说“候选交互绝无机会”。DA3：C2较弱的iso控制、全基线未就绪、重用inner和未校正多比较使小幅收益不足晋升；C1显著探索性退化只适用于这套固定日程。成本报告原错误及修复明确披露，不修改原锁定数据。有限两项完结，无新增第四方向、loss/scorer或旧解冻。

证据：C1_execution_20261024/DECISION.json、C2_execution_20261024/DECISION.json及总completion_review/。
