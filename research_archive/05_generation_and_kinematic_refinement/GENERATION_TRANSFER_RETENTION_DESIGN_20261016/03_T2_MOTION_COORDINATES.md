# T2：相同信息的运动坐标能否改善修正迁移？

状态：DESIGN_ONLY。只新增C×seed42/43/44的3runs，与本系列A比较；A完整且合同兼容才能复用。

## 1. 为什么选择这个有限结构测试

U1说明平均20条候选有利，不等于关键覆盖候选得到有利修正。当前head用P2+native位置展开；本试验问：把同一条路径写成运动增量，能否降低学习修正时对位置模式的依赖？

这里刻意不同时加入额外观测、候选之间attention、MoE、rotation-normalization、新loss或保留项。否则结果很难解释。它比再调残差幅度更接近“共享映射如何使用运动信息”，但只是运动表示的一次有限检验，不是完整运动理解模型。

过去candidate-state、matching、selector阴性仍有效；不能仅换名称宣称新方向。本设计的区分点是：在U0完整K20、已解除近零安全投影的同一输出空间内，采用**信息可逆、维数/参数严格相同**的输入重参数化，不引入新的可修复性teacher或旧slot选择任务。

## 2. 唯一因素：24维位置改为24维速度

缓存已核对past最后点为局部原点0。令p0=0，p1…p12为immutable native未来候选：

\[
v_t^0=(p_t^0-p_{t-1}^0)/0.4,\qquad p_t^0=0.4\sum_{j=1}^tv_j^0.
\]

A输入 `[P2_96, flatten(p^0)_24]`；C输入 `[P2_96, flatten(v^0)_24]`。P2不旋转、不换编码器；native base仍是原p0路径，输出控制仍用原全局局部坐标中的integrated_acceleration，不变成速度增量输出的新算子。

速度feature计算精度固定：先按原生产边界取native float32；在float64中差分除0.4，最后cast float32作为feature，再用FIT-only float64统计标准化。保存三个精度阶段的契约。独立double变换必须可逆；float32量化的重建误差单独报告，不能声称逐bit等价或改native base。

零速度、停车、反向、转弯均保留，不按速度剔除样本；无除以速度、heading不确定或小速度floor。v1保留，不能只用后11个加速度丢失初速度。不能把native预测速度误称真实未来速度。

这是信息量不增加的试验：即使成功，也更接近有限预算下的归纳偏置/条件数改善，而不是获得了新观测。若失败，不能据此否定显式历史、交互或所有运动表示。

## 3. 对照可辨识性

- A/C权重初值逐tensor一致，参数25304、全部控制和预算一致；各自位置/速度标准化不同是唯一输入表示因素的组成部分。
- A的raw统计/训练来源/生成输出必须与T1 A完全一致；统计实现不能因C开发而重算覆盖A文件。
- 变换可逆只在信息层成立；非线性优化、各维缩放和有限25epoch路径不同。不得宣传“已隔离纯运动语义”或“共享干扰唯一根因”。
- C不使用B的权重、标签辅助输出、保留项或warm start。B再优秀也不能在T2偷偷换对照。

## 4. 测什么

除共同协议全指标外，输出以下事前诊断：

1. native-failure上固定native j0初始dJ，ALL20均值，最终fixed-j0 ΔJ，及对应rescue是否真实发生；近tie不确定层保留。
2. 原生成功lost、不同G0个数、near-boundary原三层，不只看修复成功案例。
3. FIT→outer的方向改善差。两人口bootstrap独立抽track，共享modelseed；此项描述性95%CI，不替代主family24判定。
4. inner与CAL同样表格；CAL不选择超参数，也不能在outer阴性时升为primary。
5. 轨迹早中晚误差与全部五motion：输入速度化不保证输出更平滑、更安全。

## 5. 预注册判定

**TRANSFER_SIGNAL**要求C−A在outer change同时满足：

1. coverage平均增加≥2pp，三个seed均不下降，family24差区间下界>0；
2. rescue rate平均增加，且lost rate平均不增加、其family24差区间上界≤+2pp；
3. fixed-j0 actualΔJ平均减少，三个seed方向一致，family24差区间上界<0；
4. 所有源/ID/数值审核通过；全motion/protected group完整披露。

2pp是用于决定是否做一次组合的局部信号，不是+5pp最终覆盖目标。若只有ALL20平均方向改善而coverage无改善，判定“平均路径改善未转化为关键候选收益”，不进入D。

如果FIT更好、outer/CAL不变或更差，说明这次表示重参数化没有解决迁移，停止此表示分支，不再追加速度+加速度+jerk拼接、P2宽度、学习率扫描。若FIT和outer都无改变，不能宣称信息不够，只能报告在当前预算/架构下没有可检测收益。
