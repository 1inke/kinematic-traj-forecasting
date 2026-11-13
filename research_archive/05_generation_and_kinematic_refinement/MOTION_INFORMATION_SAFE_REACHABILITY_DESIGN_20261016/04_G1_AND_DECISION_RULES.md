# 04｜G1条件生成验证与系列决策

状态：DESIGN_ONLY。只有F0和I1双门成立才执行G1；不满足时记录NOT_TRIGGERED，不将其计作已完成实验。

## 1. 触发合同

必须同时具备：

1. F0 `SAFE_WITNESS_PRESENT=true`：真实生产精度重构的非零安全救回存在，完整FIT portfolio的五运动学通过，保存与源审核完成。
2. I1 `INFORMATION_READOUT_SIGNAL=true`：按03文档的family2、逐seed效应和保护门成立。
3. 两阶段无未解释的source/ID/资源/算术问题；G1自身实际预检、资源及≤30分钟全流程预算通过。

F0一个正例仅排除“所测域绝无正例”，远不足以保证+5pp机会或能学习。双门只为有限G1提供理由，不是成功预告。

## 2. G1的唯一臂间因素

继续I1的S静态历史/H真实历史、相同146维输入、28632参数、共同FIT统计、三seed paired init。两臂均fresh训练，不从I1、F0或任何旧head warm-start。

重新使用U0的全K20有界时间耦合生成：

\[
z_{jt}=f_\theta(P2,p_j^0,h)_t,\quad
u_{jt}=\frac{\tanh z_{jt}}{\max(1,\|\tanh z_{jt}\|_2)},
\]
\[
r_{jt}=\frac{0.20}{144}\sum_{q=1}^{t}[2(t-q)+1]u_{jq},\qquad
\hat p_{jt}=p^0_{jt}+r_{jt}.
\]

保持head float32、生产控制/累加与native相加的原float64边界，具体转换顺序与U0科学实现逐行核验。初始zero-last层使全native20 identity；原slot顺序不变。

两臂共同恢复原L0：

\[
L_0=\frac1B\sum_i\left[\min_{j\le20}(ADE_{ij}+0.5FDE_{ij})
+0.05\frac1{190}\sum_{j<k}\operatorname{relu}(0.3-d_{ijk})^2\right],
\]

其中d是两完整候选mean-path distance；hardwinner tie取第一个physical slot。实际标量与梯度须对原U0 L0复核，不从旧四候选版本错误复制6pair分母。

原batch256、25epochs、AdamW参数、seed42/43/44、FIT/outer/inner/CAL及固定epoch25不变。没有新keep、物理罚项、投影、fallback、GT选路、追加候选或scorer。

**跨I1→G1同时恢复输出约束和候选损失，是任务验证，不是单因素因果对照。** 每个阶段内部S/H才是可解释的单因素比较。若I1阳性G1阴性，不能据此唯一归因hardwinner、输出界或某一个梯度机制。

## 3. 为什么需要新的S对照

旧T2 A是120维/25304参数；本G1是146维/28632参数及明确末速度占位。不可将旧A直接当唯一对照。G1新增S/H×3seed共6次训练，全部fresh，同seed完整参数和批次配对。旧A/B/C仅作背景，不当新的独立重复。

G1不使用F0的GT求解路径作teacher，不能让可见监督标签、样本选择或质量筛选隐含依赖F0结果；全部18909 FIT正常训练，不只训练成功oracle样本。

## 4. 预注册比较与完整成功门

outer change为primary，两个比较H−S、H−native，每个比较4项：coverage、minADE、minFDE、minJ，共固定family8；20,000次同ID paired target-track×seed交叉bootstrap，各双侧99.375% percentile区间。没有运行其他臂也不缩family。

完整生成成功必须同时满足：

- H−S coverage平均至少+2pp、每seed不为负、同时CI下界>0，作为显式历史的候选层面收益门；
- H−native每seedchange coverage≥+5pp，minADE和minFDE各≥5%改善；H−native coverage CI下界>0，minADE/minFDE/minJ同时CI上界<0；
- 原生成功lost≤1%，overall/stable/transition误差伤害≤1%，以及全部原同路径、重复和保护规则；
- 全部FIT/outer/inner/CAL、全部group、每seed五pooled motion literal≤native1.01；不得以H比S更安全代替对native达标；
- 没有数据/source/ID/保存/资源合同偏差；首次全门通过后，另做scene/sample相关性和三seed充分性审查。

所有百分比以各自native或S同group原值作分母；零分母报告绝对差和undefined比例，不加floor制造通过。对原量为0的安全量只接受修正量≤0。

如果只H−S改善但H仍不及native，是“相对对照改进”，不是模型成功。内层和CAL必须全报，但不从中挑一个新主指标。不同阶段的family校正不构成整条适应性研究路径的确认性FWER保证。

## 5. 最终分支表：只按证据走大方向

| F0 | I1 | G1 | 下一判断/动作 |
|---|---|---|---|
| 有安全见证 | 信息信号成立 | 尚未做 | 执行已定义的有限G1，不加第四臂 |
| 有安全见证 | 信息信号未成立 | 不触发 | 更支持当前读出/迁移困难；核对FIT拟合与外层反例后结束本系列，未来可考虑上游状态表示/数据支持，不直接判“历史无信息” |
| 未找到或预算阻碍 | 信息信号成立 | 不触发 | 可读出的预测信息尚未有足够安全局部出口证据；记录安全域未确定，未来另立更宽输出域或可验证安全结构方案，不盲重跑原射线 |
| 未找到或预算阻碍 | 信息信号未成立 | 不触发 | 两问题仍未解决，不能选定唯一根因；结束并报告不确定性，不自动MoE/scorer/更多诊断 |
| 有安全见证 | 信息信号成立 | 覆盖改善但motion/保留失败 | 信息利用有进展、生成整合仍不安全；不部署、不以更低标准宣布成功 |
| 有安全见证 | 信息信号成立 | 全门失败且无可靠相对收益 | 指定信息通道未转化为候选收益；不再扫同一head/步长/epoch，不归因唯一loss根因 |
| 有安全见证 | 信息信号成立 | 所有门通过 | 报告突破候选，先做相关性与复核；现有重复benchmark证据不足以宣称独立泛化/理解意图/SOTA |

本系列最多F0+I1+条件G1，最多12个新模型。无论哪个分支结束都出总报告，列已执行/未触发/预算或科学阻碍，不以“每三项审核”作为必须继续凑实验的理由。未来上游解冻、新数据、扩大输出域均是新设计，不在这里自动授权。
