# 分组查询 Q 阶段实际结果

执行与保存审核完成；科学门：Q_STOP_MECHANISM_GATE_FAILED。

Q1单组与Q3三组：seed42/43/44，各fresh8epoch；FIT18909、inner1856；部署固定组0 K20。新参数82008/82648，部署82008。原端点线性hinge不变。

## 主比较：inner change，同一候选ADE≤0.2且FDE≤0.4

每seed覆盖差(pp)：[4.82897384305835, 2.0120724346076457, 1.0060362173038229]；平均2.6157pp。
固定family3校正98.3333%区间：[-0.019235481714305273, 0.08248400300556116]；固定seed敏感性：[-0.009575832250728228, 0.06686471600092889]。区间单位为比例。

|seed|split|group|N|Native|Q1|Q3|Q1 lost native|Q3 lost native|
|---:|---|---|---:|---:|---:|---:|---:|---:|
|42|FIT|overall|18909|10858|11326|11150|1663|1877|
|42|FIT|stable|10767|7414|7557|7292|915|1122|
|42|FIT|transition|3417|1507|1613|1664|351|350|
|42|FIT|change|4725|1937|2156|2194|397|405|
|42|inner|overall|1856|932|982|996|168|180|
|42|inner|stable|978|601|602|596|103|119|
|42|inner|transition|381|165|194|190|32|29|
|42|inner|change|497|166|186|210|33|32|
|43|FIT|overall|18909|10858|11391|11404|1683|1657|
|43|FIT|stable|10767|7414|7682|7528|869|943|
|43|FIT|transition|3417|1507|1626|1629|351|342|
|43|FIT|change|4725|1937|2083|2247|463|372|
|43|inner|overall|1856|932|1021|985|166|182|
|43|inner|stable|978|601|645|600|88|111|
|43|inner|transition|381|165|185|184|41|36|
|43|inner|change|497|166|191|201|37|35|
|44|FIT|overall|18909|10858|11472|11434|1649|1699|
|44|FIT|stable|10767|7414|7648|7611|896|912|
|44|FIT|transition|3417|1507|1689|1683|328|325|
|44|FIT|change|4725|1937|2135|2140|425|462|
|44|inner|overall|1856|932|1021|1044|150|148|
|44|inner|stable|978|601|636|651|86|85|
|44|inner|transition|381|165|193|196|27|28|
|44|inner|change|497|166|192|197|37|35|

所有连续误差、原成功保留、完整GTfree聚簇、有效good和五pooled运动学/接续见ALL_GROUPS.csv及每人口metrics/sample_metrics；相对门逐条在DECISION.json。辅助组与K60 union只作不等预算诊断。

Q3 Native严格安全：False。M：NOT_TRIGGERED_Q_GATE；E：NOT_TRIGGERED_PENDING_QUALIFIED_M。机制门不是Native安全或重大突破门；不升级baseline。

## 同助手反方核查（不是外部评审）

DA2：Q3增加640训练参数及三组decoder计算。即便有收益，仍混合了训练计算/优化几何影响，不能证明纯监督数量的因果中介。绑定相同query夹具只排除loss乘3/跨60取min的实现伪效应。完整预算和吞吐见BENCHMARK.json。

DA3：主组无提升不能推论观测不含信息；辅助组或并集提升也不能冒充K20部署增益。须保留FIT与inner分歧、每seed反例、lost和运动退化；不挑最好seed/组或缩统计家族。

限制：旧F16 h3/v2已见全部FIT及inner；目标track不交不保证上下文不交；inner反复用于开发，三seed相关窗口bootstrap仍是探索，不是独立wholemodel泛化、心理意图理解或SOTA。保存数据重算不是全部模型/梯度再次回放。AI助手完成实施和分角色核查，非外审。

原major双5%、原成功lost≤1%、protected误差1%、五motion/junction1%与相关性审核均不放宽；旧baseline/historycycle11/长期未完成目标保持。无额外组数/宽度/epoch/损失扫描。
