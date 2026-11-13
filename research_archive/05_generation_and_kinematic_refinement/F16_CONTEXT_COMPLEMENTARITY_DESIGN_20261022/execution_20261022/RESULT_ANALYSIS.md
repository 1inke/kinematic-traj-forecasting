# F16同源上下文公共基线：S1实际结果

执行完成；科学阶段门：**BASELINE_NOT_READY**。

A使用210维观测/冻结token压缩输入，82008新参数，三个fresh seed各8epochs。零输出初始化不是nativeidentity。旧F16/decoder/scorer和KIN不更新。

|split|seed|group|N|native good|A good|rescue|lost|gate|
|---|---:|---|---:|---:|---:|---:|---:|---|
|FIT|42|overall|18909|10858|11326|2131|1663|FAIL|
|FIT|42|stable|10767|7414|7557|1058|915|FAIL|
|FIT|42|transition|3417|1507|1613|457|351|FAIL|
|FIT|42|change|4725|1937|2156|616|397|FAIL|
|inner|42|overall|1856|932|982|218|168|FAIL|
|inner|42|stable|978|601|602|104|103|FAIL|
|inner|42|transition|381|165|194|61|32|FAIL|
|inner|42|change|497|166|186|53|33|FAIL|
|FIT|43|overall|18909|10858|11391|2216|1683|FAIL|
|FIT|43|stable|10767|7414|7682|1137|869|FAIL|
|FIT|43|transition|3417|1507|1626|470|351|FAIL|
|FIT|43|change|4725|1937|2083|609|463|FAIL|
|inner|43|overall|1856|932|1021|255|166|FAIL|
|inner|43|stable|978|601|645|132|88|FAIL|
|inner|43|transition|381|165|185|61|41|FAIL|
|inner|43|change|497|166|191|62|37|FAIL|
|FIT|44|overall|18909|10858|11472|2263|1649|FAIL|
|FIT|44|stable|10767|7414|7648|1130|896|FAIL|
|FIT|44|transition|3417|1507|1689|510|328|FAIL|
|FIT|44|change|4725|1937|2135|623|425|FAIL|
|inner|44|overall|1856|932|1021|239|150|FAIL|
|inner|44|stable|978|601|636|121|86|FAIL|
|inner|44|transition|381|165|193|55|27|FAIL|
|inner|44|change|497|166|192|63|37|FAIL|

所有组的minADE/FDE/J、完整GT-free聚簇、有效good、五项pooled运动学、逐窗rescue/lost ID保存在各run的metrics与sample_metrics；阶段门逐项布尔在DECISION.json。

## 结论与停止边界

任一seed/人口/组失败就不启动B/E_bal，不延长epoch、不换初始化或接口。科学失败不等于执行失败。

## 同助手反方检查（不是外审）

DA2：基线失败可能来自零初始化、短8epoch优化、recent+mean压缩或模型配方；本设计无同容量history-only臂，不能独证F16信息无用。native参考的上游暴露与输入路径不同，不能作为公平容量上限。

DA3：不挑最好seed或只看change。净覆盖不代表逐窗保留；即使部分motion更好，任一五项门失败也不称安全。保留所有反例。上游h3/v2已见inner，FIT/inner目标track互斥非context互斥，重复inner探索不构成独立wholemodel泛化。

实际worker全流程575.911s；S0成本见source_audit/RESULT.json，训练/每epoch记录见history。源与旧产物初末SHA核验在PRESERVATION.json。原major_status=NOT_ESTABLISHED；baseline/historycycle11/原总目标不变。

报告由同助手研究设计/统计角色审查，不是外部审稿。
