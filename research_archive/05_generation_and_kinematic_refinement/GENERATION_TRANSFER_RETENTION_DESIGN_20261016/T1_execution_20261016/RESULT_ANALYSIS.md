# T1结果：保留有局部改善，未达到预注册迁移信号；资源合同存在偏差

分析日期：2026-10-16，日本标准时间。六个fresh训练已结束；本报告不把计算SUCCESS等同于模型成功或全部执行合同通过。

## 1. 执行与证据范围

worker与exit均SUCCESS：12:30:48.489445→12:59:25.782480，墙钟时间差28分37.293秒；RESULT另外登记perf_counter elapsed1734.106秒，两种记录分别保留，不推断时钟差异原因。13:08:19.1928605单次完成核验确认进程已结束、任务Ready、XML0trigger、stderr0；不再查询该PID。

A/B各3seed、每run25epoch，共6模型/150history/11100更新；24科学jobs＋4native参考，84主要保存输出。原完整ARITHMETIC_AUDIT对全28job质量、原GT-free complete-link聚簇、五pooledmotion、方向/分层、keep、controls及全部统计抽样独立复算PASS。2304sources初末SHA不变。

补充completion_review只做CPU/NumPy：全28job保存质量与计数max0、全原源和完成输出SHA前后验证；12个A科学job的所有保存数组与历史U0 coupled bitwiseEXACT。新鲜重复执行不构成独立泛化，未重放所有旧模型/梯度，也不是外部评审。完整96组/全部ID清单见completion_review/SUMMARY.json。

## 2. 原生变化组完整好候选覆盖

标准固定为同一候选ADE≤0.2 AND FDE≤0.4；三个数依次seed42/43/44。不同split分母不能混用。

|人口|N|原生good|A good|B good|A rescue/lost|B rescue/lost|
|---|---:|---:|---|---|---|---|
|FIT|4725|1937|2171/2197/2161|2149/2187/2165|358/124; 374/114; 343/119|290/78; 324/74; 295/67|
|outer|253|110|107/105/113|112/111/108|15/18; 11/16; 19/16|15/13; 15/14; 14/16|
|inner|497|166|173/173/174|169/171/174|26/19; 23/16; 23/15|22/19; 21/16; 25/17|
|original_CAL|700|243|221/214/225|232/232/234|22/44; 18/47; 23/41|17/28; 20/31; 20/29|

## 3. Primary outer-change机制门

固定family24，即便C/D未执行也不缩家族；20000 PCG64(seed20261016)同ID paired track×3seed抽样，全部primary有效draw20000。普通95%不是替代门。

|B−A指标|点估计|family24同时区间|
|---|---:|---|
|coverage差(pp)|0.790514|[-5.633803, 6.824677]|
|lost率差(pp，分母原生成功110)|-2.121212|[-13.878967, 7.002942]|
|rescue率差(pp，分母原生失败143)|-0.233100|[-8.478292, 6.901000]|
|固定native-j0实际ΔJ差|-0.015105|[-0.044055, 0.015897]|

RETENTION_SIGNAL未通过：lost平均只降2.1212pp，未达3pp；三个seed分别降4.5455/1.8182/0pp，不是三seed严格下降。同时区间跨0；rescue均值略降且非劣区间下界约−8.48pp低于−2pp门；coverage第三seed下降。不能把宽区间说成“保留项已证明完全无效”，也不能据点估计宣传已迁移。

B对native绝对coverage平均仅+0.131752pp，family3区间[−4.741778,+4.515928]pp；不是每seed≥+5pp。ADE/FDE/minJ的family9差区间均跨0，无≥5%误差改进证据。因此无重大模型成功；原baseline/historycycle11/goal不升级。

## 4. 训练内与迁移、保护—救回的交换

FIT change的B把lost由124/114/119降至78/74/67，但rescue同时由358/374/343降至290/324/295。原生失败上的生成收益被部分压缩，不能只报lost减少。FIT整体good却增加（A11657/11638/11593→B11700/11743/11722），这是反例，不能说所有训练组都变差。

outer change B的good112/111/108对A107/105/113：前两seed改善，第三下降。inner change169/171/174对A173/173/174更弱或相同。CAL change232/232/234全部优于A221/214/225，仍全部低于native243；CAL仅事后反例，不可升为primary或选择权重。

B outer change的native-success lost仍13/14/16，即11.82/12.73/14.55%，远高于1%保护门。B固定原生J最佳候选的实际ΔJ比A在三seed均改善，但其family24区间跨0，且相对native本身仍是正ΔJ（变差），不能用此替代完整coverage。

B outer change平均修正幅度约0.0464–0.0493m（A约0.0559–0.0568m），并非全部回退identity。精确原生相等窗口/候选比例逐组保存在IDENTITY_FRACTIONS.json；实际质量交换而非仅基于“近identity”的主观标签判断。

## 5. 运动学与有效候选簇

所有seed/split/group五pooled指标保留literal native×1.01标准，不因当前去投影而降低要求。下表逐组列出失败项；平均加速度改善不等于全门通过。有效good簇也不能抵消完整覆盖或成功保护失败。

|cell|seed|split|group|N|good/native|R/L|有效good簇均值|失败motion|protected ADE/FDE通过|
|---|---:|---|---|---:|---|---|---:|---|---|
|A|42|FIT|overall|18909|11657/10858|1450/651|0.857687|accel_mean|True|
|A|42|FIT|stable|10767|7825/7414|788/377|1.058884|accel_mean|True|
|A|42|FIT|transition|3417|1661/1507|304/150|0.637694|accel_mean|True|
|A|42|FIT|change|4725|2171/1937|358/124|0.558307|accel_mean|True|
|A|42|outer|overall|1035|632/625|57/50|0.859903|accel_mean|True|
|A|42|outer|stable|604|438/434|31/27|1.066225|accel_mean|True|
|A|42|outer|transition|178|87/81|11/5|0.651685|accel_mean|True|
|A|42|outer|change|253|107/110|15/18|0.513834|accel_mean|True|
|A|42|inner|overall|1856|954/932|112/90|0.713362|accel_mean|False|
|A|42|inner|stable|978|610/601|61/52|0.896728|accel_mean|False|
|A|42|inner|transition|381|171/165|25/19|0.627297|accel_mean|True|
|A|42|inner|change|497|173/166|26/19|0.418511|accel_mean|True|
|A|42|original_CAL|overall|2879|1474/1490|134/150|0.721778|accel_mean|False|
|A|42|original_CAL|stable|1535|993/994|72/73|0.964821|accel_mean|False|
|A|42|original_CAL|transition|644|260/253|40/33|0.501553|accel_mean|False|
|A|42|original_CAL|change|700|221/243|22/44|0.391429|accel_mean|True|
|B|42|FIT|overall|18909|11700/10858|1232/390|0.847216|accel_mean|True|
|B|42|FIT|stable|10767|7863/7414|668/219|1.045695|accel_mean|True|
|B|42|FIT|transition|3417|1688/1507|274/93|0.632719|accel_mean|True|
|B|42|FIT|change|4725|2149/1937|290/78|0.550053|accel_mean|True|
|B|42|outer|overall|1035|627/625|49/47|0.834783|accel_mean|True|
|B|42|outer|stable|604|429/434|22/27|1.021523|accel_mean|True|
|B|42|outer|transition|178|86/81|12/7|0.629213|accel_mean|True|
|B|42|outer|change|253|112/110|15/13|0.533597|accel_mean|True|
|B|42|inner|overall|1856|940/932|100/92|0.692888|accel_mean|True|
|B|42|inner|stable|978|610/601|58/49|0.873211|accel_mean|True|
|B|42|inner|transition|381|161/165|20/24|0.585302|accel_mean|False|
|B|42|inner|change|497|169/166|22/19|0.420523|accel_mean|True|
|B|42|original_CAL|overall|2879|1499/1490|118/109|0.723515|accel_mean|True|
|B|42|original_CAL|stable|1535|1001/994|68/61|0.955049|accel_mean|True|
|B|42|original_CAL|transition|644|266/253|33/20|0.517081|accel_mean|False|
|B|42|original_CAL|change|700|232/243|17/28|0.405714|accel_mean|True|
|A|43|FIT|overall|18909|11638/10858|1415/635|0.851869|accel_mean|True|
|A|43|FIT|stable|10767|7764/7414|732/382|1.047738|accel_mean|True|
|A|43|FIT|transition|3417|1677/1507|309/139|0.636523|accel_mean|True|
|A|43|FIT|change|4725|2197/1937|374/114|0.561270|accel_mean|True|
|A|43|outer|overall|1035|617/625|51/59|0.821256|accel_mean|True|
|A|43|outer|stable|604|426/434|26/34|1.011589|accel_mean|True|
|A|43|outer|transition|178|86/81|14/9|0.623596|accel_mean|True|
|A|43|outer|change|253|105/110|11/16|0.505929|accel_mean|True|
|A|43|inner|overall|1856|959/932|113/86|0.711207|accel_mean|False|
|A|43|inner|stable|978|614/601|64/51|0.895706|accel_mean|False|
|A|43|inner|transition|381|172/165|26/19|0.603675|accel_mean|True|
|A|43|inner|change|497|173/166|23/16|0.430584|accel_mean|True|
|A|43|original_CAL|overall|2879|1454/1490|120/156|0.707885|accel_mean|False|
|A|43|original_CAL|stable|1535|990/994|74/78|0.944625|accel_mean|True|
|A|43|original_CAL|transition|644|250/253|28/31|0.498447|accel_mean|False|
|A|43|original_CAL|change|700|214/243|18/47|0.381429|accel_mean|True|
|B|43|FIT|overall|18909|11743/10858|1291/406|0.862288|accel_mean|True|
|B|43|FIT|stable|10767|7874/7414|692/232|1.066407|accel_mean|True|
|B|43|FIT|transition|3417|1682/1507|275/100|0.635645|accel_mean|True|
|B|43|FIT|change|4725|2187/1937|324/74|0.561058|accel_mean|True|
|B|43|outer|overall|1035|631/625|49/43|0.840580|accel_mean|True|
|B|43|outer|stable|604|433/434|22/23|1.029801|accel_mean|True|
|B|43|outer|transition|178|87/81|12/6|0.646067|accel_mean|True|
|B|43|outer|change|253|111/110|15/14|0.525692|accel_mean|True|
|B|43|inner|overall|1856|933/932|91/90|0.699892|accel_mean|True|
|B|43|inner|stable|978|603/601|55/53|0.873211|accel_mean|False|
|B|43|inner|transition|381|159/165|15/21|0.603675|accel_mean|False|
|B|43|inner|change|497|171/166|21/16|0.432596|accel_mean|True|
|B|43|original_CAL|overall|2879|1489/1490|120/121|0.731157|accel_mean|True|
|B|43|original_CAL|stable|1535|998/994|68/64|0.971336|accel_mean|True|
|B|43|original_CAL|transition|644|259/253|32/26|0.503106|accel_mean|True|
|B|43|original_CAL|change|700|232/243|20/31|0.414286|accel_mean|True|
|A|44|FIT|overall|18909|11593/10858|1386/651|0.855836|accel_mean|True|
|A|44|FIT|stable|10767|7784/7414|749/379|1.056562|accel_mean|True|
|A|44|FIT|transition|3417|1648/1507|294/153|0.632719|accel_mean|True|
|A|44|FIT|change|4725|2161/1937|343/119|0.559788|accel_mean|True|
|A|44|outer|overall|1035|642/625|65/48|0.853140|accel_mean|True|
|A|44|outer|stable|604|441/434|31/24|1.049669|accel_mean|True|
|A|44|outer|transition|178|88/81|15/8|0.651685|accel_mean|True|
|A|44|outer|change|253|113/110|19/16|0.525692|accel_mean|True|
|A|44|inner|overall|1856|957/932|113/88|0.712284|accel_mean|False|
|A|44|inner|stable|978|614/601|62/49|0.898773|accel_mean|False|
|A|44|inner|transition|381|169/165|28/24|0.611549|accel_mean|False|
|A|44|inner|change|497|174/166|23/15|0.422535|accel_mean|True|
|A|44|original_CAL|overall|2879|1472/1490|140/158|0.721084|accel_mean|False|
|A|44|original_CAL|stable|1535|981/994|75/88|0.953094|accel_mean|True|
|A|44|original_CAL|transition|644|266/253|42/29|0.521739|accel_mean|True|
|A|44|original_CAL|change|700|225/243|23/41|0.395714|accel_mean|True|
|B|44|FIT|overall|18909|11722/10858|1239/375|0.855730|accel_mean|True|
|B|44|FIT|stable|10767|7852/7414|665/227|1.056933|accel_mean|True|
|B|44|FIT|transition|3417|1705/1507|279/81|0.635353|accel_mean|True|
|B|44|FIT|change|4725|2165/1937|295/67|0.556614|accel_mean|True|
|B|44|outer|overall|1035|623/625|43/45|0.849275|accel_mean|True|
|B|44|outer|stable|604|431/434|18/21|1.048013|accel_mean|True|
|B|44|outer|transition|178|84/81|11/8|0.646067|accel_mean|True|
|B|44|outer|change|253|108/110|14/16|0.517787|accel_mean|True|
|B|44|inner|overall|1856|945/932|98/85|0.695582|accel_mean|True|
|B|44|inner|stable|978|609/601|58/50|0.876278|accel_mean|True|
|B|44|inner|transition|381|162/165|15/18|0.587927|accel_mean|False|
|B|44|inner|change|497|174/166|25/17|0.422535|accel_mean|True|
|B|44|original_CAL|overall|2879|1487/1490|115/118|0.722820|accel_mean|True|
|B|44|original_CAL|stable|1535|992/994|63/65|0.948534|accel_mean|True|
|B|44|original_CAL|transition|644|261/253|32/24|0.521739|accel_mean|True|
|B|44|original_CAL|change|700|234/243|20/29|0.412857|accel_mean|True|

B运动学失败计数（48个seed×split×group，五项分别统计）：`{'speed_p95': 0, 'accel_mean': 48, 'accel_p95': 0, 'jerk_mean': 0, 'jerk_p95': 0}`。

outer change B平均加速度相对native仍约+10.81%至+11.32%，A约+12.29%至+12.52%；代价减小但仍严格失败。固定少量FIT batch的L0/keep输出坐标梯度cos为正，不能称参数梯度冲突已排除，也不能当有限AdamW因果证明。训练history为preupdate加权值，不能当最终人口loss；后者保存在各job done.population。

## 6. 资源偏差：停止下一训练而不覆盖原报告

本任务E=3297603712bytes（3.071133GiB），1.3E=3.992473GiB；原RESOURCE登记peak_private_bytes=4502712320（4.193478GiB），比上限高约5.03%。原报告仍标PASS，但guard只比较当前private/RSS，累计记录的生命周期peak没有参与超限判断。

没有保留每次当前snapshot/峰值时间，不能反推峰值发生时刻或确定具体分配来源；需核对peak口径并在新版本补足已记录峰值检查/保守估算。可用物理内存最低约13.40GiB，故不能声称电脑内存耗尽或要求用户关闭程序。

原RESOURCE/PREFLIGHT/SHA/worker结果全部保留，不把原SUCCESS改写成失败，也不把资源合同偏差抹掉。新RESOURCE_DEVIATION_REVIEW.json明确该偏差，严格无偏差执行完成暂不记1；六训练已完成/数学分析完成另列。T2尚未实施/启动，须等直接人类确认新版本资源审核与重估，不重训T1，不静默提高E或改固定1.3倍。

## 7. 同助手反方检查（非外审）

**DA2 — 证据与反例：** 不能因outer机制门失败就抹去CAL保留改善、FIT整体覆盖增加和fixed-slot三seed点估计改善；但也不能用这些事后人口替代primary。负面论据最强处是第三seed覆盖与救回下降、保护仍远超门及motion持续失败。

**DA3 — 最终压力测试：** 该keep也可视为一般正则，无法独证“缺失保护loss就是根因”。三seed/重复窗口/F16已见/target非context独立/outerCAL适应性复用+无新holdout限制全部保留。资源报告PASS与峰值矛盾是执行合同重大问题，必须阻止自动T2，不因科学结果已可读就放行。

## 8. 当前分支

T1：RETENTION_SIGNAL_NOT_ESTABLISHED；D因T1门失败不触发。T2在科学问题上仍独立可做，但当前RESOURCE_REVIEW_REQUIRED，未实施、未预检、无worker或ETA。下一次只能先在新目录核对峰值口径、修正guard/估算并保留原源，再按原C对A方案走实际GPU/恢复/预算门；不扫keep权重或新MoE/scorer。

来源：本目录RESULT.json、ARITHMETIC_AUDIT.json、STATISTICS.json、PRESERVATION.json、RESOURCE.json、completion_review/SUMMARY.json/REVIEW.json/IDENTITY_FRACTIONS.json及各run保存路径/metrics/history；辅助分析AI执行，同助手反方审核，非外部独立评审。
