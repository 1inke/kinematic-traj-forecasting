# T1结果：相对质量惩罚未达到预定目标

## 结论

**执行完成、证据审核通过；科学结论为TRANSFER_UNRESOLVED，MECHANISM_SUPPORTED=false，BASELINE_NOT_READY。**

固定配方P_relative未能稳定降低inner变化组的相对退化，也未保住A的覆盖收益。其平均主指标R降低6.20%，低于事前20%目标，seed43反向，配对track+seed和固定seed敏感性95%CI都跨0。前两个seed明显损失覆盖增益，第三个seed有正例，但不能挑它宣布机制成立。

本有限系列到此完成，不延长epoch、不扫描惩罚系数，不自动启动B/E_bal、scorer、新安全项或更广解冻。没有升级baseline、重置historycycle11或宣布长期总目标完成。

## 1. 实际执行与有效比较

- 2026-10-23 14:05:49.945917启动，14:25:01.802038 worker/exit双SUCCESS，墙钟约19分12秒；RESULT记录科学主流程1150.084秒。14:28:39.6161232的一次完成核验确认任务Ready、无该worker进程、XML0、新stderr0；之后不再查询完成PID或旧任务。
- A_replay/P_relative×seeds42/43/44，各fresh8epochs，共6runs、48历史行、3552 optimizer steps；一个正式实验，不是6项独立实验。每run只评FIT18909和inner1856，共12模型人口输出，唯一窗口20765，跨臂/seed重复记录124590不当独立样本。
- 相同82008参数、210维观测输入、K20直接位置、初始化/统计/顺序、原hardQ与endpoint线性diversity、AdamW/clip5/8轮。P唯一新增.2 mean relu(Jstudent−detach Jnative)，不复制native坐标、不在forward输入GT/nativeJ、不物理投影或筛轨迹。
- 三个fresh A最终路径对旧A全部bitwise一致，历史共同字段EXACT；控制可复现。fresh重复并非新的独立数据泛化证据。旧F16没有再实例化/前向，只有来源与缓存保护。

## 2. 主指标：有小幅平均下降，但远未满足完整AND条件

R=mean max(Jstudent−Jnative,0)，J=min_slot max(ADE/.2,FDE/.4)，每个J都来自一条完整候选，不拼ADE/FDE或时段。

|inner变化组，497窗/35条track|A_replay|P_relative|P相对A变化|
|---|---:|---:|---:|
|seed42 R|.420108|.365256|−13.06%|
|seed43 R|.313019|.317664|+1.48%|
|seed44 R|.343304|.326754|−4.82%|

三seed平均R相对降低6.2018%，并非每seed百分比的简单平均。

- P−A主点估计：−.0222526。
- 20000次共同inner全track+配对seed bootstrap，95%CI **[−.0725033,+.0173602]**。
- 固定3seed、仅track抽样敏感性95%CI **[−.0494319,+.0022569]**。
- 两种抽样各20000有效、0无效，变化组35条track超过20条最低门。区间跨0不是统计程序不足或零分母错误，而是当前证据不能稳定确认收益；也不能据此证明真实效应恰为0或排除所有有用惩罚。

FIT整体R三个seed也有下降：A .190615/.159094/.164173→P .180217/.143264/.148121；最终FIT population base均降低。但FIT变化组seed44的R .253060→.255061反向，因此不能简写为“FIT完全解决，只剩泛化”。history是batch加权preupdate记录，不能和最终population loss混作同一分母/时间点。

## 3. 覆盖与保留：均值R变好不等于覆盖收益被保住

### inner变化组（497窗，native成功166）

|seed|A覆盖|P覆盖|变化窗数|A救回→P救回|A丢失→P丢失|保留A相对native净增|
|---|---:|---:|---:|---|---|---:|
|42|186|167|−19|53→51|33→50|5.00%|
|43|191|175|−16|62→46|37→37|36.00%|
|44|192|197|+5|63→61|37→30|119.23%|

原要求每seed保留至少90%的控制正净增；前两个失败。P原生成功丢失率分别30.12%/22.29%/18.07%，仍远超过单列的严格1%保留目标。seed44正例主要是少丢7个旧成功，救回反而少2个，不是补全与保留同时全面改善。

inner整体（1856窗，native932）：A982/1021/1021→P948/977/1029，分别−34/−44/+8。P的旧成功丢失184/168/138，对应A168/166/150；前两个又未满足lost不增加。整体净增保留率32.00%/50.56%/108.99%，同样只有seed44达到90%门。

P相对native的覆盖平均仍为正：inner整体+2.83764pp、变化组+2.74983pp；但相对A分别−1.25718pp/−2.01207pp，其family16同时区间都跨0。既不能宣称无能力，也不能把仍有正净增当成相对惩罚的稳定收益。

## 4. 连续误差与旧人口：改善位置存在交换

|inner变化组|A minADE|P minADE|A minFDE|P minFDE|
|---|---:|---:|---:|---:|
|42|.306638|.296535|.578088|.543290|
|43|.281449|.288935|.518984|.530572|
|44|.291134|.286928|.511826|.533488|

P−A平均minADE −.002274m、minFDE −.000516m，均未获得family16校正区间的稳定改善支持；seed43两项、seed44 FDE未满足≤1.01A门。对native，P变化组minADE平均仍恶化+.031181m、minFDE+.076561m，family16配对同时区间分别[+.002747,+.058354]m/[+.007641,+.142086]m。本地探索下有退化证据，但不将其包装成独立场景确认性结论。

按**旧A/native事先冻结**的四类人口分析P−旧A，不按P重新分组：

|旧inner变化人口|seed42平均J变化|seed43|seed44|
|---|---:|---:|---:|
|共同成功|+.204743|+.040609|+.046580|
|旧A救回|+.336283|+.337551|+.252779|
|旧A丢失|−.410301|−.242310|−.171125|
|共同失败|−.192683|−.004769|−.050551|

正为更差，负为更好。P在原A丢失人口平均改善，却同时伤及原共同成功和原A救回人口；共同失败J也有降低，但seed43的minADE/FDE反而略增，seed44该人口FDE也增。小平均J改善和成功阈值附近的丢失可同时出现，不能把平均单侧惩罚当逐窗硬保留保证。这是当前结果的交换模式，不是对共享head干扰、梯度责任或容量的唯一因果证明。

## 5. 运动学与候选集合

P的FIT/inner×四组×三seed共24项完整G_base **0/24通过**；五motion联合也0/24，accel_mean全部超native1.01门。A也是0/24。P42 FIT变化组coverage1916还低于native1937，不能只报inner平均净增。

|inner变化组P相对native变化|seed42|seed43|seed44|
|---|---:|---:|---:|
|speed_p95|+4.46%|−3.76%|−6.69%|
|accel_mean|+33.48%|+36.28%|+28.51%|
|accel_p95|−51.95%|−39.33%|−53.95%|
|jerk_mean|−45.57%|−49.04%|−50.26%|
|jerk_p95|−72.39%|−65.71%|−72.95%|

因此不能说“所有运动项都更坏”，也不能因jerk/p95更低宣称安全。固定J-slot对GT的加速度向量误差P .246866/.248977/.246700，仍低于native .261044，但相对A只有seed42更低；所选GT候选和ALL20集合安全门不是一回事。D0/D1的解释保持，不为本结果修改阈值或平滑GT。

全部最终epoch的20槽都有WTA责任；P各组平均distinct约19.452–20。槽位活跃/几何不同不代表有效互补或运动安全，也不支持一概归因候选完全坍缩。

## 6. 统计完整性、反证与证据边界

- 主终点及固定seed敏感性、全部16次级比较完整保留，见[BOOTSTRAP.json](BOOTSTRAP.json)、[SECONDARY16_CI.csv](completion_review/SECONDARY16_CI.csv)。不以某个未校正95%区间替换family16区间。P-A reference-success lost率以A成功为分母，P-native以native成功为分母；这与工程门中“双方对native成功丢失率之差”不同。自参考lost=0，不把这种条件丢失区间排除0解读为一般模型优劣的显著性结论。
- CI按共同inner全76条sequence+target track与配对3seed抽样，先每seed窗口加权再seed均值；变化组35条track。F16上游已见inner、track不等于上下文/录制独立、窗口相关、重复使用inner、仅3seed等限制仍在，没有独立wholemodel泛化或可靠scene级推断。
- 同助手DA1回顾：D0共同失败主贡献仅支持做此固定测试，不能预设相对J惩罚有效；本结果未达到其入口之后的机制门。
- 同助手DA2：若只看seed44或FIT base下降会得出过强结论；报告seed42/43的覆盖损失、R反向、旧救回人口受损及运动学反例。也不以阴性结果否定所有质量约束。
- 同助手DA3：最强反方是8epoch固定优化、参数化或未知更好配方未被排除。因此只判“本配方未获支持”，不宣布观测信息无用、未来不可预测或共享head唯一根因；不以此为由事后延长/扫描。本次local主门和完整G_base/major分别判断，不能相互替代。

按academic-research-suite与nature-statistics进行的上述核查由同一助手完成，不是外部独立审稿。

## 7. 完成证据与资源

- [RESULT](RESULT.json)、[DECISION](DECISION.json)、[ARITHMETIC_AUDIT](ARITHMETIC_AUDIT.json)、[PRESERVATION](PRESERVATION.json)及每run完整done清单保留。
- [completion_review/REVIEW.json](completion_review/REVIEW.json)：615源及2个CPU预检扩展、141个done绑定文件SHA前后不变；12保存人口全部同slot质量/覆盖计数/ID救回丢失、48历史行聚合与六checkpoint CPU元数据审核PASS，质量最大误差4.54747e−13；20k完整抽样顺序EXACT，主/16次级/固定seed所有保存draw和CI独立复核最大误差8.32667e−17。
- 原12份完整GTfree聚簇/pooled五motion和几何独立审核均PASS，新补充不冒称再做第三遍完整聚簇/几何，不实例化模型、不forward/backward/optimizer或重训。
- 生产RESOURCE PASS：保留private峰3408257024bytes≈3.174GiB、RSS峰2586419200bytes≈2.409GiB，E6GiB按1.3动态规则；保存观测physical最低15223042048bytes、commit最低14109589504bytes，不是固定6GiB门。新CPU完成补充另估E4GiB，实测private保留峰1014652928bytes≈.945GiB，不能用它代替生产峰。
- 原12:xx D0、旧S1、旧环境、科学源、失败证据均不改。完成仅清理本次f16-tail-t1自动跟进，旧暂停任务不动；没有自动下一实验。

完整每seed/split/group指标：[ALL_GROUP_RESULTS.csv](completion_review/ALL_GROUP_RESULTS.csv)；五motion：[ALL_MOTION.csv](completion_review/ALL_MOTION.csv)；固定旧人口：[FROZEN_OLD_A_SUBSETS.csv](completion_review/FROZEN_OLD_A_SUBSETS.csv)；模型合同：[CHECKPOINTS.json](completion_review/CHECKPOINTS.json)。原始逐窗数据、rescue/lost ID、J/Q全路径/早中晚与运动时序在各run保存文件中。
