# DGeo四臂复核分析

结论：几何特征+模式中心化确实改变了表示，并带来比上一轮更明显的总体误差改善；但收益主要来自stable，change保护/覆盖和物理极端尾部没有解决，四臂均不满足原完整筛选门。差分相对普通几何门控有配对点增益；后验关系打分头本次未显示增益。

完整worker/exit/RESULT及GEO_EXTRA_REVIEW/FINAL_SEAL均SUCCESS。4fresh seed42各8轮/2720，共32轮10880更新；worker实耗16.156min，实际前检/成本/回归加worker合计约19.884min，低于本次45min规划预算（不含代码实施及非计算对话时间）。原167来源与338去重来源/输出SHA复核一致，全部8科学人口审核收据齐。

|模型|总体ADE|总体FDE|总体成功窗|change ADE|change FDE|change成功窗|
|---|---:|---:|---:|---:|---:|---:|
|B0|0.204214|0.361499|2179/4753|0.263121|0.444397|350/960|
|DG-Base|0.205628|0.365312|2190/4753|0.262689|0.446761|360/960|
|DG-GeoGate|0.198403|0.350782|2264/4753|0.268317|0.464513|319/960|
|DG-GeoDiff|0.197951|0.347337|2287/4753|0.266980|0.466316|334/960|
|DG-GeoScore|0.198343|0.350489|2251/4753|0.268498|0.467760|302/960|

## 1. 去共线成功，但不能等同预测目标成功

固定128 dev输入的qk off-diagonal cosine：DG-Base约.9865→.9960；GeoGate约-.0466→-.0398；GeoDiff约-.0459→-.0419。GeoDiff负权41.875%→39.634%，不再是上一轮1.59%。原所有新增参数第2步已连通，数值/初始等价与精确恢复通过。这支持“实现确实工作”，不证明去共线是全部精度变化的独因果。

## 2. 总体收益是真实点估计，但主要来自stable

GeoDiff总体ADE相对B0约下降3.07%、FDE下降3.92%，coverage+108/4753=+2.272pp。其分解是stable+98、transition+26、change-16；change ADE恶化1.47%、FDE恶化4.93%，coverage350→334。GeoGate整体+85但change-31；GeoScore整体+72但change-48。此现象在FIT也存在：GeoGate/GeoDiff/GeoScore的change成功1495/1585/1457，均低于B0的1660，不应只解释为dev统计门过严。
差分相对GeoGate总体ADE下降约.228%、总体多23/change多15成功窗，符合事先相邻比较的点信号；但仍没有修复相对B0的change退化。DG-Base的change360/960反而高于三种几何臂，但自身总体/稳定FDE及尾部门不通过，不自动挑它改名方法。

## 3. 评分效果必须固定共同支持分母

各自已覆盖change集合：B0106/350=30.29%，GeoGate115/319=36.05%，GeoDiff101/334=30.24%，GeoScore104/302=34.44%。GeoGate相对B0共同覆盖267窗上113/267 vs99/267，仍有14次Top1净增，故不是全靠分母变小；但这不抵消其丢失83个B0成功change窗。
新增关系打分器应对比GeoScore与GeoGate：共同覆盖292个change窗，GeoGate111/292=38.01%，GeoScore104/292=35.62%；三者共同覆盖253窗也111→104。GeoScore还比GeoGate少17个change成功窗，故该评分头本次未获支持。不同训练模型候选坐标也不同，共同窗口不等同一候选池，不能再称纯固定候选排序因果隔离。

## 4. 支持交换与运动风险未解决

change rescue/lost：DG-Base32/22、GeoGate52/83、GeoDiff51/67、GeoScore45/93；GeoDiff旧好丢失67/350=19.14%，不是小净变化。总体matched事件及Jmargin全保留在SUPPORT_EXCHANGE.csv，不用净增掩盖。
GeoDiff总体接续acc p95从1.4706降至1.3204，确有变平滑的部分；但速度p95从1.2828升至1.3791，速度超阈样本6760→12420（相同1140720候选×时间点分母，非独立人）。内部acc p95/p99有下降，但极端超阈631→2515，正excess条件p95幅度+82.72%。这不是所有运动整体恶化，也不是纯条件分母截断：更稀少的极端尾部确实扩大。C0的经验阈不是生理硬安全上限，原统计和验收规则不因解释改变。
四臂rate增量均<=+1pp，但可估幅度项有真实失败；零参考jerk仍NOT_ESTIMABLE_NO_FLOOR，不补PASS。GeoGate/GeoDiff/GeoScore的原连续ADE门还因change超过1%害而失败，不能把未晋级只归于运动门。

## ARS DA2/DA3与统计边界（同助手角色，非外审）

Cannot cherry-pick overall gains: GeoDiff +108 overall = stable+98 transition+26 change-16; change FDE+4.93% and oldgood loss67/350. Actual feature de-collinearity/negative attention works numerically but not the protected scientific objective.
GeoDiff has descriptive adjacent improvement vsGeoGate, not full success; GeoScore common-change ranking worsens111->104 on292 windows. Rare acceleration extremes and speed increase remain despite smoother junction and some bulk quantiles. No automatic seeds/V/F or causal/intent claims.
One seed42/arm, dev4753 overlapping windows/248tracks, change960/102tracks, repeated dev/sisterfootage unknown. No confirmatory p/CI, no independent whole-scene generalization. Nominal structure and shared B0 weights do not isolate geometry from centering or other engineering changes.

当前结论：保留B0为开发对照、不升级baseline，不自动补seed43/44/V/F或扫描宽度/LR/loss。可保留“几何联合干预有总体误差信号、差分相邻对照有点增益、GeoGate共同change排序有局部信号”，但不能宣称目标机动预测、旧支持保留或安全成功，也不能反推所有几何/差分方法无效。全部文件/模型/失败/日志/自动记录和旧PAUSED保持。
