# T2速度表示实验与A/B/C系列结果分析

结论：**T2执行完成且审核通过，但迁移机制未成立；T1保留机制也未成立，因此D/T3不触发，有限系列结束。** 不能把执行PASS写成模型成功。旧baseline、history cycle11和总体研究目标不变。

## 1. 做到了哪一步

T1完成A/B×3seed共6个25epoch训练；T2只新训C×3seed共75epoch/5550steps，A经过SHA和训练合同认证后只读复用，不重训A，不把重复对照当独立泛化。系列共9个实际完成训练、225epoch；T1有历史资源合同偏差，不能称9个全部无偏差正式执行。严格本次新C完整执行计1，T1旧attempt/deviation单列。

C唯一改变：将候选native24位置特征改为可逆差分/.4速度，P2不变、同25304参数head、全20候选积分修正、相同训练目标和优化预算；不是增加观测信息。两臂位置界.20(t/12)^2仍在，S0逐时物理约束/统一alpha/fallback不在，运动学只评价、不用于回退。本轮没有加载旧F16做新forward、没有改旧模型、teacher、scorer或数据。

primary为固定epoch25的outer change（253窗口/68 target tracks）；FIT训练18909，outer1035，inner1856，原CAL2879；四人口合24679unique IDs。inner/CAL只posthoc，不挑checkpoint。

## 2. 完整好候选覆盖

同一完整候选必须同时ADE≤.2、FDE≤.4；不能拼不同slot的ADE/FDE。每格三数依次seed42/43/44。

| 人口（change） | n | native好窗口 | A：位置+L0 | B：位置+keep | C：速度+L0 |
|---|---:|---:|---|---|---|
| FIT | 4725 | 1937 | 2171 / 2197 / 2161 | 2149 / 2187 / 2165 | 2181 / 2224 / 2164 |
| outer | 253 | 110 | 107 / 105 / 113 | 112 / 111 / 108 | 108 / 111 / 109 |
| inner | 497 | 166 | 173 / 173 / 174 | 169 / 171 / 174 | 166 / 168 / 169 |
| original_CAL | 700 | 243 | 221 / 214 / 225 | 232 / 232 / 234 | 223 / 230 / 229 |


C在FIT change相对native增加5.164pp / 6.074pp / 4.804pp，确有拟合能力；但outer为-0.791pp / 0.395pp / -0.395pp，CAL为-2.857pp / -1.857pp / -2.000pp。因此不是“取消约束仍完全不会改轨迹”，而是这种拟合收益没有稳定转化为外层完整覆盖。

反例必须保留：C外层overall638/634/640均高于native625，inner change166/168/169也不低于native166；不能写成全群体崩溃。CAL change虽均比A好，仍全部低于native243，且C inner change低于A三个seed。

## 3. T2预注册主检验

20,000次PCG64 seed20261016，same-ID paired模型seed×target-track交叉bootstrap，先每draw窗口加权、再seed均值；所有draw有效。family24固定，不因D不跑而缩小；以下pp为百分点。

| C−A：outer change | 均值 | family24 同时区间 |
|---|---:|---|
| coverage | 0.3953 pp | [-4.7812, 5.1908] pp |
| lost_rate | 0.3030 pp | [-6.5400, 7.7048] pp |
| rescue_rate | 0.9324 pp | [-5.6030, 7.2327] pp |
| fixed_j0_actual_deltaJ | -0.0060 | [-0.0319, 0.0163] |


C−A coverage逐seed为+.3953/+2.3715/−1.5810pp，平均+.3953pp，未到+2pp机制门、第三seed下降、同时区间跨零。lost均值反而增加.3030pp；rescue均值+.9324pp但区间跨零。fixed-j0实际ΔJ三个seed的C−A点估计均改善，这是正反例，但其同时区间仍跨零，不能据此宣称迁移信号。

绝对C−native：coverage均值−.2635pp，family3区间[−6.0832,+4.7872]pp；minADE均值+.00039554、minFDE+.00084133、minJ+.01094757，family9区间分别[−.00542521,+.00593441]、[−.01369704,+.01757655]、[−.02106097,+.04354824]，均跨零。没有可靠的绝对改善，也不能把宽区间称等价或无能力。

FIT→outer方向差敏感性两人口独立抽track/sharedseed：C−A的nativefailure fixed-j0 initialdJ差为−.00323469，95%区间[−.02555285,+.01755427]；allchange fixed-j0 actualΔJ差−.00293496，区间[−.01904128,+.01246767]，均仅描述、均跨零，不能替换主family。

## 4. 救回与丢失不能只看净增

| 人口 | C rescue（42/43/44） | C lost（42/43/44） | 原生成功lost率 | C平均加速度相对native |
|---|---|---|---|---|
| FIT change | 362 / 395 / 366 | 118 / 108 / 139 | 6.09% / 5.58% / 7.18% | +12.13% / +12.14% / +12.15% |
| outer change | 15 / 16 / 18 | 17 / 15 / 19 | 15.45% / 13.64% / 17.27% | +12.63% / +12.55% / +12.64% |
| inner change | 23 / 24 / 23 | 23 / 22 / 20 | 13.86% / 13.25% / 12.05% | +11.76% / +11.53% / +11.54% |
| original_CAL change | 22 / 25 / 25 | 42 / 38 / 39 | 17.28% / 15.64% / 16.05% | +11.59% / +11.62% / +11.74% |


C outer每seed最终good=110+rescue−lost，即108/111/109。原本成功110窗口中丢失17/15/19（13.64–17.27%），远大于≤1%保护门。C所有48个seed×split×group的原生成功保留门均失败；A/B也各48/48失败，B的局部保留改善不等于达标。

T1 B的FIT change lost由A124/114/119降至78/74/67，但rescue也由358/374/343降至290/324/295。outer B−A coverage均+.7905pp、lost−2.1212pp、rescue−.2331pp、fixed-j0ΔJ−.0151051；各family24区间均跨零，RETENTION_SIGNAL=false。保留与救回折中存在，但不是当前keep已解决问题。

## 5. 时间结构、候选多样性与运动学

C outer change的固定native-J最近slot实际ΔJ仍为+.030784/+.026719/+.039120（大为差），ALL20平均ΔJ却为−.104139/−.097368/−.107661。这说明“所有候选平均变好”不能替代“最关键候选变好并过完整门”，也不支持把所有候选方向统一判错。Q与J各自选一条完整slot，早中晚段不拼接oracle。

C全48个seed×split×group均只有accel_mean超出native1.01；另外四pooled项speedp95、accelp95、jerkmean、jerkp95均通过。A/B各48组也同为accel_mean失败，不应写成全部运动学崩溃。C outer change加速度均值+12.554–12.640%，比A的+12.285–12.523%略高；不能用p95/jerk通过抵消均值失败。

C outer change去重有效good簇均值.5178/.5257/.5059，distinct19.7194/19.7075/19.6917；不是通过复制20条凑覆盖。簇数也不能抵消samecandidate覆盖/保护/运动门失败。C该组平均位置修正约.0579–.0582m，不是完全退回native。

保护错误方面，C36个overall/stable/transition的seed×split单元中14个有ADE或FDE>native1%（主要inner/CAL），逐项见JSON与全组表。原major覆盖每seed+5pp、误差每seed5%收益、同时CI、lost≤1%、protected errors及五motion1%全门均未达到，不升baseline。

## 6. 执行、恢复与资源可信度

成功worker与exit均SUCCESS，开始2026-10-16 18:58:35.162590HK，结束19:15:31.613003HK，墙钟16m56.450s；RESULT elapsed1011.330s另列，不推断两个计时差异原因。3个新C模型SHA、75history/5550steps、12新C+12复用A+4native=28jobs/84主输出齐全。

ARITHMETIC_AUDIT独立重算28保存job的samecandidate质量、原GT-free全K20 complete-link(.10 meanpath/.20 endpoint)、五pooled motion、方向、keep、history合同及全部20k bootstrap，通过，审计耗时475.468s。保存算术不等于全模型/梯度再次回放。eval权重及buffers精确不变由原执行合同审核。

2560source继承旧digest、初末不变，source digest `189bb8f62e3270a63a2e388f95a3480645670a4723996e3a2fb2f477dd854c94`。本次completion_review另做36科学job的144组samecandidate质量计数/救回丢失ID/pooled motion与9训练history核对，最大数值差0；2618个源及完成文件SHA前后再次通过。该补充没有Torch/模型forward/backward/optimizer，未再做聚簇或bootstrap全重放；不能声称外审。

本次资源按1.3×任务估计，而非固定6GiB：E4.578535GiB，1.3E5.952095GiB；生产peak_private4.174679GiB，peak_RSS3.318951GiB，system最小可用physical14.985687GiB、commit21.583168GiB，当前及保留生命周期peak门通过。补充CPU峰值单列，不冒充生产峰值。

旧T1 peak_private4502712320bytes超过其当时1.3E4286884825.6bytes而旧guard误报PASS的历史偏差永久保留；T2修了retainedpeak检查且采用明列估算，不能回写洗掉T1。旧T2协议描述/manifest position文件名失败来源也完整保留，本成功恢复只修正确速度stats文件名，未改科学因素或门。

半程唯一检查实际19:05:16.587（scheduled19:04:07.950），当时已75epoch/28saved、stageT2_PRIMARY_BOOTSTRAP，尚未完整。前轮中断导致控制记录/完成节点未持久化；本次从原工具输出补存halfway_check，不重复OS查询、不冒称按时完成定点。本次人类“分析结果”触发只读完成文件核验，未查询已完成PID/task。

## 7. 系列结论与下一分支

| 层级 | T1 B−A | T2 C−A |
|---|---|---|
| 科学计算与保存审核 | 完成，但历史资源合同偏差保留 | 完成、执行合同通过 |
| 预注册机制信号 | 未成立 | 未成立 |
| 原重大成功全部门 | 未达到 | 未达到 |

D/T3=NOT_TRIGGERED，两个机制门未同时成立；不是尚待排队。至此按原文档结束A/B/C九runs，不扫描keep系数、速度再加accel/jerk、宽度/LR/epoch或slot阈值，不重训旧实验，不提前改scorer。

这轮支持的最窄结论是：**当前固定keep与可逆速度表示，没有在这套冻结表征、native候选支撑、训练预算下解决完整轨迹迁移与保留；FIT拟合收益仍伴随加速度均值代价。** 它不能唯一归因loss、共享head或场景不可迁移，更不证明未来不可预测。

若另立后续研究，应按原指南上升到“冻结P2/现有native支撑是否含足够可迁移修正信息”的问题，结合此前反证设计可辨识对照；不是在本系列自动启动新loss/MoE/scorer/旧模块解冻/新数据。现有权限没有自动扩展到这些新范围。

局限：F16已见原数据，target-disjoint≠context-disjoint；3scene、3seed、相关窗口，outer/CAL适应性复用，统计为探索而非独立wholemodel泛化/SOTA/心理意图或唯一根因证明。

## 8. 文件与反方检查

- [全144组附表](ALL_GROUPS.md)，[完整保存摘要与ID/薄层](completion_review/SUMMARY.json)，[补充复核](completion_review/REVIEW.json)。
- [原统计与全部探索区间](STATISTICS.json)，[原独立审核](ARITHMETIC_AUDIT.json)，[保存结果](RESULT.json)。
- [DA2/DA3反方检查](completion_review/INLINE_DA2_DA3.md)在综合前逐段形成：保留C的FIT/overall/CAL/关键slot正例，同时检查净增掩盖lost、family混用、T1资源偏差等风险；均由同一助手执行，非独立外审。
- 方法采用 [Academic Research Suite](<USER_HOME>/.codex/skills/academic-research-suite/SKILL.md) 的证据—反证分离流程；不把工具执行PASS等同科学成功。

