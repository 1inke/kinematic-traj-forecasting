# 01｜测试协议与固定反事实

状态：DESIGN_ONLY。本轮属于NONFORMAL保存输出诊断，不增加模型训练数、不升级baseline。

## 1. 证据源与输入范围

唯一W0根目录：

`MODE_PRESERVING_CONDITIONAL_REFINEMENT_DESIGN_20261017/W0_execution_write_recovery_20261018/`

必须只读绑定：

- [JOBS.json](../MODE_PRESERVING_CONDITIONAL_REFINEMENT_DESIGN_20261017/W0_execution_write_recovery_20261018/JOBS.json)：640题实际files路径及SHA。
- [sample_registry.json](../MODE_PRESERVING_CONDITIONAL_REFINEMENT_DESIGN_20261017/W0_execution_write_recovery_20261018/sample_registry.json)：原32失败+16成功登记。
- [FULL_FIT_AUDIT.json](../MODE_PRESERVING_CONDITIONAL_REFINEMENT_DESIGN_20261017/W0_execution_write_recovery_20261018/FULL_FIT_AUDIT.json)：既定13份选择，`chosen`的key是FIT数组index，不是sample_id。
- `portfolio.npz`：完整FIT native、原W0 paths、GT、IDs及group/track/scene/sequence。
- `RESULT.json`、`source_manifest.json`、`ARITHMETIC_AUDIT.json`、`completion_review/REVIEW.json`及旧输出SHA。

46题曾从父`W0_execution_20261017`兼容复用，因此必须沿JOBS.files的真实路径读，**不能假设所有npz位于当前child/attempts**。禁止执行旧common/run模块顶层造成写入或resource/PID副作用。

基于本轮已读索引的预期：640状态条目、18份含actual controls/paths的方案、17严格合格、14不同窗口；17合格方案覆盖13窗口。若实际登记不符，先核对来源，不删记录凑数。

完整640题只做manifest/状态核对，无actual paths的622题不伪造反事实，不重求解、不列作本诊断“反事实失败”。16原成功identity对照只验原存储和未修改一致性，不虚构其修复解。

## 2. 分析单位与层级

**Primary：原W0已选的13个合格方案，一窗口一方案。** 不重新依据target-only是否通过选方案；不按本轮结果改变原chosen。

**Secondary：全部17个合格方案。** 同窗口多方案只作敏感性，汇总时另报按窗口“至少一份target-only通过”的计数，不把17当17个独立窗口。

**Negative case：** sample22057/slot7的原实际路径质量合格但accel_p95失败，完整走同样42种配置，单列结果。不能因为消融后改善就追溯改写其W0判定。

总体基数同时保留32个注册失败窗口；只在13个已找到见证窗口中的比例必须写明条件分母。无训练seed、不做假三seed统计。

## 3. 42种配置：只复制完整slot，不缩放、不重新优化

设同一窗口原生集合为N，保存W0实际集合为J，原求解目标物理slot为j。所有集合shape固定20×12×2，native与actual均用存储数值，禁止先转float32损失精度。

| 名称 | 定义 | 数量/方案 | 目的 |
|---|---|---:|---|
| N | 原生全部20条 | 1 | 原生参考 |
| J | 保存的W0完整20条 | 1 | 原完整解重现 |
| T | 只将N[j]换成J[j]，其余19条N不动 | 1 | 目标修正单独保留 |
| O | 只把J[j]恢复N[j]，其余19条保留J | 1 | 非目标修正的作用 |
| DROP_k | 对每个k≠j，将J[k]恢复N[k] | 19 | 单个非目标修正对完整解的必要性 |
| ADD_k | 对每个k≠j，将T[k]换成J[k] | 19 | 一个非目标修正是否足以配合目标 |

总数4+19+19=42；18×42=756。物理slot升序遍历，GT不参与DROP/ADD次序或选择；不做greedy删除、子集搜索、按贡献排序挑最佳k、二分缩放或混合系数扫描。

即使配置数值重复也保留逻辑记录及一致性断言，可用相同tensor hash关联，但不得当重复独立证据。

## 4. 不变量与严格门

对每个原合格方案，T、DROP_k、ADD_k的目标路径都与J[j]逐元素完全相同，因此其目标ADE/FDE应相同。**若这几个配置出现目标质量变化，是实现/索引问题，不能解释成协同效果。**

每个slot由已保存合法actual控制或native零控制产生；按slot拼接保持U0独立控制的可实现域与位置界。这里的“可实现”指存在raw控制，不指共享head已经能预测这些控制。

对每个配置分别保存：

1. `target_good`：该固定j同时ADE≤.2且FDE≤.4；以及其J、质量余量。
2. `set_good`：20条中至少一条同完整候选满足上述门；不能拼不同slot最小值。
3. `control_position_pass`：控制单位盘、每步位移≤.20(t/12)^2；native slot使用零控制。
4. `window_motion_pass`：全部20条窗口pooled的speed_p95、accel_mean、accel_p95、jerk_mean、jerk_p95，每项literal≤1.01×N。
5. 全K20原SciPy complete-link D=max(meanpath/.10, endpoint/.20)，cut1的distinct/effective-good，以及native支撑距离，见02。

dt=.4；第一预测速度用局部最后观测点0到第一预测点；速度12项、加速度11项、jerk10项。p95以20×时间元素整体计算，沿原NumPy quantile线性插值，不能平均各候选p95。native指标为0时直接比较绝对值，不除零、不加评价epsilon。

T在原成功17方案中质量必须仍成功；如果T失败，只能进一步归因到严格物理门或实现/数据错误，不能说其他候选“帮助它更接近GT”。O可能仍有其他好slot，必须如实统计。

## 5. 完整FIT的四个portfolio

使用原chosen的13窗口且只这些窗口，构造：

- `FIT_N`：全native。
- `FIT_J`：原W0 portfolio，要求与原保存paths完全一致。
- `FIT_T`：13窗口仅替换既定目标slot。
- `FIT_O`：13窗口保留非目标修正、目标恢复native。

其余全部窗口保持原生。**FIT_T不按窗口物理结果回退**：它是反事实测量，失败必须保留。不从另4份合格方案替换chosen，不新增“筛过再组合”的第五portfolio。

完整18909窗口、change4725；overall/stable/transition/change分别重新计算质量、救回丢失、全K聚簇、五pooled motion。无原生成功窗口被修改，所以lost=0是构造不变量，不能声称学会保留。FIT_J和FIT_T都应保留原13个目标good，因此change覆盖1950是预期不变量，**本轮区分力主要在安全性/支撑/目标兼容性，而非再次庆祝13个救回。**

窗口p95失败不自动意味着全FIT p95失败，窗口通过也不保证全FIT p95通过。两层结果必须同时列出，不用更有利的一层替代另一层。全FIT受13/18909稀释，另报13窗口合并及逐窗口绝对超量，防止总体掩盖局部代价。

## 6. 权限和不做的事情

只使用现有获准、去标识轨迹数据；不新增受试者/外部数据/身份识别，不虚构新的伦理审批。本轮不访问新DEV/test/fold，不训练或前向旧模型，不安装/调用solver，不更改P1目标或oldsource，不恢复P2/旧worker/旧监控。

本次没有solver调用是方法范围，而不是重新建立“以后不得安装solver”的禁令。若未来需要证明目标单独修复的更广存在性，另立restricted optimization设计，本轮不会偷偷执行。
