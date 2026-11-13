# U1：修正方向、成功保留与覆盖边界归因实验

日期：2026-10-15  
状态：**DESIGN_ONLY：仅设计，未实现、未预检、未运行。**  
性质：一次有限的CPU保存输出诊断；不训练、不加载神经网络、不搜索新部署策略。这里U1仅为本新目录的阶段简称，**不是恢复旧D0的U1分支**。

## 1. 目的与边界

U0已经回答：移除S0强安全映射后，共享head能在FIT上增加完整候选覆盖，但outer/CAL未稳定改善，已有成功大量丢失，且加速度均值不合格。

下一步不再重复证明“FIT好、CAL差”，而要在一次诊断中区分：

1. **方向问题**：在评估人口中，保存修正是否从一开始就使完整候选误差变大？
2. **有限步长问题**：方向开始有益，但完整修正走过头？
3. **成功边界问题**：主要破坏的是本来只勉强合格的候选，还是已有较大质量余量的候选？
4. **保留问题是否足以解释失败**：即使完美保留native-success，现有rescue是否足以达到原覆盖目标？

最终交付必须给出一个明确的优先研究分支，允许结论为“混合机制/证据不足”，但不得据此无限追加局部诊断。一次完成后停止，不自动训练loss、gate、scorer或新head。

本实验是适应性、描述性机制诊断；结果可决定下一精确干预的优先级，但不能证明唯一因果根源。GT只能参与离线归因，不能作为部署输入、选择器标签或自动teacher。

## 2. 现有证据与不必再花计算量回答的问题

来源：[U0完整报告](../GENERATION_UNPROJECTED_TIME_COUPLED_REFINEMENT_20261015/U0_execution_20261015/audit_comparator_recovery_20261015/RESULT_ANALYSIS.md)、[实际训练协议](../GENERATION_UNPROJECTED_TIME_COUPLED_REFINEMENT_20261015/U0_execution_20261015/protocol.json)。

| 变化组覆盖相对native，三seed平均 | free_position | integrated_acceleration |
|---|---:|---:|
| FIT | +5.086pp | +5.065pp |
| outer | −1.186pp | −0.659pp |
| inner | +0.939pp | +1.476pp |
| CAL | −2.095pp | −3.286pp |

时间耦合outer rescue为15/11/19、lost为18/16/16；CAL rescue为22/18/23、lost为44/47/41。实际修正已到厘米级，不是S0的近零输出。全部96个评估单元的五运动学联合门失败，失败项为acceleration mean，其余四项未越过1%门。全部12个原主比较同时CI跨0，不得称已证实显著恶化或两个算子等价。

### 已知的保留上限算术

保持现有rescue不变，假设GT特权地找回所有lost，则最多净增rescue数。outer变化组253个窗口，5pp至少需要13个；CAL700个窗口至少需要35个。

- outer seed43只有11个rescue，完美保留也不足13个。
- CAL三个seed只有22/18/23个rescue，完美保留均不足35个。

因此**单纯消除lost不足以满足当前跨seed、跨人口目标**。这已由报告计数推出，不需要启动实验才能知道。新的审计应查清“怎样改对”，而不仅是“怎样不动”。以上不构成实际可部署保留策略，也不取消原保护门。

## 3. 冻结对象、数据合同与工作量

读取[U0 JOBS.json](../GENERATION_UNPROJECTED_TIME_COUPLED_REFINEMENT_20261015/U0_execution_20261015/JOBS.json)定位实际文件，不猜测路径和数组顺序：

- 两臂free_position/integrated_acceleration，seeds42/43/44；最终epoch25固定。
- FIT18909、outer1035、inner1856、original_CAL2879；合计24679唯一ID。
- 24个科学保存job（2×3×4）和4个native参考job；科学重复窗口行148074，不能当148074个独立样本。
- 主要分析为integrated_acceleration；free_position为完整报告的对照，不能根据结果换主臂。
- 每job读取`paths/native/gt/sample_id/groups/track/scene/sequence`，另读已有raw/controls/correction核对；缺字段就具体STOP，不重跑模型补数据。
- 强制ID、GT、原生路径、分组、物理slot顺序与原done/SHA一致。`paths-native`必须与保存correction在登记算术容差内一致，不重解释坐标单位。

原F16/decoder/scorer、六checkpoint、旧源和失败证据均不可变。继承恢复目录2092项原digest，扩入完成报告、补充和新诊断代码；不能覆盖旧digest。总数以实际清单为准，不宣称全部历史文件。

不读取新DEV/test/fold、外部数据，不改checkpoint、不恢复暂停任务、不查询旧完成PID、不修改旧自动化。全新增产物只进本设计目录下未来独立`execution_*`子目录。

## 4. 统一质量定义与分析单位

令native路径为p，保存修正后路径为q，r=q−p，GT为y。每candidate：

\[
A_k=\frac1{12}\sum_t\|p_{kt}-y_t\|,\quad F_k=\|p_{k,12}-y_{12}\|,
\quad J_k=\max(A_k/0.2,F_k/0.4).
\]

集合合格为`min_k J_k≤1`，同一完整candidate同时满足ADE/FDE，不拼不同slot。Q=ADE+.5FDE winner单独保存，不代替J。

每窗口先定义**与新模型无关**的native分层：

- native-success：C0=1；native-failure：C0=0。
- native J最近槽位`j0=argmin J_native`，并列取第一physical slot。每次比较固定这个槽位，不在修正后重新挑另一个槽位冒充同路径改善。
- 原本合格槽位集合`G0={k:J_native,k≤1}`，数量g0。

再按修正结果形成四种事件：retained、lost、rescue、unresolved。全部事件都报告；仅看lost属于结果条件筛选，不能据此解释所有窗口。

主要统计单位是窗口与track，非20个slot独立重复。所有候选级统计同时给出窗口加权版本、原分母及跨seed重复说明。

## 5. D1：修正方向与有限步长损害分开

### 5.1 精确的平方误差分解（辅助，不替代ADE/FDE）

定义`e=p−y`。同一candidate完整修正满足：

\[
\Delta MSE=\frac1{12}\sum_t(\|e_t+r_t\|^2-\|e_t\|^2)
=\underbrace{\frac2{12}\sum_t e_t^Tr_t}_{方向交叉项}
+\underbrace{\frac1{12}\sum_t\|r_t\|^2}_{步长平方项}.
\]

终点也做同样分解。交叉项负表示平方误差的一阶下降方向，不能单独说明完整修正成功；正的步长项可能抵消它。不可把MSE收益冒充ADE/FDE或J收益。

余弦只作辅助：e或r为零时标记undefined，不能加floor制造方向；保存方向交叉项原值、幅度、余弦的有效分母。

### 5.2 真正质量指标的方向导数

考察保存方向上的一维路径`p(λ)=p+λr`，只用于离线解释，不挑λ部署。定义每点范数的右导数：

\[
d_t=\begin{cases} e_t^Tr_t/\|e_t\|,&\|e_t\|>0\\ \|r_t\|,&e_t=0.\end{cases}
\]

于是`dA=mean(d_t)`、`dF=d_12`。设a=A/0.2、f=F/0.4：

\[
J'_+(0)=\begin{cases}dA/0.2,&a>f\\dF/0.4,&f>a\\\max(dA/0.2,dF/0.4),&a=f.\end{cases}
\]

**max并列时取两个方向导数的最大值，不能沿用训练argmax的ADE-first反传约定。** 此处是数学右方向导数，不是复现自动微分tie梯度。

计算全部20槽位，以及固定j0和G0的汇总；保存实际`J(1)−J(0)`、ADE/FDE变化。逐candidate按以下互斥类型归因：

1. **初始上升方向**：J右导数>0；沿该固定方向J是凸函数，精确数学上缩小正λ不能使该candidate的J优于native。
2. **开始有益、完整步长变差**：J右导数<0但J(1)>J(0)，支持沿此方向存在步长过大/非线性误差增长。
3. **实际改善**：J(1)<J(0)，并分别标记是否达到完整合格阈值。
4. **零/临界/不确定**：零导数、数值边界或其余情况，保留独立类别，不强分成好坏。

实际float64方向分类设置固定`|J'|≤1e-10`为数值不确定带，并记录`|a−f|≤1e-10`的近tie；它们不进入“确定方向”计数，原始连续值全部保留。该带只限制归因置信，不改变J≤1、科学阈值或原输出。

补充统计native-failure窗口中是否所有20条均无确定下降方向。不要把j0一条方向差推成整个候选集合都无机会。数学凸性判断限定于**当前保存的固定方向**，不否定其他方向/新模型可达空间；近tie和数值不确定窗口不能给出严格不可救证明。

### 5.3 重点比较

按native-success/failure分别比较FIT→outer、FIT→CAL、以及inner反例：方向导数、实际同slotΔJ、方向/步长类型、修正范数、终点与早中晚误差。三个seed各自报告，不先合并掩盖不一致。

没有对照干预时，上述只能定位输出几何失效类型，不能唯一归因P2信息不足、共享梯度干扰或loss不对齐。

## 6. D2：已有成功为何丢失

### 6.1 边界余量与冗余成功数

native-success窗口定义质量余量`m=1−min_k J_native,k`，按固定三档报告：`[0,0.02]`、`(0.02,0.10]`、`>0.10`。同时按g0=1、2、≥3分层。分档只是描述，不是新增合格标准。

每档给出窗口/track数、lost率、原生误差、幅度、G0中失败比例、修正后的J最近slot变化。native-failure对应报告`minJ−1`同样三档，避免将两人口原始难度差误当方向能力差。

原生余量与g0只作事后GT分层，不能作为部署gate输入。保留所有薄层，不删困难记录；窗口<20或tracks<5的层明确标记低样本，不据其确定优先分支。不用事后分箱搜索得到想要的解释。

### 6.2 原有好候选的“退出合格区间”

只对原本合格、最终不合格的槽位，求：

\[
\lambda_{exit,k}=\sup\{\lambda\in[0,1]:J_k(p_k+\lambda r_k)\le1\}.
\]

因为J沿固定方向是凸的，且λ=0合格，其合格子集是包含0的区间。固定80次二分，保存`lo`合格和`hi`不合格的括区，不把边界四舍五入后当科学PASS。最终仍合格的原生好槽位记为“[0,1]内未退出”，不是声称λ>1时也安全。

对每个lost窗口保存全部G0的退出区间，并汇总最后一条原生好候选的退出点`max_k∈G0 λ_exit,k`：

- ≤0.5：不到完整修正的一半，原有合格支撑已全部丢失；
- >0.5：较靠后才全部退出。

这不是寻找最佳缩放、不是对λ做系数网格，也不计算或部署GT选择的缩放策略。**其它原本失败的槽位可能在中间λ临时合格**，所以此量只描述原生好槽位的消失，不是完整集合覆盖沿λ的全局退出点。

幅度归因必须结合方向导数与native余量：临界好候选沿错误方向退出，不应仅称“幅度太大”；有益方向后来越界，也不能证明一个GT-free幅度预测器一定可学。

## 7. D3：保留上限与运动代价的归因

### 7.1 只做计数的特权保留上限

每个split/group/seed给出：

\[
\Delta C=(R-L)/N,\quad
\Delta C_{keep\_all\_old}^{oracle}=R/N.
\]

在原lost≤1%native-good门下，最乐观可容许丢失数为`floor(0.01*N_good_native)`；同时给出既有R是否足以满足原5pp覆盖要求、还缺多少rescue。

上限含GT特权，仅限定“保持当前rescues、仅消除lost”这个假设。不能说所有保护训练最多就达到它——新训练可能同时改变R和L。不要生成K40、native与refined并集或GT选路径的伪部署结果；本模块不需要新增候选。

### 7.2 加速度均值的可加分解

对每个窗口计算全部20×11个加速度向量的范数均值之差δa_i，固定dt=0.4。同长度、同K使完整人口差可精确分解：

\[
\Delta\bar a=\sum_{s\in\{retained,lost,rescue,unresolved\}}\frac{n_s}{N}\overline{\delta a}_s.
\]

报告四类事件的有符号贡献、分母、native绝对值和原始单位。不把全部代价归给rescue，不将负贡献裁成零。native均值为0时相对增幅undefined，保留绝对值与原literal判定。

speed/accel/jerk的p95不能按相同方式线性分摊；其分层描述必须明确不能相加重构全局p95。复用已审核五运动学结果并校验来源，本次只新增加速度均值归因，不宣称重复完成全部SciPy聚簇审核。

## 8. 统计计划：一个固定小家族，不制造新的“显著收益”

主臂固定integrated_acceleration；主对比为outer变化组减FIT变化组，4个指标：

1. native-failure中固定j0的平均J右方向导数；
2. native-success中的lost率；
3. native-failure中的rescue率；
4. 全部变化窗口固定j0的平均实际ΔJ。

1/2/4正值代表外部更差，3负值代表更差。FIT是已训练人口，差异是拟合—转移差距，不是随机化因果效应。

固定PCG64种子20261015、20000次bootstrap：每次共同抽3个paired模型seed；FIT和outer分别独立抽各自target tracks，track内保留全部窗口。不同人口ID不能强配对。各分层指标由抽样后的分子/分母重算，不能先平均每track率。某次分层分母为0则记无效；不补抽，仅报告有效次数，低于19000次则该指标CI不用于分支判断。

报告各seed原值、普通95%CI，以及4指标Bonferroni同时CI，尾部各0.05/(2×4)。free臂、CAL、inner、分箱、方向类别和退出点只作描述/反证，不加入事后显著性搜索。原U0的12比较CI原样引用，不能用本诊断家族替换它。

由于只有3seed、相关窗口和适应性复用数据，CI仍属探索性。方向指标连续数值完整保存；近tie/不确定数据在主平均值中保留数学计算值，并单列“剔除不确定仅敏感性”，不得挑更有利版本作主结论。

## 9. 预先固定的结论与后续分支

以下用于决定下一设计优先级，不自动启动训练；可以同时成立，不强迫唯一归因。

| 标签 | 必须具备的证据 | 下一设计优先级 |
|---|---|---|
| DIRECTION_TRANSFER_SUPPORTED | 主指标1同时CI下界>0且3seed外部均更差；同slot实际ΔJ及全部slot方向统计不构成直接反证 | 改变观察—候选条件化或共享映射中的一个因素；结构改变必须同时评估成功保留，不盲增宽 |
| RETENTION_GAP_SUPPORTED | 主指标2同时CI下界>0且3seed外部lost率均更高；报告余量/g0组成差异 | 把已有成功保留作为下一训练的明确共同要求或单独受控因素；不能声称仅保护即可达标 |
| FINITE_STEP_OVERSHOOT_PATTERN | lost的原生G0中多数槽位J初始导数<0而最终J上升，并在3seed一致；近tie不算此类 | 考虑一次GT-free信任幅度/适应性修正强度的精确设计，不按当前GT退出点部署 |
| FRAGILE_MARGIN_PATTERN | 3seed中≥80%lost位于native余量≤0.02，且该档lost率高于其余合并档；薄层不触发 | 优先边界稳健性与集合保留，不把样本组成解释为唯一模型机制 |
| MIXED_OR_INCONCLUSIVE | 上述证据不充分、互相冲突或CI太宽 | 报告剩余替代解释后结束本诊断；不扫分箱/epsilon/seed求结论，不凑“唯一根因” |

后三类中的50%多数和80%只是固定描述性路由阈值，不是统计显著性、物理规律或重大结果门。若同时出现方向迁移和保留问题，下一结构实验必须以保留为约束评估，而不是把两个因素同时改掉。

FINITE_STEP的“多数”按每seed全部lost窗口的所有G0槽位计算，分母包含数值不确定槽位但它们不计入分子；同时报告每窗口先求比例再平均的版本，二者不一致则只给MIXED解释。FRAGILE的80%分母为该seed全部lost窗口，不删除缺乏某一派生诊断的记录。

**在所有分支中：**原完整覆盖、旧成功≤1%丢失、protected ADE/FDE≤1%损害、全部五pooled motion≤native1.01、change ADE/FDE各5%与三seed/CI标准不变。U0当前不满足，不能因某项归因清楚而升级baseline。scorer仍不是下一优先项。

## 10. 核心实现设计与审核

建议未来代码职责（本轮不创建可执行脚本）：

```text
source_contract.py       # manifest、ID、slot、dtype、旧SHA合同；只读旧目录
direction_metrics.py     # MSE分解、ADE/FDE/J方向导数、零向量/并列
retention_intervals.py   # native余量/G0；退出区间固定80次二分
attribution.py           # 四事件、保留上限、加速度均值可加分解
cluster_bootstrap.py     # 两人口独立track + 共享seed抽样，固定4家族
run_audit.py             # 24科学job有界串行处理，原子done合同
independent_review.py    # 第二实现与全部保存数据的标量/计数复核
```

核心流程草图：

```python
for job in manifest.scientific_jobs:  # 恰好24个，不选更好的seed
    verify_source_and_job_hash(job)
    native, refined, gt, ids = read_immutable_saved_arrays(job)
    residual = refined - native
    native_metrics = same_path_metrics(native, gt)
    refined_metrics = same_path_metrics(refined, gt)
    directions = right_directional_derivatives(native, residual, gt)
    events = classify_window_events(native_metrics, refined_metrics)
    # G0中的所有槽位，而不只选择被破坏最严重的一条
    exits = bracket_original_good_exits(native, residual, gt, iterations=80)
    record = summarize_by_group_and_native_stratum(
        native_metrics, refined_metrics, directions, events, exits)
    write_new_artifacts_atomically(record)  # 永不写原job
verify_all_original_hashes_again()
```

必须先实际通过的fixtures：

- r=0、e=0、终点误差为0、J两个分量精确并列/近并列；并列右导数用max而不是第一个分支。
- 朝GT微小修正、背离GT修正、沿正确方向过冲；验证MSE恒等式、有限变化与导数区别。
- 两条不同slot分别ADE/FDE达标但无共同好candidate，必须判失败。
- 原好候选从0即退出、在中途退出、λ=1恰达边界、最终未退出；原失败槽位中途变好不能被误计为G0。
- all20同向/互相冲突、多个原好槽位、g0=0、候选并列、空分组、无效bootstrap分母。
- 独立loop与vectorized实现对实际两臂各128保存窗口核验；方向有限差分只在光滑fixture验证，不因跨max活动面搜索epsilon求PASS。
- 算术容差沿用U0保存审核2e-12用于可比标量；方向导数在极小误差处分母敏感，须保存独立逐点项比较与不确定标志，不能把2e-12当所有病态量绝对误差保证。原科学分类始终literal，无评价epsilon。
- 验证done SHA兼容才skip；坏源、坏ID、坏slot、坏输出实际拒绝。这里是分析恢复，不是训练optimizer/RNG恢复。

所有fixture和真实小样本审核均为未来实施要求，不是本文已通过事项。

## 11. 资源、时间与完成合同

遵循最新[项目资源策略](../RESOURCE_POLICY.json)：启动前available physical与commit分别≥估计任务峰值E×1.3；运行时分别扣除该任务已分配RSS/private后的剩余需求，实际RSS/private超过1.3E即STOP_REESTIMATE。**不继承旧固定6GiB门。** 不改OS/pagefile、不关闭用户程序。

峰值按实际NPZ payload、一次job解压、分块工作数组、bootstrap压缩track统计和实测RSS/private估计。每次一个job、candidate计算分块，禁止把24份完整路径一起驻留。bootstrap只操作track级充分统计，不重复读取轨迹20000次。

预检用两臂实际128块测方向/二分/归因时间，并加24job读取、初末SHA、独立复核及CI开销；planning≤30分钟才可启动。超过则报告具体瓶颈，不临时删split/seed/审核或改变80次规则。本文无实际ETA，不把上轮2分钟审核耗时当本轮生产保证。

未来执行如采用独立OSworker：唯一0-trigger任务、byte-lock、一次startup、一次预计50%检查后修正completion定点、一次完成核验；不高频poll，不查询旧完成任务。当前无启动或自动化变更。

完成必须：worker与exit双SUCCESS、24job完成、全部来源/输出SHA初末不变、独立复核、4指标bootstrap、资源记录、完整报告。没有新模型，不创建checkpoint；不将本次标成新formal训练或完成长期目标。

最少产物：

```text
protocol.json / source_manifest.json / INPUT_ID_AUDIT.json
PREFLIGHT.json / RESOURCE_ESTIMATE.json
jobs/<arm>/<seed>/<split>_candidate_diagnostics.npz
jobs/<arm>/<seed>/<split>_window_diagnostics.npz
jobs/<arm>/<seed>/<split>_summary.json + done.json
PRIMARY_COMPARISONS.json / bootstrap_draws.npz
RETENTION_CEILING.json / ACCEL_ATTRIBUTION.json
ARITHMETIC_AUDIT.json / PRESERVATION.json / RESOURCE.json
RESULT_ANALYSIS.md / RESULT_ANALYSIS.json / NEXT_BRANCH.json
```

文档必须包含全部split×group×seed反例、实际分母和薄层、未触发分支。任何新GT事后结果不能进入部署路径或成为训练teacher而不另作精确设计。

## 12. 反方检查与研究边界

本设计依academic-research-suite完成同助手架构/反方检查，不是外部同行评审，也不是实际运行审核：

- **设计检查**：这是固定输出的几何归因，不是导致泛化失败的随机化因果实验；不得把方向坏直接等同表示不足。
- **证据检查**：原生余量不同、g0不同、群体组成不同都可能解释部分差距；同slot和分层帮助定位，但不消除混杂。FIT已参与训练，不能当独立验证。
- **最终反驳**：即使找到GT知道的退出幅度，也不知道past能否预测它；保留上限也不是learned gate可实现性能。此轮结果只能指导下一单因素设计。

保留inner正例、outer seed44净增、自由臂与耦合臂各自的优劣；不把原12个CI跨0说成等价或无能力。原F16见过数据、target-disjoint非context-disjoint、只有3seed/3scene、outer/CAL已适应性复用的限制不消失。

仅在已有合法数据范围做离线计算，不接触或重新识别行人、不导出私人数据、不声称获得新的伦理审批。最后结论必须是“目前哪一种输出失败模式有证据、哪种未识别、下一次只干预什么”，而不是“模型终于理解/不理解行人”。
