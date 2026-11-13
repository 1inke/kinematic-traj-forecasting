# 新实验指导：分开验证生成空间与条件映射

日期：2026-10-15（日本标准时间）｜版本：1.0｜状态：**DESIGN ONLY**

本轮只交付设计。没有实现脚本、计算新结果、运行预检、训练模型或创建后台任务。本文不是旧 D0/U1 的恢复指令；未来执行须在新的 execution 子目录开始。

## 0. 首要决定

**不继续梯度投影，不自动增加 MoE，不再调整 selector 特征。先做一次有解析上界的短筛选；只有证明当前安全生成空间内确实存在足够的覆盖增量，才做条件映射学习对照。**

核心问题拆成两个，而不是继续笼统讨论“泛化不好”：

1. 在当前选槽与位置界下，即使知道真实未来，最多能救回多少完整轨迹？
2. 如果存在足够安全修正，原生路径自身能解释多少修正，额外过去编码 P2 又能增加多少？

本方案最多允许一次解析筛选 A、一次双臂学习 B，以及条件性迁移 V。A 不通过就停止；不能用“诊断还不够”无限追加局部试验。**若当前受限平台本身空间太小，应退出这个平台，不再修补其共享优化器。**

## 1. 证据起点与禁止重复的路线

|已有证据|允许的解释|本轮对应约束|
|---|---|---|
|D0 普通更新仅 seed43 满足干扰门；投影在42更差、43减害21.82%、44普通更新已改善|局部冲突存在，但这次投影没有稳定有用效果|U1及其条件性V保持未启动，不换投影顺序/系数重试|
|D0约95–96%候选触发位置缩放|当前可达范围可能重要，但尚非根因证据|用解析可达界量化，不凭触发率直接放宽界|
|过去知道终点的native修正，outer change覆盖110→224/253；但没有局部位置界，且部分组速度失败|存在有价值的未来条件几何见证，不代表安全、可学习或可预测|不再重复“喂GT终点能否修好”|
|T1固定模式责任相对dynamic覆盖差约−12.91pp，且FIT也差|该模式词典/固定责任方案失败|不重启固定k-means责任或T2/T3|
|R1–R5已研究可修复anchor标签与排序|选中可修复标签不等于产生好轨迹|不以Hit@4或标签AUC代替真实生成结果|
|输入支持诊断不足以单独解释CAL失败|粗输入距离内仍可能有条件分布变化|不得宣称分布问题排除，也不直接归咎共享head|

主要来源：

- [D0结果](../SHARED_MAPPING_GENERALIZATION_DESIGN_20261015/D0_recovery_resource_20261015/RESULT_ANALYSIS.md)
- [广分支几何筛选设计](../GENERATION_BRANCH_SCREEN_20261014/DESIGN.md)
- [T1结果](../MOTION_PATTERN_TRANSFER_DESIGN_20261015/T1_execution_20261015/RESULT_ANALYSIS.md)
- [分布诊断](../GROUPED_TRANSFER_DISTRIBUTION_DIAGNOSIS_20261015/RESULT_ANALYSIS.md)
- [原position3协议](../PED_FOUR_DIRECTION_SERIES_20261011/experiment1_motion_process/DESIGN.md)
- [已有native-relative射线算术](../GENERATION_NATIVE_SIMILARITY_FEASIBLE_SERIES_20261008/physical_ray.py)

引用旧源只用于读与核验，不执行旧运行器顶层代码，不改锁源、模型、报告、失败证据。

## 2. 总体顺序与权限边界

|阶段|实际计算|唯一研究目的|进入条件|
|---|---|---|---|
|A0 解析空间界|保存native/GT上的CPU算术，无训练、无数值优化|区分选槽限制与0.2m位置界限制|未来批准执行后首先做|
|A1 安全正见证|固定解析修正+既有射线约束，无模型训练、无标签求解器搜索|给出可实际构造的安全覆盖增量下界|A0的bottom4上界达到预定空间门|
|B 条件映射对照|原生路径输入 vs 原生路径+P2，双臂三seed，固定教师路径监督|测试已有P2是否提供可迁移的额外修正信息|A1在FIT通过空间门；新监督协议被纳入执行批准|
|V 迁移|先recording-time，再条件性scene，同一B协议fresh重训|验证增量不只依赖target划分|前一级质量/保护/运动学门全部通过|

本设计不授权现在运行。未来批准整个文档时，B的**新教师路径监督**必须明确包含在批准范围，不能冒称它属于旧canonical loss；无需就日常数值实现反复询问。

禁止旧F16/decoder/scorer解冻、外部数据、DEV/test/其他fold、云服务、旧暂停任务恢复。不得自动续开旧六小时窗口。新基线和原长期目标状态均不改变。

## 3. 共同数据和评价约定

复用已注册target划分：FIT18909、outer1035、inner1856、originalCAL2879。outer为固定末轮主评价，inner/CAL均仅事后报告，不选epoch、阈值、教师或架构。统计只用FIT；保留全部原ID、group、track、scene及其分母。

同一native输入多seed重复时先核验EXACT，再在A中只算一份，不把重复缓存当模型重复。B使用三个fresh seed42/43/44，不能复用D0临时副本。

坐标为末观测点原点的focal坐标；12步、dt=.4；原生槽位顺序不变。原bottom4来自冻结logits及既有稳定tie规则，其余16原生路径不动。不得根据GT重选部署槽位。

覆盖定义：

\[
J_i(C)=\min_{k\in C}\max(ADE_{ik}/.2,FDE_{ik}/.4),\quad good_i=1[J_i\le1].
\]

禁止拼接来自不同slot的minADE与minFDE。另报告minADE、minFDE、Q与J各自同一slot的早/中/晚误差、rescue/lost、原GT-free全K20 complete-link(.10 meanpath/.20 endpoint)有效good与distinct。数量不是越多越好。

五motion保持原12v/11a/10j、dt=.4、全部窗口×候选×时间的pooled速度p95、加速度mean/p95、jerk mean/p95；不能平均窗口p95替代总体p95。所有seed、split、group均须literal≤1.01×native；不添加评价epsilon/floor。原major的change ADE/FDE各≥5%、protected overall/stable/transition≤1%harm等门保持。

## 4. A0：用解析界先判断这个平台值不值得继续

### 4.1 精确的放松空间，而非另一次GT曲线拟合

对native候选p[k,t]与GT y[t]，令：

\[
b_t=.20(t/12)^2,\quad e_{kt}=\|p_{kt}-y_t\|,\quad d^{LB}_{kt}=\max(e_{kt}-b_t,0).
\]

则任何逐时刻修正范数不超过b_t的轨迹，均满足：

\[
ADE_k\ge\overline{d^{LB}_{kt}},\quad FDE_k\ge d^{LB}_{k,12}.
\]

这两个界可以由**同一条逐点朝GT投影的路径同时达到**：

\[
r^{ball}_{kt}=\begin{cases}
0&e_{kt}=0\\
\min(1,b_t/e_{kt})(y_t-p_{kt})&e_{kt}>0.
\end{cases}
\]

因此它是“12个时刻独立二维球、无时间basis、无物理约束”的精确质量最优值；对position3、Bernstein或加入安全约束的模型，只是乐观上界。**不能称它为当前模型精确最优值，也不能称安全上界已可实现。**

### 4.2 两个嵌套端口，严格同候选预算

1. `UB_bottom4`：只有旧bottom4使用上述理想误差，其他16仍用原生误差，取同slot J最小值。
2. `UB_all20`：20槽均允许相同位置界修正。只用于定位选槽瓶颈，不是GT-free选槽策略，不进入部署。

二者都是K20，不构造K24大库。应逐窗口验证：native coverage≤UB_bottom4≤UB_all20；原生good不丢。保存逐slot ADE/FDE下界，不能只保存三个总数。

### 4.3 一次性分支规则

定义FIT change的净覆盖增加，统一用百分点pp；**5pp是新研究的实用空间筛选阈值，不是证明统计显著，也不是降低原major。**

- `UB_all20−native <5pp`：0.2m局部球即使知道GT也不足以支持预期增量。结束本系列；退出当前小幅局部修复平台，不继续其梯度、selector、MoE变体。
- `UB_bottom4−native <5pp`但all20≥5pp：定位到固定bottom4的明显机会限制。停止本设计的B；不根据GT选top4部署，不回到R1–R5循环。后续须另立原生支撑分配/非局部生成设计。
- `UB_bottom4−native ≥5pp`：仅表示有几何空间，继续A1，尚不能训练B。

只用FIT决定门；outer/inner/CAL在规则锁定后一次性描述性报告。它们不用于选择阈值、槽位、半径或下一臂。边界小差异按原精确比较记录，不四舍五入成通过。

## 5. A1：把乐观上界变成安全正见证

### 5.1 固定构造，零搜索

仅对原bottom4，以r_ball为提案，调用独立隔离的既有native-relative射线算术，得到：

\[
w_{kt}=p_{kt}+\alpha_k r^{ball}_{kt},\quad0\le\alpha_k\le1.
\]

alpha不使用GT再搜索，它只由native和提案决定；GT已经用于构造提案，因此**整个w仍是oracle教师，不是GT-free预测**。

以相同差分算子D取v/a/j，逐位置要求：

\[
\|Dp+\alpha Dr\|^2\le1.01^2\|Dp\|^2.
\]

沿用所引用physical_ray.py实际12+11+10个约束及稳定二次根分支、零预算规则与内部数值余量；不得照抄历史文字中的“31”而漏掉实际33项。公式是强充分约束，**比五项pooled门更严格**，不能把它未找到见证说成所有安全路径不存在。数值证书只解释浮点误差，科学评价仍literal。

每条候选alpha独立，但全窗口最终K20必须独立重算五motion。若实际逐对应导数literal检查或该窗口五项literal检查有任何失败，整个窗口教师回退native20，记录精确原因；禁止调alpha直到通过。报告回退率；超过1%窗口则标为构造数值异常，停止实施检查，不靠大范围回退隐藏缺陷。

其余16原路径EXACT。因为每一步仍位于native到GT的线段上，理论上每个对应点的GT距离不增加；保存重读必须验证samepath质量不退化及native-good lost=0。任何实质反例是实现/坐标问题，不是可接受噪声。

### 5.2 最少必须报告的结果

- native、UB_bottom4、UB_all20、witness_bottom4的全组C/J/ADE/FDE及分母。
- 原生失败中的可救数量、全组rescue/lost；不只报条件命中率。
- 每候选alpha、零alpha比例；限制来自v/a/j哪一类、零native预算、位置界；保留并列活动项。
- 安全见证相对于解析机会的兑现率；分母为UB_bottom4−native，零分母写NA。
- 全量五motion、有效good簇、非零修正幅度；只报告非零不算质量收益。

### 5.3 A1进入B的门

FIT change witness−native≥5pp；overall/stable/transition各覆盖不下降；全部FIT组五motion literal通过，native-good lost=0，所有保存/来源审核通过。不得选最好seed——A没有模型seed重复。

不通过则停止B。明确区分：

- 上界小：该位置界/槽位平台空间不足。
- 上界大、固定安全见证小：只说明本构造/充分安全约束没有兑现空间，**不可证明安全空间小或观测信息无用**。
- 安全见证大：才有资格测试条件映射，不能先宣布“模型有能力理解”。

本轮到此最多一次CPU筛选，不扩成新的逐anchor求解器、R1标签生成或未知U样本搜索。

## 6. B：同一安全生成器，检验额外观测信息的价值

### 6.1 为什么不是旧T1、R5或D0的改名

- 不学修复标签排序；直接生成实际12步轨迹。
- 不给20种k-means模式固定GT类别；各输出始终对应同一native物理槽位。
- 不投影组梯度；新问题是安全连续修正的条件可学习性。
- 不比较新的B与旧position3来宣称单因素收益；本B内部两臂才是配对对照。

### 6.2 固定结构与两臂

共同MLP120→128→64→24/GELU，理论25304参数，实际预检重新计数。输入为标准化P2_96与当前native12×2路径24维；共同FIT-only统计，不单独拟合不同臂统计。共享head对四个旧bottom4槽逐一调用，不输入slotID/group/未来GT/logit/scene/track。

|项目|A：native-only|B：native+P2|
|---|---|---|
|native24输入|真实|相同真实|
|标准化P2通道|固定0|真实|
|参数、buffers、初始化、batch顺序、输出几何|配对相同|配对相同|
|监督|相同FIT安全教师w|相同FIT安全教师w|

P2归零是从训练开始的消融，不是仅测试时制造OOD。两臂名义参数相同，但有效容量和优化不同，A的P2输入列梯度解析为零属预期，不能伪造全参数正梯度。native本身已包含历史/社会信息，故这测试的是**P2在native之外的增量价值**，不是“是否存在所有过去信息”。

raw输出reshape12×2；u=tanh(raw)/max(1,L2norm(tanh(raw)))；r=b_t*u。再用与A1相同的GT-free射线映射和实际literal回退规则输出p+alpha*r；回退不得读GT，规则训练/推理一致，需真实GPU分支和保存路径审计。未选16 EXACT，最终K20。最后一层零初始化，两臂P0完整native20。

这是明确更换为12步自由修正+安全映射的**新平台**，不是沿用position3的六输出。该平台仅为保证教师路径处于其几何放松域并测试映射；它与旧basis之间不作纯因果效果比较。tanh有限参数未必达到球边界，教师可行不保证有限网络可精确实现。

### 6.3 新监督：只用FIT教师，不能混用外层GT

固定损失：

\[
L_{teacher}=\frac1{4B}\sum_{i=1}^{B}\sum_{k\in bottom4_i}
\left[ADE(\hat p_{ik},w_{ik})+.5FDE(\hat p_{ik},w_{ik})\right].
\]

全四槽等权、全FIT窗口参与；identity教师也参与，不只采rescue窗口，不做class balancing、winner重选、额外diversity/protection、soft assignment或物理penalty。**这是有意新增的教师路径监督，不是canonical hardmin loss，不能拿两者训练loss数值直接比较。** 双臂同一教师和损失才能解释P2增量。

FIT GT仅生成FIT教师与训练评价；outer/inner/CAL教师若计算，只能放入独立只读诊断目录，训练代码API必须拒绝这些split。不得把外层oracle路径作为输入、蒸馏目标、归一化来源或checkpoint选择依据。

AdamW lr=.001、wd=.0001、batch256、shuffle、25epoch、无clip/scheduler；seeds42/43/44；全部fresh、不warmstart；主结果固定25。总6 runs/150epochs。除学生轨迹外保存实际alpha与回退，防止“安全”只是输出native。

### 6.4 哪些结果分别说明什么

同时报告FIT与outer的教师误差、native→student coverage、学生兑现的教师增量比例；比例不截断到[0,1]，分母≤0为NA。核心结论仍用绝对C/J/ADE/FDE，不只用比例。

|观察|能支持的解释|行动|
|---|---|---|
|两臂FIT都不能兑现教师空间|当前网络/监督/优化/边界组合仍有问题|停止；不能归因跨轨迹泛化，不加长epoch扫描|
|两臂FIT好、outer都差|存在训练到目标人口的映射迁移缺口|保留结果，不自动宣布不可预测或MoE有效|
|A、B都好但相近|native输入已经足够支持这一修正；P2增量未证实|不能称历史无用；不为凑阶段自动V|
|B在FIT更好、outer不比A好|P2增量不能稳定迁移|停止该P2条件映射方案|
|B对A及native均稳定有用且安全|现有观测编码能提供额外可迁移修正信息|进入V；仍不是心理意图理解证明|

### 6.5 进入V门

outer change：B−A每seed覆盖>0、平均≥5pp且校正CI下界>0；B−native每seed覆盖>0且校正CI下界>0。B的change ADE/FDE均不劣于A；overall/stable/transition的ADE/FDE每seed对A与native均≤1%harm；B全部四split×四group×三seed五motion literal通过。必须有真实非零rescue，不能仅更接近native。

这些是继续研究门；最终major还须原change ADE/FDE各≥5%等全部要求。阴性不加P2噪声强度/hidden width/教师权重/步数或另一安全余量变体。

## 7. V：通过后才做完整迁移，不拿局部probe当泛化

先recording-time既有精确ID：train16104、outer566、inner2134；通过同类门后才scene-family：train16337、outer1561、inner1773。CAL2879一直posthoc。

每split重新只用本train生成教师、共同统计与fresh双臂三seed，25epoch；不得复用target权重/教师/statistics，不能让新的outer进入训练。P2/native来自原冻结F16，F16历史见过池内数据，因此只评newhead迁移，不是整个模型全新未见泛化。

每个split先固定同A0/A1的源侧空间门；没有足够源侧见证就记录未触发，不改半径/槽位硬做。time失败不做scene；最多额外12 runs。不同outer ID不跨split强配对，3scene不是大量独立场景。

## 8. 统计、审计和失败解释

A是全指定人口的确定性几何计算，不伪造三个seed或显著性；可描述性报告track聚合分布，不用新CI换触发规则。

B/V预留24个主比较：每阶段B−A与B−native，各outer change C/ADE/FDE/J四项。20,000次PCG64 seed20261015，whole target-track与三个模型seed交叉配对重采样、window-weighted；time主抽timeblock，track敏感性分开；双尾Bonferroni .05/24，未运行阶段不缩family。95%只描述。共享同次配对索引并保存SHA；三个seed和历史适应性不能被大量bootstrap伪装成确认性证据。

oracle/student配对图必须写明未来信息不对称。所有组都报，不能只选change好数；安全回退造成零修正时写native identity而非“高质量生成”。

训练前必须通过：

- source/split-ID/native-slot/坐标dt对齐；相关旧digest继承而非覆盖。
- A0闭式与逐点构造独立一致；零距离、刚好边界、独立minima假成功、bottom/all嵌套fixtures。
- A1非零曲线/零导数预算/各motion拒绝/回退/无GT推理接口fixtures；数学质量单调与实际literal门分开。
- B真实GPU8batch、paired权重与buffers、P0原生identity、合法梯度；A的屏蔽通道零梯度单列；独立教师loss与坐标梯度核验；硬活动面FD切换不得冒称通过。
- 双臂真实513FIT+12inner子集两epoch连续/中断，全部参数/buffers/optimizer、TorchCPU/CUDA/loader/NumPy/Python RNG、history EXACT，坏source/output/恢复契约真正拒绝。只代表子集恢复。
- 实际保存128条数据的独立完整quality/原SciPy聚簇/five-motion/固定16/几何及回退复核。

完成时独立CPU审核所有输出、checkpoint source/IDs/stats/objective/初始化/optimizer-RNG绑定、150history、CI、源与输出SHA；不得称所有模型/梯度二次回放或外审。source/ID/资源/数值实质异常STOP留证据，不调整科学阈值求通过。

## 9. 最小实现结构与核心接口

建议未来目录（本次未创建这些可执行模块）：

```text
execution_YYYYMMDD/
  protocol.json                 # 执行前将本设计数值与源SHA固定
  data_contract.py              # split白名单、坐标、ID、slot、旧SHA
  analytic_envelope.py          # A0闭式lower errors与两个coverage上界
  safe_witness.py               # A1、只读隔离ray、literal校验/回退
  teacher_store.py              # fit_only=True；外层教师禁止返回训练API
  model.py                      # 共同120→128→64→24与P2输入开关
  train.py                      # 固定teacher全4槽loss；不含外层GT加载
  evaluate.py                   # actual K20、samepath、clusters/motion
  audit_saved.py                # 独立NumPy保存算术与来源审核
  preflight.py / benchmark.py
  run_manifest.json / runtime_state.json
  RESULT_ANALYSIS.md
```

关键伪代码仅表达契约，不是经过预检的可运行实现：

```python
def envelope_bound(native, gt, selected4):
    # native[N,20,12,2], gt[N,12,2]; native origin must be last observed point
    e = norm(gt[:, None] - native, axis=-1)
    b = 0.20 * (arange(1, 13) / 12.0)**2
    d_lb = maximum(e - b, 0.0)
    j_native_slot = maximum(mean(e, -1) / .2, e[..., -1] / .4)
    j_ball_slot = maximum(mean(d_lb, -1) / .2, d_lb[..., -1] / .4)
    # same slot supplies ADE and FDE; untouched16 use their original errors
    j_bottom = where(selected4_mask, j_ball_slot, j_native_slot).min(-1)
    j_all = j_ball_slot.min(-1)
    return j_bottom, j_all

def build_fit_teacher(native, gt, selected4, split):
    assert split == 'FIT'
    proposal = project_gt_to_per_time_balls(native[selected4], gt)
    repaired = native[selected4] + native_relative_ray(native[selected4], proposal)
    teacher = scatter_into_native20(native, selected4, repaired)
    teacher = native_fallback_on_literal_violation(teacher, native)  # no GT here
    assert_samepath_quality_monotone(teacher, native, gt)  # label audit only
    return teacher[selected4]

def predict(native, p2, use_p2, stats):
    z = concat(standardize(p2, stats.p2) if use_p2 else zeros96,
               standardize(native_selected24, stats.path))
    r = bounded_free12(mlp(z))
    pred4 = native_selected + native_relative_ray(native_selected, r)
    pred20 = scatter_into_native20(native, selected4, pred4)
    return native_fallback_on_literal_violation(pred20, native)  # GT-free
```

A0不调用model，A1不调用optimizer；B部署predict签名不得出现GT、教师、group、ID。教师路径反向只到学生，教师不可学习。回退与alpha活动面必须进入训练日志。

## 10. 时间、资源与交付纪律

现在没有实测ETA，不沿用D0八分钟或过去训练吞吐。A先用8个真实128窗口chunk测完整构造/质量/保存审核；B每臂4warm+32measure FIT256，加全split评估/聚簇/CI/初末SHA与独立审核估时。**每个阶段planning>30min则不启动**，不能去掉FIT、审计、epoch来凑预算，也不自动沿用过去一次性长预算批准。

process private≤6GiB、system physical和commit各reserve6GiB；长期约17GiB可用偏好保持。不改OS/pagefile、不kill用户程序、不查已完成或暂停任务。只新目录写入，锁源后不能静默修改。

未来后台运行采用唯一0trigger独立worker、byte-lock、一次startup、一次预计elapsed半程、一次实际完成定点；健康无变化静默。执行成功需worker/exit双SUCCESS和完整审核，不等于科学门通过；不重置baseline/historycycle11/旧goal，不自动恢复旧automation/E3。

最终报告必须有：设计对照、所有人口分母、A上下界/安全见证、B实际生成及P2增量、反例、motion全部门、未触发阶段、真实耗时与资源、源SHA。A/B/V均不满足时应明确结束本系列，而不是再给一串微调。

## 11. 设计反方检查与结论边界

本节为academic-research-suite辅助的同助手设计检查，非外部独立评审、非实际预检PASS。

**检查1：问题与方法。** 原草案容易把“投影求不到”误当“无安全路径”；现明确A0是放松上界，A1仅充分安全正见证，负结果分开解释。A不回答理解或可预测性。

**检查2：最强替代解释。** B−A包含P2通道可用性与有效优化差异；native输入已携带观测信息；教师监督与旧WTA不同。故B阳性只支持此固定安全映射中P2增量有用，不证明旧失败唯一来自共享head或旧loss。

**检查3：是否又是小变体循环。** 当前平台若连理想上界/固定安全见证都不足，直接停止；有空间才投入一次六run学习。禁止恢复R1–R5/D0或调界/节点/专家数。三阶段不是必须凑齐的任务列表。

最重要的剩余限制：射线充分约束对低加速度/静止native可能非常保守；小幅球空间先验本身不能创造远离native的大转向/停止模式；F16已见数据、重复outer/CAL、窗口相关和三个seed仍限制推广。若本系列停止，下一份架构方案应改变问题表示并重新论证，而不是以停止结果证明“行人未来不可预测”。

**推荐实际第一步仅A0；A0通过才A1。B和V全部条件性，不提前启动。**
