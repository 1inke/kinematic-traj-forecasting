# 端点修正的可行更新诊断：不牺牲旧成功，还能学到多少救回？

日期：2026-10-31  
状态：**DESIGN_ONLY，未实现、未预检、未训练**。本文不是执行授权或预检通过证明。  
建议执行目录：`qoder_work/ENDPOINT_FEASIBLE_UPDATE_DIAGNOSIS_EXEC_20261031/`。

## 0. 本轮只回答一个问题

> 在相同轻量共享网络、相同RS目标和有限训练预算下，如果不再允许训练更新牺牲已有成功窗口，模型还能否学出非平凡的安全救回？

这不是再加一个loss权重，而是把“平均软惩罚”与“更新必须满足训练集可行性”分开检验。先完成一个有限诊断，不直接进入全量训练、学门控、换MoE或改scorer。

本轮结构：

1. **O0：构造无共享限制的GT可行见证**，只复用已有END49动作，不训练。
2. **U1：三seed共享网络的可行更新实验**；对照复用已完成的RS三seed，避免重复训练。
3. **U2：固定检查点的逐窗口输出缩步探针**，不训练、不部署，帮助解释全局更新为什么受阻。

最多新增3个4770参数tiny模型，每个256次更新提议，共768次；实际接受更新数另报。没有自动正式训练分支。任何阴性结果均不能证明共享网络数学上不可能或输入不包含信息。

## 1. 证据起点与尚未解决的问题

最新[最终报告](../qoder_work/ENDPOINT_SET_UTILITY_RETENTION_EXEC_20261030/RESULT_ANALYSIS_FINAL.md)及配套表已显示：

| 安全执行S，Native为69/128 | seed42 | seed43 | seed44 |
|---|---:|---:|---:|
| M：覆盖／救回／丢失 | 70／8／7 | 71／9／7 | 69／7／7 |
| R：覆盖／救回／丢失 | 70／8／7 | 71／8／6 | 71／8／6 |
| RS：覆盖／救回／丢失 | 70／9／8 | 72／9／6 | 73／10／6 |

- 连续保留代理R比M降低约65%–66%；RS的救回代理比R降低约53%–56%，但执行保留仍失败。
- RS连续C丢失均为9；量化Q及guard S丢失8／6／6。最终lost的阶段码全部为Native/C/Q/S=`1000`，损伤首先出现在连续预测。
- RS transition FDE恶化约10.55%／14.22%／17.25%；物理通过不能代替准确性保护。
- 当前只完成固定128训练窗、97条track的tiny，不是正式泛化实验；旧H−G仍未测，不能据此判定观测信息不足。

目前竞争解释至少包括：平均目标允许局部牺牲；共享更新影响多个窗口；当前优化路径难到达兼顾解；以及连续输出与离散安全执行的错位。本文只定位前述有限训练过程，不试图一轮证明唯一根因。

## 2. 冻结范围与数据边界

所有比较沿用原tiny128、97个sequence|target tracks、2560候选，不能重选更容易的窗口。Native原好69、原坏59；旧安全oracle覆盖90是已有诊断结果，不预设本轮构造也必然通过全部分组门。

不变项：

- H输入40维：原past16＋native path24，使用原train3103统计及同一特征变换；不重新用tiny拟统计。
- 共享网络`40→64→32→2`，GELU、tanh，4770参数；同seed初始化逐张量一致，末层原零初始化。
- 原K20、END49动作表、最近动作规则、identity、literal七项候选guard、物理单位与dt=.4不变。
- 连续路径为`native + .4*(t/12)*u`，head float32、路径与J float64；不增加径向裁剪、STE或guard反向。
- 原RS目标及系数1/1、原teacher、batch32、原seed42/43/44和sampler批序、AdamW lr=.001、weight_decay=.0001、clip5保持。
- 旧F16、decoder、scorer及旧head均不实例化训练；只读取锁定输入、标签、统计及必要权重收据。

本轮不读取hold/inner GT，不运行正式train3103训练或val挑选，不扩大数据或恢复旧PAUSED任务。O0/U2中的GT明确只用于训练人口的诊断，绝不作为部署输入。

## 3. 统一定义

对窗口i、槽k、完整12步路径定义：

\[
J_{ik}=\max(ADE_{ik}/0.2,\;FDE_{ik}/0.4),\quad good_i=1[\min_k J_{ik}\le1].
\]

不能用不同槽的minADE和minFDE拼成good。保留独立minADE/minFDE作为误差指标，但与samecandidate coverage分开。

- `B_i={k:J_native[i,k]≤1}`：原好支持集合，训练标签。
- C：连续路径；Q：原END49量化；S：原guard后路径。
- `lost = Native_good & ~S_good`，`rescue = ~Native_good & S_good`。
- 所有ties沿原实现：J分量ADE-first，min按原physical slot，动作并列按冻结动作表次序。
- 科学比较literal，无epsilon；算术重算容差另列，不能用于放宽good或motion。

## 4. O0：先给出不依赖共享映射的有限见证

### 4.1 构造规则

只读取旧tiny的逐槽49-action安全/J表及对应源绑定，不做新动作搜索。

1. 原好窗：20条全部保持Native，保证不主动消耗旧成功支持。
2. 原坏窗：在全部槽、全部原安全动作中，找同槽J最低的`(slot,action)`；仅当该J≤1，修改这一槽，其余19槽保持Native。
3. 并列按`J、physical slot、冻结action index`升序；无可救回动作则整窗identity。
4. 用实际动作方程与原guard重新构造输出，不能只读取标签中的“safe=true”就宣布通过。

这是使用未来GT选择动作的**查表见证**，不是预测器，也不是新teacher；禁止训练网络去模仿它，禁止将其混入U1损失。

### 4.2 必须审核的层次

- identity/49动作可构造性、同ID和原动作范围、K20及19槽不变；
- rescue/lost和原好完整保留；预期21个旧oracle可救回窗口应可由单槽见证覆盖，若不符先核对旧标签语义及来源，不能强写90；
- 全population/group五pooled motion、junction、原保护误差，以及全K聚簇/effective-good。

即便逐候选safe，也不能预设pooled分位数通过。若O0旧成功全保留、覆盖有救回，但分组误差或物理门失败，应标为`COVERAGE_WITNESS_ONLY`，不是完整可行解；不扫描另一个槽/动作凑全PASS。

O0失败不证明整个有限bank无可行组合，只说明这项固定构造未给出完整见证。来源/ID异常则STOP；科学阴性仍可按固定U1完成有限诊断，报告上界解释受限。

## 5. U1：同RS目标下的可行参数更新

### 5.1 对照与数量

| 名称 | 参数映射 | 更新方式 | 本轮工作 |
|---|---|---|---|
| Soft-RS | 原共享H4770 | 原AdamW，无可行性接受测试 | 复用旧RS42/43/44，核SHA与合同，不重训 |
| Feasible-RS（F-RS） | 相同共享H4770 | 同一AdamW提议＋预注册缩步/拒绝 | 3个fresh tiny |

不复用旧RS最终权重作为起点，因为它已违反保护约束。F-RS从对应seed原零输出初始化开始；旧控制的初始参数、输入统计及256批ID作为配对来源。

对照检验的是**训练更新接受机制**。它同时影响实际步长、有效更新次数和Adam状态，不得宣称纯粹隔离了loss权重、网络容量或梯度冲突。Soft-RS与F-RS按相同256次提议预算比较，不伪称相同接受更新数或计算成本。

### 5.2 可行集合：在训练全128窗上检验

每个候选参数状态必须同时满足：

**A. 原支持及执行保留**

- 每个原好窗：`min_{k∈B_i} J_C[i,k]≤1`；至少一个原成功支持连续路径仍合格。
- 同窗安全执行S：`min_all20 J_S[i,k]≤1`。

不强制保住每个原好槽，也不强制维持Native原最优J的全部余量。RS软损失仍保留其原margin定义；这里新增的是更新接受的合格下界。

**B. 已注册分组误差保护，仅在S判断**

- overall/stable/transition的minADE、minFDE均≤Native×1.01；
- change的minADE、minFDE均≤Native；
- 每组的五pooled motion及junction mean/p95均≤Native×1.01。

空组标NOT_ESTIMABLE；真实128的组组成须预先核实，不能删除小组使门通过。以上不是对未来未知样本的保证；它只是训练时参数状态的经验可行性。

**C. 基本有效性**

所有输出有限、固定K20、原guard完整执行、源/ID不变。没有“如果GT发现损伤则在推理时回退Native”的额外操作。训练接受的是一个全局参数向量，之后预测所有窗口必须只靠观测输入。

### 5.3 固定提议、回退和记账

固定256次提议，按原batch32×4批×64遍的sampler顺序；不因拒绝重抽batch或无限重试。

每次提议：

1. 保存当前参数θ、AdamW状态m、全部RNG及该批preupdate诊断；计算原RS loss、梯度、clip5；执行一次原AdamW得到`θ_hat,m_hat`。
2. `Δ=θ_hat−θ`，只试固定α序列`[1,1/2,1/4,1/8,1/16,1/32,1/64,1/128]`，按此顺序。
3. 每次从同一θ构造`θ_trial=θ+αΔ`，不能在上一个trial基础上累积缩步。保持参数原dtype，实际读取更新后参数，不能以double理想值代替float32生产值。
4. 用全128窗重新前向，执行原C/Q/S并计算第5.2节所有门。接受第一个全部满足且参数实际改变的trial，不按救回数量或验证效果选α。
5. 接受时提交`θ_trial`及`m_hat`；Adam step只增加一次。α只缩放参数位移，moment不乘α，此为明确注册的新优化器语义。
6. 若8个trial全拒绝或位移全为bitwise零，恢复θ与m，接受步数不增加；sampler仍消耗该batch，提议计数增加。不能把拒绝记为训练更新成功。

不加入Armijo、投影求解器、随机扰动、额外方向、α扫描或训练延长。若全部拒绝，输出真实停滞结果；不能为得到非零收益更改门。

**RNG约定：**提议前保存S0；一次训练前向/梯度/Adam提议后保存S1。trial采用eval/no_grad，必须无随机消耗，并以RNG前后EXACT核验。接受后保留S1；全拒绝时参数/Adam回S0，但RNG保留S1且sampler已前进。检查点显式保存这套事务状态，不笼统称“恢复全部训练时钟”。异常中断按已提交状态恢复，不能半提交。

每个trial记录全部失败位，而非首个失败就停止统计，以区分：C支持、S保留、protected ADE/FDE、motion/junction、非有限、量化平台和零位移。可以跳过已经失败trial的昂贵聚簇，但不得遗漏科学门；聚簇不用于提议接受，只在规定保存点完整计算。

### 5.4 固定保存点

保存提议0、64、128、192、256后的状态；主结果只用256，不挑中途最好模型。每点保存C/Q/S、u、量化动作、guard位、samecandidate指标及全组指标；最终完整聚簇。

每次拒绝保存提议Δ的摘要及trial证据；为控制存储，只额外保存64/128/192/256四次的完整`u_current、u_full_proposal`，供U2。每个seed最终checkpoint含256提议历史及实际接受数。不得把“64遍数据”误报为“256次参数更新”。

## 6. U2：逐窗口缩步的非部署诊断

目的：判断“一次全局共享参数位移”是否把不同窗口的可接受修改绑在一起。它不是训练一个按窗口ID查表的模型。

对U1每seed固定64/128/192/256提议，一共12个快照，全部执行以下固定规则，不按结果挑快照：

1. 对每窗在输出空间构造`u_i(α)=u_current_i+α*(u_full_proposal_i−u_current_i)`；α为同8项序列，加末项0。
2. 只在原好窗要求第5.2A两个保留条件；每窗执行原guard、有限性检验；原坏窗不按GT救回与否筛α。
3. 每窗取第一个通过上述局部条件的α，不能选其GT最优α。原好窗使用GT判断仍使该规则不可部署，明确标`GT_DIAGNOSTIC_ONLY`。
4. 拼成一个K20/window集合，再检查全部population/group误差、物理和聚簇。局部门不能保证全局分位数或分组误差，失败照报，不再调整各窗α。

同时保存同快照α=1的输出空间提议及α=0当前状态。比较GT窗口级缩步与实际全局参数缩步，不计算“GT挑最佳快照”的主结果。

重要限制：非线性网络满足`f(θ+αΔ)≠f(θ)+α(f(θ+Δ)−f(θ))`通常成立；U2也取消了不同窗口必须由同一参数向量产生的要求。因此U2只给出输出空间局部见证，不能把改善全归因共享权重、不能声称找到了网络可实现解。

## 7. 预注册问题、指标与结论分级

### 7.1 非平凡性是核心，不把强制零丢失当创新

F-RS零丢失、分组保护通过是接受规则的直接结果，不是独立实验发现。真正待检验的是：**满足这些条件后还剩多少救回，是否仍在学习？**

主表按每seed列出：

- final256 rescue/lost/coverage，以及Soft-RS、Native和O0参照；
- 接受数、拒绝数、接受α分布、bitwise零位移数、量化动作未变数；
- rescue／原safe-oracle可救回21窗（若O0来源核对改变，应STOP查来源，不事后替换分母）；
- 与Soft-RS配对救回差、旧成功损伤消除数、每个原坏窗是否被不同方法救回；
- 参数MSE、LR/LS、全部C/Q/S转换、非identity比例；非identity本身不是收益；
- 所有group minADE/FDE、同槽minJ、完整coverage、五motion/junction；
- 完整K20 complete-link `.10 meanpath/.20 endpoint`、distinct/effective-good、J最优单轨迹早/中/晚误差；固定`.3/.5`只作敏感性。

### 7.2 结果级别，不自动推进正式训练

| 状态 | 事先固定判据 | 可得结论 |
|---|---|---|
| `NONTRIVIAL_FEASIBLE_SHARED_WITNESS` | 三seed全部final安全rescue≥1、全部约束通过 | 当前共享H至少存在训练内非平凡可行实例 |
| `PROMISING_TINY_FEASIBLE_UPDATE` | 在上项基础上，三seed rescue分别≥旧RS的9/9/10 | 在此tiny上可保留既有救回规模且不牺牲旧成功 |
| `LIMITED_FEASIBLE_TRADEOFF` | 至少一seed rescue>0，但不满足上两行 | 有限可行更新成立，收益或稳定性不足 |
| `IDENTITY_OR_STALLED` | 全部seed final rescue=0 | 这套有限更新路径未给出有效实例，非不可能性证明 |
| `AUDIT_OR_SOURCE_BLOCKED` | 来源、数学、恢复、保存或资源实质失败 | 不解释科学正负，保留失败来源 |

任一级别均不改旧90%拟合STOP、H−G未测、TIME/P3状态，不升baseline，不宣称理解运动。新判据是诊断分类，不是降低历史部署/major门。

### 7.3 解释分支

- F-RS成功：削弱“当前共享结构根本不能兼顾”的绝对说法，支持更新约束是可行干预；仍需另立跨track检验。
- O0完整通过，F-RS停滞，U2出现非平凡完整见证：支持研究共享更新耦合或选择性修正，但优化方向、参数化、有限步长均是竞争解释，不直接上MoE。
- F-RS停滞且U2也无收益：只能说明当前RS提议方向及固定搜索没有找到可行进展，不证明所有模型/输入无解。
- C保留容易通过但S/量化频繁失败：给出执行一致性障碍，仍不移除guard。
- 原支持门通过但protected-group频繁失败：说明“是否good”之外，组误差保护仍是独立限制，不能只优化coverage讲成功。
- O0仅coverage见证通过：降低完整安全可行性的结论强度，不把逐候选oracle误作全集合上界证书。

### 7.4 统计范围

本轮不做正式显著性宣称或新held主检验。128训练窗/97tracks有相关性且反复用于探索；3seed不是3份独立数据。逐seed、逐track计数为主，报告所有窗口，不用重复seed扩大样本量。

如需描述区间，只允许固定整track配对bootstrap2000draw、RNG20261031、三seed均值固定不重采seed，报告普通95%探索区间；分母无效draw记NA且不补抽，有效<1900则NOT_ESTIMABLE。不据该区间选择模型、α或后续阈值，旧正式family6保持NOT_RUN。

## 8. 核心代码接口与事务要求

以下是设计伪代码，不是已验证生产脚本。

```python
ALPHAS = (1., .5, .25, .125, .0625, .03125, .015625, .0078125)

def predict_observed(model, normalized_features):
    # 输入不含GT、原成功mask、oracle、ID；与旧H相同。
    return model(normalized_features)

def training_feasibility(cqs, training_labels, native_reference):
    # 只用于训练参数更新接受；不得被inference.py调用。
    # 返回全部窗口/组失败位及原始标量，不只返回bool。
    return independent_contract_checks(cqs, training_labels, native_reference)

def propose_and_commit(engine, batch, all_train):
    old = snapshot_parameters_optimizer_rng(engine)
    adam_propose_once(engine, batch, objective="original_RS", grad_clip=5.)
    proposal = snapshot_parameters_optimizer_rng(engine)
    delta = subtract_parameters(proposal.params, old.params)
    trials = []
    for alpha in ALPHAS:
        assign_parameters(engine, affine_from_old(old.params, delta, alpha))
        if parameters_bitwise_equal(engine, old.params):
            trials.append({"alpha": alpha, "reason": "ZERO_DISPLACEMENT"})
            continue
        cqs = frozen_original_decode_full_training(engine, all_train)
        verdict = training_feasibility(cqs, all_train.labels, all_train.native_ref)
        trials.append(verdict)
        if verdict.all_pass:
            commit_proposed_optimizer(engine, proposal.optimizer)
            restore_rng(engine, proposal.rng)  # trial不得消耗随机数
            return {"accepted": True, "alpha": alpha, "trials": trials}
    restore_parameters_optimizer(engine, old)
    restore_rng(engine, proposal.rng)
    return {"accepted": False, "alpha": 0., "trials": trials}
```

sampler推进与proposal_count在外层恰执行一次，accepted_count只在接受时加1。Adam的step计数必须与接受数对齐；上面snapshot不得把已推进的sampler回滚。实现需明确区分model/optimizer状态、sampler状态和随机数状态，不能直接用旧整包restore覆盖新时钟。

建议模块：`contracts.py`、`feasible_update.py`、`oracle_witness.py`、`output_probe.py`、`evaluate_saved.py`、`audit_saved.py`。旧loss只读引用并绑定SHA；若shim会触发旧顶层写入，先静态隔离纯函数，不能执行旧启动脚本。

## 9. 实施前必须完成的核验

### 9.1 来源与数值

- 精确继承上轮5371相关生产来源及最终V2封存165输出的原digest，合并重叠只去重、不覆盖冲突；实际数量以清单为准，不称全部历史文件。
- 128IDs顺序、97tracks、Native69、teacher/END49、统计、特征与batchIDs逐项核对。
- 当前H初始化输出必须Native identity；RS标量、所有4770梯度和第一Adam完整提议与旧实现EXACT。若不符先STOP定位，不能换源后继续称配对对照。
- 新接受检查要用第二套NumPy实现，从路径重算quality、所有group pooled运动学/junction和失效位。预检完整128表的第二实现审核必须在生产前完成；禁止重演上一轮后补时间披露。
- loss精度沿上轮：独立float32 MSE后端界2e-7、double R/S与保存几何2e-12；原科学比较零epsilon。没有修改loss，不为回退分支做“连续梯度正确”的虚假证明。
- exact ties、空组、Native零运动、量化边界、α缩步后参数不变、全部拒绝、第一项接受、第八项接受均有固定fixture。
- 训练GT改动可以改变接受决定，但必须不改变同一θ的观测前向与原解码；在不调用接受器的独立推理进程验证这一点。

### 9.2 精确恢复

真实固定128人口、2遍8次提议，提议5后中断；连续与恢复的参数、Adam、sampler、所有CPU/CUDA/Python/NumPy RNG、提议/接受/拒绝计数、α历史、全部trial指标和最终路径EXACT。另用固定合成fixture保证覆盖拒绝及α<1接受，不能因真实8次全接受就声称拒绝恢复已测。

模型当前无buffer，不虚称统计buffer；外部statistics需要SHA绑定。trial中断只允许回退到上一个完整提交点，原子写checkpoint。坏source、坏数据、坏协议、坏checkpoint、坏输出必须实际拒绝，完成skip只认完整源兼容SHA。

### 9.3 资源与预算

沿用当前`resource_guard_v3`的实测inventory/E、预警及启动许可逻辑；不恢复过时机械6GiB硬门，不改OS/pagefile，不kill用户程序或删除旧证据。

必须分别benchmark：普通RS提议、完整128 C/Q/S与全部可行性门、最坏8trial、全部聚簇、O0/U2、来源SHA/复核IO。throwaway不保存为科学模型、不按收益选择设置。

保守规划式：`3×256×(T_proposal + 8*T_feasibility) + O0 + U2 + 完整审核/保存/SHA`；另报由实测接受率得出的描述估计，但启动准入用保守预算。预计阶段>30min先报告构成并停止启动，不减审核/提议次数绕短预算，也不沿用上轮72.8秒当本轮ETA。本设计无声称已测耗时。

若未来明确实施且短预算通过：短任务前台完成即可；确需后台时sole按需0trigger、一次startup、一次50%预计elapsed与一次完成核验，期间安静，不查询旧完成任务、不恢复旧PAUSED。

## 10. 保存与完成审核

```text
ENDPOINT_FEASIBLE_UPDATE_DIAGNOSIS_EXEC_20261031/
  README.md / DESIGN_SNAPSHOT.md / protocol.json / runtime_state.json
  INPUT_SOURCE_LOCK.json / source_manifest.json / INPUT_AUDIT.json
  preflight/gradient/ / transaction/ / resume/ / saved_pipeline/
  PREFLIGHT.json / PRELAUNCH_REVIEW.json / BENCHMARK.json
  O0_witness/                    # GT查表，非模型
  F_RS_42/ F_RS_43/ F_RS_44/     # 三个fresh模型
  U2_output_probe/               # 12固定快照，无新训练
  PROPOSALS.csv / TRIAL_REJECTIONS.csv / ALL_GROUPS.csv
  WINDOW_EVENTS.csv / SUPPORT_STRATA.csv / METHOD_COMPARISON.csv
  completion_review/REVIEW.json
  RESULT_ANALYSIS.md / RESULT_ANALYSIS.json / DECISION.json
  EXECUTION_ACCOUNTING.json / RESOURCE.json / FINAL_PRESERVATION.json
```

保存0/64/128/192/256检查点、原始u、C/Q/S、动作和失败位、单槽指标、初始及最终checkpoint/optimizer/RNG/数据统计和loss源绑定。禁止只保存最优epoch或成功trial。

完成后第二实现CPU完整重算O0、三个最终模型和12个U2快照的全部质量、聚簇、五motion、junction、分组保护与覆盖；中间trial逐一重算接受所用科学标量，可按保存u重构路径，不要求所有trial重复聚簇。审核先后时序、科学/夹具/失败前提议与更新数量分别列出。

只在全部源/ID/恢复/保存审核通过后解释科学结果。第二实现算术不是所有网络梯度第二次回放，也不是外部独立评审。保留任何失败源码、日志与partial输出；不得修改旧封存清洗记录。

## 11. 后续方向与明确停止边界

本轮结束必须先交付上述三层证据的综合报告；**无自动全量训练、无自动门控/MoE/第四loss或新的α集合**。

- 若有稳定共享可行见证，下一份协议才考虑跨完整track训练/验证和不使用GT的条件修改；训练集可行性不会自动迁移。
- 若只有O0/U2见证，不应宣称“模型只差训练技巧”或“加gate一定有效”；先明确共享优化与输出级选择的差距，再决定是否值得一个结构级测试。
- 若没有新增有效证据，停止这条固定端点小修正的局部循环，综合已完成可达性/条件信息/时间形状证据重新选择大分支，不无限续tiny。

历史major change ADE/FDE各≥5%、protected 1%、相关性审核及全部运动学门原样保留；本轮无论哪一级别，都不是major PASS、baseline升级或论文创新完成。

## 12. 反方自检与研究诚信

**DA1／问题边界：**硬接受会强制保留，因此零丢失不是独立成果；已把救回、非平凡性和代价设为核心。该有限方法失败不能证伪所有共享可行解。

**DA2／替代解释：**回退改变有效步长和Adam状态；U2还改变参数耦合及非线性插值。因此不把差值直接当“共享干扰的纯因果效应”，保存完整提议/拒绝证据。

**DA3／信息泄漏：**O0/U2使用GT，仅为训练内上界/局部诊断。U1的训练接受使用训练GT是监督机制，部署前向不得调用它；禁止按未来GT逐窗identity fallback。tiny反复探索和F16历史暴露不能被三seed消除。

本文由AI助手依据本地一手实验资料形成；反方检查为同助手自检，非外审。不新增数据、参与者或个人身份信息，沿用项目既有数据使用与去标识约束，不虚构伦理批准。

## 13. 来源索引

- [上一轮设计](ENDPOINT_SET_UTILITY_RETENTION_DESIGN_20261030.md)
- [最终报告与时序披露](../qoder_work/ENDPOINT_SET_UTILITY_RETENTION_EXEC_20261030/RESULT_ANALYSIS_FINAL.md)
- [最终连续代理](../qoder_work/ENDPOINT_SET_UTILITY_RETENTION_EXEC_20261030/FINAL_CONTINUOUS_PROXIES.csv)
- [全组C/Q/S](../qoder_work/ENDPOINT_SET_UTILITY_RETENTION_EXEC_20261030/ALL_GROUPS.csv)
- [逐窗阶段事件](../qoder_work/ENDPOINT_SET_UTILITY_RETENTION_EXEC_20261030/WINDOW_EVENTS.csv)
- [实际RS目标](../qoder_work/ENDPOINT_SET_UTILITY_RETENTION_EXEC_20261030/objectives.py)
- [实际训练器](../qoder_work/ENDPOINT_SET_UTILITY_RETENTION_EXEC_20261030/trainer.py)
- [H网络与零末层初始化](../qoder_work/ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029/code/model.py)
- [前轮支持失败归因](../qoder_work/ENDPOINT_SUPPORT_FAILURE_ATTRIBUTION_EXEC_20261030/RESULT_ANALYSIS.md)
- [当前最终SHA封存](../qoder_work/ENDPOINT_SET_UTILITY_RETENTION_EXEC_20261030/FINAL_SEAL_CHECK_V2.json)

所有数值是既有结果或明确的设计预算，本文没有新增实验结果。静态文档检查不能替代未来真实GPU、事务恢复及独立算术预检。
