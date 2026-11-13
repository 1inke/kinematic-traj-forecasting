# 下一轮实验：安全端点修正的可学性与成功保留

日期：2026-10-29  
状态：**DESIGN_ONLY — 本文未启动实验、未修改冻结源码、未恢复自动任务。**  
定位：原“端点／时间可观测性系列”的END学习阶段续接，不是另开动作、网络和损失联合搜索。

## 0. 一页执行摘要

**核心问题：在固定Native K20和已审计安全动作集合下，原始历史能否帮助预测有用的端点校正，并把这种能力迁移为完整好轨迹覆盖的净增益？**

本轮只做以下事项：

1. 保留已完成P0、GPU梯度与恢复证据，核源后按依赖范围复用。
2. 将不可实现的tiny双分层抽样明确修订为“64活跃训练窗＋64个未被选中的训练窗”，保持128窗、256更新和原90%拟合门。
3. 预检通过后，只训练END的G/H×seed42/43/44，共六个小模型；各固定8epoch。
4. 六个final与推理合同全部封存后，一次性评价train、val、hold、inner，完成独立保存审核和分组统计。
5. 根据参数预测、动作效用、覆盖、成功保留和运动代价联合下结论；结束本轮，不自动追加变体。

**明确不做：TIME训练、P3联合、新动作网格、增大修正幅度、换loss、加宽网络、增加epoch、训练scorer、解冻F16、新数据或恢复任何旧PAUSED任务。**

这里的“单因素”是G/H之间是否提供显式原始历史；tiny抽样修订是预检可实现性修复，不是新的性能因素。

## 1. 已有证据与研究边界

依据[本次RESULT_ANALYSIS](../qoder_work/ENDPOINT_TEMPORAL_OBSERVABILITY_EXEC_20261029/RESULT_ANALYSIS.md)、[P0决策](../qoder_work/ENDPOINT_TEMPORAL_OBSERVABILITY_EXEC_20261029/p0_completion_review/DECISION.json)及[独立保存审核](../qoder_work/ENDPOINT_TEMPORAL_OBSERVABILITY_EXEC_20261029/p0_completion_review/REVIEW.json)：

| 动作类／人口 | 变化窗口数 | Native好窗口 | safe oracle好窗口 | 新救回 | 涉及轨迹数 |
|---|---:|---:|---:|---:|---:|
| END train | 721 | 283 | 417 | 134（18.5853pp） | 78 |
| END val | 256 | 113 | 157 | 44（17.1875pp） | 23 |
| TIME train | 721 | 283 | 288 | 5（0.6935pp） | 5 |
| TIME val | 256 | 113 | 113 | 0 | 0 |

END已通过原train至少15个变化窗口、至少5条轨迹的机会门。TIME未通过，保持 `STOP_LOW_SAFE_OPPORTUNITY_IN_THIS_ACTION_CLASS`；P3保持 `NOT_TRIGGERED_TIME_P0_STOP`。不能为了把系列跑全而训练TIME。

上述是GT知情的有限动作oracle，不是可部署模型。identity合法且按同槽最小J选择，因此oracle零lost是构造性质，不是学会了保留成功。val也没有给出任何学习算法的泛化结果，只说明同一动作类的安全改善机会并非仅存在于训练人口。

END train非identity教师为50087/62060候选（约80.71%），3103/3103窗口至少含一个活跃候选；候选级仍有11973个identity标签。不能将“所有窗口活跃”写成“所有候选都要改”，也不能沿用“identity标签占绝大多数”的旧解释。[分层实测](../qoder_work/ENDPOINT_TEMPORAL_OBSERVABILITY_EXEC_20261029/TINY_STRATUM_ADMISSION.json)

END当前精确并列及J差≤1e-6的多动作近并列计数为0，但J差≤0.01的较宽等效集合仍有非零直径。多解风险应保留为解释项，不能预先认定它是唯一原因。[歧义表](../qoder_work/ENDPOINT_TEMPORAL_OBSERVABILITY_EXEC_20261029/completion_review/P0_AMBIGUITY.csv)

### 1.1 这次到底在学什么

END为 `p'_t = p_t + (t/12)δ`，dt=0.4s。未来速度整体增加 `δ/4.8s`，最大范数约0.0833m/s；在精确数学下，未来内部加速度与jerk不变，但历史末速度至未来首步的衔接会变化。

因此，本轮检验的是**轻量的端点／位移／平均速度偏差校正**。它不是新的停走时序生成器，也不必保持原转向角完全不变。即便结果阳性，也不能直接写成“模型理解了行人意图”或“首次生成了复杂运动过程”。

## 2. 变更登记：只修tiny抽样，不移动科学门

建议修订标识：`END_TINY_SAMPLING_AMENDMENT_V2_20261029`。

| 项目 | 原合同／现状 | 本轮设计 |
|---|---|---|
| 第一组 | 64个活跃train窗口 | 不变 |
| 第二组 | 核心实现解释为64个非活跃窗口，实际为0 | 从第一组之外的train窗口按同一固定哈希取64个；允许活跃 |
| 总样本 | 128个不同train窗口 | 不变 |
| tiny模型 | H、seed42、原4770参数结构 | 不变 |
| tiny预算 | batch32、256更新 | 不变 |
| tiny主门 | 非identity候选MSE较初始化下降≥90% | 不变 |
| 正式训练／动作／安全／评价 | 原END合同 | 全部不变 |

当前所有train窗口均活跃，V2等价于按原固定哈希取前128个不同train窗口。两个64组只是保留第一组的抽样兼容性，**不再称正负平衡分层**。

原 `STOP_TINY_STRATUM_INSUFFICIENT` 收据和源码保留；原tiny更新数仍为0。本修订不能回写成原预检已经通过。

## 3. 固定数据与使用顺序

| 人口 | 总窗口 | change窗口 | 用途 |
|---|---:|---:|---|
| train | 3103 | 721 | 正式拟合、输入统计、标签均值、tiny与预检 |
| val | 993 | 256 | 固定报告；不选checkpoint，不调阈值 |
| hold | 3825 | 1023 | 主评价人口，change为主比较 |
| inner | 1856 | 497 | 次级完整评价与保护门 |

继承原opt4096的train/val身份划分，不重抽split。原Native K20、past8×2、future12×2、米制局部坐标与dt=0.4保持。group沿用stable/transition/change，时间歧义记录不删除。

必须核准确sample IDs、行顺序、sequence|target和原轨迹不交叉约束。数量一致不是身份认证。只用train拟合40维统计，float64 mean/std，std<1e-6置1；训练输入float32。val/hold/inner不能参与统计、标签均值、样本权重或方案选择。

**GT开放顺序：**

1. 继续使用已经封存的train/val END标签与掩码。
2. tiny与六个正式训练仅使用train监督；val只按合同作预检／记录。
3. 全部六个final、stats、解码函数、评价和审核源码封存后，才开放本轮hold/inner GT。
4. hold/inner先封存GT-free masks，再关联GT计算同动作类的教师及oracle参照；不再用held机会数改变已训练方案。
5. 一次完整报告所有人口和seed，不挑最好的seed替代原主结果。

旧F16已见过原数据，hold已有历史分析暴露；本次是防止进一步按held调方案的运行隔离，不称全新盲测。target-disjoint不等于邻居／时间上下文独立。

## 4. 阶段A：来源复用与当前版本准入

### 4.1 可以保留的证据

- 完整P0 mask、标签、oracle和独立保存算术。
- 已通过的实际END G/H GPU8梯度、配对初始化、zero identity检查。
- 已通过的两arm真实513 train＋12 val、两epoch34更新／19步中断恢复。
- 坏source/stat/split/output拒绝收据。

依据：[GPU8](../qoder_work/ENDPOINT_TEMPORAL_OBSERVABILITY_EXEC_20261029/END_learning_preflight/GPU8.json)、[RESUME](../qoder_work/ENDPOINT_TEMPORAL_OBSERVABILITY_EXEC_20261029/END_learning_preflight/RESUME.json)。它们是已有检查，不是本轮重新运行的结果。

### 4.2 不能直接继承一个总PASS

新执行放入独立目录，例如 `qoder_work/ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029/`；本文不创建该执行目录。继承4451来源清单及learning_source_extension中的原digest，扩入P0结果、旧预检收据、修订文档和新适配器；不覆盖任何旧hash。

先做明确diff／AST和依赖审查：

- 若只有tiny索引选择及新报告／控制适配改变，几何、guard、teacher、特征、模型、损失、训练恢复核心与真实输入/stat SHA均一致，可将旧GPU／恢复证据标为 `INHERITED_EXACT_DEPENDENCY_MATCH`，不重复完整P0。
- 同时执行新抽样单元检查、新源码绑定检查及固定小批次桥接；旧源码digest与新源码digest均保存。
- 若训练／统计／数据读取／数值算子／恢复核心发生变化，则重跑受影响的实际检查；不能只因diff很短就继承。
- 若旧loader因全目录code_digest而拒绝新版本，应增加显式、逐项核验的迁移适配器，不关闭校验、不改写旧done或伪造新的旧hash。

当前版 `PREFLIGHT.json` 必须逐项区分“继承且依赖一致”“本轮新跑”“未完成”，不能把所有项目写成本轮重验。source、ID或算术不一致先停，不通过更换容差／换样本解决。

## 5. 阶段B：tiny拟合预检V2

### 5.1 确定性抽样

教师非identity按原动作ID定义；当前END也可与归一化target是否非零逐项交叉核验。

沿用原哈希字符串：`endpoint-temporal-tiny-v1|<sample_id>`，UTF-8、SHA256，按hex digest升序、sample_id作确定性次级排序。保留旧tag是为了保留原第一组，不表示抽样政策仍是V1。

```python
# 设计伪代码；实施在新版本，不编辑旧冻结文件。
active = (teacher_action_id != END_IDENTITY_ID).any(axis=1)
ranked = sorted(range(N_train), key=lambda i: (
    sha256(("endpoint-temporal-tiny-v1|" + str(sample_id[i])).encode("utf-8")).hexdigest(),
    int(sample_id[i]),
))
first = [i for i in ranked if active[i]][:64]
used = set(first)
second = [i for i in ranked if i not in used][:64]
selected = first + second
assert len(first) == 64 and len(second) == 64
assert len(set(selected)) == 128
assert set(sample_id[selected]) <= set(registered_train_ids)
```

当前END第一层足够，第二层足够；若执行时不符，视为身份／标签／依赖版本问题，不借val或TIME样本填充。保存全部128 IDs、哈希、track、group、两组归属、候选级active/identity数量和选择源码SHA。tiny不同窗口可来自同一track，必须披露；它是拟合检查，不是泛化评估。

### 5.2 固定拟合与报告

- 只训练一个H seed42 tiny模型，fresh初始化；正式模型不能从tiny热启动。
- 128窗口，每窗全部20候选、全部原标签；不改窗口权重，不删除identity候选。
- batch32、AdamW lr0.001/wd0.0001、clip5、固定256更新，相当于128窗64轮。
- 冻结两个读数：初始化0步、最终256步。中间loss可完整保存，但不选最佳步数。
- 主门沿用旧实现：在这128窗的非identity候选上，二维归一参数平方误差均值相对初始化下降≥90%。初始值为0时必须报告原始值并检查合同，不用epsilon制造下降率。
- 同时报告全体、active、identity候选MSE；identity初始MSE可能为0，只报绝对变化，不作无定义相对下降。
- 补齐raw／guarded教师动作命中、J-regret、identity regret、非identity执行比例、逐项guard拒绝和tiny覆盖/lost。命中与regret是必要报告，不在看完结果后新造更容易或更难的通过阈值。

若90%门未过，登记 `STOP_TINY_FIT_V2`：tiny已真实执行但当前拟合预检不足；不自动调LR、宽度、epoch、损失或另抽128窗。可以对已保存输出作一次有限原因核对，报告参数拟合／量化／guard／监督效用的区别；本轮不再训练替代模型。

若门通过，也只说明这个小训练子集可拟合，不是历史信息信号、不是迁移成功。继续完成当前源下保存重读、拒坏输入、实测资源和吞吐等剩余预检，全部完整后才正式训练。

## 6. 阶段C：唯一正式实验——END G/H配对学习

### 6.1 对照与训练合同

| 项目 | G | H |
|---|---|---|
| 原生候选输入 | 当前slot的12×2坐标 | 相同 |
| 原始历史 | 标准化后前16维置0 | 8×2原始历史，标准化后保留 |
| 网络 | 40→64→32→2，GELU，输出tanh | 相同 |
| 参数 | 4770名义参数 | 4770名义参数 |
| seed | 42、43、44 | 与G一一配对 |
| 正式训练 | train3103，8epoch，batch32，97batch/epoch，776更新 | 相同 |
| 优化器 | AdamW .001，wd .0001，clip5，无scheduler | 相同 |
| primary checkpoint | 固定final8 | 相同 |

共六个正式模型、48个model-epoch、4656次更新。不为TIME预留模型，不追加第二个网络。tiny和预检模型单独计数，不算正式重复。

同seed的G/H全部初始参数、统计、batch ID顺序保持一致；G历史通道无输入，因此有效连接／容量不同，不能宣称严格等有效容量。所有20候选共享同一个MLP，不输入slot ID、sample ID、scene、group、logit、GT或oracle原因。原Native已经含冻结历史信息，G不是“无历史模型”。

输入为 `[flatten(past8), flatten(native_slot12)]`，共同train-only统计；最后一层0初始化，初始输出为identity。G标准化后遮历史，不喂伪物理零轨迹。训练全体窗口，不专挑失败／change或按oracle收益加权。

教师为已经审核的逐候选安全最小J动作参数 `u*=δ*/0.4`。损失固定：

`L = mean_window(mean_slot(mean_coordinate((u_pred-u*)²)))`。

不加protection、motion、diversity、新STOP BCE或49-way分类项；否则无法回答原G/H问题。正式训练损失对全部20槽求平均，所以“候选平均变准”与“集合多救回一个窗口”不是同一目标，报告必须同时检查。

### 6.2 动作与安全政策

- END49原整数圆盘：i,j∈[-4,4]且i²+j²≤16，δ=0.1(i,j)m；冻结原实现的数值网格，不改为连续偏移。
- 选归一参数欧氏距离最近的动作；精确并列优先identity、较小参数范数、原动作索引。不在近似并列时加epsilon改选择。
- `p'_t=p_t+(t/12)δ`，t=1..12；identity精确复制Native。
- 对选中动作的speed p95、accel mean/p95、jerk mean/p95和junction逐项literal比较原候选×1.01；原阈值0就按0比较。
- 任一失败整槽回退Native，不另找第二动作；guard不接收GT，不以原生是否“真实good”决定修改。
- 保持K20及slot顺序；不把原20＋新20的K40结果当主性能。

保存每窗每槽的连续参数、raw/执行动作ID、raw/guarded路径、全部失败位。对照基线为Native identity及**train全窗口／全候选教师参数均值**，两者均不训练，均经过同一有限动作和guard政策。均值基线按原代码的float32教师target→float64累加口径锁定，不新增更有利的求均值办法；不按scene/group/held拟不同均值。

## 7. 阶段D：最终评价与防止“平均指标掩盖效用”

### 7.1 保存规模与主要端口

六模型×四人口＝24个model-population结果集，每个都保存raw和guarded。Native与train-mean基线各四人口，分别唯一保存，不复制三份假装模型seed。safe oracle仅作有限动作机会参照，不计入部署模型排行。

### 7.2 主指标：历史增量是否有用

主人口为hold change1023：

1. `ΔMSE = MSE_H − MSE_G`：每窗先平均20槽和两个归一参数。
2. `Δregret = regret_H − regret_G`：regret为执行动作同槽J减安全最优动作同槽J，再窗内平均20槽。
3. `Δcoverage = Coverage_H − Coverage_Native`：至少一个**同一完整候选**同时ADE≤0.2m且FDE≤0.4m。

前两项每seed<0、三seed均值<0且校正CI上界<0，才通过显式历史增量信号门；另要求H每seed regret低于identity。H−identity regret完整给描述CI，但不添加一个事后显著性主门。

G和H都完整评价对Native／train-mean的效果。若G有效而H无额外收益，可报告“候选几何可利用、显式历史未见增量”，但不能把G事后替成原H主假设已通过。

### 7.3 部署推进门：全部同时满足

1. hold change的H guarded覆盖增益每seed>0，三seed平均≥2pp，原家族校正CI下界>0。
2. hold与inner每group每seed的 `lost/native_good≤1%`。原好为0则标NA并保留计数，不补平滑分母，也不声称已证明该组保护率。
3. hold与inner整体／stable／transition的minADE及minFDE均≤Native×1.01；change两误差均不得恶化。
4. hold与inner每group每seed五pooled motion各≤Native×1.01；新增junction mean和p95也各≤Native×1.01。
5. inner change每seed净覆盖增益≥0。

逐候选guard通过不等人口pooled p95必然通过，所以第4项必须实际计算。执行identity导致全安全不等有效改善。2pp是本系列推进门，不能替代历史major要求；宣称大突破仍需原change ADE/FDE各≥5%、保护、相关性与原全部门另审。

### 7.4 必须保留的解释性分层

各人口／seed／raw-guarded均报告：

- overall/stable/transition/change：原好、救回、丢失、保留、净覆盖、各分母。
- 原成功／原失败窗口：参数误差与regret是否只在原成功窗口下降。
- 原失败中的safe-oracle可救回／不可救回：实际捕获了多少机会；保留逐窗IDs便于复核。分类由既定GT指标生成，只作事后描述，不进入forward或训练采样。
- 教师active／identity候选：候选级分母单列，不能冒窗口数。候选子集指标先窗内求均值，空子集窗标NA，跨窗汇总同时给有效窗口数；tiny原候选均值门另列，不能混分母。
- scene及原已保存精细模式／终点可达类别：只有显式ID对齐且定义未改才复用；缺侧栏列UNAVAILABLE，不重新用held挑定义。
- 每slot修正参数幅度、非identity比例、raw→guarded拒绝比例及七个失败位；不会只报第一个原因。
- 原SciPy全K20 complete-link：`D=max(meanpath/.10,endpoint/.20)`、cut1，distinct、有效good和重复；有效good不是越多越好。
- 单J最优完整轨迹的早4／中4／末4误差及slot；不拼接分段oracle。

必须并列minADE、minFDE、minJ和同槽coverage，说明前两个min可能来自不同候选。`.3/.5`只做原固定敏感性，不换成主阈值。raw安全外动作regret允许为负，另列说明；不能与safe regret混表。

## 8. 统计合同：TIME停止也不缩小家族

- 整sequence|target轨迹为重抽样单位，先在整个固定人口track框抽样，再施加group mask；报告frame/group track数。
- 20,000次draw，RNG seed20261028；G/H/Native共享track索引，再配对重抽三个模型seed；另给只抽track、固定三seed均值的敏感性。
- group空draw标invalid不补0，有效不足19,000则标不足可估，不自动补抽到通过。
- 家族仍为原END/TIME×MSE/regret/coverage六项。TIME未执行标NOT_TESTED，**不把分母从6改成3**。
- 双侧Bonferroni分位点为0.05/(2×6)与1−0.05/(2×6)，另给未校正95%描述CI。主门使用校正区间。
- 只对预设hold change主比较作主结论；其他人口、group、分层为完整保护检查和描述，不能从中挑显著项替换主比较。
- scene只有三个且不均衡，不能把窗口或候选当场景独立重复。bootstrap及三seed不消除旧数据复用／窗口相关限制。

## 9. 结果分支：一轮之后如何决定

| 实际结果 | 允许的结论 | 本轮结束后的建议，不自动执行 |
|---|---|---|
| tiny V2未过 | 当前固定网络／监督／预算的tiny拟合未过 | 先解释优化、尺度与参数效用；不宣称输入无信息 |
| tiny过，全train仍无改善 | 小子集可拟合不等全人口映射可学 | 检查全人口监督和表达，不自动加宽加epoch |
| train改善，val/hold无稳定改善 | 当前映射迁移不足 | 下一轮再考虑更稳定条件表示或泛化设计 |
| H参数更准但regret不改善 | 参数目标的改进没有转成动作效用 | 下一轮研究参数监督／量化决策的错位，不先加scorer |
| regret改善但覆盖不增 | 槽级改进没有转成集合跨阈值收益 | 核对是否只改善原好／冗余候选，再决定集合目标 |
| 救回增加但lost超门 | 新信息可能有用，成功保留仍不足 | 下一轮独立研究保留机制，不放宽lost门 |
| G有收益，H无额外收益 | 当前候选包含可用校正信息 | 可将更轻G作为后续路线，但非H机制成功 |
| H信号与部署门全部通过 | 支持当前END类的可迁移历史条件校正 | 再做独立确认和完整成本评估；不自动复活TIME/P3 |

无论哪种结果，都不自动新增第四loss、增大δ、换动作索引、调安全阈值或用不同hold子集寻找阳性。

## 10. 审核、轻量成本与执行控制

### 10.1 当前版本预检的剩余项目

tiny完成后，须补齐实际保存读回及第二实现的参数／动作／路径／guard核对、所有基线、全指标与CI小fixture、source/ID/stats/output损坏拒绝、资源inventory及吞吐。现有标量geometry/guard审核不能冒充全质量／聚簇／CI审核。

独立GPU／恢复旧证据按第4节处理；无需无理由重跑已完整P0。但新hold/inner的mask与标签仍需真实计算和独立审核，不能从train/val通过外推。

### 10.2 全量完成标准

六个25或旧80epoch模型不属于本实验；本次必须是六个**8epoch、776更新**final。逐checkpoint绑定：source/split/stats/teacher/action/guard/初始化/optimizer/全部RNG/批序/history。

最终独立CPU审核包括24个模型人口结果的全部保存连续参数与动作、raw/guarded几何、逐项失败位、同槽ADE/FDE/J、覆盖事件、全SciPy聚簇、五pooled motion/junction、bootstrap索引及draw重构；基线和新held教师同样审核。检查所有旧依赖和新不可变输出初末SHA。

执行收据、数学审核、预测性信号、部署推进、历史major分别给状态；不得用 `SUCCESS` 一词包办。保存算术审核不是所有模型和梯度再次回放，也不是外部评审。

### 10.3 资源与耗时

沿用resource_guard_v3，E取本阶段实测峰与最大载荷inventory；1.3为预警，不恢复旧6GiB硬上限。单模型／单枚举任务、有界批次，不修改OS/pagefile、不kill用户软件、不删除旧结果腾空间。

预热4批、计时32批，分别测训练、参数推理、动作解码、安全检查、全K聚簇、held标签枚举、独立审核、bootstrap和文件IO。训练6×776更新只是总耗时的一部分；旧P0约624秒不是新任务ETA。本轮文档不虚构分钟估计。

另外保存新模块CPU单线程／GPU的batch1与32推理计时、明确MAC口径和参数量；可固定50 warmup＋500 measured，设备同步、无写文件混入计时。模型4770参数不是整套F16系统总参数；旧主干未重新测时，完整系统成本标NOT_REMEASURED，不把Native缓存查表说成端到端延迟。

未来获得执行要求且预检全部通过后：独立唯一按需0trigger worker；一次startup、一次50%预计elapsed、一次完成核验。按真实剩余阶段修正一次性时间节点，不高频轮询。原已完成P0进程与旧PAUSED任务不查询、不重启；本轮文档不创建这些任务。

## 11. 建议目录与交付清单

```text
ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029/
  README.md / protocol.json / AMENDMENT_TINY_V2.md
  SOURCE_REUSE_REVIEW.json / source_manifest.json / split_manifest.json
  tiny_v2/SELECTION.json / checkpoint.pt / METRICS.json / TINY_REVIEW.json
  preflight/PREFLIGHT.json / BENCHMARK.json / RESOURCE_INVENTORY.json
  G/42|43|44/checkpoint.pt / run_manifest.json / history.json
  H/42|43|44/checkpoint.pt / run_manifest.json / history.json
  FINAL_RELEASE.json
  predictions/<arm>/<seed>/<population>/...
  baselines/<identity|train_mean>/<population>/...
  held_teachers/<hold|inner>/...
  metrics/ALL_GROUPS.csv / STRATA.csv / RESCUE_LOST_IDS.csv
  statistics/bootstrap_indices.npz / CI.json
  completion_review/REVIEW.json / SOURCE_PRESERVATION.json
  RESULT_ANALYSIS.md / RESULT_ANALYSIS.json
```

`FINAL_RELEASE`必须实际逐一绑定六final和本次冻结评价代码，并引用TIME停止证据；不能沿用要求十二模型的空等待，也不能手写总PASS跳过模型核验。某模型未完成不得开held；受阻时输出真实未完成报告。

最终报告开头必须回答：tiny是否实际通过；六模型是否完成；H是否优于G；是否优于identity／均值；收益是否迁移；原成功丢了多少；安全门是否全部通过；TIME/P3为何未执行；成本是否只是新模块口径。

## 12. 预先反方检查与研究诚信说明

**DA1／设计风险：**tiny抽样在看到数据分层后修订，不能再称原先预注册双分层完全按计划执行。应在任何tiny更新前保存修订与IDs；不按拟合效果选样本。当前请求只交付设计，旧文件不改。

**DA2／证据风险：**END oracle有134/44个救回不证明历史可预测其修正方向；每候选同GT监督也可能改善大量无关候选而不提高集合覆盖。G/H、regret、原好保留和完整K20指标共同约束解释。

**DA3／结论风险：**H不优于G不能推出历史没有信息；H优于G也不证明心理意图理解。END实质为低维偏差校正，TIME阴性只针对当前动作和安全合同。即使全部门通过，仍是反复使用研究人口上的探索证据，不能直接称SOTA或独立whole-model泛化。

本文件由AI助手按academic-research-suite的方法学与反方流程形成；审查为同助手内联检查，非外部独立评审。只使用既有授权数据与去标识化缓存，不接触新受试者；原数据许可／机构伦理适用性沿用项目记录，本文不虚构已获新的伦理审批。论文对外材料不披露可识别个体信息。

## 13. 继承文档

- [原系列设计](ENDPOINT_TEMPORAL_OBSERVABILITY_DESIGN_20261028/01_SERIES_DESIGN.md)
- [原核心动作／训练合同](ENDPOINT_TEMPORAL_OBSERVABILITY_DESIGN_20261028/02_CORE_CONTRACTS.md)
- [原统计与停止标准](ENDPOINT_TEMPORAL_OBSERVABILITY_DESIGN_20261028/03_EVALUATION_AND_DECISIONS.md)
- [已有代码及生产未完成项](ENDPOINT_TEMPORAL_OBSERVABILITY_DESIGN_20261028/04_CODE_GUIDE.md)
- [P0结果报告](../qoder_work/ENDPOINT_TEMPORAL_OBSERVABILITY_EXEC_20261029/RESULT_ANALYSIS.md)
- [原tiny停止收据](../qoder_work/ENDPOINT_TEMPORAL_OBSERVABILITY_EXEC_20261029/END_learning_preflight/TINY.json)

冲突处理：本设计仅替换tiny第二组的选择规则并收敛到END；其他科学常数沿用已经审核的原合同。若发现旧文档、代码和收据仍存在未说明冲突，先做来源调和，不从中挑更容易通过的一版。
