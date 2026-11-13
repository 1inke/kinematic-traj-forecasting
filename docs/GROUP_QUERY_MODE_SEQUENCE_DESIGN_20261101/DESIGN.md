# 固定K20下的分组监督与候选交互：下一阶段实验设计

日期：2026-11-01｜状态：**DESIGN_ONLY**。本文件不是执行收据。

## 0. 决策摘要

先进行一个有边界的训练机制实验：**同一直接生成解码头，单查询组 G=1 对三查询组 G=3；推理始终只用预先固定的主组0，输出20条完整轨迹。**

只有发现可重复、未进一步牺牲安全性的主组收益，才开展“全连接模式交互对因果模式交互”。早匹配责任分配放在最后，且要求前序已经有合格的生成质量。禁止一开始同时叠加分组、因果注意力、EMTA、新评分器、保护loss和原生混合策略。

本轮借鉴文献，不复现驾驶场景整套网络。论文依据及反例见[PAPER_REVIEW.md](PAPER_REVIEW.md)。不再延长END49小端点系列。

## 1. 为什么换这个问题

### 已有证据

1. 最新离散动作H相对G有训练内收益，但change仍9/23，无稳定保护；已经监督全部合法动作，不支持将失败统一归因于WTA梯度稀疏。
2. 旧 ContextK20 的inner change覆盖166→186/191/192，说明直接生成头能产生补充好路径；同时lost33/37/37、误差与运动门失败，说明“更多覆盖”还不是“可用的候选集合”。
3. 旧ContextK20未通过自身门，旧蒸馏B/E_bal未运行。本研究另立训练机制诊断，**不把旧门改成PASS，不恢复旧任务**。

### 新的三个问题

- Q：训练期增加独立的查询责任路径，能否改善固定主组，而不是仅辅助组的并集？
- M：候选之间交换隐藏信息是否有效；固定单向顺序是否比同成本的双向交互更有效？
- E：若生成已稳定，选“第一个合格候选”承担回归责任，是否优于永远选最小误差者？

Q、M、E不是预设答案。无收益、更多塌缩、只有FIT收益、只有大并集收益、安全退化，均是有效停止结论。

## 2. 固定边界与数据

- 旧F16、原decoder、原scorer权重不变；本轮训练新头，不训练编码器、不使用新地图或新数据。
- 输入沿用已验证210维：8步二维past展开16维、末步速度2维、最近token96维、平均token96维。速度dt=0.4。只使用预测时可见信息。
- 原FIT 18,909窗训练，原inner 1,856窗评价；完全继承原ID与sequence|target-track定义，核对不交叉。统计只由FIT拟合。
- inner是反复使用过的开发验证人口，不宣称全新确认集、场景独立泛化或三seed等于三次独立数据实验。
- 本系列不新开outer、original CAL、外部test。与最新tiny128/训练3103分支的人口不同，不能直接作配对差分。
- 推理K=20、T=12、dt=0.4，整条轨迹并行输出；GT/group/ID/logit不进入新头forward。新头无地图输入。
- 这不是“保留native16+新增4”。它生成完整新K20，初始零输出**不是native identity**。native20只是同ID固定参照，不与新K20并成部署集合。

输入、原split、统计与原始单位不能仅凭路径相同认定一致，必须保存SHA与显式ID。历史归档输出若确有复核需要，从现有无损档案按需恢复；禁止因文件已归档而重训旧模型。

## 3. 公共模型：保留已知实现，避免同时换架构

依据：`qoder_work/F16_CONTEXT_COMPLEMENTARITY_DESIGN_20261022/execution_20261022/core.py` 中 `ContextK20` 及同目录 `protocol.json`。

```text
210D观测特征 → FIT标准化 → MLP(210,128,128)
                                      ↓ 广播至每个query
query[g,k]:16D → concatenate → MLP(144,128,128)
                                      ↓
                        [M阶段才插入模式交互]
                                      ↓
                            Linear(128,24)
                                      ↓
                            K×12×2完整轨迹
```

激活GELU；query初值N(0,0.02²)，末层权重和bias零初始化。G=1共82,008可训练参数。G=3仅新增两个20×16查询表，共82,648参数；导出主组后仍82,008。上述数字是解析计数，实施时须由实际模型参数清单核对，不是已构建模型的实测结果。

这里查询是可学习模式token，不是原文的粗轨迹anchor query。它是**分组监督机制移植**，不是 Group Query 全模型复现。不得为贴近论文同时增加anchor、地图、第二阶段细化。

公共训练固定：fresh seeds42/43/44；8 epochs；batch256；AdamW lr=0.001、weight_decay=0.0001；clip_grad_norm=5；固定末epoch8。无早停/最优inner checkpoint/额外训练预算。

公共每组损失：

\[
Q_{igk}=ADE(\hat Y_{igk},Y_i)+0.5FDE(\hat Y_{igk},Y_i),
\]
\[
L_g=\operatorname{mean}_i\min_k Q_{igk}
+0.05\operatorname{mean}_{i,k<l}[0.3-\|\hat y_{igk,T}-\hat y_{igl,T}\|]_+.
\]

端点diversity是**线性hinge**、190对的均值，不是之前另一分支的平方hinge。Q阶段总loss为各组均值，不是求和；clip在均值loss反传后仅做一次。相同epoch/update/batch order不等于相同训练FLOPs，须两种成本都报告。

## 4. Q：首先实施的6模型实验

|臂|训练|推理|改变因素|
|---|---|---|---|
|Q1|一组20查询|组0，K20|公共对照|
|Q3|三组各20查询，每组独立WTA|固定组0，K20|额外独立查询组及共享梯度|

每臂3seed，合计6个新模型、48科学epochs、按18,909/batch256每轮74步，共3,552科学更新。预检和benchmark更新单列。不得拿历史Q1旧权重冒充此次配对对照；旧输出只能核对实现兼容。

### 配对与非退化要求

1. Q1、Q3共享trunk/fuse/head、组0query、mean/std初值逐tensor EXACT；相同科学batch顺序。
2. 辅助组1/2使用独立固定子种子初始化，不能消费或改变主模型/loader RNG。禁止三个组完全复制且共享同一query参数，否则平均loss可能退化成原单组。
3. 各组独立算winner与diversity。不能在60候选上取一个min，也不能跨组算190以外的pair。
4. 测试GT改变不得影响预测；辅助组删掉后主组输出必须与原模型组0一致（同计算后端/批次）；导出代码在构造查询前就只保留主组。
5. 不按inner结果挑主组，不报告“最好组”作为方法，不用GT给各组赋予stop/turn语义。

### 核心接口伪代码

以下为架构合同，非已执行的生产脚本。

```python
def forward(features, train_groups=True):
    h = trunk((features - mean) / std)          # [B,128]
    q = queries if train_groups else queries[:1]  # [G,20,16]
    B = len(h)
    hh = h[:, None, :].expand(B, 20, 128)
    outputs = []
    for query_group in q:
        qq = query_group[None].expand(B, 20, 16)
        z = fuse(torch.cat([hh, qq], dim=-1))  # [B,20,128]
        # M阶段在这里调用相同的组内mix，不能跨组。
        outputs.append(head(z).reshape(B, 20, 12, 2))
    return torch.stack(outputs, dim=1)         # [B,G,20,12,2]

def grouped_objective(paths, gt):
    # 不允许 paths.reshape(B, G*20, 12, 2) 后取一个winner。
    losses = [original_Q_and_linear_diversity(paths[:, g], gt)
              for g in range(paths.shape[1])]
    return torch.stack(losses).mean()

def deploy(features):
    return forward(features, train_groups=False)[:, 0]
```

Q阶段参考实现逐组使用相同B×20计算形状，保持主组导出核验清晰；不得先将三组flatten成B×60再要求所有CUDA后端逐位相同。后续若矢量化，须单独核验数值与梯度等价及真实速度，不静默改变已锁实现。逐组是训练期计算，部署只有一组，不引入20次逐模式循环。

### 必须记录的机制证据

- 每组20槽winner次数、并列率、归一化winner熵、从未获胜槽数；按epoch而非只最终。
- 主组有效good cluster、完整覆盖、原成功lost，以及全部运动指标。
- 固定无质量选择的FIT hash128诊断批次，在epoch0/1/4/8记录各组对共享head/fuse的梯度范数、余弦；零梯度余弦记NA，不伪造0。
- 各组在同窗选中轨迹的几何距离/交集覆盖。槽号相同不代表轨迹相同，槽号不同也不等于不同模式。
- 辅助组单独成绩和60候选union仅作训练诊断，明确不等预算；主结论只看组0 K20。
- “熵上升/梯度不同”只是机制线索，不能代替覆盖收益或证明因果中介。

### 不训练新模型的反事实实现夹具

在固定真实8窗、固定非零权重、禁dropout条件下，构造三个完全相同且参数绑定的query组：均值loss与单组loss及共享梯度应一致。与独立query组对照，证明程序没有通过loss乘3或跨组取min产生假收益。这是数值夹具，不是第七个科学模型。

## 5. M：只在Q有主组信号后测试模式交互

### 触发规则

Q3−Q1须达到第8节的“机制推进门”。未通过则结束Q分支，不自动扫描G、宽度、epoch或加入EMTA救结果。

若通过但尚未达到native严格安全门，允许本阶段**一次机制诊断**，因为模式交互正是待检验的集合质量因素；该允许不构成部署许可、不覆盖旧失败。M仍失败即结束，不进入E。

固定G=3及Q全部参数/训练/输入/损失，增加一个小模式模块：Q/K/V各128→32，4heads×8D，输出32→128，残差相加，无dropout、无新增LayerNorm，组间禁止交互。

|臂|模式注意力mask|其他|
|---|---|---|
|M-full|每组20模式双向全连接|同模块、同初值、同预算|
|M-causal|固定query索引0…19，下三角含对角|同上，不用GT排序|

两臂各3seed fresh从配对初始态训练8epochs，不从Q最终模型warm-start。固定索引不是“最可能模式顺序”，无论文的置信度重排；必须叫**固定顺序因果模式交互**，不能冒称完整Parallel ModeSeq。

该模块增加16,608参数：三个(128×32+32)及(32×128+128)。G3训练99,256、主组推理98,616；实际实现须验证。输出投影零初始化可使初始残差为零，但QKV初始梯度可能为零，合法梯度核验须分阶段，不能要求所有参数第0步正梯度。

```python
def mix_modes(z, causal):                 # [B,G,20,128]
    B, G, K, D = z.shape
    x = z.reshape(B * G, K, D)           # 组视作batch，不能并进K
    q, k, v = project_to_four_heads(x)   # [B*G,4,20,8]
    score = (q @ k.transpose(-1, -2)) / (8 ** 0.5)
    if causal:
        score = score.masked_fill(strict_upper_triangle(K), -torch.inf)
    h = torch.softmax(score, dim=-1) @ v
    return (x + out(join_heads(h))).reshape(B, G, K, D)
```

该实现仍通常计算完整20×20矩阵，再mask；**不能声称因果mask降低了真实FLOPs**。所有时间点仍一次输出。记录CPU/GPU实测延迟、MAC口径、峰值；不只报参数量。

主配对比较M-causal−M-full隔离方向性mask；二者分别对Q3无交互参照的比较只能说明增加交互模块的综合效果，包含新增容量与优化变化。不得将后者称为纯交互因果。

M-causal未满足机制推进门，不挑M-full替它进入E；保留full的正/负结果，结束本预定因果路线。

## 6. E：条件性早匹配责任诊断，不直接照搬EMTA

只有M-causal同时通过机制推进门及第8节native安全门，才允许E。否则记录 `NOT_TRIGGERED_GENERATION_NOT_READY`，不把“少跑一个实验”写成未尽职责。

固定M-causal结构、G=3、数据、优化器、diversity，比较两臂各3seed fresh：

- E-WTA：每组最小Q者承担Q回归。
- E-early：同一组内，若有候选同时ADE≤0.2、FDE≤0.4，选固定索引最小的合格候选承担**同一个Q回归**；无合格候选则回退最小Q。

这只改变责任分配，**不改成minJ回归、不加新分类/排序头、不改旧scorer、不按GT重新排列部署输出**。训练GT可用于选择监督对象，推理不得传GT或选择oracle。

```python
good = (ade <= 0.2) & (fde <= 0.4)       # [B,G,K]，同candidate
fallback = q.argmin(dim=-1)             # ties取最小索引
first = good.to(torch.int64).argmax(dim=-1)
winner = torch.where(good.any(-1), first, fallback)
task = q.gather(-1, winner[..., None]).squeeze(-1).mean()
loss = task + 0.05 * within_group_linear_endpoint_hinge(paths)
```

额外报告：有匹配窗比例、E/WTA winner-switch、选择Q增加量、早槽责任集中、后槽饥饿、完整集合质量。早匹配可能仅把梯度挪到早槽而没有互补收益；这属于可证伪结果。

不能从原文2%或20%等假设switch率移植门限。switch小不说明算法等价，switch大也不证明更好。本地版本没有原文置信度重排/焦点分类/排序loss，报告中必须注明适配差异。

## 7. 预检、预算与审计合同

### 先做预检，后给ETA

1. 当前输入source/split/stat单位、F16缓存SHA与显式ID；输入函数签名审计，避免import旧顶层代码写旧目录。
2. 实际GPU真实8批FIT，验证finite、shape、组0配对、组边界、部署导出、loss与梯度；Q1在同权重下与原ContextK20基线一致。初始零head不冒称nativeidentity。
3. 独立NumPy公共loss；固定非零权重double有限差分两eps 1e-4/1e-5，记录winner/hinge/clip活动面。跨面时报告不可用，不通过挑eps/改容差洗成PASS。
4. 每臂真实hash513FIT+12inner子集连续/中断两epoch；参数、buffers、Adam状态、CPU/CUDA/NumPy/Python/loader RNG、顺序、history全EXACT。不能称完整FIT两轮。
5. 实际128窗输出保存读回，完整samepath质量、原SciPy聚簇、全五motion和junction双实现；GT变更不改变forward/导出输出。
6. 错source/ID/stats/checkpoint/output必须被reader拒绝；分析SHA-skip不是训练RNG恢复。

### 短预算，不用文献速度代替实测

每臂4warm+32throwaway FIT256测吞吐，另测保存、完整聚簇、bootstrap、最终SHA和CPU审核。按本阶段6模型、48epochs、全FIT/inner输出估计完整工作量与临时空间。

每阶段保守完整预计≤30分钟才允许生产。若超预算，标 `SKIPPED_BUDGET` 并报告分项，不暗减epoch/seed/人口来赶时间；本设计不批准长预算训练。实现/写代码的人时与实际科学运行耗时分开。

资源沿用最新 **warning-first v3**：以实测峰及载荷建立E清单，1.3为预警，不恢复旧固定6GiB硬停；持续压力和紧急条件沿用已确认实现。禁止改OS/pagefile、锁内存或kill用户进程。

实施时每阶段独立新执行目录、当前科学源锁；唯一worker，启动一次确认，单次50%预计elapsed检查，完成节点核验，按实际剩余修正；不日复一日/高频轮询。不恢复历史PAUSED。**本次文档交付不创建任务或自动跟进。**

## 8. 评价、统计与阶段门

### 必须统一保存的指标

每split×seed×overall/stable/transition/change：

- samecandidate good：ADE≤0.2且FDE≤0.4；J=min_k max(ADE_k/.2,FDE_k/.4)。
- minADE、minFDE单独保留，不假设来自同一个候选；J赢家的early/middle/late误差不拼slot。
- native→主组、配对对照→主组的rescue/lost/retained；lost分母为参照成功窗，零lost不等于连续误差无损。
- 原SciPy complete-link，D=max(meanpath/.10,endpoint/.20)，全K20；有效good、重复/不同cluster，不用候选数最大化代替质量。
- 全人口pooled speed_p95、accel_mean/p95、jerk_mean/p95，dt=.4；原接续junction指标继续保存。分母、quantile、dtype和空组处理锁定，literal科学门无epsilon。
- primary为inner change同路径coverage；FIT是学习机制与安全诊断，不替代inner。

### 机制推进门：相对配对新对照，不等于方法成功

三项预定主比较为Q3−Q1、M-causal−M-full、E-early−E-WTA，不增加最好组/最好seed比较：

1. inner change覆盖三seed都正，平均至少+2个百分点。
2. 同track配对、同seed配对的探索bootstrap校正CI下界>0。
3. inner各组minADE/minFDE及FIT/inner各组五运动指标，每seed不超过该配对对照1.01倍；inner各组覆盖不低于对照0.99倍，lost/对照成功≤1%。

均不通过则不推进。条件阶段仍保留完整负例。相对一个不安全对照“不更坏”并不等于绝对安全。

### Native安全与最终成功门

另外单列对immutable native的严格保护：FIT/inner所有组、所有seed五motion及既有junction门；inner overall/stable/transition minADE/minFDE≤native1.01、覆盖≥native0.99、lost/native成功≤1%。任何NA不自动PASS。

最终大突破仍需原change minADE及minFDE各≥5%改善、三seed一致及联合CI、同路径覆盖/有效候选/全部运动与相关性复核。+2pp只是本系列机制推进阈值，不替代原major标准、不自动升级baseline。

### 统计口径

预注册20,000次whole sequence|target-track bootstrap，配对臂使用同抽样；同时配对重采样三个模型seed，另报告固定三seed的track-only敏感性。RNG20261101，window-weighted后seed平均；相关窗口不是独立样本。

三项计划主比较统一Bonferroni：双侧98.3333%探索区间，分位[0.0083333333,0.9916666667]；跳过的阶段不把其alpha挪给已运行阶段。有效draw<19,000或分组有效track<20标NOT_ESTIMABLE，不删难例修复。其他指标/组为完整描述与硬安全门，不虚构大量未经校正的显著性。

这些区间仍是反复使用开发集下的探索证据，不是事后选择被校正成确认性。若方法真正成功，另立锁定评估协议再谈外推；本设计不开放新test。

## 9. 结果解释分支：避免再无限局部调试

|结果|可支持解释|下一步|
|---|---|---|
|Q辅助组/60并集好，主组无提升|训练多样性未转移到部署主组|停止Q路线，不用union冒K20收益|
|Q只改善FIT|监督机制有训练内效果，迁移未解决|不自动加组数/epoch，不称信息不足|
|Q提升主组但误差或安全更坏|责任分配与完整集合目标仍不一致|门失败停止；不靠论文名义放宽|
|Q过机制门，仍未过native安全|有训练机制线索，尚非可用方法|仅允许预定M一次，不部署|
|M-full与causal均比Q好，彼此无差|一般交互/容量可能有用，顺序优势未证|不宣称Mode-as-Sequence机制获证，E不触发|
|M-causal过相对门但native不安全|顺序可能有帮助，集合质量仍不足|E不触发；停止本系列|
|E只改善覆盖却伤minFDE/支持|重现指标冲突|拒绝E，不新增scorer掩盖|
|完整质量和成本均过门|固定预算训练机制具有实证价值|另立独立确认与轻量端到端验证|

最多三个条件阶段，每阶段6模型；不是一次性授权18模型必跑。Q失败即止，禁止第四个G/排序/宽度/阈值变体。即使所有阶段均未成功，也应形成明确的候选生成训练机制结论，而不是回到单槽端点无限调试。

## 10. 输出合同与实现文件建议

未来实施目录建议：`qoder_work/GROUP_QUERY_MODE_SEQUENCE_EXEC_20261101/`，当前不得伪造已存在执行结果。

```text
protocol.json / source_manifest.json / split_ids.npz / statistics.npz
models.py              # 分组query、组内mix、仅主组导出
objectives.py          # 原线性hinge与分组均值；E独立assignment
preflight.py           # 实际GPU、分组/导出、FD、真两epoch恢复
train_stage.py         # Q/M/E显式白名单，不隐式扫参
evaluate_saved.py      # 全人口质量、聚簇、运动、责任统计
completion_review/     # 第二实现CPU保存算术、CI、SHA审计
run_manifest.json / checkpoint.pt / metrics.json / assignment_stats.json
DECISION.json / RESULT_ANALYSIS.md
```

checkpoint必须绑定source、split、统计、初始化、group数/组0身份、mask/order、loss公式和系数、optimizer/clip、batch顺序、所有RNG。保存完整各组训练诊断与唯一主组部署预测；不得只保留最好模型。

执行成功要求worker/exit双SUCCESS、全部科学模型与输出清单完整、全保存算术和SHA审核通过；审核PASS与科学门PASS分栏。CPU保存审核不冒充所有模型/梯度二次回放。

## 11. 反方检查（同助手，非外审）

**DA1：是不是换回已失败模型再赌一次？** 本研究确实选旧A为结构参照，因其有跨track覆盖信号且能隔离组监督。它不合格是已知事实，native安全门不撤销。若Q仍不改善，直接终止，不能以新论文为理由扩训。

**DA2：是不是更多算力/参数才变好？** Q3多640训练参数和约三组decoder计算，不能声称纯监督数量的因果识别。报告固定update及真实成本，提供绑定相同query的解析/数值反事实；正式算力匹配研究不在本次短预算中，限制如实写明。

**DA3：是不是又只优化覆盖忽略模型理解？** 所有误差、lost、运动学、有效good、失败场景均联合报告。注意力、梯度、winner熵只是诊断，不能证明心理意图理解。只有实际固定K20、低成本、保留支持并提高困难运动覆盖，才有资格形成项目的实证创新点。

## 12. 本轮实际完成与未完成

- 已完成：核查两篇用户原始PDF、对照官方论文页面；检查旧A实际架构/协议与最新离散效用失败；制定分阶段设计和接口。
- 未完成：新模型实现、GPU预检、资源/ETA实测、训练、导出、质量/CI复核。
- 未改动：旧科学源/权重/结果/阈值、PAUSED任务、baseline和长期goal。

**推荐第一步只做Q的实际实现与预检；不是立即把三种论文技术全部装进模型。**
