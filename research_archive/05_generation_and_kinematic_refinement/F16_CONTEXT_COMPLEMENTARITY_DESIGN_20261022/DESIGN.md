# 实验设计：同源上下文公共基线与 R/N 重加权

## 1. 决策背景与新问题

旧 E1 三 seed 均值：FIT overall native10858、A8071、B8603、E8193；inner overall native932、A667、B703、E662。E 的 inner change 比 B 多约0.872pp，但整体少约2.191pp，原成功丢失30%–40%；inner全部44个模型×group五运动学联合门失败。

这支持结束旧 history-only 配方，不支持认定学生容量上限，也不支持单纯扩大补全监督。新问题分两层：

- **Q1：在既有 F16 上下文可用的条件下，一个固定的轻量 K20 解码器能否先形成合格公共基线？**
- **Q2：若基线合格，同教师集合、同总辅助系数下，R/N 组间重加权是否带来安全且稳定的完整覆盖增量？**

Q1 不回答“增加上下文的纯因果作用”：本包不额外训练 history-only 同容量对照，旧 E1 仅描述性参考。Q2 只在新公共模型内归因。不要将不同信息的旧 E1 与新 E_bal 做直接机制比较。

## 2. 范围与探索性质

只使用原 target-grouped FIT18909、inner1856。原 outer1035、original_CAL2879、cal_v11 不参与本包新模型评价、选择或统计拟合。保留原全部 ID、原分组，不删难例。

已有 native/h3/v2 的上游暴露不会被新名字或新 head 消除。S0 必须查清**实际采用的编码缓存**是否有相同暴露，不能把另一条缓存链的结论直接套用。即使可重放，其结果也只支持“该冻结表征/既有重复使用人口条件下的新解码器比较”，不是 whole-model 未见泛化。

不修改或训练旧 F16 encoder/decoder/scorer，不训练教师，不新增数据、外部 fold、场景图、soft winner、物理损失或候选过滤。本包所有学生从头初始化，非旧 head warm-start。蒸馏教师只在训练端出现；推理输出学生自身20条，不追加为K40，不进行GT选择。

## 3. S0：来源与公共输入

### 3.1 输入定义

固定焦点行人原8步、96维编码 token：`Z ∈ R^(8×96)`。公共输入：

`x = concat(flatten(past_local[8,2]), (past[-1]-past[-2])/.4, Z[-1], mean_time(Z))`

维度为16+2+96+96=**210**。上下文来自同一个固定 F16 checkpoint/预处理链，不随学生 seed/arm 变化。不拼 native 预测坐标、教师轨迹、未来 GT、group、scene、track、ID 或 logits 到 forward。

使用 recent+mean 是确定性压缩接口，类似原解码器读取近期与汇总 token 的方式；**不是原解码器结构复刻，也不保证完整保留8个token信息**。本包不同时测试 mean-only/full-token/attention 等接口。

past/native/GT 全部使用已经核实的焦点坐标和米单位；最后观测点为零。不能混 raw940 与 prepared token 缓存。字段名、offset 和 producer 路径必须来自真实源码核验，不允许仅靠“shape看起来对”绑定。

### 3.2 S0 审核与阻碍

锁定实际编码 checkpoint、预处理源码、缓存、native、每个 KIN seed、GT、split、身份表的文件SHA和显式ID顺序。统计只能由FIT18909拟合。

优先使用有显式ID的既有token缓存。若只有合并特征，核验其真实导出字段和数据流；若需冻结重放验证，单模型、无梯度、固定ID哈希抽128窗，不按质量选。原始模型 forward 必须仅接收观测，不把未来标签用于新邻居选择；既有legacy选择问题单列披露。

预检基于生产同设备/同固定batch256计算，小尾批补零后丢弃padding输出。缓存与实际frozen forward对齐容差预设2e-5；GT/ID/顺序离散值精确匹配，失败保留来源，不换另个缓存求通过。

缺少可验证token或producer时，S0写具体BLOCKED并结束，不自动降级成history-only，也不启动旧encoder训练。上游已知暴露不阻止**明确限定的探索实验**，但阻止把结果称为干净泛化；当前文件身份无法绑定则直接阻止运行。

## 4. 公共学生、统计与训练配方

所有臂共用一套新学生：

|部分|固定设计|
|---|---|
|输入统计|210维，FIT-only float64 mean/std，std<1e-6置1，存float32 buffers|
|共享编码MLP|210→128→128，GELU|
|候选query|20×16，可学习，初始化Normal(0,.02)|
|融合MLP|144→128→128，GELU|
|输出头|128→24，线性，weight/bias零初始化|
|输出|float32局部完整20×12×2位置；不裁剪/平滑/投影|
|预计可训参数|82,008，必须以实现实数断言；不是整模参数|
|初始化|seeds42/43/44，三臂每seed完整state_dict逐项相同|
|训练|8 epochs，batch256，AdamW lr=.001/wd=.0001，global grad norm clip5|
|数据顺序|每seed独立保存8个shuffle permutation，三臂复用，不由臂运行顺序改变|
|最终checkpoint|固定epoch8；inner仅评价、不选checkpoint、不早停|

固定8 epochs是短预算筛选，不是收敛证明。输出零初始化意味着初始20条路径完全相同；保留该明确设计而不事后更换初始化求成功，预检与正式日志必须记录ties和槽位分化。失败只能说明当前配置/预算未建立可靠基线。

基础损失复用旧E1**实际**函数：

`L_task = mean_i min_k (ADE_ik + .5 FDE_ik)`

`L_div = mean_(i,k<l) relu(.3 - ||p_ik,end - p_il,end||)`

`L_base = L_task + .05 L_div`

注意是20槽190对的**终点距离线性hinge**；不是其他系列的全路径平方hinge、4槽六对或noclip配方。

## 5. S1：先训练公共基线 A

只训练A×3seed，保存3checkpoint与6份最终FIT/inner完整输出。旧A仅供表格参考，不作新统计主对照。

### 基线可用门 G_base

以下为**本包新增的保守继续投资门**，不是旧实验追溯标准，不是“重大突破”标准。每个seed、FIT和inner、每个overall/stable/transition/change组，全部满足：

1. `coverage_A >= .99 * coverage_native`（相对1%护栏，不是减1个百分点）；
2. `mean minADE_A <= 1.01 * mean minADE_native` 且FDE同理；
3. 五项pooled运动学各自 `M_A <= 1.01*M_native`；
4. 非空组、完整有效K20、所有源/ID/数值/保存审核通过。

另外完整报告native→A的rescue/lost；G_base净覆盖接近不等于逐窗成功保留。该限制在S2另设严格lost门。

**任一失败：结束本包于BASELINE_NOT_READY，不训练B/E。** 不延长epoch，不重新归一化，不删某seed，不扫描结构。结果若改善但未过门，可报告线索，不能绕门继续。此门可能较严格且native信息/暴露条件不同；它是成本决策，不是统计非劣证明或学生能力上限。

S1通过后A结果锁定，S2的B/E从对应同seed初始state重新训练，不能从A最终权重续训。S2执行需使用本设计的固定协议，不再依据A结果调λ、输入或教师。

## 6. S2：唯一因素——R/N组间权重

### 6.1 同一教师全集

每个FIT窗口先根据原native20和GT同路径门构造：

- R：native已成功窗中的native好候选。
- N：native失败窗中的KIN好候选。
- T=R∪N。各窗最多20条有效教师，固定按原物理slot升序保存，不平均轨迹，不按学生输出重选成员。

native成功窗中的KIN好候选对B/E**都排除**，排除数量单列。这消除旧B/E教师成员混杂。不再追加物理筛选教师；quality-only教师并非集合安全保证，五motion作为结果门保持。

KIN seed与学生seed对应，所有臂在同seed使用同缓存。跨seed变化同时包含初始化与KIN实例变化，报告为**3个配对pipeline实例**，不能宣称单纯随机初始化方差。

### 6.2 三臂

|臂|训练目标|数量|
|---|---|---:|
|A|L_base|复用S1三run|
|B|L_base + L_union|3run|
|E_bal|L_base + L_bal|3run|

教师到学生距离 `d(t,S)=min_k(mean_h||t_h-S_kh||+.5||t_end-S_k,end||)`。两组距离和S_R/S_N、数量n_R/n_N均为当前真实训练batch的有效教师条数。

- B：`(S_R+S_N)/max(n_R+n_N,1)`。
- E_bal：两组都有时`.5*S_R/n_R+.5*S_N/n_N`；仅一组有时用其均值；全空时0。

两臂辅助总系数λ=1，无扫描。只有相对组权重改变，不同时过采样、加宽、改teacher、改clip、增加物理损失。

同teacher且权重为`n_R/(n_R+n_N)`与`n_N/(n_R+n_N)`的分解应与B等价，这是单元测试，不增加训练臂。E_bal不是恒等式，改变梯度方向、尺度及方差；相同λ不等于相同梯度范数。

每epoch记录R/N有效计数、总及分项loss、preclip梯度范数、裁剪触发率、WTA/teacher责任和ties。固定哈希FIT诊断子集1024窗每epoch记录覆盖/保留/补全，不用于选择模型；全FIT和inner只在最终固定checkpoint完整评估。

## 7. 评价、主比较与统计

### 7.1 固定评价

主评价人口inner1856，其中change497。FIT18909为拟合诊断，不和inner合成一个总体。全组均报告：同candidate ADE≤.2 AND FDE≤.4、J=min_k max(ADE/.2,FDE/.4)、rescue/lost/retained、minADE、minFDE、GT-free完整K20 complete-link(.10meanpath/.20endpoint)、effective-good、五pooledmotion。

v/a/j为12/11/10项，dt=.4，prepend最后观测点，p95使用linear，必须全人口汇总不能平均逐窗p95。native零基值直接比较绝对不等式，比例NA；不加科学epsilon。固定.3/.5只敏感性。

### 7.2 唯一 primary 与继续门

唯一primary：**inner change的 E_bal−B 完整覆盖差**，单位百分点。

本包局部机制阳性需要同时满足：

1. primary平均增量≥2pp，三个seed差均>0；
2. 预设共同track+paired-pipeline bootstrap的95%区间下界>0，固定pipeline敏感性下界也>0；
3. E_bal相对A在inner change平均差≥0，三seed均不降低；否则不能将“优于较差的B”称为有效互补；
4. inner每seed每组E_bal的coverage≥.99×A且≥.99×native；minADE/FDE≤1.01×A且≤1.01×native；
5. 每seed每组E_bal相对native及A的lost/参考成功数≤1%；空成功分母标NA并解释，不自动PASS；
6. FIT/inner每seed每组五pooledmotion均≤1.01×native；
7. 来源、模型冻结边界及完整保存审核通过。

2pp是本设计选择的最小值得继续研究幅度，**不是功效分析得出的可检测阈值，也不是收益预测**。只有497个change窗与少量相关track，可能无法建立清晰区间，应允许INCONCLUSIVE。

通过局部门不等于项目原major通过。原changeADE/FDE各≥5%改善、protected≤1%伤害、原生同路径覆盖/重复候选/五motion及相关性审核等要求保持，不因本包primary换成coverage而撤销。未完成原major全套审核时`major_status=NOT_ESTABLISHED`，绝不自动升级baseline。

### 7.3 相关性与多比较

track键为sequence+pedestrian，不把窗口或20条候选当独立样本。每次bootstrap在inner全部track中有放回抽取相同数量track，同一draw的track重数用于所有臂和三个pipeline；再有放回抽3个pipeline索引并对所有臂共用。保留窗口加权：各pipeline在抽中group窗口上求均值，再取pipeline均值。

固定20000draw、RNG20261022；记录无group分母draw并拒绝重抽该draw，若无效比例>5%或有效track<20，标INSUFFICIENT，不报可靠主区间。另给固定三pipeline、只重抽track的95%敏感性区间。primary仅1项、不挑另一敏感性翻案。

预定secondary family共8项：inner E_bal−B与E_bal−A × overall/change × ADE/FDE。给每项Bonferroni百分位区间[.003125,.996875]及原始效应量；作为探索性familywise近似，不保证有限样本精确覆盖。其他group、J、子集、effective-good为描述性，不从大量输出筛显著。

所有继续门为AND，安全护栏是描述性确定比较，不宣称统计非劣已证。S1筛选、重复使用inner、仅3pipeline、同recording多track依赖都会限制区间；这些区间不能建立独立未知场景泛化。不要因p值或CI过门就称运动理解。

## 8. 运行规模、资源与状态

S0先做来源/模型接口/teacher形状、真实GPU梯度、真正子集2epoch恢复与保存审核；不得在预检中选择科学效果最好的实现。

S1：3训练×8epoch，最终6模型人口输出。S2追加6训练×8epoch、12输出；总计最多9训练、72epochs、18模型人口最终输出。每epoch诊断小子集、native参考输出另计，不混充科学run数。若S1失败仅3训练，后续标SKIPPED_BASELINE_GATE。

运行前实测4warm+32throwaway FIT256 step、inner完整前向和128窗完整聚簇/运动学/独立审核、源SHA与磁盘临时/最终需求。不得套旧E1“15秒/模型”作整套ETA，旧22份补评约1239秒已表明审核IO不可忽略。

资源沿用最新明确的**任务峰值估计E、1.3动态remaining与Windows保留峰值**规则，原实现见证据修复包`common.py`，审查后复用。E需包含Torch/完整输入、单模型+optimizer、teacher/输出/独立审核临时数组；逐模型、bounded chunk，不能以已有冻结E4GiB直接保证训练足够。此包不重新引入被覆盖的固定30分钟/固定6GiB旧限制。不得改OS/pagefile、关闭用户程序。

当前是文档交付，不安排任何定时任务。以后若获执行请求，只有实际预检与实测ETA通过后才sole OS worker、一次startup、一次50%elapsed检查及一次完成核验；源锁后科学变更必须留失败及修订来源，不盲重启。

## 9. 结果分支与终止

- S0身份/来源无法对应：BLOCKED，停止。已知暴露但映射确实则EXPLORATORY_ONLY，不偷换成干净泛化。
- S1失败：当前同源上下文接口/架构/8epoch不足以建立合格基线。不能唯一归因编码信息无用，也不能随后自动换接口、初始化或训练时长。
- S1通过，B/E都差：原基础能力存在，但这套quality-only集合监督没有安全增量，停止该监督配置。
- E只优B不优A：重加权可能缓解普通蒸馏伤害，不是互补成功。
- E质量改善但lost/motion失败：带代价的局部改进，不放宽门，不自动加平滑修成另一方案。
- E全部局部门通过：允许建议独立清洁上游/新人口复核，不能自动执行或宣称已完成创新。

不在本包自动运行第四臂、λ扫描、容量×采样组合、场景条件或新scorer。保存负结果后结束。

## 10. 同助手设计反方检查

1. 最强质疑是新学生仍可能从头学不好。S1明示是短预算可用性门，失败停止，不能把这种可能性藏到S2后解释。
2. recent+mean并非完整F16信息，且模型比旧学生多24576参数。故旧E1比较只描述，不宣称纯上下文因果；B/E内部才严格同信息同容量。
3. teacher按GT质量筛选并不保证学生集合安全。安全只由最终全组实际路径确认，不用教师17/24池门结果替学生背书。
4. 上游暴露与反复inner探索仍在。即使通过，本包仅给有限机制线索，论文需要独立验证。

本设计由同助手依研究设计与统计技能整理，非外部审稿；未进行功效计算、预检或训练。
