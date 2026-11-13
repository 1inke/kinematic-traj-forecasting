# “泛化与共享映射”分支：有限更新干扰与稳定学习对照

2026-10-15，日本标准时间；版本1.0；**DESIGN ONLY，未实现、未预检、未执行**。

本文件补充一个可实施的具体分支，不恢复失败T1的T2/T3，不修改旧锁定协议。本次请求是补方案，不是启动训练。未来实施应在新execution子目录进行。

## 1. 把大分支改成一个可以被否定的问题

当前问题不是“是否马上增加MoE”，而是：

> 在同一共享candidate-conditioned head中，降低一批窗口的训练误差，是否会通过共享参数更新损伤另一批轨迹；改变冲突梯度的合成方式，能否在不增加模型容量、不改变候选构造和loss数值定义的情况下，提高可迁移覆盖？

这是“跨样例共享更新干扰”子假设，不囊括表示不足、可达集合不足、条件分布变化等全部泛化问题。若试验阴性，不宣称共享映射已经排除，也不自动切换另一组路由参数。

## 2. 最新证据与此前方案的修正

1. [分布诊断](../GROUPED_TRANSFER_DISTRIBUTION_DIAGNOSIS_20261015/RESULT_ANALYSIS.md)：FIT→CAL支持外约5%–7%，不足以单独解释全部失败；共同支持人口中的差距仍在。但摘要支持内不等于P(Y|X)相同，不能说“分布问题已经排除”。
2. [三项总复盘](../GROUPED_TRANSFER_DISTRIBUTION_DIAGNOSIS_20261015/FULL_THREE_STUDY_REVIEW.md)：target/time/scene变化覆盖的grouped-random原探索CI均跨0，scene整体覆盖有更明确负信号；不是三次确认性证明共享head错误。
3. [T1结果](../MOTION_PATTERN_TRANSFER_DESIGN_20261015/T1_execution_20261015/RESULT_ANALYSIS.md)：固定完整模式责任相对dynamic覆盖平均−12.91pp，校正区间为负，且FIT本身也更差。原型固定责任路线停止，不重新包装成“稳定共享映射”。
4. [安全结构参照](../PED_FOUR_DIRECTION_SERIES_20261011/experiment1_motion_process/RESULT_ANALYSIS.md)：position3全部36个既有评价组五motion通过，但outer change仅110→111，尚非有效突破；acceleration3失败。此前candidate routing也未解决问题，不因“进入共享映射分支”就默认增加专家。

本设计采用position3作为**已有安全记录的受限实验平台**，不是新最佳模型。它的bottom4、0.2m界和低自由度可能限制改进；阴性只能约束这一平台。选择它是为避免在M2/T1已经明显不安全的全路径生成器上混合优化与运动学结构两个因素。

## 3. 三步计划，先诊断后训练

|阶段|具体操作|回答的问题|运行条件|
|---|---|---|---|
|D0 有限更新干扰|三个已完成position3 checkpoint的隔离副本，每个8个固定batch包，观察0/1/4步|局部梯度冲突是否实际变成跨track质量损害？|未来获准执行后，先做此项；非完整重训|
|U1 同架构优化对照|普通梯度合成 vs 对称冲突投影，2臂×3seed×25epoch|改变共享更新能否改善覆盖与成功保留？|D0触发且实际预算通过|
|V 迁移验证|先时间隔离双臂6runs，再条件性场景隔离6runs|改进能否超出原target split？|U1通过质量/保护/运动学门；time通过才scene|

最多18个fresh正式run/450epochs，**不是一次提交整个矩阵**。D0只是隔离副本的短步参数干预；确实会做backward和临时optimizer step，不能叫“纯离线无训练计算”，但不覆盖旧checkpoint、不部署副本、不续训旧worker。

## 4. 固定平台：先不动结构、责任或候选预算

严格沿用[原position3设计](../PED_FOUR_DIRECTION_SERIES_20261011/experiment1_motion_process/DESIGN.md)：

- 共享MLP120→128→64→6，GELU，24,134参数；P2观察96维+当前immutable native path24维。
- 三个位置残差节点t=4/8/12，原tanh/L2节点处理、线性插值和原单trajectory lambda，位置界0.20(t/12)^2；不扩大界、不换basis。
- fixed bottom4按旧native logits及physical-slot tie规则；其余16路径EXACT、最终K20。低logit不代表冗余，必须报lost。
- 原hardmin4(ADE+.5FDE)+.05×6pair平均平方hinge(.3)。不换winner、不改为J或固定责任、不加保护/物理loss。
- target FIT18909、outer1035、inner1856、原CAL2879；fresh seeds42/43/44、25epoch、batch256、AdamW .001/wd.0001、无clip/scheduler、末轮主结果。NNfloat32、保存几何float64。
- 标准化FIT-only；forward无未来GT/group/scene/track/ID。旧F16/decoder/scorer冻结且不实例化训练，immutable cache和相关SHA保护。

新处理仅发生在**训练gradient aggregation**。原stable/transition/change标签可以把每窗口loss分成三组，但只在loss侧使用，不能进入预测forward。没有学会预测这些GT组，也没有新增部署路由器。

## 5. D0：不用cosine替代真实有限步结果

### 5.1 固定选择与身份

采用原position3三个seed的最终checkpoint，均完整使用，不挑最好seed。复制到新进程隔离对象，原文件只读。旧optimizer如果存在也不用于“恢复旧训练”；此次统一使用新建AdamW原超参、零moment状态，明确这是在末轮参数附近的局部干预，不是重放原训练时刻。

将原FIT的(sequence,target_track)按SHA256固定tag `shared-map-d0-track-v1`排序，前floor(0.8×track数)为source tracks，其余为probe tracks。按身份切，不按质量。检查二者track无交集；同scene/time/context仍可能相关，不能称独立泛化。

source窗口按SHA256 `shared-map-d0-source-v1|sampleID`排序，取前2048条，顺序切成8个不重叠256窗口包。probe取保留tracks的全部窗口。不得为了满足结果而均衡或重新抽样。若source不足2048或probe<500窗口/30tracks，STOP记录，不改变比例。它们均是已训练过的FIT人口，D0只检测跨track的更新干扰；不使用outer/CAL来决定算法。

### 5.2 每个包做什么

每个seed×包从同一原checkpoint重新开始，两个隔离副本分别执行普通合成和下节投影合成；相同batch连续4步，新建optimizer。保存0/1/4步source和probe完整预测及指标，所有包报告，不按不利结果选包。

每步记录三组梯度内积/范数、投影次数、实际AdamW参数位移、winner/lambda活动面变化。另以NoUpdate副本重复前向，输出应EXACT，确认不是随机层/读取问题。

**主要看有限步，而不是负cosine：**

- source原完整窗口loss是否下降？
- disjoint-track probe的J、ADE/FDE、good-rescue/lost是否变差？
- probe中干预前已经good的固定集合，有多少被破坏？集合按step0确定，不能按step4重新筛选。
- 投影是否减轻这种损害，还是只是让模型几乎不更新？

K4训练Q与K20评价J不同，都要保存，不能用训练Q改善代替覆盖。GT仅loss/评价，NoUpdate和两种更新的probe身份一致。

### 5.3 D0预定触发，非显著性结论

先对每seed的8包等权汇总step4相对step0的改变。需要三个seed均满足普通更新source平均loss下降且probe平均J上升；并且每seed至少6/8包方向相同。此时才称“这个局部平台有可重复干扰信号”。

再要求投影在三个seed均保持source平均loss下降，下降幅度至少达到普通合成的80%；probe平均J损害比普通减少至少50%，且step0-good的平均lost数不增加。满足才允许U1。

这些是事前研究触发阈值，不是统计功效证明或新的科学成功门。8包共享模型/probe，不是8个独立实验；不虚构显著性CI。若仅有负cosine、有限步无损害，或投影只靠不学习减少损害，则停止，不启动U1。零moment局部结果不外推所有训练时段。

## 6. U1：保持共享网络，只改变冲突梯度合成

### 6.1 精确算法

每个窗口原loss为ell_i，包括其自己的K4 winner与6pair diversity。batch大小B，三个非空组g的大小n_g，定义：

\[
L_g=\frac1{n_g}\sum_{i\in g}\ell_i,\quad q_g=\frac{n_g}{B}\nabla_\theta L_g.
\]

普通A：`grad = sum(q_g)`，必须数值核验等于原batch平均loss的直接梯度。不把三组简单等权，不改变原人口权重。

投影B：为每组保留原q_g，拷贝h_g=q_g；依预定顺序遍历其他非空组j。若h_g·q_j<0且q_j·q_j>0，执行：

\[
h_g\leftarrow h_g-\frac{h_g^Tq_j}{q_j^Tq_j}q_j;
\quad grad=\sum_g h_g.
\]

stable/transition/change索引0/1/2；第s步的顺序为循环移位 `[s mod3,(s+1)mod3,(s+2)mod3]`，每个g按此序跳过自己。投影参考始终为原q_j而非已经改过的h_j；空组直接跳过，范数零跳过，不添加任意epsilon。点积与投影使用float64，再转原参数grad dtype，finite检查。两臂都实际计算分组梯度，使实现/计时口径明确。

它属于[Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)的冲突投影思想；这里的样本权重、三运动组及确定循环顺序是本地预定适配，不冒称论文原实验的逐项复现。文献提出这种优化工具，不证明它适合本项目。

AdamW仍原超参/weight decay；不得投影后另做norm rescale、clip、LR调度或接受/拒绝试步。记录合成梯度范数与实际参数更新范数。由于AdamW预条件和weight decay，梯度投影**不保证**每组有限步loss下降，后续投影也可能重新产生之前的冲突；实际评价决定成败。

### 6.2 唯一因素与局限

|因素|A|B|
|---|---|---|
|输入、所有参数/buffers、初始输出|配对相同|配对相同|
|预测结构、native16、K20、可达范围|相同|相同|
|每窗口loss、dynamic winner、diversity|相同|相同|
|batch与人口权重|相同|相同|
|更新梯度|原加权和|冲突投影后和|

不增加专家或容量；投影会同时改变方向、幅度、训练轨迹和有效优化速度。因此它测试“干扰缓解优化包”，**不是纯方向因果隔离，也不能成功后就证明共享head结构是唯一根因**。

6个fresh run，不从D0副本或旧训练末轮接着跑。逐epoch保存责任、loss、冲突率及更新量；初始化zero完整native20必须保存。主结果固定25，不挑第5/10/15/20轮。完整FIT最终输出另存，不用minibatch日志冒充FIT泛化。

### 6.3 继续门

outer change：B−A完整覆盖三seed均正，平均≥5pp，校正CI下界>0；B相对native每seed覆盖净增加，change ADE/FDE均不劣于A；outer overall/stable/transition ADE/FDE每seed对A及native均≤1%harm。B全部四split×四group×三seed五pooled motion≤native1.01。

同时报告native-good→lost与native-bad→rescue；不能只减少lost却不增加change覆盖就称突破。满足上述条件才进入V。最终major仍额外要求每seed相对native change ADE/FDE各≥5%改善及其校正区间等原全部门，不降低。

若cosine变好但覆盖不变、或FIT更好而outer不改善，就拒绝这一优化方案；不继续调投影顺序、group定义或参数。若motion失败，同样停止，不临时加penalty。

## 7. V：相同机制，fresh时间与场景隔离验证

只在U1通过后，先沿用已有recording-time grouped精确ID（train16104/outer566/inner2134）做普通/投影×三seed；通过相同门才做scene-family non-zara train16337/outer1561/inner1773。CAL2879均posthoc。参照ID来源：[time](../GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment2_recording_time/DESIGN.md)、[scene](../GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment3_scene_family/DESIGN.md)。

每个split重新FIT-only标准化、fresh网络与optimizer；不复用target训练权重/统计，避免训练身份进入新的outer。旧F16/cache仍见过旧数据，只是newhead迁移。只复用ID，不沿用旧80epoch、91596参数、physical-projection或loss混合包。

time隔离不是target/context独立；source-index不是秒；scene只有zara一个族。跨split不将不同outer ID强配对、不合并为独立行人。若time失败不自动scene，任何阴性不自动重启MoE/固定责任/旧M2。

## 8. 指标、统计、审计与成本

共同完整候选：J=min_k max(ADE/.2,FDE/.4)，good=J≤1，同一slot；报告Q/J分别选的同路径早中晚、rescue/lost、全K20原SciPy complete-link .10meanpath/.20endpoint有效good簇。独立minADE/FDE不拼接。五motion原12v/11a/10j、dt.4、全窗口×K×时间pooled，非窗口p95平均。所有组全部保存，不降低1%门。

U1、V-time、V-scene各B−A/B−native的outer change C/ADE/FDE/J，共24primary family；20,000次PCG64 seed20261015，target/scene整targettrack×三seedcrossed配对，time整timeblock×seed；Bonferroni两尾.05/(2×24)，未运行阶段不缩family，95%仅描述。D0不放入确认性family，不将局部触发当独立显著结论。历史适应性探索与3seed限制继续存在。

实际GPU预检必须包含：普通分组和原loss全梯度一致、投影解析fixtures/零组/零范数/顺序、实际有限步与allfinite、全部paired/frozen/16identity/bounds；每臂真实513FIT+12inner子集两epoch连续/中断ALLparams_buffers_optimizer及TorchCPUCUDA/loader/NumPy/Python RNG/history EXACT。只有子集通过，不能写全FIT两轮恢复。

每正式阶段6×4split=24finaljobs，6个checkpoint、150history；source/split/stats/init/objective/gradient-rule/order/optimizer/RNG完整manifest绑定。新增只新目录，旧科学/模型/输出/失败全SHA保留；锁源后不得静默改。保存后独立CPU复算质量/全聚簇/motion/几何/CI/来源，非所有模型梯度再回放或外审。

D0与每个双臂正式对照分别实测估时，planning>30min则停止该项预算门，不削减审计/epoch或默认沿用旧长预算。没有当前实测ETA。process private6GiB、系统physical/commit各reserve6GiB、单模型有界输出，不改OS/pagefile、不kill用户程序。未来唯一0trigger OSworker、一次startup/预计半程/完成定点，不高频轮询旧PID。当前没有worker或自动化动作。

## 9. 结果树与反方结论

|观察|能支持什么|不能支持什么|下一步|
|---|---|---|---|
|负cosine，但有限步无跨track损害|局部方向冲突存在|共享映射导致失败|停止该干扰子分支|
|有限步有损害，投影只让学习停滞|存在局部权衡|缓解机制有效|停止，不正式训练|
|D0改善、U1无coverage收益|局部效果不能外推整训练|训练已理解运动|停止|
|U1安全改善、迁移失败|这一人口内优化有价值|普遍泛化/独立场景理解|保留反例，停止自动扩张|
|全部通过且major/相关性审查满足|共享更新优化有实际可迁移价值|唯一root或心理意图|再讨论新基线，而不是自动动scorer|

ARS同助手设计检查：DA1保留position3可达范围限制；DA2保留T1阴性、CAL反例与支持内不等于条件分布一致；DA3用真实有限步和fresh迁移约束cosine解释。文档可作为未来实施方案，不是外部评审PASS或实验预检PASS。它明确覆盖共享映射的一个子机制，不能假装已经解决整个泛化分支。
