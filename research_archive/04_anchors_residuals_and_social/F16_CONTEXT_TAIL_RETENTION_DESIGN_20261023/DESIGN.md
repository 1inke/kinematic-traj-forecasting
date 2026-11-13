# 实验设计：固定预算候选生成的相对退化控制

版本v1，2026-10-23；仅设计，所有新实验结果均未产生。

## 1. 证据起点与待检验问题

前序A为82008参数上下文解码器，不使用教师损失。inner整体native932/1856→982/1021/1021，变化组166/497→186/191/192；变化组救回53/62/63、丢失33/37/37。旧成功丢失约20%–22%，但不是全部平均误差恶化的主要来源。

三个seed平均inner变化组minADE净退化+.033455m，minFDE+.077077m；共同失败子集贡献分别+.036911m、+.084388m，其他子集合计部分抵消。共同失败数量278/269/268，不能称为“少数极端异常点”。

native的加速度集中于v5−v4和v9−v8，而新A分散到更多时间步。因此accel_mean增加、accel_p95和jerk降低可以共存；这既不证明新模型更差，也不证明更真实。原注册五项运动学门仍然失败，不作追认修改。

本系列三个问题：

1. D0：退化主要是终点方向、幅度、沿途形状还是时序问题？是否同样发生在FIT？
2. D1：新增运动变化是否与GT对应，还是与GT不一致的额外弯曲/变速？
3. T1：仅增加训练期原生相对质量惩罚，能否降低退化而不抹掉覆盖收益？

T1是一个待验证优化假设，不预设一定有效，不把之前“替换成joint winner损失”的失败隐去。此处保留原Q主目标，新增相对native的单侧J惩罚；它仍同时影响梯度分配与目标几何，不是纯责任分配因果实验。

## 2. 范围和固定条件

- 仅当前FIT18909和inner1856，原IDs、track、原stable/transition/change分组、坐标和dt=.4不变。不得读取outer、original_CAL、cal_v11或额外测试人口来选方案。
- 输入210维：过去8点16维、最后观测速度2维、最后token96维、8token均值96维。保持实际旧源的float32特征生成与FIT-only统计。
- 固定210→128→128 GELU、20×16查询、144→128→128 GELU、128→24零输出层，82008参数。每窗输出20×12×2，不改变候选预算，不拼K40。
- 冻结F16及所有旧decoder/scorer，T1只训练新学生。部署不运行native decoder或KIN；native参考只在训练loss与事后评价使用。
- 固定seeds42/43/44，8epochs，batch256，AdamW lr=.001、wd=.0001、clip_norm=5，无scheduler，epoch8为唯一主checkpoint；不靠inner挑epoch。
- 初始权重、FIT-only统计、每epoch顺序与旧A按seed锁定。两臂fresh初始化相同；不从旧epoch8继续训练，不warmstart旧模型。
- 原Q主损失与190对endpoint线性hinge .3、权重.05逐项保持，不能混入此前平方hinge、4slot、物理投影或class balancing协议。

该设计并未隔离“context vs容量”，不支持声称上下文是唯一收益来源。未来若做匹配信息消融需另设方案；本系列不增加这条训练分支。

## 3. D0：保存输出的误差退化定位

### 3.1 数据和逐窗质量

只读取旧三seed×FIT/inner的完整saved paths、GT、native、ID、track与原分组；必要时读取同源past作观测方向参照。无Torch、模型前向或训练。逐slot重算并对应保存字段：

\[
Q_{ij}=ADE_{ij}+0.5FDE_{ij},\quad
J_{ij}=\max(ADE_{ij}/0.2,FDE_{ij}/0.4),\quad
J_i^*=\min_jJ_{ij}.
\]

成功为J*≤1。minADE、minFDE分别报告，但不拼成成功轨迹。所有同路径诊断使用一次选定的J最小slot，exact tie取最小原slot；早中晚不能重新选择slot。Q-winner仅作完整另列的敏感性，不能挑其中更好值。

### 3.2 分解和尾部

每split×seed×原组输出共同成功、救回、丢失、共同失败四类的N、track数、mean/median/p90/p95、质量变化、对全组误差变化贡献。贡献必须加总等于全组变化。

另存误差增大部分与减小部分的总量，固定报告最坏1%/5%/10%窗口对正退化总量的贡献。排序仅用于事后分布描述，不删窗口、不训练重采样、不以最坏subset作为主结果。分母为0则NA，不用epsilon制造百分比。

### 3.3 同路径方向、幅度、形状

设GT终点向量g=y12，预测终点p。对||g||≥.2m：

- 有符号沿向误差：(p−g)·g/||g||，正值为沿GT终点方向超出；这不等于全部语义overshoot。
- 横向误差：二维cross(g/||g||,p−g)，同时给绝对值。
- endpoint方向角：仅||p||、||g||均≥.2m时定义，其他保存undefined及分母；不将零位移判错方向。
- 幅度误差：||p||−||g||，全部窗可定义；与沿向误差不要混用。
- 形状残差：e_t=(p_t−y_t)−(t/12)(p12−y12)，报告mean||e_t||，保留原始ADE；它是去端点线性误差诊断，不是修正后的部署轨迹。
- 早/中/晚分别1–4、5–8、9–12步ADE，固定同一J-slot。

全部类别允许重叠。不得将几种连续量强制分成互斥“根因占比”，不得因某一量最大便宣称模型不懂转弯。

### 3.4 观察状态分层

以最后观测速度分固定区间[0,.1)、[.1,.5)、[.5,1)、[1,+∞)m/s，另按sequence描述。分层不进入训练，不更改原loss组，不使用inner结果重新定边界。每层给n与track数，小层只描述。

### 3.5 D0→T1入口

除source/ID/数值正确性外，要求inner变化组三个seed的共同失败子集对全组minADE或minFDE的贡献均为正，且其平均贡献至少一项为四类中最大的正贡献。该条件已由前次局部inner复算支持，本轮作用是完整固化，不是独立新发现或新的显著性检验。

不满足则不自动训练P_relative，报告为何当前假设不成立；不得在D0后改惩罚定义来适配结果。

## 4. D1：新增运动是否符合真实时间过程

与D0共用同一遍读取，不重跑模型，不生成新候选。

### 4.1 两类评价严格分开

**集合门：** 原ALL-K20、全人口pooled的speed_p95、accel_mean、accel_p95、jerk_mean、jerk_p95，dt=.4，12/11/10项，linear quantile，literal≤1.01native不变。

**解释指标：** 固定J-slot与GT的运动向量误差，不用它替代集合门或作为部署slot选择器：

\[
E_v=mean_t\|v_t-\hat v_t^{GT}\|,\quad
E_a=mean_t\|a_t-a_t^{GT}\|,\quad
E_j=mean_t\|j_t-j_t^{GT}\|.
\]

另报告加速度幅值误差mean| ||a||−||aGT|| |、逐时刻加速度幅值，以及加速度向量误差的逐时刻分布。先比较原生和新A各自固定J-slot，再对全部20槽平均同样指标作补充；后者不是候选分布似然，不能要求20条都匹配唯一GT。

### 4.2 时间质量分布

对11项加速度幅值归一化为时间质量分布u、uGT，定义时间一阶运输距离：

\[
W_{time}=.4\sum_{k=1}^{10}|CDF_u(k)-CDF_{uGT}(k)|.
\]

两个总加速度恰为0时记0；只有一方为0时NA并另报比例，不加floor归一化。此量忽略幅值，必须和E_a及幅值误差一起解释。GT来自有限采样坐标，差分放大噪声；保持原坐标，不临时平滑。它不是意图/onset语义真值。

### 4.3 开始处的衔接

从同源past计算最后观测速度，另报||v1−vobs||和首个预测速度方向差（两向量模长≥.05m/s才定义角度）。这是附加诊断，不能偷偷替换原11项加速度中的任何一项。

### 4.4 固定判断语言

- 三seed同路径E_a均低于native而集合门失败：写“与所选GT路径的运动一致性改善，但原非劣约束未满足”，不能改门宣布安全。
- 三seedE_a均高于native：支持继续定位错误运动变化，不能直接认定高频抖动或加速度幅值越小越好。
- seed方向不一致：写混合证据，不强选结论。
- 变化组J-slot不能代表所有槽；三个seed/许多窗口不等于三个独立场景。

D1只决定解释，不从多指标中挑一个最差项自动加入训练；T1定义在读取D1结果前固定。

## 5. T1：一个训练因素——原生相对同路径质量惩罚

### 5.1 两臂与唯一差异

|臂|损失|其它|
|---|---|---|
|A_replay|原A的L_base|完全相同|
|P_relative|L_base + L_rel|只增加下式|

每臂3seed，最多6fresh runs。旧A保存输出是历史基准与对齐检查，不替代当前配对控制。运行源/环境不兼容时不能悄悄拿旧A充当fresh对照。

原生参考的J*只由固定FIT native20与FIT GT计算并detach：

\[
L_{base}=mean_i\min_j(ADE_{ij}+0.5FDE_{ij})
+0.05\,mean_{i,j<k}[0.3-\|p_{ij,12}-p_{ik,12}\|]_+.
\]

\[
L_{rel}=0.2\,mean_i[J_i^*(student)-stopgrad(J_i^*(native))]_+.
\]

λ=1固定；0.2为将无量纲J换回米量级，不是新的评价阈值。无组内分别归一化、hard-case采样、margin、temperature、保护系数扫描。所有FIT窗口参加，native失败的窗口也参加；不使用旧A的lost/neither标签训练。

### 5.2 为什么这样设计

零惩罚意味着该训练窗student J*≤native J*。若native成功，则student同样成功；若native失败，则限制连续J进一步退化。它不强迫复制原生坐标，不必在推理运行native，不增加部署参数/候选/分支。

但这是**零惩罚时逐窗的数学性质**，有限训练中不能保证零惩罚，更不能保证inner满足。更小J也不保证minADE与minFDE各自均降低、所有模态保留或五项运动学通过，必须实际评价。

### 5.3 与已有失败路线的区别和边界

- 不替换原Q主目标；不只是将winner换成J，不复刻原四slot position3实验。
- 不蒸馏KIN或native坐标，不新增轨迹教师集合；这是训练期参考性能惩罚。
- 仍是一种相对质量重加权，不能因数学形式新就宣称新算法创新或唯一根因。
- 不同时加物理、平滑、保留分类器、残差锚定、后处理或native20+new20组合。
- 不能只评价通过质量筛选的轨迹；全部K20运动学必须保留。

### 5.4 主终点与工程保留门

主终点：inner变化组R=mean[max(0,J*_student−J*_native)]，比较P_relative−A_replay。

局部机制支持条件全部满足才记MECHANISM_SUPPORTED：

1. R三seed均下降，(mean_seed R_A−mean_seed R_P)/mean_seed R_A≥20%，且主配对track/seed探索95%CI上界<0；固定seed敏感性CI上界亦<0。分母为0记NO_POSITIVE_REGRESSION_TO_REDUCE，不能判改善。20%是本次事前工程筛选目标，不是历史已证最小有效差或正式功效结果。
2. inner整体和变化组，各seed保留至少90%的A_replay相对native正覆盖净增：C_P−C_N ≥ .9(C_A−C_N)。若控制净增≤0，禁止除零或以此判通过，记CONTROL_GAIN_NOT_REPRODUCED。
3. inner整体及变化组，各seed的native-success lost率不高于A_replay。
4. 同两组，各seed minADE与minFDE不高于A_replay的1.01倍；不允许降低J却显著损害另一个误差指标。

工程筛选不是baseline升级。仍全量报告FIT/inner四组每seed绝对指标，不因某组差而删组。

### 5.5 控制一致性

A_replay使用旧A绑定的初始状态、顺序、统计和相同有效实现。固定256计算批次和原环境；对最终paths/history逐项验证。预检需证明新增统计不改变控制loss/全部梯度；RNG独立于诊断。

若控制最终未bitwise复现：保留新结果，先记录环境与来源差异，不反复训练求吻合；状态CONTROL_REPRODUCTION_UNRESOLVED。不得使用与旧A预定义subset的强比较或作“只改唯一因素”的历史归因，直到来源被解释。fresh A/P内部配对证据也必须带该限制。

## 6. 统计与同ID解释

### 6.1 固定分析人口

主推断仅inner变化组，全部窗口，无按GT删例。报告n_window、n_unique_sequence_target_track、各seed与场景构成。3seed只是优化重复；上游F16已见inner且inner反复探索，不称独立全模型泛化、确认性检验或可靠scene级CI。

D0/D1各子集是结果定义的事后描述，不对它们堆砌显著性。T1附加报告旧A/native事先冻结的四类subset上P的变化；不可按P重新划四类后伪称同人口改善。主结果始终ALL。

### 6.2 主bootstrap

20000 draws，NumPy default_rng(20261023)，sample单位sequence+target_track。一次draw从inner全track全集等概率有放回抽同样数量track，其权重共享全部臂、seed、组；同时配对有放回抽3个seed索引。每seed按抽样窗口数加权算指标，再取seed均值，最后P−A。不要给每臂独立抽样，也不要把3×1856视为独立窗口。

主95%percentile CI [.025,.975]；固定三个seed不重抽的track-only敏感性另列。该bootstrap不能解除同录制/邻居上下文相关与小seed数问题。

每组实际track<20或zero-denominator draw比例>5%时记INSUFFICIENT；不补0、不随意换组或抽到成功为止，不能通过CI门。非有限数视数值阻碍，不作为离群点删除。

### 6.3 次级family16

比较P−A、P−native × inner overall/change × coverage/minADE/minFDE/该参考success条件lost率，共16项；Bonferroni percentile [.0015625,.9984375]，与未校正描述性区间区分。lost使用对应参考成功数作分母。全部16项报告，不以某项显著代替主终点。

与GT运动一致性、每scene、速度分箱和错误分解仅描述；本系列不再增加隐性显著性搜索。统计软件版本在执行时读取并写入，不在设计中虚构已测数值。

## 7. 结论分支：不把局部机制和完整成功混起来

|结果|结论及下一步|
|---|---|
|D0来源或指标不能复现|STOP_SOURCE_OR_METRIC，先保留并解释；不训练|
|D0入口不满足|NO_SUPPORT_FOR_RELATIVE_REGRESSION_TEST；停止此预定训练，不换目标求通过|
|P只改善FIT，inner无明确改善|TRANSFER_UNRESOLVED；不加epoch/扫λ|
|P降低R但覆盖净增保留不足|COVERAGE_REGRESSION_TRADEOFF；不可宣布互补成功|
|P局部门通过但原生五motion仍失败|MECHANISM_SUPPORTED_BASELINE_NOT_READY；保留收益，不能自动蒸馏/scorer|
|P恢复旧成功但共同失败误差仍坏|RETENTION_ONLY，说明相对J目标未充分处理连续误差|
|P完整公共基线门通过|BASELINE_READY_FOR_SEPARATE_COMPLEMENTARITY_DESIGN；另立B/E或其它方案，不在本包自动执行|

完整公共基线门沿用旧G_base：FIT/inner四组、每seed，coverage≥.99native；minADE、minFDE及五pooledmotion≤1.01native。原major还要求变化组ADE/FDE各≥5%改善、protected总体/stable/transition≤1%伤害、3seed一致、联合CI与同路径覆盖、相关性审查等，不能被本局部门替代。原生成功lost≤1%的更严格保留要求单独报告，不追认成上一S1的门。

## 8. 实施边界、资源和结束

本次仅文档。未来执行D0/D1先测真实128窗读取/算术/保存吞吐，T1先测4warm+32throwaway训练batch与完整保存审核开销；所有模型丢弃、不选结果。不能把此前9m36当新承诺，更不能只用31秒训练时间估总时长。

按现行任务inventory估E、1.3E和动态剩余physical/commit、Windows retained峰值守卫；不恢复过时固定30min/6GiB/六小时限制。禁止OS/pagefile调整、kill用户程序。预计成本过大先报告阶段范围与成本，不擅降epoch或砍审核。

获得执行请求且预检PASS后才按唯一0trigger独立worker、一次startup、一次预计elapsed50%检查、一次修正completion定点流程运行。完成必须worker/exit双成功+保存独立审核，再报告；不能重复旧PID、旧任务或旧half节点。无变化静默。本包最多六run后必须总结，不无限延伸。

## 9. 同助手反方检查记录（设计阶段）

1. 范围检查：共同失败已知退化不是新独立证据；D0仅固化并定位，禁止称确认性发现。旧A同容量上下文因果仍未解决。
2. 方法检查：J非劣不能保证独立ADE/FDE或物理安全；已加入完整报告、反退化门和旧门，不以loss下降替代结果。
3. 结论检查：相对native惩罚也可能限制真正新行为、损害收益；固定90%覆盖净增保留、报告负例并停止扫参。GT运动一致性是事后oracle诊断，不等于部署识别或运动理解。

以上按academic-research-suite及nature-statistics进行单助手设计核查，不是外部审稿、实际预检PASS或公开预注册。主要文献新颖性不属于本次任务，也未做外部检索。
