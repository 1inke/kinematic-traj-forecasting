# 共同协议：固定什么、比较什么

状态：DESIGN_ONLY。任何“必须通过”都是未来执行条件，不是已PASS。

## 1. 数据与冻结范围

只复用U0原target-grouped划分：FIT18909、outer1035、inner1856、original_CAL2879，合计24679个唯一ID。原change分母分别4725、253、497、700；执行时逐ID核验，不能只对总数。

- FIT是唯一训练与标准化来源。outer固定epoch25为primary；inner、CAL只posthoc，不选checkpoint、表示、保留权重或阈值。
- 保留原stable/transition/change全部样本与原顺序定义，不删难例。所有结果逐seed、split、group报告。
- F16、原encoder/decoder/scorer、native20与slot顺序、缓存、既有旧权重/布尔buffer均不改。新head只读冻结P2和原生候选。
- 不使用新的DEV/test/fold/外部数据，不执行旧模型训练，不使用GT生成部署gate。对缓存与source建立SHA保护；不冒称未实例化的旧模型又做过GPU重放。
- frozen F16已见过旧数据；target-disjoint不等于context/time-disjoint；重复outer/CAL和适应性研究只支持探索，不是独立泛化或SOTA证明。

## 2. 固定架构与训练

以已核对的 `GENERATION_UNPROJECTED_TIME_COUPLED_REFINEMENT_20261015/U0_execution_20261015/model.py` 为科学参考，不导入会写旧目录的顶层代码。

共享head：120→128→64→24，GELU，25304可训练参数。120=冻结P2的96维+每条native路径的24维。每window完整K20，各slot共享head；无新slot embedding、gating、MoE、scorer。

最后一层zero-init，所有arm起始输出严格为native20；同seed权重逐tensor配对。输出raw为12×2，float32计算；控制量及最终轨迹float64：

\[
u_t=\tanh(z_t)/\max(1,\|\tanh(z_t)\|_2),\qquad
r_t={0.20\over144}\sum_{j=1}^{t}[2(t-j)+1]u_j,\quad \hat p_t=p^0_t+r_t.
\]

保持每步位置界0.20(t/12)^2、原完整K20、原槽位。不新增物理投影、fallback、残差缩放、滤波或原生候选拼接。该界不是运动学安全保证。

训练：seed42/43/44，25epochs，batch256，shuffle；AdamW lr0.001、wd0.0001、betas(0.9,0.999)、eps1e-8、foreach=false，无grad clip。相同样本批次与更新次数，未满batch保留；固定epoch25，无early stopping。每阶段按合同保存全部optimizer与Torch CPU/CUDA、loader、Python、NumPy RNG。

原目标：

\[
L_0={1\over B}\sum_i\min_{k\le20}(ADE_{ik}+0.5FDE_{ik})
+0.05\,{1\over190B}\sum_i\sum_{j<k}[\max(0,0.3-\bar d(\hat p_{ij},\hat p_{ik}))]^2.
\]

原Qwinner先取同一slot，ties按原physical slot升序。不能悄悄换为minADE与minFDE分别取slot；不能从早期bottom4目标混入loss。B/D仅增加T1指定一项，其余逐源码/标量/梯度核验。

## 3. 输入统计和信息合同

A/B使用同一FIT-only位置统计；C/D使用同一FIT-only速度统计。P2统计四者完全相同。候选部分统计分母为18909×20，P2分母18909；原float32 feature转float64累加，std<1e-6→1，不用outer/inner/CAL重拟合。

同算法不同数值的标准化属于T2表示干预的一部分，不能声称四者数值buffer完全相同；同表示对内必须exact。跨表示模型初始权重相同，但非零初始化测试输出不要求相同。所有统计绑定训练ID与feature版本。

## 4. 科学指标与分母

令每slot \(J=\max(ADE/0.2,FDE/0.4)\)，同完整路径J≤1才good。

1. 每split/group的覆盖率、native→arm的rescue/lost/retained/unresolved及IDs；记录 \(coverage_{arm}-coverage_{native}=(R-L)/N\)。
2. lost rate分母为native成功数；rescue rate分母为native失败数；不得把这两个率直接相减当coverage变化。
3. minADE/minFDE可分别作为标准oracle指标报告，但二者的slot可能不同，不能构成完整候选成功证据。同时报告minJ和J-closest单slot早/中/晚误差。
4. 完整GT-free K20聚簇：原SciPy complete-link，D=max(meanpath/0.10,endpoint/0.20)，cut1；再事后评估有效good簇、重复候选、distinct。不是越多越好。
5. 所有seed/split/group、完整候选人口的pooled speed p95、accel mean/p95、jerk mean/p95，dt0.4；全部≤immutable native×1.01，literal比较，不加epsilon。零native时报告绝对差并按原门，不造百分比。
6. U1同定义fixed原生j0、ALL20、原good集合G0的初始dJ与最终ΔJ；这些仅事后诊断，不能用于选择部署路径。FIT与outer不按不同ID强配对。
7. 整个FIT最终checkpoint也保存并重算，不以batch训练loss下降冒称人口coverage提升。

每run保存4split，共12科学jobs/cell。A/B/C为36jobs，若D执行则48jobs；每split native参考只保存一次，4个参考不混入科学job。18/27等旧任务job计数不套用。

## 5. 统计预注册与选择限制

outer change为唯一primary群体；A/B/C/D同ID、同seed配对。使用20000次PCG64(seed=20261016) paired target-track与3 modelseed bootstrap：抽取3个seed并有放回抽取outer tracks，保留track内全部窗口；所有cell/contrast共享draw。保存draw/ID顺序SHA。按抽取后的窗口加权计算各seed指标，再取seed均值；缺分母draw记invalid，不补抽。valid<19000则不能触发统计分支。

主family固定24项：六个对比B−A、C−A、D−A、D−B、D−C、交互D−C−B+A，分别评估coverage rate、lost rate、rescue rate、fixed-j0 actualΔJ。使用Bonferroni family24双侧区间，尾概率0.05/(2×24)；不因D未执行而缩小family。常规95%区间同时展示但不能替代主区间。

另设绝对效果family9：B/C/D相对native的outer change ADE、FDE、minJ，Bonferroni9区间。最终重大成功必须同时满足这组和下述所有门；不把多个family任一阳性包装为全系列显著。原U0区间仍原样保留，不能改其family重算以求通过。

inner/CAL及其他group逐seed完整展示，按同算法提供95%探索区间并明确非primary。CAL不参与T3启动选择，也不选任何超参数；明显反例不得隐去。没有新holdout，因此整个系列不能变成确认性证据。3seed和68个outer change tracks限制统计能力，宽CI即“不确定”，不是等效或无作用。

## 6. 三种判定必须分开

- **执行通过**：源/ID/恢复/算术/完整输出全部通过。
- **机制信号**：T1/T2预注册的局部判定通过，可支持一次组合检验；仍可能motion FAIL，不得升级模型。
- **重大模型成功**：outer change相对native coverage每seed≥+5pp，另对arm−native coverage给family3(B/C/D)区间下界>0；family24中的arm−A改善不能代替这个绝对证据。change ADE/FDE每seed各改善≥5%，family9 ADE/FDE/minJ差上界<0；overall/stable/transition误差伤害≤1%，native-success lost≤1%，全部五运动学门≤1%，三seed一致，完成相关性审查。所有split/group完整披露，CAL失败不得称“跨场景成功”。

这套绝对标准保留既有严格要求；机制门不是放宽重大成功标准。没有通过时baseline/historycycle11/既有goal均不升级。

## 7. 预算、权限与来源

每阶段先测4warm+32measure吞吐与保存/聚簇/CI/审计IO，再报实测规划；>30分钟不自动启动，不通过削减epoch、seed或审核来压预算。此文不承诺预计分钟数。

采用最新[RESOURCE_POLICY.json](../RESOURCE_POLICY.json) task_memory_1p3_v1，不恢复旧固定6GiB门：启动physical与commit各≥1.3E；运行中分别要求余量≥1.3max(E−本任务RSS/private,0)，实际进程占用>1.3E停止重估。E含参数、optimizer、缓存、有界批次、压缩临时文件与审核峰值。不得改OS/pagefile或关闭用户程序。

未来执行须独立新目录、source/input/output SHA、全旧相关digest继承不覆盖；先完成预检再锁源。只在被明确要求执行后启动唯一零trigger worker，ONEstartup→ONE预计50%→ONE真实完成核验。本文不创建、修改或恢复自动任务。
