# 01｜证据与共同合同

状态：DESIGN_ONLY。所有PASS、触发门和输出文件均为未来执行要求。

## 1. 新设计由哪些结果推动

change完整好轨迹要求：同一候选ADE≤0.2m且FDE≤0.4m。三数对应seed42/43/44。

| 人口 | native | T1 A原目标 | T1 B加keep | T2 C可逆速度输入 |
|---|---:|---|---|---|
| FIT / 4725 | 1937 | 2171/2197/2161 | 2149/2187/2165 | 2181/2224/2164 |
| outer / 253 | 110 | 107/105/113 | 112/111/108 | 108/111/109 |
| inner / 497 | 166 | 173/173/174 | 169/171/174 | 166/168/169 |
| CAL / 700 | 243 | 221/214/225 | 232/232/234 | 223/230/229 |

- C−A outer覆盖均值仅+0.3953pp，family24 CI[-4.7812,+5.1908]pp；普通95%CI也跨0。不是仅多重校正造成阴性。
- C outer原成功lost为17/15/19，占原成功13.64%–17.27%；平均加速度增加12.554%–12.640%。另四项pooled运动学通过，不能写成所有运动学都崩溃。
- T1在FIT减少lost也减少rescue；outer保留迁移信号未成立。
- C outer overall、inner change等有正反例；不支持“所有样本都没有改善”。

### 避免重复的历史实验

| 已完成证据 | 能支持什么 | 不能据此推出什么 | 新系列区别 |
|---|---|---|---|
| A0：独立位置球all20 oracle，FIT1937→3108、outer110→166，但oracle运动学失败 | 位置界内存在质量机会 | 存在安全解，或全部安全解不存在 | F0保留实际积分耦合，并核验安全正例 |
| S0：33逐点导数约束+单射线，change零rescue，alpha几乎0 | 该充分安全映射非常保守 | 原五项pooled门下无安全修正 | F0允许20条路径联合变化，不采用单一ray alpha |
| M1：显式动态GRU+goal生成未优于静态对照 | 该端到端goal结构未证明历史收益 | 历史没有预测信息 | I1去掉best-of责任与生成可达域，仅检验直接读出 |
| G2：学习过程相对G1改善outer change，但仍低native且运动失败 | 时间过程有用，目标正确不够 | 安全生成和迁移已解决 | G1仅在前两项证据成立后进行完整候选检验 |
| M2：联合goal/process未可靠优于冻结goal | 该联合优化不足以解决问题 | 必须更深、更宽或更大MoE | 本系列不继续扩容量 |
| U1：overshoot只解释约11%–13%的outer lost | 不像主要是步长过大 | 已定位唯一表征/优化根因 | 不扫步长 |

## 2. 数据和冻结范围

- 只复用target-grouped：FIT18909、outer1035、inner1856、original_CAL2879；合计24679唯一ID。
- 原change分母4725/253/497/700，执行时验证每个ID、GT、groups、track、slot顺序，不只对总数。
- I1/G1只用FIT训练、拟合输入统计；outer固定epoch25为primary；inner/CAL仅posthoc，不选择checkpoint、结构、系数。
- F0仅读取FIT的轨迹和GT；不在outer/CAL寻找oracle正例。F0抽样是明确依赖native成功状态的诊断抽样，不是无GT部署策略。
- 冻结P2为原8个observed tokens均值96维；原生候选20×12×2、dt=0.4s。原raw past应为对齐的8×2局部观测，末点0；必须验证缓存与原source IDs映射。
- 旧F16不实例化、不新forward、不训练；只能声称缓存与相关旧文件SHA保护，不能声称旧全模型GPU重放。
- 真实历史只取这8个已观察点，不使用未来、groups、scene、ID或logits作为head输入。

## 3. 共同评价

\[
J_{ij}=\max(ADE_{ij}/0.2,FDE_{ij}/0.4),\quad good_i=1[\min_jJ_{ij}\le1].
\]

独立minADE与minFDE仍报告，但不得拼成一条合格路径。rescue/lost/retained均基于同ID；lost分母是native成功数，rescue分母是native失败数，覆盖差分母是该完整group人口。

候选评价按顺序：同路径覆盖 → GT-free有效good簇 → scorer。聚簇仍是全K20原SciPy complete-link，D=max(meanpath/0.10,endpoint/0.20)，cut1。五项pooled运动学为speed p95、acceleration mean/p95、jerk mean/p95；速度12步、加速度11步、jerk10步，严格沿用既有算子和quantile实现。每项literal≤同population native×1.01，没有评价epsilon或floor。

G1完整成功沿用原门：change每seed覆盖至少+5pp、ADE/FDE各至少5%收益、相应同时CI、overall/stable/transition≤1%伤害、原成功lost≤1%、全部五运动学每seed/group≤1%，以及原同路径/重复控制审核。首度全部通过仍需相关性与3seed充分性审核，不能直接称SOTA。

I1是无界诊断读出，不要求它作为安全生成器通过上述门；其结果不得参与baseline升级。F0是GT辅助正例，同样不是模型收益。

## 4. 资源与执行限制

- 采用T2修复后的task_memory_1p3_peak_v2口径：预先明列任务估计E，当前及Windows retained peak RSS/private均不得超过1.3E；不能只看当前内存，不能运行中抬E。
- 系统可用physical与commit各保留≥6GiB；“约17GiB可用”是规划背景，不把旧峰值当此次实测。
- 一个模型/求解器、有界batch/chunk；不得修改OS/pagefile、关闭用户程序或绕过拒绝。
- 单阶段预计包含完整审核>30分钟则不启动，不减少预注册样本/solver slots/epoch使其勉强达标。
- 本次只创建设计；若以后执行，每阶段新执行目录、source lock、真实预检、独立按需零trigger worker、一次startup、一次50%预计时间检查和一次完成节点。未执行不得报告ETA或running。

## 5. 原始结果来源

- [T2完整分析](../GENERATION_TRANSFER_RETENTION_DESIGN_20261016/T2_execution_manifest_recovery_20261016/RESULT_ANALYSIS.md)
- [A0位置机会](../GENERATION_FEASIBILITY_CONDITIONING_DESIGN_20261015/A0_execution_20261015/RESULT_ANALYSIS.md)
- [S0安全射线](../NATIVE_SUPPORT_WIDE_REFINEMENT_DESIGN_20261015/S0_execution_20261015/RESULT_ANALYSIS.md)
- [M1历史对照](../MOTION_STATE_GOAL_PROCESS_DESIGN_20261014/M1_execution_20261014/RESULT_ANALYSIS.md)
- [G2过程生成](../GENERATION_BRANCH_SCREEN_20261014/G2_frozen_goal_process_20261014/RESULT_ANALYSIS.md)
- [M2联合生成](../MOTION_STATE_GOAL_PROCESS_DESIGN_20261014/M2_execution_20261014/RESULT_ANALYSIS.md)

T1旧生命周期峰值超过当时1.3E却被guard误标PASS的偏差继续保留；不以T2审核通过清洗旧记录。
