# 01｜证据基础与共同科学合同

状态：DESIGN_ONLY。所有新门槛是本系列事前规定，不追溯改变旧实验结果。

## 1. 为什么不是继续增加历史输入或改 scorer

### I1：均值预测能力不等于候选支撑能力

来源：[I1 RESULT](../MOTION_INFORMATION_SAFE_REACHABILITY_DESIGN_20261016/I1_execution_authorized_20261017/RESULT.json)、[统计](../MOTION_INFORMATION_SAFE_REACHABILITY_DESIGN_20261016/I1_execution_authorized_20261017/STATISTICS.json)、[保存算术审核](../MOTION_INFORMATION_SAFE_REACHABILITY_DESIGN_20261016/I1_execution_authorized_20261017/ARITHMETIC_AUDIT.json)。

- outer change 共 253 窗口、68 tracks。真实历史 H 的 mean20 MSE=1.009374，末速度占位 S=1.015272，CV=1.417287。
- H 相比 S 仅改善约 0.581%；H−S 的 family2 区间为 [-0.046206, 0.045750]，跨零。不能认定新增显式历史有可靠边际迁移收益。
- H 相比 CV MSE 低约 28.78%，但 S 也低约 28.37%。P2/native 本来含历史，S 不是“无历史模型”。
- outer change 原生覆盖 110，H 只有 38/42/35；CAL change 原生 243，H 只有 56/77/46。outer change 的候选 distinct 约 3.56/4.21/3.72，原生约 19.72。
- S/H 的 96 个 seed×split×group 五运动学检查都通过，覆盖却严重下降。

因此，“更安全、更小平均 MSE”不足以判定候选生成改善。全20路径都拟合同一 GT 的回归目标具有趋同倾向，这是目标局限，不是已发现程序 bug。也不能反推历史无信息或所有生成模型都失败。

### F0：没有通过的见证，不等于安全域为空

来源：[F0 分析](../MOTION_INFORMATION_SAFE_REACHABILITY_DESIGN_20261016/F0_execution_solver_tree_recovery_20261017/RESULT_ANALYSIS.md)、[保存复核](../MOTION_INFORMATION_SAFE_REACHABILITY_DESIGN_20261016/F0_execution_solver_tree_recovery_20261017/completion_review/REVIEW.json)。

640 次求解：616 infeasible、2 inaccurate、18 optimal、4 SolverError。18 个 optimal 中 14 个因原始控制范数超单位盘而拒绝；4 个生产精度重构后五运动学通过，但 ADE/FDE 越界约 1.47e-8～1.31e-7 米，仍严格拒绝。正式 accepted=0。

原目标 min acceleration 没有主动追求质量余量，固定缩放及 float32 重构可能破坏边界解。W0 只解决这个有限提取问题；不是替旧结果放宽 epsilon。616 infeasible 也不能全部归因浮点，更不能外推到更大的 pooled 安全域。

### T1/T2：训练内改善存在，保留与迁移没有一起解决

来源：[T2 全系列报告](../GENERATION_TRANSFER_RETENTION_DESIGN_20261016/T2_execution_manifest_recovery_20261016/RESULT_ANALYSIS.md)。

原 A 在 FIT change 从 1937 提升到 2171/2197/2161，但 outer 为 107/105/113，原生 110；CAL 为 221/214/225，原生 243。T1 的 success-keep 减少部分 lost，也同时减少 rescue，迁移信号未成立。T2 速度重表达仍没有可靠收益，acceleration mean 不合格。

本系列非负责路径几何保护与 T1 的“保住至少一条旧好路径”不同：前者对所有非负责 slot 约束相对自身 native 的漂移，不只保护已经成功的窗口；它也可能过度抑制修复，必须接受失败。

项目历史还记录了 soft-set/energy-mixture 负结果，以及旧架构上 native winner 切换仅约 FIT 3.3–6.7%、CAL 1.6–3.9% 的描述性检查（见项目 `.memory` 对旧诊断的记录）。这些不是本架构全训练期的切换率；但足以反对预设“频繁换 winner 就是主要根因”。P1 测的是责任政策效果，不是验证一个已成立的根因。

## 2. P1/P2 共用模型，不追随 I1 改目标或输入

| 项目 | 固定合同 |
|---|---|
| 输入 | P2 observed 96 + immutable native path 24，共120；每个 slot 使用同一共享 head |
| 网络 | 120→128→64→24，GELU，25,304 个可训练参数；末层零初始化 |
| 候选 | 全20原生候选，一一对应输出20条，不删槽、不加K、不改原slot顺序 |
| 输出 | float32 head；控制及 native 加法采用原 U0 已验证 float64 路径 |
| 控制 | u[t]=tanh(z[t])/max(1, L2(tanh(z[t]))) |
| 积分 | r[t]=(.20/144) Σ(q=1..t) [2(t−q)+1] u[q]；p=native+r |
| 位置界 | 每时刻 L2(r[t])≤.20(t/12)^2；不是五运动学保证 |
| 训练 | AdamW lr=.001、wd=.0001、betas=(.9,.999)、eps=1e-8、foreach=false；batch256、shuffle、25epoch、无grad clip |
| 随机性 | seeds42/43/44，配对初始化、batch IDs/order、dtype、优化器及预算 |
| 选择 | 固定 epoch25 primary，inner/CAL 不选checkpoint、系数或臂 |
| 原模型 | 不训练/改写旧 F16、decoder、scorer；只用已认证 immutable cache，不冒称旧GPU全重放 |

P1 精确复用原 target-grouped IDs：FIT18909、outer1035、inner1856、CAL2879；change 分母为4725/253/497/700。统计只用 FIT。复用原位置输入 statistics 前必须证明 ID、特征表达式及源SHA一致；不满足则实现前停止核对，不能拿速度版或 I1 的146维统计替代。

数值顺序按已读原实现固定：输入float32特征转double后减mean/除std，再转P2 dtype送head；先执行float32 `tanh(z)`，随后转double做单位盘norm/归一化和积分。不能改成double tanh后再声称与旧U0精确同一前向。

P2 只换已注册的时间隔离 population，统计在 P2 自身训练16104 IDs上重新计算，两臂共用；详见04。不同阶段不能把不同ID当配对样本。

未来GT仅进训练目标及事后评价，不进入 head、推理路由或候选选择。group/ID/track/scene 不进 forward。候选责任标签只属于训练 loss；没有 GT-free 的“自动挑对 owner”承诺。

## 3. 每一条结果必须采用的质量定义

对同一完整 slot j：

- ADE_j=12步欧氏距离均值，FDE_j=末步欧氏距离。
- Q_j=ADE_j+.5FDE_j，用于训练责任。
- J_j=max(ADE_j/.2,FDE_j/.4)，J*=min_j J_j。
- good_i=1[J*≤1]；不能把不同候选的 minADE 与 minFDE 拼成成功。
- minADE/minFDE 各自仍报告，但其最优 slot 可以不同，不能代替 good。
- rescue=N(native失败且新成功)，lost=N(native成功且新失败)。rescue率以原失败数为分母，lost率以原成功数为分母；净coverage变化=(rescue−lost)/全部窗口数。

原 SciPy GT-free complete-link D=max(meanpath/.10, endpoint/.20)、cut1，全K20聚簇；effective-good/distinct 作为第二层，不以越多越好，不靠复制候选。scorer 暂不改，原 logits 不是新路径概率。

五 pooled motion：speed p95、acceleration mean/p95、jerk mean/p95；dt=.4，逐seed、逐split、overall/stable/transition/change 全报。评价采用原差分起点及分位数实现，严格≤1.01×native，native零分母直接用绝对不等式，不加评价 epsilon。

## 4. 最终重大成功标准不变

预先指定 D 的 primary outer change：每seed覆盖至少比native高5个百分点；minADE、minFDE各降低至少5%；相应校正区间支持改善。overall/stable/transition 的 ADE/FDE/J 不得比native恶化超过1%；各组原生成功丢失率≤1%；全部四split、所有seed/group五运动学都合格。coverage区间family3、三个error区间family9的保守口径不缩小，详见05。

最终重大门的保护/lost/五motion须检查全部四split、每seed及每组；01所述primary覆盖及误差收益门针对outer change。还须通过 P2、同路径审核、相关性与仅3seed充分性检查，才讨论下一阶段验证；不是直接升级baseline。P1 的“机制候选门”低于重大成功门，是为了决定是否值得复核，绝不能改名为突破。

## 5. 范围与伦理

只用既有获准轨迹数据及去标识ID；无新受试者招募、无身份重识别、无外部下载。沿用数据许可，不凭本设计宣称已获得新的伦理审批。报告属于计算实验/基准研究，不冒充临床试验或已公开注册研究。

这些是基于已见结果制定的本地预注册设计，非盲态独立确认性检验。三scene、重复窗口、F16已见原pool、outer/CAL多次使用限制始终保留。
