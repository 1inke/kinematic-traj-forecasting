# 下一系列实验：固定K20端点修正的集合效用与成功保留监督

日期：2026-10-30  
状态：**DESIGN_ONLY，仅设计与核心代码接口，尚未实施。**  
建议执行目录：`qoder_work/ENDPOINT_SET_UTILITY_RETENTION_EXEC_20261030/`

## 0. 执行摘要

本轮不再寻找新轨迹结构。固定当前4770参数H网络、观察输入、END49动作、K20预算和安全过滤，只改变训练目标，依次比较：

| arm | 目标 | 回答的问题 |
|---|---|---|
| M | 原候选平均参数MSE | 原始对照 |
| R | MSE＋窗口成功保留 | 能否减少关键成功支持被修坏？ |
| RS | MSE＋窗口成功保留＋原失败窗口救回 | 能否在保留基础上，把收益变成完整覆盖？ |

每个arm固定seeds42/43/44。先在已封存tiny128上各训练256更新，共9个短诊断模型；只有预定tiny效用门通过才进入3 arms×3 seeds的完整train3103、8epoch比较。不能挑一个seed或最好的arm临时进入下一阶段。

本轮唯一实验因素是**监督目标配置**。R−M定位保留项，RS−R定位救回项，RS−M评价完整目标组合。RS−M包含两项变化，不能把其改善全归因某一项。

TIME继续STOP，P3不触发；不训练旧F16/decoder/scorer，不新增候选、不换输入、不放宽运动门、不扫系数。本轮不检验H−G历史增量，原G/H实验仍为NOT_TESTED。

## 1. 为什么现在做这个实验

[最新归因报告](../qoder_work/ENDPOINT_SUPPORT_FAILURE_ATTRIBUTION_EXEC_20261030/RESULT_ANALYSIS.md)的固定128训练窗结果：

- Native/C连续/Q量化/S安全过滤/O安全oracle覆盖分别69/68/67/70/90；
- 最终7个lost均只有1个Native-good支持，且在连续C阶段已坏；其中5个支持的教师为identity；
- 36个唯一支持槽平均ΔJ=-.08819，反而变差；1257个原好窗失败槽及1180个原坏窗槽的平均J显著降低；
- 123个原好候选占2560的4.80%，唯一支持36槽占1.41%。这是均匀损失计权上的稀少性，不是已经测得的梯度冲突因果；
- 当前8个rescue若理想地全部保留，覆盖会计上限77，仍低于O的90。因此不能只保留、完全不救回；
- 安全过滤Q→S救3、丢0，不支持先放宽安全门。

本轮假设：**窗口级保留与救回监督，比单纯候选平均参数拟合更符合完整K20的实际用途。**竞争解释包括拟合容量不足、局部条件缺失、连续训练与离散执行不一致；新目标并不保证有效。

## 2. 与旧90% tiny门的关系：必须明确改变的是问题

旧任务研究“安全oracle参数是否可拟合、H是否优于G”，active MSE下降90%是其正式训练准入条件。当前实际下降80.3173%，`STOP_TINY_FIT_V2`永远保留。

本设计是归因后的**新监督机制实验**，不是恢复旧G/H任务，也不是将旧90%改成80%。它直接测试成功保留/救回，故另设事前效用筛查，原90%结果仍完整报告但不作为本新研究唯一入场条件。即使RS更有效但参数MSE更差，也只能称集合效用有改善，不能称旧拟合门已过。

未来要求“执行本文件”才代表同意这项明确的新研究合同。本轮文档交付不是执行授权，不创建worker、自动任务或训练目录。

## 3. 所有不变项

| 项目 | 固定合同 |
|---|---|
| 输入 | H：past8×2＋当前native slot12×2，共40维 |
| 统计 | 已封存train3103全20槽共同mean/std，不重新拟合、不用val/held |
| 网络 | 40→64→32→2，GELU＋tanh，共4770参数，末层零初始化 |
| 共享 | 20槽共享同一MLP，不加slot/ID/scene/logit/GT标签输入 |
| 训练teacher | 已审核safe END49逐槽最小J动作的归一参数u*=δ*/.4 |
| 优化器 | AdamW lr.001、wd.0001，clip norm5，无scheduler |
| 批次 | 32完整窗口，保留最后不足批，不按group/成功重采样 |
| seed | 42、43、44，三arm相同seed逐参数相同初始化、相同batch顺序 |
| 执行 | 最近END49动作；exact tie沿用identity、范数、索引优先；原guard失败则该槽Native |
| 候选预算 | 固定K20，slot顺序不变，无K40或GT选槽 |
| 评价 | 同路径ADE≤.2 AND FDE≤.4，原五motion及junction literal1.01 |

模型为float32；原MSE保持原float32计算路径。新轨迹几何/J损失用float64计算，梯度回传float32参数；R和RS的差别只能是救回项。M执行原标量与梯度路径，不能为了统一dtype重写M而又宣称bitwise复现。

对连续值、nearest-action、guard三端分别保存，不能把新损失定义的continuous输出冒充实际部署输出。

## 4. 新目标的精确定义

### 4.1 记号与连续代理

每窗i有20条原生轨迹p_ik，GT未来y_i；网络预测归一端点参数u_ik。训练用连续路径：

\[
\tilde p_{ikt}=p_{ikt}+\frac{t}{12}\,0.4u_{ik},\quad t=1,\ldots,12.
\]

定义同槽：

\[
J_{ik}(u)=\max\{ADE(\tilde p_{ik},y_i)/.2,\ FDE(\tilde p_{ik},y_i)/.4\}.
\]

原生J为J^N_ik。原好支持集B_i={k:J^N_ik≤1}，b_i=1[B_i非空]。从已保存safe teacher读出逐槽最佳J^*_ik，令q_i^*=min_k J^*_ik。identity存在，因此q_i^*≤min_k J^N_ik，数值复核不符则STOP。

**B、b、q*只由训练GT构造监督，不进入forward、解码或部署选择。**不按当前预测改变原好/原坏分组。

### 4.2 原MSE

\[
L_M=\frac1B\sum_i\frac1{20}\sum_k\frac12\|u_{ik}-u^*_{ik}\|^2.
\]

全20槽保留，不删除identity或失败槽，不改teacher。

### 4.3 成功保留项：保住至少一个原成功支持

对原好窗：q_i^N=min_{k∈B_i}J^N_ik，q_i^B(u)=min_{k∈B_i}J_ik(u)。定义：

\[
L_R=\frac1B\sum_i b_i\,[\max(0,q_i^B(u)-q_i^N)]^2.
\]

它要求原成功支持集合中至少有一条保持不劣于原最好J，而不是冻结所有原好槽。使用原q^N而非仅阈值1，保留已有安全余量，避免训练刚好贴着good边界。

这比“允许任意新槽补替原支持”的纯集合保留更保守，是本轮明确选择：检验**原生支持保留**，不宣称是唯一正确保留形式。本轮不增加另一个全K min保留变体。

原坏窗该项为0。分母始终为batch窗口数B，不除以当前batch原好窗数；空组自然0，不做动态权重放大。

### 4.4 救回项：推动原失败窗的完整路径接近可行机会

令q_i(u)=min_{k=1..20}J_ik(u)：

\[
L_S=\frac1B\sum_i(1-b_i)\,[\max(0,q_i(u)-q_i^*)]^2.
\]

目标q*来自原safe oracle，而非任意要求每个窗口都达到0或盲目强迫不可救回窗口good。仍包含原坏但safe oracle未达good的窗口：它们可降低误差，但单独报告，不能把其改善冒充救回。

每窗口至少一个同槽J被推动，不分别优化minADE与minFDE后拼接。实际梯度为当前minJ槽承担；不是强制预测“oracle slot类别”，不加新assignment分类器。

### 4.5 三arm与系数

\[
M:L=L_M;\qquad R:L=L_M+L_R;\qquad RS:L=L_M+L_R+L_S.
\]

所有量无量纲，新增两项系数固定为1，表示本轮一个明确配置，而非已证明最优平衡。记录三项数值、实际梯度范数及clip触发，不做梯度自适应归一、动态权重、温度或系数扫描。

若新增项尺度导致原MSE完全失去作用或训练不稳，应报告这一负结果并停止，而非中途调整系数。

### 4.6 非光滑、guard和连续代理的限制

- max(ADE/.2,FDE/.4)精确tie选ADE；min候选tie选最低physical slot，保持确定梯度；
- norm在精确0处采用框架合法零次梯度；hinge边界采用0次梯度，不用epsilon改变科学质量；
- 无softmin、straight-through、guard反向或把最近动作当可微；
- 连续u坐标各∈[-1,1]但二维范数可>1，路径可能越过.4圆域；不在训练中新增径向clip，因为那会改变结构映射；
- 新目标仅是执行效用的**连续代理**，不是实际guarded coverage的精确可微优化。参数越界、量化翻转、guard拒绝必须完整报告；若代理收益在S消失，本方案失败，不取消guard。

## 5. 核心代码设计

以下为损失接口伪代码，不是已经验证的生产脚本。必须在新目录实现并经过第8节预检；旧源码不改。

```python
import torch

def continuous_j(native, u, gt):
    # native [B,20,12,2], u [B,20,2], gt [B,12,2]
    envelope = torch.arange(1, 13, device=u.device, dtype=torch.float64) / 12
    paths = native.double() + .4 * envelope[None, None, :, None] * u.double()[:, :, None, :]
    d = torch.linalg.vector_norm(paths - gt.double()[:, None, :, :], dim=-1)
    ade, fde = d.mean(-1), d[..., -1]
    # dim max, not elementwise maximum: explicit ADE-first exact tie.
    return torch.stack((ade / .2, fde / .4), dim=-1).max(dim=-1).values

def objective(u, teacher_u, native, gt, native_j, safe_best_j, arm):
    mse = ((u - teacher_u)**2).mean(dim=(1, 2)).mean()  # original float32
    if arm == 'M':
        return mse, {'mse': mse.detach()}
    assert arm in ('R', 'RS')
    j = continuous_j(native, u, gt)
    nj = native_j.detach().double()
    best = safe_best_j.detach().double().min(dim=1).values
    support = nj <= 1.0
    covered = support.any(dim=1)
    ret = j.new_zeros(j.shape[0])
    rows = torch.where(covered)[0]
    if rows.numel():
        # No all-inf rows enter the min; do not multiply inf by a zero mask.
        jb = j[rows].masked_fill(~support[rows], float('inf')).min(dim=1).values
        qn = nj[rows].min(dim=1).values
        ret = ret.index_copy(0, rows, torch.relu(jb - qn).square())
    rescue = j.new_zeros(j.shape[0])
    rows_bad = torch.where(~covered)[0]
    if rows_bad.numel():
        qr = j[rows_bad].min(dim=1).values
        rescue = rescue.index_copy(0, rows_bad, torch.relu(qr - best[rows_bad]).square())
    lr, ls = ret.mean(), rescue.mean()
    total = mse + lr if arm == 'R' else mse + lr + ls
    return total, {'mse': mse.detach(), 'retention': lr.detach(), 'rescue': ls.detach()}
```

独立NumPy实现必须从native与GT重算native_j，不信任手写mask。部署函数签名仍只有past/native/parameters/actions，不得接收上述GT辅助量。批内混组、空组和候选并列均要保存测试。

训练循环沿用原批次及RNG逻辑，但新checkpoint要额外绑定loss版本、两个系数、q*与B来源SHA、三分量history、winner/component/activity记录和clip前梯度范数。不能仅替换loss函数而沿用旧source通过标签。

## 6. 两阶段实施与数量

### 6.1 阶段T：固定tiny上的监督机制筛查

- 人口仍为原SELECTION的128窗/97tracks；全部20槽和503个identity候选保留；
- M/R/RS各42、43、44，共9个fresh tiny；同seed初始化/每一批IDs完全配对；
- batch32、256updates=64epoch，每模型固定末256步，不选最佳epoch；共2304updates；
- seed42 M必须对旧H42的初始化、history、最终参数及保存数组进行兼容性复现。预期bitwise，仅当前源/环境精确匹配才宣称；不一致先定位，不把改后的结果当新随机重复掩盖；
- 原H42旧模型不续训，其他arm不热启动；tiny不能用于正式模型初始化。

tiny每模型报告原90% active-MSE状态，但新效用门在任何结果产生前固定如下。

**进入阶段F须全部满足：**

1. 三seed均R的lost≤M，且至少一seed严格减少；
2. 三seed均RS的rescue≥R，且至少一seed严格增加；
3. 三seed均RS的lost≤M、rescue≥M，且RS覆盖严格高于Native（不是仅低lost）；
4. 三seed均RS overall `lost/native_good≤1%`：tiny有69原好，此门等价lost=0，明确是严格筛查，不事后改成允许1例；
5. 三seed均RS在stable/transition/change及overall五pooled motion与junction mean/p95满足literal1.01；protected overall/stable/transition的minADE和minFDE不超过Native1%，change两者不得恶化；
6. 所有代码、数据、真实恢复、独立保存审核和资源前置检查通过。

这是小训练集的实用筛查，不是显著性检验，不证明迁移。tiny失败时结束本轮，保存三arm全部结果；不换128窗、不加轮次、不改系数、不尝试第四loss。若R好而RS坏，只能报告保留项有限阳性和救回项负结果，不能事后把R改为预注册主方法进入F。

为什么门严格：前一轮已出现明显训练内损伤；本轮先要求这套目标在原小样本上同时体现机制与非平凡效用，再支付完整训练/审核成本。失败不证明所有集合目标无效。

### 6.2 阶段F：通过T后才运行的完整训练

| 数据人口 | 窗口数 | change数 | 用途 |
|---|---:|---:|---|
| train | 3103 | 721 | 拟统计、监督与训练 |
| val | 993 | 256 | 固定final评价，不挑checkpoint/系数 |
| hold | 3825 | 1023 | 主评价，九final锁定后才开放本轮GT分析 |
| inner | 1856 | 497 | 次级迁移与保护检查 |

使用原split及ID，不重划。所有数据历史上已有探索暴露，不能宣称全新独立测试；本轮封存仅防本轮再调参。

- M/R/RS×3seed，共9个新正式模型；
- 全train、固定8epoch、batch32，每epoch97updates，末批31，776/model；共6984updates；
- primary固定final8，val/hold/inner均不选择模型；
- 所有9final/初始化/源/统计/teacher/loss/optimizer/全部RNG封存后，才计算本轮held oracle和结果；
- 输出9×4=36个模型人口，各有连续C、原量化Q与安全S三个端口；主结论只用S。

Native身份与原train全teacher均值基线各四人口，唯一保存，不复制seed伪装重复。原P0已完整的train/val teacher复用，hold/inner缺失教师需按旧END49安全合同新算并独立审计，不将缺失写成通过。

完整上限：9tiny＋9formal，模型优化更新9288；预检子集模型另列。不得自动加G、TIME或额外seed。

## 7. 正式评价：机制、迁移、部署分别结案

### 7.1 六个预注册主比较

固定主人口和方向，不在结果出来后换组：

| 编号 | 人口与比较 | 预期方向 |
|---|---|---|
| H1 | hold overall：R−M的lost/native-good | <0 |
| H2 | hold overall：RS−R的rescue/native-bad | >0 |
| H3 | hold overall：RS−M的coverage | >0 |
| H4 | hold change：RS−Native的coverage | >0 |
| H5 | hold change：RS−Native的minADE | <0 |
| H6 | hold change：RS−Native的minFDE | <0 |

使用整sequence|target track配对bootstrap，20,000 draws，RNG20261030，各arm共用抽样索引和配对seed抽样；另给固定3seed均值、只抽track敏感性。六项家族双侧Bonferroni分位点`.05/(2*6)`与`1-.05/(2*6)`，同时列普通95%描述区间。无分母的draw记invalid，有效不足19000则NOT_ESTIMABLE，不补0或补抽到显著。

这是新的六项效用家族，**不是旧END/TIME×MSE/regret/coverage家族的替代完成**。旧H−G仍未检验，历史试验多次探索不能由本次校正完全消除。

机制证据：H1/H2分别要求三seed同向且校正CI排除0；二者分开给状态。完整组合相对M效用：H3同样要求三seed正且CI下界>0。H4−H6完整保留，不以其中最漂亮的一项代替其他失败。

### 7.2 部署推进门全部保留

RS须同时满足：

- hold change coverage每seed>Native、均值至少+2pp、H4校正CI下界>0；
- hold及inner各group每seed lost/native-good≤1%；空原好分母标NA并说明无法证明，不平滑；
- hold/inner overall、stable、transition的minADE/FDE≤Native×1.01，change两指标不恶化；
- hold/inner各group每seed全部五pooled motion及junction mean/p95≤Native×1.01；
- inner change coverage每seed≥Native；
- RS相对M有H3组合效用证据，不只靠Native对比掩盖原M已能做到的改进。

原历史major change ADE/FDE各5%、相关性及全部保护门另列；本轮2pp推进通过也不是“大突破”或baseline升级。若仅保留项机制阳性而主部署未过，只报告部分机制证据，不交付成功模型。

### 7.3 不允许净增掩盖损伤

每arm/seed/population必须完整报告：

- Native good、rescue、lost、retained、unresolved；coverage=rescue−lost的净变化与分母；
- 原唯一支持、多支持、原好窗失败槽、原坏窗槽四类，再交叉identity/active；
- safe oracle可救回/不可救回原坏窗的实际捕获；不只报全部坏槽平均J；
- 参数MSE、精确动作命中、safe regret、非identity比例、参数范数越圆域、七guard位、C/Q/S阶段事件；
- 原SciPy全K20 complete-link `.10 meanpath/.20 endpoint`，distinct和effective-good，不将更多簇自动当更好；
- 同槽J最优完整路径早/中/晚误差；minADE与minFDE可能来自不同槽，要与同槽coverage分开；
- 所有group五pooled motion dt=.4、junction、固定.3/.5敏感性，不换主阈值删难例。

保留训练内全部15旧事件的跨arm对比，但它们只是事后诊断案例，不作调参/模型选择依据。正式held失败同样全量保存，不只挑这15例的修好效果。

## 8. 实施前真实预检，不用单测代替

### 8.1 源和数据

继承最新归因执行的原manifest、v2最终封存及其5140来源链，不覆盖旧digest；锁定实际输入、统计、teacher、loss源码、协议和环境。只读加载旧纯函数，禁止执行旧脚本顶层启动/写目录。新修复只新namespace并保存失败来源，不改旧scientific source。

### 8.2 数学和梯度

- 合成原全好/全坏/混合/唯一和多支持、max/min exact tie、零距离、空组；
- M与原冻结代码标量、全部参数梯度、初始和固定非零输出EXACT；
- 新R/RS独立NumPy scalar及解析/自动微分交叉核对；固定double参数两个eps1e-4/1e-5，完整记录norm/max/min/hinge活动面切换；
- 遇活动面切换不宣称FD通过，也不搜索更多eps求过。用预先定义的光滑合成fixture检验导数，真实点记NONSMOOTH_CHECK_INCONCLUSIVE，逐项报告数学核验覆盖范围；
- 精确tie采用确定次梯度做单独单元测试，不用跨面FD证明唯一导数；
- 实际固定前8train窗GPU，三arm全部合法连接分阶段有限梯度；LR初层零梯度与零末层初始化的解析原因分开记录；
- GT/支持标签只能影响loss，不得影响forward、动作解码或安全门；改变GT的反例测试必须使预测构造不变。

### 8.3 恢复与保存

三arm真实固定hash513train＋12val子集两epoch，中断点19updates，参数、统计buffer、optimizer、TorchCPU/CUDA/loader/Python/NumPy RNG、批序和新三分量history全部EXACT。不是完整3103两轮，也不是从旧训练继续。

正式恢复合同绑定GT/safe-best/support摘要，任何来源/统计/label/loss变动必须拒绝；完成模型只有完整SHA及来源兼容才skip。保存路径、动作、同槽quality、full clustering/motion的实际128窗独立读回PASS。预检不能借用正式held来证明可用。

### 8.4 资源与短预算

沿用resource_guard_v3实测inventory/E及预警，不恢复旧机械6GiB硬门，不改OS/pagefile、不关用户软件、不删除证据。

每arm4warm＋32训练批实测，另测解码guard、全聚簇、来源SHA、held标签和复核IO。报告阶段T和阶段F各自ETA；任一阶段规划>30分钟先报告耗时构成，不自动启动长任务，也不减少epoch/审核以“满足预算”。新4770参数不是完整F16总成本；旧主干未重测则标NOT_REMEASURED。

未来获执行要求且真实预检通过后才sole按需0trigger后台任务，一次startup/一次50%预计elapsed/一次完成核验，按实际剩余修正下一定点后退出静默。tiny短任务可前台完成，不为它造无意义轮询；旧完成任务和PAUSED不动。

## 9. 审核与文件交付

每checkpoint绑定：source/split/stat/teacher/Native支持及oracle-J摘要/loss公式与系数/init/seed/optimizer/全部RNG/batch顺序/epoch/history。文件不足则不算完成。

```text
ENDPOINT_SET_UTILITY_RETENTION_EXEC_20261030/
  README.md / DESIGN_SNAPSHOT.md / protocol.json
  source_manifest.json / INPUT_AUDIT.json
  code/objectives.py / trainer.py / evaluate_saved.py
  preflight/gradient/ / resume/ / pipeline/ / RESOURCE.json
  tiny/M_42 ... RS_44/
  tiny_review/REVIEW.json / TINY_DECISION.json
  formal/M_42 ... RS_44/       # 仅tiny门通过时存在
  final_release.json         # 九final封存后才允许本轮held分析
  saved/{arm_seed}/{train,val,hold,inner}/
  completion_review/REVIEW.json
  ALL_GROUPS.csv / SUPPORT_STRATA.csv / CI.json
  RESULT_ANALYSIS.md / RESULT_ANALYSIS.json / DECISION.json
  FINAL_PRESERVATION.json / RESOURCE.json
```

第二套CPU保存算术完整复核：C/Q/S路径、nearest动作、失败位、同槽quality、全聚簇、五motion/junction、全部窗口事件、三loss分量及固定checkpoint选择、bootstrap索引和draw。训练源与所有旧输入初末SHA不变；输出逐文件封存读回。不把保存算术称所有模型/梯度二次重放，不把同助手第二实现称外部评审。

若停止于tiny，formal/held/主CI标NOT_RUN，写清9tiny的实际完成数，不用“整套成功”。如果模型数学/ID/源发生实质异常立即STOP保留失败，不通过改门或改hash求PASS。

## 10. 结果如何解释

| 结果 | 支持的判断 | 不支持的判断 |
|---|---|---|
| R减少lost，RS进一步增加rescue，且held完整通过 | 当前固定结构中集合监督比候选均匀拟合更有效 | 已证明理解意图/所有场景泛化 |
| R保护改善但RS救回无增量 | 保留是可干预问题，新增覆盖仍受限 | 单靠保护已解决互补生成 |
| tiny有效、held失败 | 训练内目标对齐未迁移 | 输入必然不含信息 |
| 连续C改善，Q/S不改善 | 连续代理与离散安全执行仍错位 | 应去掉guard或放大动作范围 |
| RS全identity或救回下降 | 平凡保留，不满足研究目标 | 物理全PASS就是成功 |
| 训练更难或三项尺度互相压制 | 当前固定系数/目标组合不成立 | 所有集合效用损失无效 |
| R/RS都未改善关键支持 | 当前干预未支持目标错位可修复解释 | 唯一根因已经排除为输入问题 |

本轮结束后先总结全部三arm及旧负结果，不自动第四系数/网络宽度/训练步数变体。下一动作需由新的具体证据决定，不为了凑齐研究次数继续。

## 11. 反方检查、伦理和轻量化叙事

**DA1：**此方案根据已见tiny损伤提出，是适应性探索。固定旧15例用于说明，不用于优化。独立held本轮封存不能抹去整个项目此前对其暴露。

**DA2：**R/RS保留原MSE却新增路径损失，会改变梯度尺度和责任分配；三arm能分离两个新增项，但不能独证“只有关键样本权重”或“只有某条梯度冲突”是因果。连续代理存在越圆域和guard落差，必须如实报告。

**DA3：**零修改是最容易保留成功的解，所以必须独立要求新增rescue/coverage和held门。GT只用于训练和审核，任何按真实成功槽决定部署修改都是泄漏，不是互补生成方法。

若实验成功，合理技术表述是“在相同轻量网络与固定候选预算下，通过集合效用监督改善原生支持保留与新增覆盖”；不是增加复杂网络，也不预先宣称论文创新已完成。推理图不增加loss计算，但必须实测解码/guard成本，不仅报参数量。

本设计由AI助手辅助整理，反方检查为同助手自检。仅沿用合法既有去标识轨迹，不新增参与者、外部数据或个人信息；数据许可与机构伦理要求沿原项目核实，不编造IRB批准。旧模型、结果、失败、日志及自动记录全部保留。

## 12. 来源

- [最新归因报告](../qoder_work/ENDPOINT_SUPPORT_FAILURE_ATTRIBUTION_EXEC_20261030/RESULT_ANALYSIS.md)
- [归因决策](../qoder_work/ENDPOINT_SUPPORT_FAILURE_ATTRIBUTION_EXEC_20261030/DECISION.json)
- [收益分层](../qoder_work/ENDPOINT_SUPPORT_FAILURE_ATTRIBUTION_EXEC_20261030/gain_allocation.csv)
- [原tiny结果](../qoder_work/ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029/RESULT_ANALYSIS.md)
- [原输入与统计实现](../qoder_work/ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029/code/features.py)
- [原训练与RNG恢复](../qoder_work/ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029/code/train.py)
- [原解码与guard入口](../qoder_work/ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029/code/inference.py)
- [旧G/H续执行合同，仅作历史来源](ENDPOINT_LEARNABILITY_CONTINUATION_DESIGN_20261029.md)

本文件不编辑以上任何来源，不表示新的目标函数已数值通过。只有未来明确实施本设计后，才能报告实际预检、启动、耗时或实验结果。
