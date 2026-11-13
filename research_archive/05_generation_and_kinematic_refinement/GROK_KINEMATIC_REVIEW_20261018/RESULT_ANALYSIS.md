# 新运动学分支复核：互补性成立，全面优越与部署门尚未成立

日期：2026-10-18。范围：用户所述其他助手新增实验，在项目内定位到下列7个同日目录。文件未统一标注作者，本报告按目录认定证据，不凭命名认证作者身份。

**结论：应认真跟进“原生支撑＋另一类运动学候选＋固定K集合选择”，但先冻结并补齐验证，不直接部署或再扩模型。** 这是比继续微调P1 anchor更有希望的新方向；它仍没有达到项目原先全部质量、变化组、保护和运动学门。

## 1. 本次实际做了什么

- 阅读7组RESULTS、关键设计、数据加载、模型、损失、限幅、选择器及评价代码，核对三seed JSON。
- 执行新CPU脚本[保存数组复核](audit_cached_outputs.py)，只读已有native/GT/ID/split及三个KIN outer缓存；无Torch、模型前向、optimizer或训练。
- 独立重算native20、KIN20、union40在3seed×4组共36行的同候选coverage、rescue/lost、minADE/FDE、完整SciPy complete-link簇/有效good簇及五pooled运动学。
- 可比旧JSON数值最大差0，10个本轮输入/脚本SHA前后不变。机器可读结果：[SAVED_CACHE_REVIEW.json](SAVED_CACHE_REVIEW.json)。这是当前保存数据算术复核，不是历史训练源锁、所有旧文件或外部评审。
- KIN `.npy`没有嵌入ID/模型SHA：当前按实际loader的outer顺序解释，尺寸、缓存ID和报告计数相符，但不能补造历史checkpoint→cache绑定。
- 选择器目录未发现最终HGB序列化、每窗口40分数、所选20索引或所选完整路径。因此其JSON能够交叉核对计数，却**不能完成选择器逐组运动学/置信区间/全保存路径复算**。没有偷偷重新拟合来冒充原模型。
- 默认Python检查缺SciPy后，使用目录README指定的既有`<PYTHON_ENV_ROOT>/envs/astra2/python.exe`成功完成。未安装依赖；未启动worker、查询旧PID或修改旧结果。

另发现`GEMINI_EXP`中的README/RESULT文件仅1字节，无可分析实证，不纳入7组结果。

## 2. 七组实验的证据摘要

共同outer1035窗口，原生good625/1035=60.3865%；变化组253窗口，原生110/253。这里native是已有F16候选，而BASE是新训直接位置MLP，不是同一个基线。

| 实验目录（20261018） | 核心变化 | outer overall结果 | 判断 |
|---|---|---|---|
| KINEMATIC_DECODER_EXPLICIT_INTEGRATION | 直接位置 vs 加速度积分＋一致性辅助 | BASE平均41.64%；KIN56.91%；KIN整体运动门0/3 | 相对弱BASE有结构配方收益，未优于native |
| KINEMATIC_DECODER_SPEED_CLAMP | FIT统计速度/加速度限幅＋motion penalty | KIN585/585/583个good，平均56.46%；旧整体门3/3 | 更平滑但低于native；逐组安全有反例 |
| KINEMATIC_DECODER_POTENTIAL_PROBE | 32epoch LONG及一次LONG_RELAX | 平均56.14%/56.46%，均低于native | 此有限配方未突破，不证明理论天花板 |
| MODE_DIVERSE_KINEMATIC | 4 mode×5 slot及mode diversity | 平均约54.7%；原报告整体门3/3 | 本版本阴性，div loss到0不等于模式塌缩已被证明 |
| NATIVE_ANCHORED_KIN_RESIDUAL | 原生全20加积分残差 | 630/631/618；rescue33/32/35，lost28/26/42 | 均值仅+0.129pp，seed44下降；不能称稳定保留成功 |
| HYBRID_NATIVE_KIN_UNION | native20∪KIN20 | 721/723/724；新增96/98/99，lost0 | 明确互补见证，但K40不等预算；lost0是保留全部native的构造 |
| UNION_SELECTOR_K40_TO_K20 | FIT训练HGB＋条件选择 | 同一cond_boost_11_9：664/675/675 | 等K20覆盖探索阳性，错误尾部/保护/模型选择偏差需补核 |

上表训练阶段多数只保存汇总/模型，不能把每个“整体门3/3”写成项目全组严格门通过。

## 3. 最有价值的新证据：互补候选池

| 变化组／253 | native20 | KIN20 | union40 | K20 cond_boost_11_9 |
|---|---:|---:|---:|---:|
| seed42 | 110 | 85 | 128 | 112 |
| seed43 | 110 | 100 | 135 | 119 |
| seed44 | 110 | 92 | 129 | 115 |

KIN单独仍丢掉很多native好窗口，但救回了18/25/19个变化窗口。并集的变化组minADE为.22949/.22655/.22583（native .24915），minFDE为.36789/.36693/.36427（native .42938）；因此不只是“多了几条完全重复轨迹”。完整聚簇复核也显示有效good簇增加。

不过“union K40 oracle K20覆盖一样”并非已经学会选择：只要池里有good，GT oracle保留一条就能达到池覆盖，这是定义上的上界。它不证明GT-free排序已掌握这些信息，更不证明新候选来自新观测或心理意图理解。

相比P1的A/B大动作但不安全、C/D近identity，这条新分支展示了另一条路线：**不用把全部原生模式改好，先获得互补来源，再判断如何保持必要支撑。** 但两系列架构、loss、预算和保护机制不同，不能拿其结果作单因素因果对照。

## 4. 最重要的负面证据：覆盖增加不等于误差整体改善

同一选择器cond_boost_11_9的原始JSON：

| 指标 | native | seed42 | seed43 | seed44 |
|---|---:|---:|---:|---:|
| overall good | 625 | 664 | 675 | 675 |
| rescue | — | 56 | 64 | 67 |
| lost | — | 17 | 14 | 17 |
| overall minADE／m | .17548 | .19836 | .19442 | .19569 |
| overall minFDE／m | .29663 | .35989 | .35253 | .35117 |

覆盖平均+4.47665pp；但minADE平均**恶化11.79%**，minFDE平均**恶化19.52%**。原成功丢失率分别17/625=2.72%、14/625=2.24%、17/625=2.72%，不是零损失，也未达到此前≤1%的保留要求。overall误差保护门已不满足，不必等缺失审计才知道尚非major成功。

净增加的139个“seed×窗口”计数中，stable贡献30+37+34=101，transition贡献7+4+11=22，change贡献2+9+5=16。约72.66%净增来自stable；这些是跨seed重复窗口，非139个独立样本。不能用overall +4.48pp替代“复杂变化理解显著改善”。

阈值覆盖和平均oracle误差可以朝相反方向变化：一些窗口被推过.2/.4门，同时其他窗口丢失更精确或远尾所需候选。此解释与结果一致，但哪些tail导致误差增大还需每窗口所选路径才能定位，当前不能断言唯一原因。

## 5. 原“运动学PASS”只检查整体：本轮补出了失败组

实际`kdec/metrics.py`在进入group循环前对全population只计算一次motion；循环只保存各组质量。因此旧PASS是overall，不是overall/stable/transition/change全部通过。

本次独立缓存复核：

| 保存端口 | seed | change speed_p95/native | literal1.01门 |
|---|---:|---:|---|
| KIN20 | 42 | 1.01258149 | FAIL |
| KIN20 | 43 | 1.05157268 | FAIL |
| KIN20 | 44 | 1.00456195 | PASS |
| union40 | 42 | 1.00623970 | PASS |
| union40 | 43 | 1.02229773 | FAIL |
| union40 | 44 | 1.00250232 | PASS |

其他已复核组/指标见JSON，不能把上述失败概括成所有物理指标都坏。选择器的组门**未知**，不是已知失败，也不是可由KIN/union门推断通过。选择子集会改变分位数，必须拿最终20条算。

加速度限幅的是控制a_t；速度投影后实际(v_t−v_{t−1})/.4未必等于a_t。所谓soft_clamp实际上是分段径向cap，不是全局光滑函数。所有安全结论仍以最终路径复算为准，不能仅凭“运动学积分”名称保证五门。

## 6. 可辨识性与报告问题

### 6.1 GT-free前向不等于没有用outer选择方法

`eval_selector.py`逐seed用outer coverage选`best_name`，再按outer三seed平均选`best_overall`。当前脚本共有21个GT-free策略（18条件策略＋3几何策略），还有不同版本输出记录。训练函数的确仅用FIT GT，没有发现把outer GT作为HGB特征；但**策略/主方法选择使用了outer结果**。两件事必须区分。

原DESIGN主要列几何/FIT logistic方案，未把最终HGB cond_boost_11_9锁为唯一主方法。不能把事后统一同一方法就称为独立确认性。seed42该同一方法提升3.768pp，不是其逐seed最佳cond_top_11_9的4.06pp。

这里没有证明报告数字造假；问题是验证集适应性和不完整证据。后续不能继续在这个outer上挑更好11/9、12/8、boost或listwise方案再称确认性。

### 6.2 HGB不是报告所称的软质量回归

实际生成`soft=max(exp(−max(ADE/.25,FDE/.5)), good)`，随后HGB使用`yb=(soft>=.5)`训练分类器。非good意味着ADE>.2或FDE>.4，故指数≤exp(−.8)<.5；good被设成1。因此此标签**恰等于good二元标签**，不是连续soft-quality回归。

加上正负重采样，predict_proba不等于已校准的未来概率；再乘端点互补距离后更是排序启发式。它未直接优化set leave-one-out效用、误差尾部或罕见支撑。这与threshold覆盖好而minADE/FDE坏相容，但尚非因果证明。

### 6.3 新BASE并非强原生模型，也不是纯结构单因素

BASE参数57,432，KIN63,624；KIN初始加速度0对应CV（受cap），BASE位置0对应stationary。KIN额外有一致性辅助，clamp版本还有motion penalty，而BASE没有。双方训练梯度clip5、endpoint线性diversity，也不同于此前canonical无clip/path平方hinge。故相对BASE的15pp改善是整套配方/初始化/容量/优化的比较，不能全部归因“理解了动力学”。

KIN只有past和末速度输入，没有邻居。还需区分是否胜过同预算CV/简单平滑候选、是否利用额外历史，而不能从低jerk推出社会交互理解。

### 6.4 不能用FIT与outer原始coverage接近证明迁移充分

两集合本身难度、native coverage及scene组成不同。应比较相对同集合native的gain、rescue/lost及分组条件量；小的绝对coverage差不是迁移性证明。32epoch两变体未提升只能说明该有限配方没有观察到上行，非“理论天花板”。

### 6.5 审计与可重现性尚有缺口

checkpoint保存state_dict/arm/seed/限幅参数，但未完整绑定训练source、split IDs、所有objective参数、optimizer及ALL RNG。缓存按文件存在直接复用，未校验checkpoint/input source digest。选择器未持久化HGB/score/index，旧JSON缺组运动学与组误差详情。当前没有看到与主项目等价的源锁、独立全保存算术、track聚簇CI或exact-resume证据；**缺证据不等于这些检查都一定失败**。

## 7. 后续优先级

**第一优先：冻结当前cond_boost_11_9，完成证据闭环，不新增生成器、不继续扫描选择政策。** 具体步骤见[下一测试设计](NEXT_VALIDATION_DESIGN.md)。补齐最终路径/索引/HGB、逐组安全、同路径质量、错误尾部与聚簇CI，明确数据曾参与方法选择。

**第二优先：用固定候选池验证条件信息是否真正有边际价值。** 只在固定训练内分组验证/既有inner与CAL后验范围比较native、KIN、固定几何选择与锁定条件方法；先不改变generator、loss、K或参数预算。需要重新拟合时必须是新版本，不能伪造原HGB重现。

**暂不优先：再做更大残差、加anchor系数、扩大epoch、多mode或新增scorer网络。** 新结果最直接暴露的是支撑整合与评价选择问题，不是已证明容量不足。

原W0协同设计已补全：[设计入口](../SAFE_REPAIR_COOPERATION_AUDIT_DESIGN_20261018/README.md)。它保留为可选机制诊断，不应阻挡对这条新互补分支的优先复核。本次没有执行W0消融、没有重训任何模型，也不启动新自动任务。

## 8. 同助手反方复核

- 正面观点：三seed缓存的KIN-only rescue真实存在，变化组也存在，不应因审计缺口否认有价值的新候选来源。
- 最强反驳：K40天然扩大预算，K20主方法在outer被挑选，而且平均误差变坏、局部motion不齐；尚不能作为全面突破或部署结论。
- 综合：推进“互补支撑集合”大分支，但先固定方法和完整验收合同。使用academic-research-suite的证据/反方检查框架，全部由本助手完成，非外部评审。

## 9. 主要本地证据入口

- [显式积分原报告](../KINEMATIC_DECODER_EXPLICIT_INTEGRATION_20261018/RESULTS.md)、[限幅原报告](../KINEMATIC_DECODER_SPEED_CLAMP_20261018/RESULTS.md)、[32epoch潜力原报告](../KINEMATIC_DECODER_POTENTIAL_PROBE_20261018/RESULTS.md)。
- [多模式原报告](../MODE_DIVERSE_KINEMATIC_20261018/RESULTS.md)、[原生残差原报告](../NATIVE_ANCHORED_KIN_RESIDUAL_20261018/RESULTS.md)。
- [并集原报告](../HYBRID_NATIVE_KIN_UNION_20261018/RESULTS.md)、[选择器原报告](../UNION_SELECTOR_K40_TO_K20_20261018/RESULTS.md)。
- [选择器训练与outer择优代码](../UNION_SELECTOR_K40_TO_K20_20261018/kdec/eval_selector.py)、[HGB实际标签](../UNION_SELECTOR_K40_TO_K20_20261018/kdec/hgb_ranker.py)、[只计算整体motion的评价代码](../UNION_SELECTOR_K40_TO_K20_20261018/kdec/metrics.py)。
- [split/ID加载](../KINEMATIC_DECODER_SPEED_CLAMP_20261018/kdec/data.py)、[模型与初始化](../KINEMATIC_DECODER_SPEED_CLAMP_20261018/kdec/models.py)、[训练目标及checkpoint字段](../KINEMATIC_DECODER_SPEED_CLAMP_20261018/kdec/train.py)。

本报告保留原报告的探索性正面结果，同时纠正其证据范围，不覆写任何原文件。
