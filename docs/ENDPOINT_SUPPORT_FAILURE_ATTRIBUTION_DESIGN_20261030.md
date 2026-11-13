# 下一轮实验：端点修正的关键候选损伤与监督可辨识性归因

日期：2026-10-30  
状态：**DESIGN_ONLY；本文件不代表实验已执行或新训练已获启动。**  
建议执行名：`ENDPOINT_SUPPORT_FAILURE_ATTRIBUTION_EXEC_20261030`  
性质：固定已有输出的有限离线诊断，非新正式训练、非泛化结论。

## 0. 本轮要回答什么

当前不再同时尝试更大的网络、更多训练轮次和新损失。先利用已完成的 H42 tiny V2，回答三个具体问题：

1. 原来承担成功覆盖的候选，在哪一步被破坏：连续参数预测、离散动作选择，还是安全过滤？
2. 参数误差和平均 J-regret 的收益，主要发生在关键成功候选，还是不影响集合成败的其他候选？
3. 当前观察输入非常相近时，所需修正是否也相近？表面上的标签差异究竟代表功能冲突，还是多种同样有效的动作？

本轮交付一个有证据的分支决定，而不是再把 tiny 调到通过：

- `PRIORITIZE_FIT_DIAGNOSIS`：优先检验固定映射的拟合能力或优化预算；
- `PRIORITIZE_SET_UTILITY`：优先检验候选平均目标与成功保留之间的落差；
- `PRIORITIZE_DECODING`：优先检验当前量化/过滤接口；
- `PRIORITIZE_INPUT_LABEL_COMPATIBILITY`：优先检验输入与监督的匹配；
- `MIXED_OR_INCONCLUSIVE`：当前证据不足，不强行归为唯一根因。

上述状态只是下一轮建议，不自动触发训练。TIME441继续停止，P3不触发，原六个 G/H 正式模型仍未运行。

## 1. 已有证据与不能越过的边界

来源：[tiny V2报告](../qoder_work/ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029/RESULT_ANALYSIS.md)、[保存审核](../qoder_work/ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029/completion_review/TINY_REVIEW.json)、[最终封存核验](../qoder_work/ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029/completion_review/FINAL_SEAL_CHECK.json)。

| 已知事实 | 数值／状态 | 不允许的推论 |
|---|---|---|
| 固定训练子集 | 128窗、97条track、2560候选 | 不是128个独立行人，也不是2560个独立样本 |
| active / identity教师候选 | 2057 / 503 | identity不是窗口原成功的同义词 |
| active参数MSE | .377369956 → .074276728，下降80.3173% | 未达90%，不是完全没学习，也不是输入无信息 |
| guarded平均J-regret | .565533311 → .194834317 | 平均候选收益不是完整K20覆盖收益 |
| Native / raw / guarded完整好窗 | 69 / 67 / 70 | 净增1不是可靠部署收益 |
| raw rescue / lost | 8 / 10 | 不省略损伤只报救回 |
| guarded rescue / lost | 8 / 7 | 7/69原成功损失仍约10.14% |
| change组 | 23窗，9→9，救1丢1 | 小分母不能支持变化泛化结论 |
| 当前五pooled motion与junction | guarded四组通过 | 物理通过不是准确性保留 |
| 正式G/H、hold/inner与主CI | 未执行 | 不声称H胜G或迁移失败 |

上轮 END P0 的安全oracle机会保留；它是访问GT的有限动作可达证据，不能与当前模型收益混称。原90% tiny准入、科学质量门和运动学阈值全部不改。本轮诊断通过也不会把 `STOP_TINY_FIT_V2` 改为PASS。

## 2. 冻结范围、数据与来源合同

### 2.1 唯一人口

主分析只使用原 `tiny_v2/SELECTION.json` 和 `indices.npz` 封存的128个训练窗口，顺序、sample ID、sequence+target track身份全部一致。禁止重新抽样、借用val/hold/inner、按结果筛出“更适合分析”的窗口。

每个窗口保留全部20条候选；报告总体、stable/transition/change、原成功/原失败及完整事件四分区。不删低速、静止、边界、无法找到近邻的窗口。

允许只读原 prepared 文件以定位这128个ID及既有共同train统计。若NPZ容器包含更多训练行，记录容器访问，但科学计算严格限制在128行。不得新增其他人口的GT评价。

### 2.2 输入清单

原执行根：`qoder_work/ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029/`。

- `tiny_v2/initial.npz`、`final.npz`：连续参数、动作ID、raw/guarded路径和已保存指标；
- `tiny_v2/SELECTION.json`、`indices.npz`、`METRICS.json`、`history.json`；
- `completion_review/TINY_WINDOW_EVENTS.csv`、`TINY_WINDOW_CANDIDATE_STRATA.csv`、审核和封存收据；
- 原P0根 `qoder_work/ENDPOINT_TEMPORAL_OBSERVABILITY_EXEC_20261029/` 中的 `prepared/X_train.npz`、`GT_train.npz`、`statistics.npz`及对应128行的END标签、安全mask、动作表；
- 原来源manifest及namespace修复来源。checkpoint只核文件SHA，不加载模型或pickle权重，不需Torch。

执行前生成只读输入manifest，保留5140来源原digest及其继承链，新脚本和本设计单独扩展锁定，不能替换旧digest。若源发生变化，逐项报告并STOP，不自动刷新hash使其通过。完成后再次核验输入及全部新增输出SHA。

### 2.3 明确禁止

无Torch、模型forward/backward、optimizer、训练、恢复旧训练、延长tiny、扫LR/width/loss、重新枚举全部P0、扩大数据集、修改已保存预测、自动部署GT归因规则。不得查询旧完成PID/任务/日志或恢复PAUSED自动任务。

全部新代码、表格和图只写入新的执行根，旧失败和结果不覆盖。设计文件本轮只写入docs，不创建生产运行目录。

## 3. A0：小规模输入及算术复核

先复核全部128ID、97 tracks、2057/503候选分母，按独立NumPy实现从原路径重算same-path ADE/FDE/J及事件。必须复现第1节的69/67/70与救回/丢失计数，不能通过舍入修改good定义。

设候选轨迹为p，真实未来为y：

\[
J(p,y)=\max\{ADE(p,y)/0.2,\ FDE(p,y)/0.4\},\quad good(p)=\mathbf1[J\le1].
\]

实现仍逐项检查 `ADE<=.2 AND FDE<=.4`，同一候选，不拼接两个不同槽位的独立最小值。窗口覆盖为20个good的逻辑OR。

这不是重新运行原训练审核；是新诊断输入与算法的桥接。出现不一致必须先留失败收据，暂停归因。

## 4. A1：连续预测→离散化→安全过滤的阶段分解

### 4.1 固定五个端口

| 端口 | 定义 | 用途 |
|---|---|---|
| N | immutable Native20 | 基线 |
| C | 保存的连续归一参数u：`p+(t/12)*(.4u)` | 新增离线连续诊断，不是部署动作 |
| Q | 原END49最近动作解码，即旧raw | 真实量化输出 |
| S | 原literal guard，不通过则该槽恢复Native，即旧guarded | 真实执行输出 |
| O | 既有END49逐槽safe oracle | 有GT上界参照，不是模型 |

C不做新径向裁剪。tanh的两个坐标各在[-1,1]不保证二维范数≤1，因此C可能超出原.4m圆形动作域。必须记录范数超界、位置界及七失败位；超界C只能解释中间表示，不能称“取消量化就得到合法改进”。不得静默clip后仍称原连续输出。

Q的动作顺序、exact tie优先规则、49个格点和S的失败位完全继承原实现；不加入第二优动作搜索。O优先读取封存结果，缺少必要字段时只为这128窗独立重构125440个END动作（128×20×49），单独记录新增算术，不称全P0重跑。

### 4.2 两层归因

候选级：保存每槽五端口的ADE、FDE、J、good、参数误差、动作ID、七失败位。对固定同槽报告：

- 连续偏差：`J(C)-J(O)`；
- 量化差：`J(Q)-J(C)`；
- 过滤差：`J(S)-J(Q)`。

三项代数求和必须等于 `J(S)-J(O)`。C或Q可能不安全，前两项可以为负，禁止强行截为非负再称严格因果损失预算。

窗口级：分别记录N→C、C→Q、Q→S和N→S的coverage转移表。每一步都重算20槽OR，不沿用前一阶段winner。对最终7个lost和8个rescue逐例展示所有阶段，不只抽“典型”图片。

“回归损伤”“量化损伤”“过滤损伤”允许在一个窗口共存，并可能被下游逆转。使用阶段路径码（如1→1→0→0），不强行把7个lost塞入互斥单原因箱，也不将这些转换当真实行为因果。

### 4.3 关键成功支持集，不只取一个winner

\[
B_i=\{k:good(N_{ik})=1\},\quad m_i=|B_i|.
\]

- `m=0`：原失败窗口；
- `m=1`：唯一成功支持槽，真正的单槽覆盖关键点；
- `m>=2`：多槽支持，失败意味着所有原成功支持最终均不再成功且没有新增成功槽。

对每个B中的槽记录教师identity/active、预测偏移、J到阈值的原始余量、每阶段good以及是否被guard恢复。若多槽共同丢失，全部列出，不只选minADE槽。

同时保留原minJ槽（exact tie最低物理slot），用于连续误差比较；minJ槽与唯一覆盖支持不是同一概念。

### 4.4 两种限定的反事实，不搜索组合

在S集合中，分别将每个槽k恢复为对应Native槽，生成20个单槽恢复集合。全128窗均做，共2560个集合。报告恢复覆盖、损失新增覆盖及minJ差；不按结果选一个槽用于部署。

另做20个“仅槽k采用S、其余19槽保持N”的集合，共2560个集合，报告单独修正的救回与损伤。前者是最终集合背景下的恢复效用，后者是原生集合背景下的独立修改效用；不称二者可加、Shapley值或真实训练因果。

**恢复某个原成功槽必然能救回lost，是定义上的结果。**有信息量的是损伤出现的阶段、支持槽数量、教师标签与预测偏差，而不是“oracle恢复成功”本身。禁止把GT选恢复槽做成新方法。

## 5. A2：平均拟合收益究竟落在哪里

### 5.1 冻结分层

先按Native结果定义以下互斥候选类，不用修正结果定义“关键”：

1. 唯一成功支持槽：m=1且k属于B；
2. 多重成功支持槽：m>=2且k属于B；
3. 原成功窗口中的失败槽：m>=1且k不属于B；
4. 原失败窗口全部槽：m=0。

再与教师identity/active交叉；stable/transition/change另列。每格报告候选数、窗口数、track数。空格写NA，不填0。最终lost/rescue等事件是附加描述分层，明确其按结果选择的性质。

### 5.2 指标和分母

每槽定义：

\[
\Delta MSE=e_{initial}-e_{final},\quad
\Delta J=J(N)-J(S).
\]

分别报告每层的均值、中位数、分位数、正负比例及对全体总和的有符号贡献。总贡献接近0时比例写NA，不用绝对值掩盖负贡献。报告候选加权结果用于与原MSE严格对账；另先在每窗每层内平均、再按窗口平均，避免候选数量多的窗口支配描述。

主问题不是“有多少槽变好”，而是：

- 第3类的大量小收益是否伴随第1/2类损伤？
- 第4类改善是否真正跨过完整good边界，还是依旧全部失败？
- identity教师候选的偏移是否集中在被破坏的原成功支持上？
- 精确teacher action未命中但safe J接近oracle的比例是多少？

最后一项保留既有 `J <= J_oracle + .01` 描述口径；它不是新的成功阈值，仍同时报告literal good和精确动作命中。

### 5.3 49动作上的效用与参数距离

针对固定128窗全部槽，使用既有safe masks与J矩阵，记录预测离散动作到教师的参数距离、safe action J-gap、是否越过good边界。

尤其列出“参数差较小但good翻转”和“参数差较大但效用相近”的实际分布；不依据GT挑新解码距离或新网格。图可以使用全体散点/分箱描述，但分箱只展示，不决定训练或部署。

## 6. A3：输入相近是否意味着修正相容

这一步是固定输入空间的局部一致性诊断，**不是学习模型、不是H−G主实验，也不是证明未来信息不存在**。

### 6.1 输入与距离在读取标签关联前封存

使用原train统计和原float32模型输入，不重新拟合尺度：

- G：原生路径24维；
- H：同一24维路径与观察历史16维。

标准化后定义：

\[
d_G^2=\|z^{path}_a-z^{path}_b\|^2/24,
\]
\[
d_H^2=\tfrac12\|z^{path}_a-z^{path}_b\|^2/24+
\tfrac12\|z^{hist}_a-z^{hist}_b\|^2/16.
\]

H采用预先固定的块均衡距离，不能根据标签表现选择块权。G/H近邻差异包含距离度量变化，不能直接当作历史增量可学性的因果证据。

### 6.2 跨track近邻及参考

每个候选查询仅从本128窗的其他target tracks搜索，排除整个同track及自身窗口；每个参考track只保留最近的一个候选，再取最近5条不同track。报告top1及top5平均，均为预定口径，不挑更好者。并列按distance、sample ID、physical slot排序，track键必须含sequence，不能只用局部pedestrian整数。

主口径限制同recording/sequence；另给不限制sequence的敏感性。合格track少于5时保留实际数量并标不足，0个则NA，不跨口径偷偷补齐。两者都存在共享场景或上下文相关性，不能称独立全模型泛化。

参考组：每查询对合格reference tracks按固定SHA256 tag `endpoint-support-attribution-v1|queryID|slot|referenceTrack` 排序，取至多5条，再在各track中取固定hash首个候选，不依据距离或标签。它是确定性远近无关参照，不是正式随机化因果实验，也不保证每次比近邻更远；必须报告实际距离分布。

先输出并SHA封存 `NEIGHBOR_MAP` 与参考map，再关联GT教师和J矩阵。禁止GT参与近邻选择、删例或配对。

### 6.3 不把不同标签直接当作冲突

对query与reference报告归一参数距离、动作ID是否一致、identity/active是否一致；再将reference的END动作应用于query原生槽，按query自身safe mask与J矩阵评价：

- 是否仍安全；
- 相对query oracle的J-gap；
- 是否保留query Native成功；
- query原失败时是否救回。

若不安全，先记录失败，执行口径按原规则退回identity，分别报告提议和执行效用。只比较标签不同可能把功能等价误认为冲突；交叉应用是诊断，不是部署nearest-neighbor策略。

报告邻居与hash参照的窗口级平均差，但不以某个差值自动宣称信息充分/不足。候选不独立、只有单训练seed、该tiny已参与拟合，均限制结论。

### 6.4 精确输入冲突的有限下界

另按实际模型输入float32数值完全相同分组（正负零统一，非有限拒绝；不得四舍五入制造相同）。分别检查G/H输入组内教师参数差异。每组大小n、均值target及残差平方和都保存。

对于实际相同H输入的确定性回归器，其全候选MSE至少为：

\[
L_{dup}=\frac{1}{2560}\sum_g\sum_{a\in g}\frac{\|u_a^*-\bar u_g\|^2}{2}.
\]

这是这批精确输入与标签上的经验下界，不是总体Bayes误差。没有重复输入则该下界为0，仅表示这一检查未找到硬冲突，不表示信息充分。该下界也不能直接替代原active-only 90%门。

近似近邻标签差异不具有这个下界含义。若研究人员想改用量化容差、增加距离层或新特征，应另立设计，不在本轮扫到“冲突明显”为止。

## 7. 统计、可视化和结论尺度

### 7.1 本轮以封闭人口精确描述为主

固定128窗上的事件数和阶段变化就是主要结果。对7个lost、8个rescue完整列出，不把少量个案包装为显著性或总体机制定论。97 tracks的窗口相关性不因候选数2560而消失。

主报告不做新的显著性检验，不产生一批p值筛结论。允许对预定六个描述量给探索性95%整track percentile bootstrap区间，固定5000次、RNG20261030，所有量共用整track重抽样：

1. S−N coverage；2. S lost/native-good；3. S−N minJ；
4. 唯一支持槽的窗口均值ΔJ；5. 原失败窗全部槽的窗口均值ΔJ；
6. H近邻相对hash参照的执行J-gap差（主同sequence口径，先窗内平均）。

无分母的draw记invalid，不能填0；有效draw不足4750则该CI为NOT_ESTIMABLE。所有区间都是训练子集探索区间，不做多重检验后的“显著发现”宣称，不触发原正式family6的检验或部署放行。若不实现CI，必须标NOT_RUN，不影响本轮精确归因交付。

### 7.2 最小图表

- 一张四阶段窗口转移表，保留逆转，不强制互斥流量归因；
- 一张候选四分层的ΔMSE/ΔJ与窗口coverage对应表；
- 七个lost加八个rescue的完整案例页：past、GT、原成功槽、C/Q/S/O与各阶段失败位；
- 一张近邻距离—标签差异—交叉动作效用图，标实际窗口/track分母。

图示明确C是诊断、O使用GT。不要只画“明显修错”的失败例或“改善漂亮”的成功例；病例其他槽淡显但保留可追溯完整数组。

## 8. 全集合质量和运动学的保留规则

五端口完整人口都报告同槽coverage、rescue/lost/retained、minADE、minFDE、minJ、单Jclosest槽的早中晚误差；minADE和minFDE仍注明可能来自不同槽，不能冒充完整good。

原SciPy GT-free ALLK20 complete-link距离 `max(meanpath/.10, endpoint/.20)`、有效good簇与distinct均保留。保留全部组的五pooled motion dt=.4及junction mean/p95，literal≤Native×1.01；不得对0基线加入epsilon或floor求PASS。C的不安全结果如实报告，不用于推进。

两个单槽反事实系列至少保存全部2560×2集合的覆盖、minJ、五motion/junction和事件；涉及效用/簇数结论时必须重算相应完整K20聚簇，不用单条路径指标代替集合指标。

本轮不放宽原good=.2/.4，.3/.5仅沿用固定敏感性，不按其结果选择结论。原90% tiny门、正式部署门以及历史major门仍各自独立，不能用本轮审计PASS替代。

## 9. 有限决策表：结束诊断后只建议一个主方向

| 观察组合 | 支持的解释 | 下一轮允许建议 | 不允许的跳跃 |
|---|---|---|---|
| 关键支持槽在C就变坏，Q/S没有新增主要损伤；相似输入动作较相容 | 当前映射拟合不足仍值得检验 | 单因素容量或预算诊断，二选一并事前固定 | 宣称一定只要加epoch就好 |
| 总收益集中在非关键失败槽，成功支持损伤抵消收益；教师动作仍可保留 | 候选平均监督与集合效用错位值得优先检验 | 固定结构/解码，只比较集合效用或保留目标 | 同时换head、loss和integration |
| C合法且保留成功，Q造成可重复边界翻转 | 固定最近动作接口有可定位问题 | 单独比较训练—解码一致性 | 去掉安全门或挑GT最佳动作部署 |
| Q收益被S撤销，但Q本身违反物理门 | 无约束收益不可直接兑现 | 研究safe-action感知监督 | 宣称guard错误、放宽1.01 |
| 近邻标签差异大但交叉动作同样有效 | 单一教师标签可能过于刚性 | 研究等效动作集合监督的必要性 | 用“不同标签”证明不可预测 |
| H精确相同输入存在高标签残差，或跨track局部效用不相容 | 当前输入/标签匹配存在局部限制 | 界定缺少什么条件或监督是否过细 | 推广为行人未来总体无法预测 |
| 混合、样本稀少或无清晰定位 | 当前证据不足 | 一次总结并终止本轮，不自动新实验 | 无限小变体/继续抽样直到出现清晰结论 |

不设置“多数lost归某类就自动训练”的新阈值：只有7个lost，不能用脆弱分母机械决定架构。报告必须给出支持首选方向的具体窗口数、正反例、与其竞争的解释；其次说明为何暂不选其他方向。

即使推荐新目标，也不能用本轮GT支持集作为部署输入。监督训练可以使用训练GT，但任何选择/门控的预测输入只能来自可观测信息，需要下一份精确协议约束。

## 10. 核心实现接口与核验清单

本节是接口设计，不是已实现代码，不能直接称预检PASS。

```python
def build_ports(native, continuous_param, saved_raw, saved_guarded, teacher):
    # 仅数组算术。C=.4*u的原连续映射；Q/S先核对原保存值。
    # 输出N/C/Q/S/O，C超界单独标记，禁止隐式clip。
    ...

def support_audit(native_good, ports_good, slot_j, teacher_action):
    # 完整B_i、m_i、每槽阶段码；不只保留一个winner。
    ...

def single_slot_counterfactuals(native, guarded_paths, gt):
    # 恢复k、仅修k，各20组，固定K20；GT只计算诊断效用。
    ...

def build_neighbor_map(observed_features, sample_id, track_key, sequence):
    # 不接收GT/teacher/actions/J；同track排除，每reference track至多1。
    # 固定G/H距离、top1/top5与hash参照，随后封存。
    ...

def associate_neighbor_utility(sealed_map, labels, safe_masks, j_matrix):
    # 关联动作差异及交叉应用效用，不反向改map。
    ...
```

最低合成测试：

1. ADE/FDE各自最低来自不同槽时不能判good；
2. 单一支持与多重支持分别被破坏时，B集和事件准确；
3. 槽位顺序重排但ID映射同步后，集合指标不变、并列规则明确；
4. C二维坐标合法但范数超界时只能标记，不能自动clip；
5. exact动作tie按旧priority，identity恢复路径逐元素一致；
6. stage有逆转时不误写成互斥损伤；
7. 同track多个窗口/20候选不能进入自己的近邻库；
8. 少于5个reference tracks、空分层、重复输入、正负零、非有限值各有明确结果；
9. 修改GT不会改变neighbor map及C/Q/S路径构造，只改变事后指标；
10. 全部单槽恢复与单槽独立修改的K始终为20，禁止自动枚举2^20子集。

实际预检只固定SELECTION顺序前8窗完成一次全流程，不能根据结果改选。正式离线算术由独立实现复核：候选质量、阶段代数、支持事件、反事实集合、近邻排除与并列、重复输入下界、完整运动及聚簇。数值比较沿用被比较指标既有审核容差并在运行前锁定；literal科学判断必须完全一致，禁止看差值后扩大容差。

## 11. 预算、运行与交付

### 11.1 工作量上限

- 128窗、2560候选、5端口：640个window-port；
- 两组固定单槽反事实：5120个K20集合；
- 近邻查询最多2560个，每口径不超过2559个候选参考，分块计算，不保存所有轨迹两两差张量；
- 如必要，128窗END49动作算术最多125440个，不重新扩展TIME或全P0；
- 0训练更新、0模型forward、0新模型。

执行前用前8窗预检和实测吞吐估算算术、聚簇、近邻、来源SHA及IO总时间。不能拿旧tiny训练4.719秒当本轮ETA。建议短预算规划≤30分钟；若预估超过，先报告耗时构成，不擅自删主要审核、抽样代替全128或启动长训练。

资源沿用项目v3库存/预警口径，按实际数组分块，预估及峰值分开记录；不恢复已废弃的机械6GiB门，不调OS/pagefile、不关闭用户程序或删数据腾空间。脚本运行如需后台，只有获得本轮执行指令且预检通过后才按现有soleworker/一次startup/一次预计半程→一次完成核验流程启动；短任务前台完成则不制造多余自动任务。旧PAUSED保持不动。

### 11.2 建议文件

```text
ENDPOINT_SUPPORT_FAILURE_ATTRIBUTION_EXEC_20261030/
  README.md / protocol.json / INPUT_MANIFEST.json
  SOURCE_EXTENSION.json / PREFLIGHT.json / RESOURCE.json
  ports_summary.csv
  candidate_stage_audit.csv
  native_support_sets.json
  window_stage_transitions.csv
  single_slot_restore.csv / single_slot_edit.csv
  gain_allocation.csv
  neighbor_map.npz / NEIGHBOR_MAP_SEAL.json
  neighbor_utility.csv / exact_input_conflicts.json
  all_groups_quality_motion.csv
  CASE_INDEX.md / cases/
  completion_review/REVIEW.json
  RESULT_ANALYSIS.md / RESULT_ANALYSIS.json / DECISION.json
  FINAL_PRESERVATION.json
```

报告必须说明“已执行、未执行、不适用、无法估计”分别是什么。执行完整性结案可为 `PASS_OFFLINE_ATTRIBUTION`，科学解释状态单列第0节枚举，不把“找到原因”作为审计通过条件。

### 11.3 完成标准

- 全128IDs/分母/原指标复现，输入来源初末不变；
- A1/A2/A3全部输出，包括反例、NA、所有7个lost和8个rescue；
- 原数组与新派生数组可逐ID/slot追踪，不修改原实验；
- 完整保存算术复核及资源记录，不冒称模型/梯度再次重放；
- 只推荐一个优先大分支或明确混合证据，不自动新训练；
- 旧tiny STOP、TIME STOP、P3不触发、G/H主问题未测的状态保持真实。

## 12. 反方检查与研究诚信

**DA1——选择偏差：**本轮设计已看到7个lost，是事后探索性归因。用全128窗和全部事件作参照，不能伪装新的确认性测试。tiny只有H42且用于训练，不能作泛化验证。

**DA2——定义性成功：**GT选恢复原好槽必然救回，oracle O也通过identity保证不劣；两者都不是可部署创新证据。真正需解释的是可观测输入下的预测错误与集合效用分布。

**DA3——替代解释：**近邻冲突受距离、样本稀疏、局部坐标及上下文遗漏影响；MSE收益错位不证明任何新loss必胜；C好但超界也不支持取消量化。无法排除时写并存，不为了推进而捏造唯一根因。

本设计由AI助手辅助整理，反方检查是同助手自检，不是外部独立审稿。仅用项目已许可的去标识轨迹和既有GT，公开报告只展示必要匿名轨迹，不加入外部个人信息；是否需要机构伦理手续按原数据许可与所在机构要求核实，不编造IRB批准。没有新增参与者或数据收集。

## 13. 阅读来源

- [上一轮续执行设计](ENDPOINT_LEARNABILITY_CONTINUATION_DESIGN_20261029.md)
- [tiny V2结果分析](../qoder_work/ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029/RESULT_ANALYSIS.md)
- [tiny原指标](../qoder_work/ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029/tiny_v2/METRICS.json)
- [tiny训练记录](../qoder_work/ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029/tiny_v2/history.json)
- [实际解码实现](../qoder_work/ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029/code/inference.py)
- [固定动作与tie规则](../qoder_work/ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029/code/action_geometry.py)
- [实际输入与标准化](../qoder_work/ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029/code/features.py)
- [原safe teacher定义](../qoder_work/ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029/code/build_teacher.py)

这些文件是方法和事实来源，不意味着本轮已执行其中的脚本。下一步只有在用户要求实施本设计后，才创建新执行目录并运行固定离线诊断。
