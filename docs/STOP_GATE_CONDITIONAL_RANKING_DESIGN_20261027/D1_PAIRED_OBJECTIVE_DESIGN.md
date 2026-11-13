# D1：窗口级STOP竞争＋条件编辑排序的配对实验

2026-10-27｜DESIGN_ONLY。本文是主实施协议，尚未实现、预检或启动。

## 1. 科学问题与唯一新增方法

问题不是“怎样让模型更多编辑”，而是：在相同输入、候选池、网络和推理规则下，改变监督组织方式能否提高**安全且可迁移的净覆盖**？

三臂固定：

|arm|训练目标|推理|角色|
|---|---|---|---|
|I|原逐窗口正/负动作均值BCE|401分数eligible argmax|保留STOP权重偏置的对照|
|S|原正负最高分hard ranking，margin1|同上|已有直接排序对照，防止只战胜弱I|
|D|窗口级编辑机会BCE＋编辑内部hard ranking|同上|唯一新方法|

**D没有新gate网络、额外输入或推理阈值。** 使用最高非STOP分数与STOP分数的差作为窗口gate；部署仍为同一个401分数argmax。这样不再把网络结构、输入信息和新阈值一起改变。

D改的是整个监督分解，涉及梯度分配、门控与排序项的配比及门控margin；不能称只隔离了STOP负例稀释的纯因果效应。更不能预先宣称D一定有效。

## 2. 冻结对象、人口与训练合同

固定native20/KIN20坐标、400次替换或STOP、原物理eligible和质量B；一次最多替换1条，最终K20，其余19条EXACT。F16/KIN/原decoder/scorer不训练、不实例化做新前向；现有缓存与权重来源以SHA保护。

- FIT_opt4096训练；FIT_hold3825为主评价，change1023/74条含变化track；inner1856仅次要，change497/35track。
- 固定池/模型seed42/43/44。相同seed内I/S/D初始化、原统计与每步batch窗口顺序一致。
- 原324维特征，原324→32→32→1 GELU，末层无bias且零初始化，11488参数。
- 原FIT_opt-only全eligible等窗口统计；前三百二十三维标准化、STOP位不标准化。STOP动作私有特征原始0经原统计变换后不强制归零。
- 完整401动作训练与推理，不使用前轮最多32动作的IPW近似。这一变化对本轮三臂完全相同；前轮IPW探针只作背景，不是严格配对控制。
- AdamW lr=.001，weight_decay=.0001，betas=(.9,.999)，eps=1e-8，foreach=False；clip_grad_norm=5；window batch32，shuffle，drop_last=False。
- 每run固定8epoch、1024updates；9run合9216updates。固定epoch8为唯一主终态，不按hold或inner挑epoch、seed或停止点。
- 所有9终态先锁SHA，再评价hold/inner；opt诊断中间读数固定epoch1/4/8，不据此延长预算或换目标。

不访问outer/CAL、新fold/新数据；不增加邻居、完整token、场景地图、slot embedding或rescue-only标签。旧全量I/S的重复是预注册的新控制，不继续旧checkpoint、不查询或重开旧任务。

## 3. 标签与符号

沿旧保存标签定义：每动作质量A=minADE、F=minFDE、J=min_k max(ADE_k/.2,FDE_k/.4)。覆盖必须同一完整candidate满足ADE≤.2且FDE≤.4。

- E：有效非STOP动作集合。
- B⊆E：物理合格且A/F/J均不差、至少一个严格改善、保留原成功的全部正编辑。
- T=B若B非空，否则T={STOP}。
- y=1[B非空]，只用于训练和事后评价，不进网络输入。
- s0：STOP分数；sE=max_{a∈E}s_a；m=sE−s0。
- P=B，N=E\B；p=max_P s，n=max_N s。

B仍含连续微改善，并非只含rescue。沿用这一标签是为了不同时改变任务含义；是否应改成rescue目标留待本系列之外。

## 4. 三个目标的精确定义

### I：原独立BCE

`L_I = .5 mean_{a∈T} softplus(-s_a) + .5 mean_{a∈eligible\T} softplus(s_a)`。

无负动作的窗口为0，所有窗口仍在batch均值分母中。必须复用原实现核验，不重写成全行普通均值BCE。

### S：原联合排序

`L_S = softplus(1 + max_{eligible\T}s - max_T s)`。

无负动作窗口0；max tie按action_id首个。原S也曾全量效果弱，必须保留这项反证，不仅比较tiny S。

### D：窗口机会判断＋条件排序

当E非空时：

`L_gate = softplus(-m)`，若y=1；否则`softplus(m)`。

当P与N均非空时：

`L_rank = softplus(1 + n - p)`；其他窗口`L_rank=0`。

总目标：`L_D = .5 L_gate + .5 L_rank`，最后对全部窗口取均值。

E空时两项均0，推理必STOP。P非空但N空时仍训练gate，rank0；B空时只训练gate压住最高编辑，不虚构正编辑或从GT挑一个替代。

关键差异：每个有编辑候选的窗口对STOP竞争提供一个固定系数的二元判别项，不除以负动作数量；排序项只负责在编辑之间区分正负。两个系数.5、margin1固定，不扫描。

gate使用m而非仅s0，是为了使机会判断能响应实际候选编辑分数；仅用原STOP行时看不到具体KIN动作特征。m两侧均反向传播，不detach最高编辑，不另加ground-truth oracle score。两个任务共享同一网络，仍可能发生梯度干扰，这属于明确限制。

## 5. 推理完全保持原样

对全部401分数屏蔽非eligible后argmax；STOP为action0，在相同分数时优先。等价于：E非空且sE>s0才执行最高分非STOP，否则STOP。

禁止把sigmoid(m)当已校准的未来概率，也不通过hold选择概率门限。阈值固定为分数差0是原argmax的等价写法，不是新调参自由度。

若D更常执行但保留/质量恶化，实验失败；不通过事后调高gate阈值把失败变成成功。

## 6. 同步记录，区分学到了什么

每seed、split、overall/stable/transition/change都保存：

1. 原生/模型coverage、rescue/lost/retained、A/F/J、Q与J各自单slot的早中晚误差。
2. D0定义的非STOP top1/5/10 B及rescue命中、margin、正确编辑受阻、错误编辑放行。
3. 窗口机会gate的TP/FP/TN/FN：预测为sE>s0，真值y；B机会与真rescue机会分开报告。
4. 训练loss分项、STOP/编辑分数分位数、梯度norm、max并列率、选中正/负动作；loss不是覆盖的替代。
5. 原GT-free完整K20 complete-link D=max(meanpath/.10,endpoint/.20)、有效good簇和distinct；不以复制候选或簇数更多冒收益。
6. dt=.4、补观测原点、速度12/加速度11/jerk10，全部五pooled motion与逐窗eligible。科学1.01不加floor/epsilon。

只存chosen不足以分析本问题；必须存全部401 scores、eligible/B/T、显式ID与完整终态路径。

## 7. 主统计与成功门

主要人口固定hold change，不在看结果后切回inner。主要两个比较：

1. D−S净coverage；
2. D−native净coverage。

I作为机制参照完整报告，但D−I阳性不能单独宣布成功。两项共同家族Bonferroni双侧97.5% CI，分位.0125/.9875，同时给95%描述CI。

按整个hold目标track配对bootstrap20000次，先抽track再抽配对三个seed，窗口加权、seed等权，rng20261027；另做固定seed只抽track的相同97.5%敏感性。含变化track至少20，无效draw>5%则UNDERPOWERED，不补抽。

“新目标有安全留出收益”的全部条件：

- 两项比较三seed均严格>0，均值均≥2pp；两种校正CI下界都>0。
- D在hold和inner各seed各组native-good lost率≤1%，原overall/stable/transition A/F保护≤1%恶化，五pooled motion全部literal≤native1.01；零分母明确NA，不伪造通过。
- inner变化组三seedcoverage均不低于native，作为次要方向一致性要求，不用inner反选方法。

这些是本次目标诊断门，仍不替代项目原major change ADE/FDE各≥5%、原联合CI/完整覆盖/相关性审核，也不自动满足部署成本。所有原门单独报告；仅目标门阳性但major未过，结论只能是有限机制进展。

成本限制：旧F16+KIN+editor约397100参数已高于305000预算，CPU缓存一致性仍有旧阻碍；本轮参数不增加不等于解决旧成本失败。没有预算达标就不宣布轻量部署创新完成。

## 8. 预注册停止规则

- 数学/ID/source/恢复/资源门不通过：保存失败来源，停止，不改阈值、损失或人口求PASS。
- D0科学阴性不用于临时选择另一个loss；若执行D1，严格上述唯一D。
- 训练完成后无论阳性阴性均结束有限系列，不新增第四目标、阈值或epoch扫描。
- D只有opt提升：标记迁移不足；D只更多执行：标记未改善有效决策；D仍全STOP：记录目标改变未解除问题。任何一种都不证明输入无信息。
- 所有原文件、模型、日志、枚举、自动记录保留；旧PAUSED不触动。

## 9. 资源与范围

实施前按当前inventory估计E，使用1.3倍动态remaining与Windows retained peak，做实际批次吞吐、保存/审核IO与磁盘估计；不套旧固定6GiB、30min或过期六小时限额，也不凭前轮运行时间承诺ETA。优先短预算，若完整科学审核预计明显超出该偏好，先交成本与停止点，不偷偷少评组或缩动作。

只新增独立execution目录，不改设计和旧锁源。未来实际启动后才安排唯一startup、一次预计elapsed半程及按真实阶段修正的完成节点；本设计交付不建立这些任务。
