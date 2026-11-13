# 解码端优化调研与方案评估

日期：2026-10-23。对象：`DECODER_OPTIMIZATION_SURVEY_AND_DESIGN_20261023.md`（用户提供的 Gemini 目录文件）。

本次仅评估：核对原文、项目最新报告及本地实现，并查验会议官方页面、作者论文和代码。没有启动实验、修改科学代码、重训模型或重新审核全部保存数据；原调查文档未修改。参考文献核验采用 nature-ref-verifier 流程，由同一助手完成，非外部审稿。

## 1. 总判断

**这是一份有价值的方向清单，但不是已经可靠到可以直接执行或引用入论文的调研报告，建议大修。**

它正确地把视野从细小参数调整移向：有限候选集合的联合生成、方向与时间过程分解、历史场景信息。然而，它在三处跨越了证据边界：

1. 把“相关论文采用某种机制”写成“该机制已经解释本项目失败的根因”。
2. 把“结构可能改善某个指标”写成数学保证，尤其是加速度、冷启动不退化和锚点保留。
3. 把尚需适配和核验的代码称为可直接复用，并给出未经完整参数核算的轻量化预算。

不是“论文都是假的”，也不是“这些方向都不能做”。应当保留研究假设，撤回保证性结论，拆开混杂因素，再决定少量测试。

## 2. 最新证据已改变建议起点：T1不是待执行

调查文档还建议先跑T1。实际T1已经完整完成，见[最新T1报告](../F16_CONTEXT_TAIL_RETENTION_DESIGN_20261023/T1_execution_20261023/RESULT_ANALYSIS.md)。执行与证据审核通过，但科学判断是 `TRANSFER_UNRESOLVED`、`MECHANISM_SUPPORTED=false`、`BASELINE_NOT_READY`。

| inner变化组：497窗口、35条track | seed42 | seed43 | seed44 |
|---|---:|---:|---:|
| 原生完整好轨迹覆盖 | 166 | 166 | 166 |
| A_replay覆盖 | 186 | 191 | 192 |
| P_relative覆盖 | 167 | 175 | 197 |
| A丢失原生成功 | 33 | 37 | 37 |
| P丢失原生成功 | 50 | 37 | 30 |
| A覆盖净增保留比例 | 5% | 36% | 119.23% |

主指标平均正相对退化R仅下降6.20%，低于预定20%；P−A配对track+seed 95%CI为[−0.0725033,+0.0173602]，跨0。P的24个组级五运动学联合门均未通过，主要仍是accel_mean。

更关键的是：在旧A/native事先固定的人口中，P改善了旧lost及部分neither，却伤及旧retained和rescued。这说明当前平均单侧惩罚会发生收益交换，不是逐窗口保护约束。

因此应改写为：**“固定相对J惩罚未能稳定兼顾误差与覆盖保留；下一步需检验新的机制，而不是预设T1已经切中唯一根因。”** 不应重复T1，也不能把一个配方阴性外推成所有质量约束无效。

## 3. 文献核验：哪些是真实支持，哪些是过度转述

以下核对原文所给作者、标题、会议、年份及机制。原文没有提供DOI/页码，本表不冒称完成所有书目字段的数据库审计。

| 文献 | 核验结论 | 真正可以借鉴 | 文档需要修正的表述 |
|---|---|---|---|
| DeMo，NeurIPS 2024，Bozhou Zhang等 | 标题、作者、会议相符 | mode/state queries分工及动态状态建模 | state query不等于固定低秩曲率基；原方法包含Attention/Mamba，不能据此保证本项目全部五运动学达标 |
| SmartRefine，CVPR 2024 | **作者错误：应为Yang Zhou等，不是Haosheng Zhao** | 预测质量分数、自适应修正次数及上下文 | 确实讨论过度修正，但不是原文所写sigmoid残差门控的直接复现，更无GT成功集合不退化定理 |
| aWTA，ICRA 2025，Yihong Xu等 | 书目信息相符；**实现转述有关键遗漏** | 退火软责任分配 | 作者实现对责任权重停止梯度；不能承诺彻底消除尾部发散或本项目lost |
| TUTR，ICCV 2023，Liushuai Shi等 | 书目信息相符 | mode-level建模、减少后处理、并行多模态输出 | 不是“所有GT-free选择必然失败”的证明；本项目mean/max SetMix是自拟适配，不是原方法原样复现 |
| EigenTrajectory，ICCV 2023，Inhwan Bae、Jean Oh、Hae-Gon Jeon | 书目信息相符 | 数据驱动低秩轨迹描述与锚点细化 | 低秩不自动等于平滑、低加速度或安全；也不等于与native分段基正交 |
| SingularTrajectory，CVPR 2024，Inhwan Bae、Young-Jae Park、Hae-Gon Jeon | 书目信息相符；**输入条件被误述** | 奇异空间、环境自适应锚点 | 自适应锚点使用图像分割所得可通行图，不是完整免图像历史轨迹先验方法 |
| QCNet，CVPR 2023，Zikang Zhou等 | 书目信息相符 | 查询中心编码、proposal/refinement组合 | 不能作为native好候选保护或gamma门控的理论保证 |
| SHENet，NeurIPS 2022，Mancheng Meng等 | **正式标题错误** | group trajectory bank可作为历史轨迹先验的相关工作 | 正式标题是Forecasting Human Trajectory from Scene History；完整模型还用场景图像、Swin场景编码器和跨模态交互，并非免RGB |
| NPSN，CVPR 2022，Inhwan Bae、Jin-Hwi Park、Hae-Gon Jeon | 书目信息相符 | 小型可学习采样器改善有限采样 | 潜变量采样空间的覆盖不等于本项目同路径ADE/FDE门、稀有成功保留或运动安全 |
| MemoNet，CVPR 2022（正文提到、参考文献漏列） | 应补正式引用 | 训练实例记忆、目的地检索与条件路径生成 | 不是现有二维直方图模块的等价实现，也不提供未知场景无损保证 |

关键原始来源：

- [DeMo作者全文](https://arxiv.org/html/2410.05982v1)：状态序列建模及消融，主场景并非本项目ETH/UCY的12步K20设定。
- [SmartRefine正式会议页](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_SmartRefine_A_Scenario-Adaptive_Refinement_Framework_for_Efficient_Motion_Prediction_CVPR_2024_paper.html)与[作者全文§3.2.4、§4.4](https://arxiv.org/html/2403.11492v1)：高预测质量时跳过细化、按预测质量控制迭代；原文也观察到并非每条轨迹受益。
- [aWTA作者代码](https://github.com/valeoai/MF_aWTA#5-train-your-model-with-awta-made-easy)：示例明确含`.detach()`。
- [TUTR官方会议页](https://openaccess.thecvf.com/content/ICCV2023/html/Shi_Trajectory_Unified_Transformer_for_Pedestrian_Trajectory_Prediction_ICCV_2023_paper.html)。
- [EigenTrajectory官方会议页](https://openaccess.thecvf.com/content/ICCV2023/html/Bae_EigenTrajectory_Low-Rank_Descriptors_for_Multi-Modal_Trajectory_Forecasting_ICCV_2023_paper.html)。
- [SingularTrajectory作者全文§3.4](https://arxiv.org/html/2403.18452v1)：明确通过输入图像与语义分割生成可通行图。
- [QCNet官方会议页](https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.html)。
- [SHENet作者全文§3.3](https://arxiv.org/html/2210.08732v1)：轨迹和场景图像分别进入编码器，场景编码器使用预训练Swin；[NeurIPS正式论文](https://proceedings.neurips.cc/paper_files/paper/2022/file/9e3b203e72c4e058de26d02a92a81844-Paper-Conference.pdf)。
- [NPSN作者论文](https://arxiv.org/abs/2203.13471)。
- [MemoNet官方会议页](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_Remember_Intentions_Retrospective-Memory-Based_Trajectory_Prediction_CVPR_2022_paper.html)：Chenxin Xu、Weibo Mao、Wenjun Zhang、Siheng Chen，Remember Intentions: Retrospective-Memory-Based Trajectory Prediction。

因此，文档所称“系统梳理”还缺检索日期、关键词、检索库、纳入排除规则、任务/输入/候选预算/成本对照。更准确的定位是“选择性相关工作与设计启发”。“领域已经完成单阶段范式转移”也过强：其引用中本就包含多阶段细化和扩散方法。

## 4. 会影响实验结论的六项技术问题

### 4.1 当前softWTA不等于作者aWTA

本地[src/f16/hybrid.py](../../src/f16/hybrid.py)的`set_loss`对`softmax(-q/T)`未停止梯度，而且函数还包含coverage/entropy项。它自述只是building blocks，不是可直接运行的已审核实验。

令 w_j=softmax(−Q_j/τ)，L=Σw_jQ_j。若不detach：

`∂L/∂Q_j = w_j [1 + (Σ_k w_k Q_k − Q_j)/τ]`。

反例Q=[0,1]、τ=0.25：较差候选权重约0.017986，但其导数约−0.052665。即在独立误差坐标意义上，梯度下降会推动该项误差增加，而不是“所有候选共同向GT收敛”。

作者式适配的关键语义应是：

```python
# 示意，不是本轮已实施代码；Q沿用本项目ADE+0.5FDE，因此仍是适配。
w = torch.softmax(-Q / tau, dim=-1).detach()
loss_awta = (w * Q).sum(-1).mean()
```

此时对每个Q的偏导非负，但共享参数更新仍可能伤及其他候选/窗口；也没有安全保证。文档的0.25→0.02温度只是待预注册超参数，不能称作者验证了本项目8轮下该区间。若试验aWTA，应只替换任务责任分配/损失，原diversity等先保持；不能顺便加SetMix或删排斥项。

### 4.2 “非赢家外飘导致minADE/FDE变坏”还不是已证根因

固定一组预测时，把非最优候选变得更差，并不会直接提高仍保留的minADE/minFDE。它可能通过共享参数和训练竞争间接影响赢家，但需要梯度、赢家切换或单因素对照证据。

现有endpoint hinge仅在距离小于0.3米时激活，超过margin其梯度为零，不能描述成对所有候选持续向外推。neither人口的误差贡献分解只定位“哪里变差”，没有证明“为何变差”。此外，最终20槽都有WTA责任，不支持直接断言完全模式坍缩；也不能反过来由全槽活跃排除训练早期责任失衡。

同样，S1相对早期学生同时改变输入和容量；冻结上游已见inner，不是干净的全模型未见泛化比较。

### 4.3 分段基底和正交残差都不保证accel_mean门

同样是4/8/12节点分段线性轨迹，只要把非匀速锚点坐标乘2，加速度幅值及均值便可乘2，稀疏支撑位置却完全不变。因此“与native具有相同稀疏加速度支撑”不推出“≤1.01 native”。

在固定dt与评价时间索引下：

`mean_acc = Σ_t ||v_(t+1) − v_t|| / (11 dt)`。

把一次同向速度变化分摊到多个时刻，总变差可以不变；平均加速度并不因为非零时刻增多就必然增加。若方向反复、幅度/边界改变，才可能增加。因此“S1均值上涨是折线变曲线的必然现象”应删除，改为描述性关联。选定J-slot的GT加速度误差改善也不能推广成ALL20都更真实。

`B Φᵀ=0`是位置空间正交，不推出二阶差分空间正交，更不推出逐轨迹、逐组的五运动学上界。若要残差在4/8/12节点为零，应显式设置节点约束；若要限制总变差/速度，需要另外的可验证约束。任何新增约束均会改变可达集合，必须报告coverage代价，不能为了达标悄悄投影。

低秩方向仍值得研究：它改变的是输出时间结构和归纳偏置。但r=2是否容得下完整好路径、基底是否真平滑、是否牺牲转弯/停留，必须测；“保留99%能量”本身也不是尾部轨迹误差或安全证明。

### 4.4 参数预算与结构描述不自洽

当前已实测部署路径为冻结encoder 267159 + student 82008 = **349167**参数，已经超过文档305000预算；不是82008整模。若encoder不变，满足305000最多只剩37841参数给新decoder。

按实际[core.py](../F16_CONTEXT_COMPLEMENTARITY_DESIGN_20261022/execution_20261022/core.py)保留trunk、queries、fuse，仅将128→24换为128→10（6锚点数值+4残差系数），省1806个参数，仍有**80202**，未计SetMix和gate。

文档公式SetMix输入是384维（自身+均值+最大值），而图中只有256维（自身+均值），须统一。按384→h→128、两层含bias估算，新增`513h+128`；h=16便新增8336，合计88538，再加gate。故“约78k且低于82k”未获当前结构支持。若要删层/缩宽，必须明确并做容量对照。

4维prior→16维线性层是64个weight，含bias则80；若只使用当前2维endpoint，则是32+16=48。参数增量小不等于检索延迟、建库内存、全模MAC和实际延迟小，均应另报。

### 4.5 现有场景先验接口尚不能直接实现文档方案

核对[src/f16/scene_prior.py](../../src/f16/scene_prior.py)：

- Fragment虽保存midpoint，`modes()`实际只返回endpoint直方图前m个格子中心，不返回“对应同一历史路径的终点+中点”。
- 回退到global可能有较高support；“局部不支持”不等于`support<=5`。`cold_start`当前仅表示没有eligible记录，不等同于新场景冷启动。
- 尚无文档所说的神经先验门控和自动无损回退。这些是待实现设计，而非现成属性。
- 静态FIT库、同录制过去已结束片段的因果库是不同信息条件。留出场景没有该场景训练轨迹；在线因果库还要计暖启动及收集成本。当前target的整条track排除、跨录制ID命名空间、时间单位和片段真实终点必须绑定。

只把终点先验做得更准，也不代表前12步存在同一条ADE/FDE同时合格路径。A0终点救回潜力是信息探针，不是完整候选覆盖成绩。原[DESIGN_AB.md](../../DESIGN_AB.md)还有M1、SHUF、XSCENE及NLL/置换优势等门，不能简化为只有“>10%终点潜力”。

### 4.6 “门控归零就绝不退化”需要更严格前提

条件分支gamma=0，只说明当前模型停用该分支。如果主干、queries或集合通信层已通过联合训练改变，它不会自动等于原来无先验基线。

同理，前16个query没有加先验不代表前16条轨迹保持不变：共享主干、输出头及SetMix都能改变它们。

严格输出回退需要“同一冻结基线+确实为零的附加残差/独立分支”，并逐元素验证；这只能保证选择回退时恢复基线，不能保证门控会正确识别所有有害场景。SmartRefine提供了值得借鉴的质量感知思路，不提供此处的逐窗GT保证。

## 5. 建议保留与调整的研究优先级

### 优先方向A：结构化时间过程解码，但把安全承诺改为待检验假设

这是最贴近本项目“覆盖增加、平均加速度上涨而峰值/jerk下降”的方向。首轮只比较自由坐标与固定结构化时间输出；不同时加SetMix、门控先验和aWTA。先固定可审核的基底构造与系数定义，进行有限无训练可达性/导数检查，再决定一个小型配对训练。

过去已有Bernstein、运动过程、投影、原生相似变换等阴性结果，不能以“DeMo-Basis”新名称绕过。新测试必须说明与它们在输入、完整K20生成、可达集合、容量或责任机制上的可辨识差异。若只是换基底/秩反复搜索，不值得重开。

需要回答：改善是否来自更好的时间结构，而不是容量变化或把路径限制得更接近native？覆盖/误差/原生成功保留与全部五运动学应同时报告；新增GT相符度可作诊断，不事后替代旧门。

### 优先方向B：场景历史信息探针，先不训练新网络

它有机会增加现有观测表征缺少的信息，比继续堆模块更值得快速证伪。保留原A0全部对照，严格区分静态与因果库。检查正确局部先验相对同边缘分布置换、全局heading及运动基线的增量，并单列未知场景、暖启动、低支持度。

可先并行于文档修订做设计准备，但本轮没有执行。未通过信息门就停止，不靠增加embedding、换检索半径或用GT挑库挽救。通过终点探针后，仍需完整路径及等K预算测试。

### 次优先方向C：候选集合协调/责任分配，拆成两种实验

- 正确的stop-gradient aWTA适配：固定网络和其他loss，仅变任务责任分配/温度调度，记录责任占用、切换、每槽梯度及tail/lost；不能预设消灭发散。
- SetMix：固定loss和输出参数化，分别对照跨槽信息与参数量匹配的逐槽MLP，检查互补覆盖而非只看几何分散。自身/mean/max维度及候选置换行为要明确。

二者单独有证据后才能组合；若均失败，不自动扩展温度、层宽、query数扫描。

## 6. 文档修订清单

1. 纳入T1完成结果，删除“待执行”和“机制完全命中”措辞。
2. 修正SmartRefine作者和SHENet标题，补MemoNet；将免图像历史库与视觉可通行图分开。
3. 对所有公式标明“原论文公式 / 本项目适配 / 本项目新假设”，不借论文名替代证据。
4. 撤回“彻底消除”“天然保证”“必须放弃所有两阶段”等绝对表述。
5. 修正aWTA停止梯度语义，明确是否保留其他loss，冻结温度选择来源。
6. 给出逐层参数表及端到端成本口径；305k若为硬目标，先解决与349167现状的矛盾。
7. 时间基底、SetMix、aWTA、先验分别检验，不称组合包为单变量。
8. 将本地探索与最后无上下游暴露的正式五折验证分开；候选oracle改善不直接写成理解行人意图。

**最终建议：保留“结构化时间过程＋有限集合协调＋有来源约束的场景先验”三条大方向，但不要直接执行文档当前的组合方案。先修正实现语义与证据解释，再优先选择能区分机制的最小测试。**
