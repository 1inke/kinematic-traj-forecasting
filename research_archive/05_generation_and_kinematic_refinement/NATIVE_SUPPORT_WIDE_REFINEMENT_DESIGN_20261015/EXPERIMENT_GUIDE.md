# 原生全候选支撑修正：安全机会与共享映射实验设计

日期：2026-10-15  
状态：**DESIGN_ONLY；未实施、未预检、未训练、未安排自动任务。**  
本文件是新方案，不修改或恢复已停止的 A0→A1/B、D0→U1 等旧协议。当前交付仅为设计文档。

## 1. 本次要回答的大问题

> 当前局部修正失败，是否首先因为允许修正的候选范围错误，而不是模型完全没有可用的运动信息？如果允许同一个共享模块修正全部20条原生路径，能否在严格运动学约束下获得可迁移的完整轨迹覆盖收益？

本次不是找“另外四个最好槽位”，不是增加候选预算，也不是训练评分器。两臂最终均为K20，参数、输入、损失、安全映射、训练预算一致，只改变**可修正掩码**：固定bottom4，或全部20条。

解释范围：这是“可编辑候选范围”对照。它同时改变有效可达集合、哪些winner可获得梯度、共享参数收到的监督构成；不能包装成纯粹的梯度干扰因果实验，也不能把相同参数量称为相同有效容量。

## 2. 依据与尚未解决的缺口

依据为[已完成A0报告](../GENERATION_FEASIBILITY_CONDITIONING_DESIGN_20261015/A0_execution_20261015/RESULT_ANALYSIS.md)。下表只列change组，完整A0包含全部组。

| 人口 | 窗口数 | native覆盖 | 固定bottom4几何上界 | all20几何上界 |
|---|---:|---:|---:|---:|
| FIT | 4725 | 1937 | 2024，+1.8413pp | 3108，+24.7831pp |
| outer | 253 | 110 | 113，+1.1858pp | 166，+22.1344pp |
| inner | 497 | 166 | 169，+0.6036pp | 288，+24.5473pp |
| original CAL | 700 | 243 | 251，+1.1429pp | 418，+25.0000pp |

覆盖指至少一条**同一完整候选**同时满足ADE≤0.2m、FDE≤0.4m。不能分别取两个slot的最小值拼接成功。

关键解释：

- FIT的all20新增机会为1171，固定4只捕获87，即7.43%；1084个窗口的新增机会不在当前固定4条修正域内。
- 固定4即便使用GT逐点构造、没有运动学约束，也不足5pp原目标。因此不再在旧固定4域内增加训练轮数、换优化器、扫描局部系数。
- all20的24.78pp是独立位置球的**放松几何上界**，不是安全可实现收益。A0两种上界路径均为16/16组未通过“全部五项运动学”联合门；这不等于每个组每一项都失败。
- all20扩大机会并不保证共享网络能学习，也不说明历史所有生成失败都由bottom4导致。旧M2/T1全候选生成失败、D0跨seed不一致等反例保留。

因此先做一次有限的安全机会筛查；筛查不过就不支付六次训练成本。

## 3. 阶段、产物与停止条件

| 阶段 | 操作 | 是否训练 | 进入下一阶段的依据 |
|---|---|---|---|
| S0 安全机会筛查 | 固定GT方向位置球提案，经同一原生相对安全映射；比较native、safe4、safe20 | 否，仅CPU保存数据 | 仅FIT change的safe20净覆盖≥5pp，并通过来源、同ID、数值、安全审核 |
| S1 共享映射配对 | mask4 vs mask20，3 paired seeds，25epochs | 是，条件触发的6个fresh run | outer主结果、完整覆盖、保护组和五运动学联合判定 |
| 后续分支 | 根据S0/S1结果决定跨时间/场景验证或停止 | 本文不启动 | 不自动增加第二种head、loss或评分器；新的精确协议另行形成 |

S0为非正式训练诊断，不能算一个训练seed实验。S1是一个正式配对实验、包含6次训练，不是6个独立研究问题。不存在为凑满三项而必须启动额外实验的要求。

## 4. 数据、冻结范围与身份合同

首次只使用A0已审核的target-grouped四个人口：FIT18909、outer1035、inner1856、original CAL2879，共24679个唯一ID；change分别为4725、253、497、700。

- 读取A0实际manifest定位不可变native20、GT、past/P2、旧分组、track/scene和slot映射，禁止按数组位置猜身份。
- FIT-only拟合输入统计；outer是唯一主评估集。inner与原CAL只事后评估，不选择checkpoint、超参、策略或候选掩码。
- 固定epoch25，不使用early stopping或最佳epoch选择。
- old F16、原decoder/scorer、native cache与所有旧模型/结果不修改。训练仅访问冻结缓存与新head，不声称重新验证全部旧模型GPU前向。
- 预测输入只有observed P2与当前原生候选；GT用于S0特权诊断、S1原监督损失和事后评估。GT/groups/IDs/scene不得进入head或安全映射。
- 旧logits只定义固定mask4，不当作新概率。按原生logit从高到低稳定排序，原slot处理并列；最后4个为bottom4，按physical slot升序登记。S1不得依据GT选择槽位。
- P2是冻结观察表征，可能已包含社会上下文；不能把本实验描述为只使用行人自身运动历史。

执行前须继承A0的1823项原始digest，并扩入A0完成产物和本次新源；数量以实际审计为准，不预报新的总数。不覆盖旧hash，不宣称覆盖全部历史文件。

## 5. 两阶段共用的安全输出映射

### 5.1 位置预算与离散导数

所有路径为以最后观察位置为原点的12步相对坐标，dt=0.4s；先核验缓存坐标合同。

\[
b_t=0.20(t/12)^2\text{ m},\quad t=1,\ldots,12.
\]

速度由位置前补零后差分，得到12个v；再差分得到11个a和10个j。对原生路径p和提案r，要求每个对应导数向量满足：

\[
\|D_q(p+\alpha r)\|_2\le1.01\|D_qp\|_2,
\quad q\in\{v,a,j\},\quad 0\le\alpha\le1.
\]

沿同一个候选的残差方向使用一个alpha，共33个约束。不能把旧文档的“31”简称作为实现数量；按实际12+11+10逐项核验。

### 5.2 确定性ray规则

参考只读[既有physical_ray.py](../GENERATION_NATIVE_SIMILARITY_FEASIBLE_SERIES_20261008/physical_ray.py)，原生导数向量记为x，残差导数为d：

\[
A=\|d\|^2,\ B=2x^Td,\ C=(1.01^2-1)\|x\|^2.
\]

取满足`A*alpha²+B*alpha−C≤0`的非负根并截断[0,1]；A=0时该约束给1。B≥0优先使用`2C/(sqrt(B²+4AC)+B)`的稳定形式，零分母按已登记零预算分支处理；B<0使用正根。全部约束取最小值，若小于1再乘`1−2^-20`数值内部余量。不能为提高覆盖搜索margin或余量。

新增共同数值规则：对实际输出逐候选、逐对应导数做**literal≤1.01**检查；不通过则该候选回退到自己的native路径，alpha记0并保存fallback原因。两臂、训练与评估同一规则，不按窗口整体回退，不根据GT回退。独立实现若对fallback分类不一致，停止实现核验，不加评价epsilon掩盖。

该映射保持位置预算。对应每个导数样本的范数上界在精确数学中推出组内均值与分位数上界；实际保存输出仍须完整重算五项pooled指标，不能用证明替代数值审核。

注意：这比旧“组内五统计”门更严格，是本实验两臂共同的安全子族。它可能过度保守。安全筛查失败只能否定此固定子族的充足机会，不能否定全部安全轨迹空间。

## 6. S0：先问安全机会是否足够

### 6.1 固定构造，不求解、不训练

逐candidate、逐时刻构造：

\[
r^{ball}_{kt}=\min\{1,b_t/\|y_t-p_{kt}\|\}(y_t-p_{kt}).
\]

距离为0时残差为0。随后使用第5节同一safe map得到r_safe。一次计算all20，再应用不同掩码：

- native：完全不变；
- safe4：只应用原固定bottom4的r_safe，固定16精确保持；
- safe20：全部应用各自的r_safe；
- A0 UB4/UB20：只作为已有放松上界参照，不当安全端口。

输出都是K20，不取GT最佳partner，不扫描方向、半径、根顺序，不做迭代求解。S0的GT导向路径是机会见证，**不保存为S1训练teacher、不蒸馏、不当部署输出**。

沿GT直线的alpha∈[0,1]使每一点误差不增加，应实际验证：native-good lost=0，逐窗口coverage满足native≤safe4≤safe20≤UB20，safe4≤UB4。违反则是实现或身份错误，不是科学发现。

### 6.2 唯一source gate

只用FIT change决定：

\[
100\,[N_{good}(safe20)-1937]\ge5\times4725.
\]

即至少新增237个覆盖窗口；同时源/ID、独立算术、native-good零丢失、全部FIT组literal五运动学通过。outer/inner/CAL即使更好，也不能挽救FIT gate失败。

额外报告safe20−safe4、各slot可救窗口数、alpha分布、alpha=0比例、实际非零残差比例、33约束中谁绑定、数值fallback数量及分母。它们用于说明机会损失在哪，不用于事后改参数。

S0通过只表示**这个安全路径族存在足够GT特权机会**；不意味着past可预测这些修正。S0未通过即记录`STOP_SAFE_OPPORTUNITY_UNDER_REGISTERED_MAP`，不启动S1，不自动放宽运动门或增加球半径。

### 6.3 S0产物

四个人口各保存native/safe4/safe20路径、GT/ID、slot mask、ball residual、safe alpha/fallback、导数约束与质量指标；保存`DECISION.json`的整数门算术、`ARITHMETIC_AUDIT.json`、`PRESERVATION.json`、`RESOURCE.json`、`RESULT_ANALYSIS.md/json`。不以重复native cache创建伪seed重复。

## 7. S1：同一共享模块，唯一改变可修正范围

### 7.1 模型与掩码

| 项目 | mask4 | mask20 |
|---|---|---|
| 预测输入 | observed P2 96D + 当前native path 24D | 完全相同 |
| 共享head | 120→128→64→24，GELU，25304参数 | 完全相同 |
| 实际计算 | 每窗口全部20条均计算head和safe map | 完全相同 |
| 可编辑路径 | 预先固定bottom4 | 全部20 |
| 输出/预算 | K20，未编辑16条精确native | K20 |
| 初始输出 | 完整native20 | 完整native20 |

不输入slot编号，不增加learned gate；每个slot共享同一套参数。最后层零初始化，前层按固定seed标准初始化；两臂完整参数、scaler和相关buffer配对，不warm start旧head。

共同统计：P2按18909个FIT窗口各计一次；native24按FIT×20个candidate各计一次。float32特征、float64总体mean/std累积，std<1e-6置1；二者拼为120维，用同一统计归一化再送float32 head。不得用mask4专属统计或outer/CAL重新拟合。

每步raw二维输出先float32 tanh，再转换float64做向量界：

\[
u_{kt}=\tanh(z_{kt})/\max(1,\|\tanh(z_{kt})\|_2),\quad r_{kt}=b_tu_{kt}.
\]

本实验使用free12输出，**不同时添加Bernstein或新时间basis**。两臂均经过第5节safe map，再用掩码组合；使用`where(mask, corrected, native)`保留未编辑路径，不用可能传播NaN的`native+mask*residual`。

### 7.2 损失：两臂都覆盖同一完整K20

\[
Q_{ik}=ADE(\hat p_{ik},y_i)+0.5FDE(\hat p_{ik},y_i),
\]

\[
L=\frac1B\sum_i\min_{k\le20}Q_{ik}
+0.05\frac1{190B}\sum_i\sum_{k<l}
\left[\max(0,0.3-\operatorname{mean}_t\|\hat p_{ikt}-\hat p_{ilt}\|_2)\right]^2.
\]

winner使用第一physical slot处理并列，平方hinge有190个pair。该完整K20 dynamic-winner形式可与只读[T1 objective实现](../MOTION_PATTERN_TRANSFER_DESIGN_20261015/T1_execution_20261015/model.py)核对；只借用公式，不恢复T1训练或其fixed-responsibility分支。

**重要合同变化**：它不是旧bottom4-only loss的逐位复制。两臂现在统一K20 loss，避免把“mask4→mask20”和“loss4→loss20”同时作为两臂差异。不得将新mask4与历史K4-loss实验直接当同协议对照。

mask4中若winner位于固定16条，该样本winner项对head无梯度；diversity项仍可能有梯度。这是被测editable范围的机制，不是需要强制改winner的bug。两臂初始loss应相同，初始梯度不要求相同。

不新增J-loss、teacher、固定responsibility、class balance、物理罚项、native-good监督保护项或scorer。运动安全由共同前向映射实现，不混入损失消融。

### 7.3 训练预算

- 三paired seeds：42、43、44；每seed两臂同初始参数、窗口shuffle序列。
- AdamW：lr=0.001，weight_decay=0.0001，betas=(0.9,0.999)，eps=1e-8，foreach=False。
- batch_size=256，25epochs，无gradient clipping、scheduler或early stopping，末批保留。
- 每run 74step/epoch、1850step；总6run、150epoch、11100step。两臂均计算20条，不通过只算4条制造吞吐优势。
- 仅新head训练，所有原模型与缓存不可变。资源和吞吐必须实测，不从参数量推断可用显存或生产时间。
- 六个final checkpoint分别评估FIT/outer/inner/CAL，共24个科学保存job。FIT人口指标不得用batch训练loss代替。

### 7.4 核心接口草图（设计，不是已验证生产代码）

```python
def forward(p2, native20, editable_mask, stats, head):
    # p2[B,96]; native20[B,20,12,2]; mask[B,20]
    features = concat_p2_and_native(p2, native20)  # [B,20,120]
    z = head(normalize(features, stats)).reshape(-1, 20, 12, 2)
    u = tanh(z).to(float64)
    u = u / maximum(1.0, vector_norm(u, axis=-1, keepdims=True))
    proposal = position_envelope * u
    correction, certificate = safe_native_relative_ray(native20, proposal)
    out = where(editable_mask[..., None, None], native20 + correction, native20)
    return out, certificate

def objective(out20, gt):
    # 两臂一致；不能只对mask=True的候选选winner。
    q = ADE(out20, gt) + 0.5 * FDE(out20, gt)
    return first_slot_min(q).mean() + 0.05 * all_190_pair_squared_hinge(out20)
```

`safe_native_relative_ray`只接收路径，不接收GT/group/ID/logit。safe root、literal fallback及dtype合同必须在真实代码中独立实现和测试，不能把这段草图当预检PASS。

## 8. 评估：避免安全成功掩盖覆盖失败

### 8.1 每个split、每个seed、每个组都报告

1. 同candidate覆盖、J=min_k max(ADE_k/0.2,FDE_k/0.4)、minADE、minFDE；J与Q的slot分别登记。
2. native→每臂、mask4→mask20的rescue/lost/retained，完整ID名单和分母；native成功与失败两个固定子集分别报告。
3. 最接近J的单条轨迹早/中/晚误差；不能分段选择不同slot拼接。
4. GT-free全部K20原SciPy complete-link：meanpath阈值0.10m、endpoint阈值0.20m；有效good簇、distinct簇、重复候选。簇越多不是目标。
5. 每组全部路径pooled speed p95、acceleration mean/p95、jerk mean/p95，dt=0.4。literal≤native×1.01逐seed逐组通过，不加epsilon。
6. 修正范数、alpha分位数、零alpha/fallback、实际动了几条、33约束绑定频率；固定16的bitwise保护及全K20有限性。
7. Q-winner与J-winner slot、可编辑winner占比、winner切换、loss quality/diversity分量、真实参数梯度范数和全150行训练记录。

安全路径仍可能离GT更远：S1不具有S0的GT方向单调性。因此不能根据安全证书声称native-good不丢失；需真实报告lost并接受保护门约束。

### 8.2 主统计与预注册比较

唯一主人口为outer change。预注册8个主比较：mask20−mask4及mask20−native，各自对coverage/ADE/FDE/J计算差值；coverage正为好，其余负为好。其他split、组和敏感性阈值均描述性，不替换主结果。

使用同ID配对、target-track聚簇与3个paired model seed的交叉bootstrap：PCG64种子20261015，20000次；每次有放回抽outer target tracks和3个seed，同一抽样应用所有端口，track内保留全部窗口，按窗口加权。native按同一次seed抽样复制用于差值，不当作三次独立native模型。

报告每seed原始效果、普通95%探索CI，以及对8个主比较Bonferroni同时CI：两侧分位点为0.05/(2×8)与1−0.05/(2×8)。这是注册家族，不因某项不显著或未报告而缩小。只有3个seed、重复窗口及既有F16见过的数据限制明确保留；不制造可靠独立scene推断。

### 8.3 决策层级

**执行通过**：代码、源、ID、数值、恢复、保存算术通过；不等于科学成功。

**值得进入新的跨时间/场景确认设计**须同时满足：

- outer change上mask20相对mask4平均coverage至少+5pp，并且相对mask4和native的coverage同时CI下界均>0；三个seed两种对照的coverage均正向。
- outer change的ADE、FDE相对mask4三个seed均不恶化；overall/stable/transition的ADE/FDE相对native与mask4均≤1%损害。
- 四个人口全部seed、全部组五pooled motion均通过native1%门；来源与真实保存算术全部通过。
- 报告native-good丢失、effective-good与候选重复，不能用大量复制、只展示rescue或只挑CAL好结果支持推进。

**重大结果**仍要求原change ADE与FDE各至少5%收益、三seed一致、相对native的ADE/FDE/J同时CI排除0、同路径覆盖与全部保护/运动门通过，并审查scene/track相关性。上述推进门不替代重大结果门，不自动升级baseline或完成总目标。

0.3/0.5覆盖仅固定敏感性，不因主门失败换标准。若指标间冲突，完整报告并按未通过处理，不合成新的综合分数挽救。

## 9. 必须先通过的实现审核

### S0

- 原ID/GT/native/group/track/mask逐项一致；native坐标原点与dt核验。
- 零速度/零加速度/零jerk、零残差、恰到边界、严格越界和每一种约束绑定的固定fixture；GT修改不影响safe map，只影响S0提案。
- 根公式与独立固定70次二分仅在数值fixture比较；不是生产搜索或按效果选择路径。
- 原同路径覆盖、独立NumPy运动学、原SciPy全部K20聚簇、嵌套和零lost实际保存重读；科学分类literal，单独登记算术容差，不互相挪用。

### S1

- 两臂真实GPU各8个FIT256步骤分阶段检查合法参数梯度finite/连接，零最后层造成早期前层零梯度如实记录；不能用roundoff伪称有效连接。
- 25304实际参数计数、paired全部权重/buffer/scaler、初始native20身份、mask4固定16、有限性、位置/导数上界。
- 控制用同一actual非零参数检查mask20输出与mask4可编辑4条一致；未编辑16条必须native。all-false mask是native，无NaN梯度。
- 两臂各自的完整K20 objective独立NumPy标量和Torch梯度实现核验；并列/重复native的不可微点单列fixture，不用跨活动面有限差分冒充导数证明。固定非零double副本FD使用1e-4、1e-5并登记winner/root/max活动面；活动面变化时不报告平滑FD通过、不搜索新的epsilon求PASS。
- 双臂真实513个FIT+12个inner子集2epochs连续/中断，参数、全部buffer、optimizer、TorchCPU/CUDA/loader/NumPy/Python RNG、history及diagnostic EXACT。明确不是完整18909两轮或分析恢复。
- complete-run SHA-skip、错误source/output/resume合同实际拒绝；actual128保存重读的质量、全部聚簇、运动学、安全映射独立重构。
- 全部实际通过才锁科学源；以后不得为warning、显示bug或结果不理想静默更换源码/hash。

若发生源/身份/科学含义/资源实质阻碍，保留failure和partial，停止；routine数值实现由执行者处理，但不得修改预注册科学阈值使结果通过。

## 10. 资源、流程、产物

每个生产阶段先实测4warm+32measure训练批次（S0则固定8个真实128样本块），再包含完整评估、聚簇、CI、保存重读、源SHA与IO估计，不只报GPU计算时间。**单个阶段规划超过30分钟就不启动**，报告原因，不临时减少epochs、seed或审核项。现在无生产ETA。

process≤6GiB；system physical和commit各保留≥6GiB；不改OS/pagefile，不关闭用户程序。一次一个模型，按batch有界保存。旧暂停任务保持暂停，旧完成PID不查询、不重训。

未来实际执行采用唯一按需0-trigger OS任务、byte-lock；一次startup、一次预计50%检查、一次实际完成核验。半程根据真实阶段修正ETA，禁止高频对话poll或每天重复节点。worker与exit必须双SUCCESS且审核完整，不能以epoch100%代替完成。

每run至少：`checkpoint.pt`、`run_manifest.json`、`stats.npz`、`split_ids.json`、`history.json`、`metrics.json`、`assignment_stats.json`、四人口保存路径及`done.json`。manifest必须绑定：

```json
{
  "status": "DESIGN_SCHEMA_NOT_A_RUN",
  "factor": "editable_mask",
  "arm": "mask4_or_mask20",
  "source_sha": "resolved_at_execution",
  "split_sha": "resolved_at_execution",
  "statistics_sha": "resolved_at_execution",
  "native_cache_sha": "resolved_at_execution",
  "slot_mask_sha": "resolved_at_execution",
  "winner_assignment": "full_K20_ADE_plus_0.5FDE_first_physical_tie",
  "diversity": {"pairs": 190, "type": "mean_squared_hinge", "margin": 0.3, "weight": 0.05},
  "safety": {"margin": 1.01, "constraints_per_candidate": 33, "literal_candidate_fallback": true},
  "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0001,
  "batch_size": 256, "epochs": 25, "grad_clip": null,
  "checkpoint_selection": "fixed_final_epoch",
  "bind_initial_parameters_and_all_optimizer_rng": true
}
```

完成后独立CPU审核全部24科学job的完整质量/聚簇/运动学、安全重构、固定16、paired CI、六manifest与150history及来源输出SHA。保存算术复核不是所有模型/梯度第二次回放，不是外部独立评审。研究报告保留所有seed和失败组、源清单、生产资源峰值及未触发阶段。

## 11. 如何依据结果转向，而不是继续小修小补

| 结果 | 支持的判断 | 下一步 |
|---|---|---|
| S0 safe20机会仍<5pp | 本固定半径+保守ray安全子族机会不足 | 停止此子族训练；重新考虑能改变时间形状的结构级可行生成，不扫描本图中的margin/半径 |
| S0通过；S1连FIT都不改善 | 特权机会存在，但当前共享映射/监督/优化没有学到 | 不先归咎分布偏移；汇总训练winner可编辑性、收缩与拟合证据，再选择一个不同大方向 |
| S1 FIT改善、outer不改善 | 学到训练拟合，但转移证据不足 | 优先泛化/条件分布，而不是继续增候选或改scorer；不由此独证共享干扰 |
| S1安全、outer覆盖增益不足或CI跨0 | 支撑范围改变仍不足以形成可信收益 | 不宣布理解运动；停止当前配方自动扩展 |
| S1覆盖改善但保护组/运动门失败 | 新机会伴随不可接受损害 | 不进入评分器或部署，不能放宽门槛 |
| S1全部推进门通过 | 共享模型在更广原生支撑上学到可迁移修正的初步证据 | 才为time与scene各制定相同协议的确认实验；新IDs不能与target跨研究强配对 |

不自动回到R1–R5选择器、T1固定标签、D0梯度投影、MoE、teacher蒸馏、loss/scorer并改或局部参数网格。每个失败只否定注册的范围，不外推“所有生成都无用”或“过去观察不含信息”。

## 12. 反方检查与最终可写结论

依据academic-research-suite做同助手设计反方检查，不声称外部评审：

1. **最强反驳：all20只是自由度更多。** 接受这一点。因子就是可编辑支撑范围；相同参数不消除可达集合/责任变化，结论限制为该结构配置。
2. **最强反驳：S0使用GT，学不到。** 因此S0不当teacher或成功模型，S1才检验GT-free映射。S0是有限安全见证，不是整个安全空间的精确上界。
3. **最强反驳：safe ray太保守。** 因此S0失败只停止这张安全映射，不宣称运动理解不可能；安全成功也不保证native-good保留。
4. **最强反驳：新mask4损失已不同于旧实验。** 明确记录K20公共目标；只以本次paired A/B作主要归因，不把旧K4实验拼成一个单因素链。
5. **最强反驳：反复复用outer和CAL。** 本系列是适应性探索，F16已见旧pool，target-disjoint不等context-disjoint；三seed与相关窗口不足以宣称独立whole-model泛化、SOTA或心理意图理解。

允许的积极结论是：“在固定候选预算、冻结原模型和统一安全映射下，扩大可编辑原生支撑带来了经配对验证的完整轨迹质量收益。”是否成立须由S1真实结果决定。当前没有新结果，也没有任何已通过预检的声明。
