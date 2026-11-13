# 共同合同：固定证据、统一比较、限制结论

2026-10-26｜仅设计。与四步文档共同生效，任何一步都不得用自己的指标替换本合同。

## 1. 已核对的依据

- [最新E1/E2结果](../../qoder_work/K20_SINGLE_EDIT_COMPLEMENTARITY_EXEC_20261025/E1_execution_20261026/RESULT_ANALYSIS.md)：机会足够，但训练内与留出识别均弱；运动学通过不等于保留/成本成功。
- [实际E1协议](../../qoder_work/K20_SINGLE_EDIT_COMPLEMENTARITY_EXEC_20261025/E1_execution_20261026/protocol.json)：324维、12/11/10、当前标签/物理/恢复规则。
- [实际特征源码](../../qoder_work/K20_SINGLE_EDIT_COMPLEMENTARITY_EXEC_20261025/E1_execution_20261026/feature_builder.py)：历史42＋context192已经存在。
- [E0完整见证](../../qoder_work/K20_SINGLE_EDIT_COMPLEMENTARITY_EXEC_20261025/E0_audit_resume_20261026/RESULT_ANALYSIS.md)：固定一次动作可达性，不是无GT可学性。
- [旧R4](../../PED_REPAIRABLE_ANCHOR_SERIES_20261011/R4_CROSS_TRACK_NEIGHBORHOOD_DESIGN_20261013/production_20261013/RESULT_ANALYSIS.md)：几何支持高不等于标签富集，不能复用其距离作为本轮已验证语义距离。

根执行目录旧RESULT_ANALYSIS仍是实施前口径阻碍快照，引用最终结论必须用E1子目录，不能按文件名猜最新。

## 2. 数据及冻结范围

复用原显式ID划分，不重选有利样本：FIT_opt4096、FIT_hold3825、inner1856/change497；池seed42/43/44。native和KIN坐标、DT、来源、slot顺序、标签都由原保存SHA绑定，不生成新候选。

- `track_key=sequence|decimal_pedestrian_id`；opt与hold整目标轨迹隔离，inner仍沿原身份。
- FIT_hold是主要诊断留出集；inner仅在每步全部预定模型/配置锁定后作一次次要迁移检查，不用于挑winner、阈值或epoch。两者均已被历史研究观察，不能叫全新确认集。
- 外层outer、原CAL、新fold/外部数据本轮不访问。生成器可能训练见过这些输入，隔离仅针对新诊断头；不可声称整模型未见泛化。
- 旧tokens与候选只证明固定输入，不自动证明完全past-only。实施前核对原邻居/有效掩码来源：若依赖未来存活筛选，保留此风险；可做固定缓存条件下的有限诊断，但不得给“部署时可用信息”结论。新增邻居支路必须通过第3步更严格past-only门。
- 既有运行的全部source/input/output SHA先核验；不修改旧digest。精确缺失数组只能按已批准无损档案恢复并核原SHA，不能删除、覆盖或重训重造。

## 3. 动作、标签、特征

动作0=STOP；动作`1+20*r+l`将native槽r替换为KIN槽l，其他19条EXACT不变。只一次选择，最终K20；没有序列状态、oracle fallback、事后阈值。

沿原动作eligible：完整K20逐窗五motion≤1.01native，dt=.4，补观测原点，速度12/加速度11/jerk10，linear p95，float64算术，literal门，无floor/评价epsilon。STOP始终有效。仍须另检跨窗pooled门。

质量定义固定：`A=min ADE`，`F=min FDE`，`J=min_k max(ADE_k/.2,FDE_k/.4)`；good为同一个完整candidate的J≤1。B为物理有效、A/F/J均不变差且至少一项严格改善、保留nativegood的全部动作。B非空T=B，否则T={STOP}。

**所有新拟合仍用原T标签，不在本系列改成rescue-only目标。** 同时把B拆为：真rescue、原成功保留下连续改善、原失败但尚未跨覆盖阈值的连续改善；保存margin与基率。用于解释目标对齐，不据此重新赋权或挑训练人口。

当前324维索引（Python左闭右开）：

|索引|内容|
|---|---|
|0:42|原始past位置、速度、加速度|
|42:234|最后token96＋mean8tokens96|
|234:266|native均值轨迹、endpoint协方差、五motion|
|266:323|被删/换入轨迹及差异、集合motion差、距离|
|323|is_STOP|

“geometry-only”只留234:324，共90维；它不是无信息基线，因为native/KIN自身由历史生成。is_STOP不得标准化，其余统计opt-only、每窗等权/eligible动作等权，原方差下限规则保持。步骤1复用原opt统计，不在tiny128上重新拟合。

## 4. 工作量受限的共同动作训练样本（仅步骤2/3）

树模型不展开所有约164万动作行。对每窗口eligible内，按原T分正/负：每类最多16行，无放回，固定hash排序。若STOP在该类，必须保留STOP，再取最多15个其余动作；不足则全取。hash串`info-probe-v1|pool_seed|decimal_ID|action_id`，标签只决定训练分层，不进入特征。

总行数≤4096×32=131072/seed；两模型族与步骤3共享同一动作表。推理/最终评价始终枚举全部401动作，不把GT选择的训练子集用于部署。

为使STOP强制保留不改变原类均值目标，行权重为`1/(2*n_class*pi_action)`：全取/STOP的pi=1；有STOP且n>16时其余pi=15/(n−1)；无STOP且n>16时pi=16/n。窗口两类总权重各0.5。无负类窗口训练权重0，但保留原全4096分母/结果。

固定hash抽样只是原全动作目标的有限近似，不假称与旧E1逐位相同。新的MLP和树使用同一表/同一权重，旧I/S checkpoint仅历史参照，不能作为严格模型族对照。

## 5. 指标与统计

每步报告训练/hold/inner原生与方法的全组结果；步骤1仅tiny训练内，不解锁留出。核心指标：

- hit_B / B非空窗口；STOP在B空窗口的正确率；整体选中T比例。
- 真rescue机会利用：在存在eligible安全rescue的原失败窗中，实际救回比例。这个条件人口依赖GT，只能事后描述；同时保留全部窗口净coverage。
- 已执行动作的有效/无效/有害；STOP遗漏；good rescue/lost/retained；A/F/J与完整候选簇，不以分类准确率替代路径效果。
- 每seed/group五pooled motion、native-good lost率、protected误差。原1%/major change A和F各5%及项目其他门不放宽。诊断信号与部署资格分列。

步骤2/3共六个预注册主要差值，均为**hold change净coverage**：T_full−M_full、T_full−T_geom、T_full−T_perm；C_token−C_base、C_neighbor−C_base、C_both−C_token。全系列固定这六项，即使某支路未做也不减少多重比较分母。

使用共同hold整track配对bootstrap20000次，先抽整track再抽配对三个池/拟合seed，窗口加权、seed等权，rng20261026。95%探索CI全报；“明确正信号”采用六项Bonferroni双侧99.1667%区间（分位1/240与239/240）及固定seed只抽track敏感性同界。不能把动作、窗口、三个共享native参照当独立复制。

正信号门：三个seed差值均≥0、均值≥2pp、校正区间和固定seed敏感性下界均>0；被评方法相对native每seed不降且均值≥2pp，防止退化对照制造胜利。该门是有限机制门，不是完整major，也不是功效保证。

最少20条change-containing tracks；无效分母draw>5%则UNDERPOWERED，不挑样本补抽。无效draw比例及有效draw数完整保存。inner仅作方向一致性描述，不挑inner阳性覆盖hold阴性。

## 6. 执行、资源与保存合同

本轮只设计。未来实施先新建独立execution子目录、生成REUSE_MATRIX与SOURCE_AND_ID_AUDIT；旧模型不warmstart、不解冻，旧任务不查询/恢复。不会自动安装新依赖、下载地图或新数据。

运行前列inventory E与1.3倍动态remaining/Windows retained peak，测实际小批吞吐及保存审核IO；不恢复已撤销的固定6GiB、30min、旧六小时限制。短预算优先：任何阶段预计明显超出短诊断规模，应报告实测成本与停止点，不暗减样本或跑无限变体。所有输出和失败保留，禁止删除。

每run保存input/source/split/stat/feature-mask/action-table/objective SHA、初始化、版本、完整训练history与固定终态。MLP执行真实子集两epoch全参数/buffers/optimizer/CPU-CUDA-loader-Python-NumPy RNG EXACT续训测试；该测试不是生成器恢复。树模型若无精确续训能力，预先声明仅完成模型SHA-skip，不能假称AdamW式恢复；中断保留partial并停止，不自动重训已完成模型。

所有方法保存全401分数或其可无歧义重建文件、chosen、B/T/eligible、显式ID、最终K20、分组指标；独立CPU审核动作、固定19/STOP、质量、全部原SciPy聚簇、pooled motion、标签与完整CI。源与输出初末SHA一致，done读取分支实际拒坏source/output。保存算术不是全模型重放/外审。

零末层初始化的首步前层梯度0是预期；分阶段测试连接。新的神经特征分支须经实际有限梯度和独立标量/数值夹具。数值容差锁在执行前，不能转为效果epsilon或看阴性后放宽。

机器任务只有未来获准执行且预检通过才启动；每stage唯一worker、一次startup、一次预计半程、按真实阶段修完成定点，不每日轮询。只因速度变化不重启，不重开旧科学STOP。

## 7. 复用矩阵与报告模板

REUSE_MATRIX每行列：来源文件、科学对象、ID/输入/标签/目标是否匹配、原SHA、可复用内容、不能替代什么。

- 原E0枚举/标签/物理mask：兼容SHA时可直接复用，不重新枚举约1200万旧动作。
- 原E1统计/324特征：兼容时复用；tiny新模型/采样训练/新增特征不冒用旧模型效果。
- 旧小FIT/R1–R5：只在精确合同相同才替代本轮测试；不同锚点标签不能算已经验证本轮。
- 旧邻居干预：可复用past-only构造与失败证据，不能直接当新增邻居头已成功。

每步报告必须包含：实际执行项/未执行理由、原假设、每seed全分母、原生/STOP基线、训练与留出分离、正反例、统计与来源、成本、下一步是否触发。最后只能给“当前学习不足证据 / 输入增量证据 / 仍不确定 / 条件冲突提示”等有限结论，不能给信息论不可学证明。
