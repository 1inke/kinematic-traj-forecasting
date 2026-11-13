# 窗口级互补归因与解码端场景几何条件实验设计

版本：v1.0，2026-10-24。

状态：**仅设计，未实施、未核验地图、未预检、未训练。** 本文中新阈值是事前预算决策规则，不是现有数据已证明的规律；不得据此倒改旧实验结论。

## 1. 为什么下一轮这样做

依据是[上一轮总报告](../../qoder_work/DECODER_THREE_DIRECTION_EXEC_20261024/RESULT_ANALYSIS.md)及其A/B/C分报告与completion_review。已观察到：

| 证据 | 下一轮约束 |
|---|---|
| 时间基底降低加速度，但没有稳定保住变化覆盖 | 不再扫秩或把平滑当成运动理解；本轮不叠加基底 |
| 场景端点直方图劣于运动基线与部分打乱对照 | 若继续场景，必须换成真实静态几何信息，不能换名字重跑直方图 |
| aWTA在FIT和inner均退化 | 固定hard-WTA，不扫温度、不以延长训练掩盖阴性 |
| SetMix有效好簇增加，却仅一个seed提升覆盖 | 先区分“已有成功窗口内增加好簇”和“救回原失败窗口” |
| 成本计数曾漏融合Transformer；整模已超305k | 完整计数，机制探索与轻量部署分开，不能只报小适配器参数 |

这里并没有证据证明“场景一定是唯一缺失信息”。因此先做不训练的信息检验，只有结果支持才进入生成实验。

本系列不重复旧lost-tail全面归因、候选utility搜索或GT-free槽位替换扫描。Q0只补此次C2的具体解释缺口；Q1/Q2研究路径与真实静态环境的配准信息。

## 2. 研究问题与假设

- **H0描述问题**：C2增加的effective-good是否主要出现在两臂原本都成功的窗口，而非未覆盖窗口？这只能解释本轮现象，不识别模型内部因果。
- **H1信息问题**：在同一窗口、同一批固定候选上，正确场景几何是否比固定错位几何更能区分完整好路径与坏路径，并给原本几何不合理的错误路径提供可操作约束？
- **H2生成问题**：冻结相同运动生成器，仅给同形解码修正器不同的几何输入，正确几何能否稳定提高完整K20覆盖，同时保住原有成功、满足运动学门？

不把H1端点或排序结果当成H2的生成成功。不把H2变化覆盖上升直接写成理解心理意图。

## 3. 全系列共同合同

### 3.1 数据及模型

- 使用上一轮同一FIT18909、inner1856显式ID和原分组；inner change497、35条target tracks仅作为原注册人口参考，以SHA绑定的实际ID表核对。
- FIT用于训练；inner仅固定轮次评价，不选checkpoint；本系列不读取outer/CAL/test。
- 旧F16与旧模型权重保持冻结；Q0/Q1不做任何新模型前向。Q2若需旧A隐藏特征，允许在执行授权后对现有FIT/inner做有限冻结前向，并保存精确来源。
- 所有地图/图片来源必须是独立的静态环境资料，不由FIT或inner未来轨迹画出来；不得把“无人走过”标成障碍。
- 目标track分离不等于邻居上下文独立；上游F16见过这些数据、inner重复开发的限制原样披露。
- 新输出只进入新execution目录，锁定旧source/model/cache/result SHA，不覆盖旧摘要、失败或profile。

### 3.2 质量与物理口径

对候选k：

`good_ik = (ADE_ik <= 0.20 m) AND (FDE_ik <= 0.40 m)`

`J_ik = max(ADE_ik/0.20, FDE_ik/0.40)`；窗口覆盖为任一完整候选good。不能用不同候选的minADE、minFDE拼接成功。

`rescue = ~covered_reference & covered_new`；`lost = covered_reference & ~covered_new`。

同时报告原生参考与直接配对参考，两者分母分开。所有表列窗口n、track数、seed数、good/rescue/lost绝对数及比例。

全K20的GT-free原SciPy complete-link：`D=max(meanpath/0.10, endpoint/0.20)`，cut=1。聚类不看GT；簇是否含good成员是事后GT标记。因此effective-good不是纯GT-free预测指标，也不作为最大化目标。

所有seed × FIT/inner × overall/stable/transition/change报告五项pooled运动学：speed-p95、acceleration-mean/p95、jerk-mean/p95，dt=0.4。速度使用观测末位置到首预测点及其后位移，内部加速度11项、jerk10项；velocity_join另列。调用已验证的统一口径并用独立实现核验。

五项必须各自 `new <= 1.01*native`，零native不加floor或epsilon。原G_base、major change ADE/FDE各5%改善、protected 1%、三seed一致与相关性审核仍保留；本系列局部门不能替代它们。

## 4. Q0：冻结输出的窗口级互补归因

### 4.1 输入与范围

只读上一轮C2 iso/setmix三seed在FIT和inner的12个人口输出、相同native和旧A参考。若已有审计保存字段足够，直接复用并验证SHA；缺字段才从保存路径重算，不重新前向。

不把其他失败架构全部加入一个新排名池，不搜索最优组合。

### 4.2 必须回答的四张表

**表A：配对覆盖转移。** 对每seed、split、group列四格：两者成功、iso成功mix失败、iso失败mix成功、两者失败。与native的四格另表，不能混用。

**表B：好簇增量来自哪里。** 设g为一个窗口中至少含一条good路径的簇数。按表A四格分解：

`mean(g_mix-g_iso) = sum_z n_z/N * mean(g_mix-g_iso | z)`。

逐项必须严格加回总体差。报告正贡献和负贡献，不用负分母计算“贡献率”；正贡献归一化和有符号总差同时保存。统计“更多簇但仍是同一已覆盖窗口”的绝对窗口数。

**表C：救回与丢失的运动代价。** 在固定native→iso、native→mix及iso→mix转移ID上，列J、ADE/FDE与全部候选运动学。全population门仍是主安全门，不能只报赢家运动学或只报救回子集。

**表D：原生成功支撑。** 对每臂列原生成功166个change窗口的保留/丢失、距J=1边界、旧A成功集合及反例。这里不做新的slot删除建议。

### 4.3 接口和核验

```python
def good_and_clusters(paths, gt):
    # 返回逐slot good与仅由paths决定的cluster labels；不拼接oracle
    ...

def paired_attribution(reference, treatment, ids, tracks, groups):
    # 逐ID对齐；四格分区穷尽且互斥；簇增量恒等式必须重构
    ...
```

必要fixtures：完全相同集合；复制轨迹；一个窗口增加多个好簇但覆盖不变；一个rescue同时一个lost；不同slot分别minADE/minFDE但无完整good。

产物：`WINDOW_TRANSITIONS.parquet`、`CLUSTER_CONTRIBUTIONS.csv`、`MOTION_ATTRIBUTION.csv`、`REVIEW.json`、`RESULT_ANALYSIS.md`。全部ID保留，科学比较不用按结果筛掉seed。

### 4.4 停止与用途

Q0没有“必须得阳性”的通过门。数据/算术核验完成即交付。若已有字段已回答全部问题，不另跑一轮计算。最多一次完整补充，不延伸成无限行为标签诊断。Q0结果不能证明场景路线有效，也不是启动Q2的充分条件。

## 5. Q1：真实静态几何的数据准入与信息检验

### 5.1 Q1a先查可用性，不先训练

仅审核现有合法静态图片、可通行/障碍标注、世界坐标映射及许可。先产出`MAP_INVENTORY.json`，每recording必须列：

`image_sha / annotation_sha / source_and_license / recording_id / map_date / world_unit / H_image_to_world / H_world_to_image / valid_polygon / unknown_mask / annotation_method`。

手工标注只能查看静态图，不查看未来轨迹、模型候选或救回/丢失结果。动态人物、车辆、树冠遮挡、不可判断区域标unknown；不能从单帧遮挡推断永久墙体。当前文档没有确认这些材料已存在。

标定至少使用6个可识别地面控制点拟合，另留至少5个未参与拟合的独立点验收；不得拿GT未来轨迹调整H。预定工程准入：留出点世界误差RMS≤0.10m、max≤0.20m；无法得到足够可靠点或地图有效分辨率差于0.10m则记`MAP_CALIBRATION_UNRESOLVED`，不凭上采样冒充精度。

若只覆盖部分recording，登记全人口缺失数量；不将“地图可用子集”冒充全497。Q2主人口要求本注册FIT/inner各recording均有合格地图，否则停止本方案，不自动换新数据集或挑场景继续。

### 5.2 固定场景表示

使用0.10m世界网格，三态known-free / known-static-obstacle / unknown。对已知区域构造米制有符号距离场d，正值可通行、负值障碍；unknown不填成free。

沿每条路径每相邻点（含观测末位置→首预测点）按最大0.10m间隔均匀采样，防止只查12个节点漏过穿墙。每段含两端，重复端点只计一次。查询使用固定双线性插值；格点四邻任一unknown则该查询unknown。没有known-free或known-obstacle的局部/整张退化地图显式标记，不产生无穷距离特征。

候选几何代价：

`c(p,M) = mean_s max(0, -d_M(p_s))`，单位米。

这是中心线进入已知静态障碍的深度，不包含个人体半径假设。边界安全余量、地图不确定性另列，不新调0.2m膨胀半径以求阳性。

### 5.3 三个固定条件

- **TRUE**：正确地图、正确坐标。
- **SHIFT**：同地图查询点固定平移世界坐标(+2m,+2m)；不是另选最差错位，禁止扫角度或距离。
- **NULL**：几何数值全部0，保留与其他臂相同的可用性标记。

TRUE/SHIFT比较只在两者查询均known的位置进行；完整路径可用要求两者全部密集采样点known。其余候选标unknown，不记为可通行。共同可用率是单独数据指标。SHIFT破坏局部配准但不是“纯场景内容”因果对照，平移会改变局部障碍密度，须报告其分布差异。

### 5.4 固定候选与主终点

使用旧A/C1-hard三seed的FIT与inner K20固定输出，native作为独立参考，禁止从多模型挑最好候选。只以`inner change`为主探索人口。

在每个同时有good和bad、且全部20候选共同可用的窗口内计算：

`AUC_i(M) = mean_{g good,b bad}[1(c_g<c_b)+0.5*1(c_g=c_b)]`。

先窗内等权，再人口均值；不把候选对当独立样本。NULL恒为0.5。主量为TRUE−SHIFT；TRUE−0.5为共同必要条件。置信区间按track及配对旧模型seed，固定seed敏感性另报。

这检验的是“几何是否关联候选质量”，不是轨迹生成收益。全候选都good或都bad的窗口不能估该AUC，必须列数量；特别报告native-failure及neither-good，不把它们从总体覆盖分母删除。

同时报告：bad静态穿障比例、good静态穿障比例；TRUE/SHIFT代价分布；每窗可通行且good候选是否存在；原生失败窗口的几何问题比例。GT只用于这些事后标签。

### 5.5 Q2预算准入门（一次固定，不据结果修改）

以下全部满足才建议Q2：

1. 地图来源与标定全部合格；主change完整候选共同可用窗口≥90%，所有缺失原ID单列。
2. 每seed主AUC有效窗口≥100、覆盖≥20条target tracks，否则`INSUFFICIENT_GEOMETRY_INFORMATION`。
3. TRUE AUC均值≥0.60；TRUE−SHIFT均值≥0.05；三seed差值均为正；两项区间按第7节同时控制且下界>0。
4. 可核验good候选中静态穿障率≤1%，防止错误地图把真实可行路径当障碍。
5. 主人口中至少10%的bad候选有已知静态穿障证据，避免只有稀少个案却启动全量模型实验。

这些是投入筛查门，不是场景信息存在与否的必要充分条件。没过门只能结束该表示/该人口的投资，不能宣称所有视觉、拓扑信息无效。若Q1a不可用，不以历史轨迹热图替代，不默认获取外部新数据。

## 6. Q2：只改变场景输入的解码端路径修正

### 6.1 为什么不是新一轮堆模块

冻结旧A/C1-hard每seed的82,008参数生成器与对应F16表示。使用同一完整K20输出p0和128维槽位隐藏状态h；三个臂只训练同形修正器。旧A本身存在运动超门，因此本实验是**条件机制探索**，不能以相对旧A改善代替相对native的安全门。

不重新生成底层queries、不引入SetMix/aWTA/新时间基底、不得使用GT挑slot或永久删bottom4。全部20槽位平行修正，K20不变；不把GT安全投影或GT最优步长用于部署。

### 6.2 模块精确定义

在p0的12个预测点查询四个特征：

`g_t = [clip(d/2,-1,1), clip(dd/dx,-1,1), clip(dd/dy,-1,1), known]`。

梯度以0.10m格点中心差分求得，所需邻域unknown则梯度和known均置0。世界梯度向量旋转回模型局部坐标；距离仍是米制标量。TRUE/SHIFT使用共同有效性掩码；NULL前三项0、known相同。输入路径p0也在同一局部坐标系，不能混用像素或世界绝对坐标。

`z_k = concat(h_k[128], p0_k.flatten()[24], g_k.flatten()[48])`，总200维。

共享MLP：`Linear(200,32) → GELU → Linear(32,24)`，参数共7,224。末层weight/bias零初始化；三臂所有权重、batch顺序与随机状态配对。

`u_t = tanh(raw_t) / max(1, ||tanh(raw_t)||_2)`

`r_t = 0.20*(t/12)^2*u_t`；`p_t = p0_t + r_t`，t=1..12。

特征和MLP float32，最终基路径相加float64。界仅保证位置残差，不保证不穿障或速度/加速度/jerk安全。场景查询只对固定p0进行一次，不循环迭代，不沿12步自回归。

三臂：`Q2_TRUE`、`Q2_SHIFT`、`Q2_NULL`。差异仅为地图特征对应关系。相同known掩码也给NULL，因此主归因是几何数值/配准的增量，不是地图可用性本身。

### 6.3 训练合同

- seeds42/43/44；每seed使用对应旧A冻结基模型，不复用其他seed最优模型。
- 3臂×3seed=9个fresh修正器；8epochs、batch256、shuffle、AdamW lr0.001/wd0.0001、grad clip norm5.0；FIT18909按74batch/epoch，共592步/修正器。
- 原hard-WTA `mean_i min_k(ADE+0.5FDE)`，加原190对终点的平均线性hinge `0.05*mean_{k<l} relu(0.3-distance(endpoint_k,endpoint_l))`。不改成平方hinge、不新增障碍loss、成功保护loss或物理loss。
- 地图只作为输入。GT仅用于既有任务loss及事后评价；groups、scene ID、track ID、future、logits不得进入新MLP。
- 固定epoch8为主，不按inner选checkpoint。全部八轮loss/assignment/native success保留诊断保存；没有根据结果延长训练的默认动作。

### 6.4 设计接口（不是已实现代码）

```python
def query_map_features(p0_local, local_to_world, sdf, known, condition):
    # 返回[B,20,12,4]；条件只能true/shift/null
    # local_to_world来自观测坐标规范；无GT参数
    ...

def refine_all_slots(h_frozen, p0_frozen, geometry, adapter):
    # concat200 -> shared MLP -> bounded residual
    # 不排序、不选赢家、不删slot；输出[B,20,12,2]
    ...

def original_objective(pred, gt):
    # 与锁定原hard-WTA和190对linear hinge实现逐梯度核验
    ...
```

### 6.5 必须先过的实际预检

1. source/ID/地图/H及旧输出SHA；局部→世界→局部roundtrip、known边界、障碍段穿越fixture。
2. 三臂初始输出等于同一p0；全部7,224参数配对，旧encoder/decoder/buffers完全冻结。
3. 实际GPU FIT batch256多个batch有限；末层零初始化后分阶段检查上游梯度，不把正常初始零梯度误判为断路。
4. 原loss标量和输出梯度对旧实现EXACT；残差界独立NumPy核验；地图features旋转方向fixture。
5. 每臂实际FIT513+inner12子集两轮连续/中断的参数、optimizer、所有CPU/CUDA/loader/NumPy/Python RNG及history EXACT。只是子集恢复，不冒称完整FIT两轮。
6. 每臂实际128输出保存重读，全质量/完整聚类/五运动独立复算，完成SHA-skip及坏source/output拒绝。
7. source全锁定后才生产；源变更必须新记录，不能静默重算digest掩盖变化。

### 6.6 结果分层判定

**局部条件机制阳性**须全部满足：

- TRUE−NULL及TRUE−SHIFT的inner change完整覆盖增量各≥2pp，三seed各正，第7节同时区间下界均>0。
- TRUE每seed覆盖不低于其冻结p0；原生成功lost率不高于NULL和p0，不能只看净增。
- TRUE的inner overall/stable/transition覆盖≥0.99×native，ADE/FDE≤1.01×native；全seed/split/group五motion逐项≤1.01×native。
- 正确地图下的静态穿障比例低于NULL且不比SHIFT差；共同known范围及全人口missing单列。

只减少穿障而没有覆盖收益，结论为`GEOMETRIC_PLAUSIBILITY_ONLY`，不能称互补生成成功。只比弱NULL好而低于p0，结论为`WEAK_CONTROL_ADVANTAGE_ONLY`。只有错位臂退化，不足以通过TRUE−NULL门。

**部署门独立存在**：冻结encoder267159 + 旧A82008 + adapter7224 = **356391参数**，仍超305000。即使局部机制阳性也仅能标`MECHANISM_POSITIVE_COST_NONCOMPLIANT`；不得伪称轻量化成功。成本合规压缩须之后另立实验，本轮不同时缩主干凑预算。

原major/完整G_base不改。只通过局部机制门不能提升baseline，不代表独立泛化。

## 7. 统计、分母与有限多比较

- 全部主推断以同ID差值为基础。Q0描述性分解不筛选显著性，不给每格海量p值。
- Q1两项必要主比较为TRUE−SHIFT AUC与TRUE−0.5；Q2两项为TRUE−NULL coverage与TRUE−SHIFT coverage。每阶段分别为固定两比较family。
- 每项给20,000次共同track×paired model-seed bootstrap，RNG固定20261024。每draw先有放回抽三配对seed，再有放回抽目标tracks；同一draw三臂使用完全相同权重，按所抽有效窗口数加权，最后平均seed。AUC有效窗口可能随seed变化，保存每seed实际分母；空有效draw记无效，不无限重抽，若超过1%则统计不可用。
- 单项常规95%区间只作描述；两主比较的判定各使用97.5%双侧百分位区间（1.25/98.75百分位），Bonferroni控制该阶段两个区间的family错误率目标。bootstrap近似与小track数局限明确披露，不称精确保证。
- 固定seed、只抽track的敏感性区间同表，不用更好看的区间替换主区间。三seed不是足够的独立场景重复。
- 若额外增加对比，只能标探索、不进入门；不得看完后改变family。Q1选择后Q2仍是同inner适应性探索，不包装成两阶段独立确认。
- 无地图的ID保留全人口registry；主质量覆盖分母仍为完整inner1856/change497。地图探针的条件子集分母另列，不以它替代全人口质量。

## 8. 成本、预算与执行纪律

本轮不提供伪精确生产ETA。实施时分别实测Q0/Q1保存算术、地图预处理、Q2前向/反向、保存与审核吞吐，再给阶段ETA和峰值估计。

有限上限为一次Q0、一次Q1、条件通过后一次Q2九个8轮修正器；没有第四轮变体。资源遵循项目当前1.3动态available/remaining与Windows retained-peak规则，CPU/GPU阶段分别列inventory；不得倒退使用过期固定规则，也不得改OS/pagefile或关闭用户程序。

完整成本包括encoder、冻结base decoder、adapter、地图变换/插值与传输；离线地图制作与SDF生成冷启动开销、每窗摊销开销单列。推理虽然只训练7,224参数，却仍需要运行旧A，不能省略其成本。

MAC包含Linear、QKV/O/FFN、注意力矩阵乘、bmm/einsum与融合算子，禁止重复计数，非MAC操作另列。GPU同步计时batch1和256，50warmup/500measure，CPU1thread同口径；同时给含坐标/map查询的端到端warm路径，不以缓存特征MLP-only时间冒整模延迟。

本设计不启动worker/自动化。未来获执行授权后采用sole按需0trigger worker、一次startup、一次50%预计elapsed检查，再按真实剩余修正一次完成定点；健康时不对话轮询。完成要求worker与exit双SUCCESS、所有输出SHA、独立保存算术与资源审核，不以epoch全完冒科学完成。

## 9. 产物与源码组织建议

```text
execution_<date>/
  source_manifest.json / protocol.json / runtime_state.json
  q0_attribution/      # saved-only
  q1_map_inventory/    # 来源、H、标注、unknown及独立控制点
  q1_probe/           # fixed candidate / 三条件 / 主终点
  q2_conditioned_decoder/  # 仅Q1过门后创建生产结果
  completion_review/  # 不依赖训练评价函数的保存复算
  RESULT_ANALYSIS.md
```

建议代码边界：`map_registry.py`（来源与变换）、`geometry_features.py`（GT-free查询）、`saved_attribution.py`、`geometry_probe.py`、`decoder_adapter.py`、`train_paired.py`、`audit_saved.py`。本次没有生成这些可执行文件。

每checkpoint绑定base checkpoint SHA、map/SDF/H SHA、split IDs、stats、全部超参、init、optimizer/RNG/history。额外保存p0/refinedpaths/geometry/known/slotfeatures来源/残差；不能只保存最终metrics。

## 10. 总结报告必须回答的问题

1. 新好簇是否覆盖了新窗口？增加多少，丢失多少？
2. 正确场景信息是否优于固定错位和无场景，而非只有错位造成伤害？
3. 地图几何改善是否转化为完整轨迹质量，而非只减少穿障？
4. 是否把代价转移到速度、加速度或jerk？
5. 是否在原生成功保留和三个seed上稳定？
6. 整模计算/参数是否合规；若不合规，明确仍是机制探索。
7. 哪些项目未执行、为何停止？所有否定结果和unknown来源保留。

## 11. 同助手反证检查与设计限制

**检查1：问题与方法。** 重大风险是没有可靠地图却把轨迹密度当可通行性。已设置Q1a硬阻断；材料未核验前状态必须pending，不标PASS。

**检查2：证据与替代解释。** 正确地图可能帮助避障但对方向/停走选择没有信息；Q1条件AUC排除了全坏窗口，不能推断覆盖所有失败；已强制全人口缺失与失败分母报告，并保留Q2覆盖门。

**检查3：结论与成本。** p0本身运动不安全、适配器可能无力修复；小残差还限制可达空间。因此阴性只否定本有界adapter，而非所有场景解码。整模已知超预算，不能拿机制阳性写成轻量部署成果。

最强反驳：重复inner上的数据可用性筛查、信息筛查、生成筛查可能发生适应性过拟合，即使通过也不能确认独立泛化。后续干净评测需要另立上游训练暴露可审计的数据合同，不能把已用outer/CAL重命名盲测。本系列不会自动启动那一步。

本文件是ARS辅助的方法设计及同一助手的反证检查，不是独立外审、不含新实测结果，也不声称“解码端加地图”为首次创新。后续创新是否成立取决于固定候选/计算预算下的可复现增量。
