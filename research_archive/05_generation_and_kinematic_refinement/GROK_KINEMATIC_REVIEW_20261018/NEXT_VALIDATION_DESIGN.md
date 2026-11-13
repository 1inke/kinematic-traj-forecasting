# 下一测试设计：固定互补候选池的证据补齐与确认

日期：2026-10-18。状态：DESIGN_ONLY，未启动。本文件更新研究优先级，不修改7个旧实验或P1判定。

## 大问题

在固定native20∪KIN20池内，观察条件是否能帮助保留互补的好路径，同时不牺牲原生误差尾部和变化组运动学？

先把“候选有互补性”“选择器能保留互补性”“历史有额外预测价值”“全部门通过”分开，不一步跳到运动理解。

## V0：仅补证据与保存算术，不训练

### 输入冻结

固定SPEED_CLAMP的seed42/43/44八epoch模型、原FIT18909/outer1035/inner1856/CAL2879索引、最终21策略清单及当前结果。对现存checkpoint/cache/source/GT/IDs生成新manifest，明确它是当前取证，不伪称历史训练时锁定。

首先寻找已保存的HGB、native/KIN分数和所选20索引。若只有当前发现的JSON和KIN缓存：

- 旧cond_boost路径无法精确恢复，写`ORIGINAL_SELECTOR_REPLAY_UNAVAILABLE`；不能仅用汇总反推具体slots。
- 若其他目录有实际原模型/选择索引，先核对来源和ID，再复算。
- 若必须重新拟合HGB，应转V1并标记新重现版本，不混进V0或说没有训练。

### 唯一主方法及参照

固定cond_boost_11_9：native按HGB分数取11；KIN score乘(.35+到所选native最近endpoint距离)，取9。所有并列固定原物理slot升序；若原实现不稳定argsort出现并列，记录并列，原重现用原规则，新确认版本预先锁稳定规则，二者不混。

只报告固定参照native20、KIN20、union40（不等预算）、原已存在的cond_top_11_9。不得按新结果改主方法。原21策略是历史探索家族，全部存档而非重新选best。

### 必须保存

每split/seed：显式ID/track/group、40池SHA、40scores、20selected_indices、20paths、拟合HGB、features定义、FIT-only统计/校准、配置和source/checkpointSHA。选择函数不接GT/group/scene/track。GT仅评价。

### 核心复算

1. 每group同候选ADE≤.2 AND FDE≤.4/J、rescue/lost、conditional lost/rescue分母。
2. minADE/minFDE分开报告，同时保存Jclosest完整单slot早中晚误差，不拼oracle。
3. five pooled motion每seed×group，dt.4，literal1.01native，零基值直接比较绝对值。
4. 全K20完整原SciPy complete-link .10/.20、distinct/effective-good；不能以更多簇替代好覆盖。
5. 固定5类窗口：retained-good、rescued、lost、both-fail、union-good-but-selected-fail，保存全部成员，不按展示效果挑例。
6. `delta_minADE/delta_minFDE/delta_J`全分布；正负贡献总和、最坏10%误差贡献、stable/transition/change分别报告。这个10%是事先固定描述项，不是删例或改变主均值。
7. 旧native成功被删掉的唯一good slot、其原logit rank、是否仍在40池、最终删去原因。与以前tail attribution联系，不能把candidate score等同集合冗余。
8. 旧报告数字、全部保存输出SHA和源前后不变；新实现独立算术，不导入有写入副作用的旧入口。

### 统计

按同ID配对、targettrack聚簇与3seed层级抽样；同一track抽样应用于所有seed，另报每seed/scene描述。固定primary为overall和change各coverage、minADE、minFDE、mean min-J，共8个“锁定selector−native”差值；NumPy PCG64 seed20261018，100000次draw，同draw有放回抽3个seed与原track数个track，按每track抽取次数重复其全部窗口，再计算窗口加权差值及seed平均。使用Bonferroni family8的百分位区间[.003125,.996875]；普通95%只作副参考，不替换主区间。无change窗口的draw记无效且不补抽，报告有效数；若无法形成该指标有效分布则CI缺失，不能作通过证据。其他方法/组只作描述，不偷偷增加主对比后仍用family8。

outer曾参与选择，所有区间均明确为探索性、不能消除方法选择偏差。只3scene/3seed，不能伪造独立scene总体结论。不能据此将V1的历史反复使用inner/CAL改称全新确认性。

V0完成即给复核报告，不因为可能缺少selector产物就静默重新训练生成器。

## V1：必要时只重建并冻结选择器，生成器不动

这一步包含新HGB拟合，非“完全无训练”。不改KIN、native或四个参照。

1. 固定当前HGB max_depth6/learning_rate.08/max_iter200/min_samples_leaf40/l2.1、同原二元good标签与采样规则；声明当前所谓soft标签实际退化为good，不暗换新目标。
2. 从现有FIT内按完整target track划出固定校准部分，哈希tag/比例/ID一次登记；训练和校准的native与KIN输入保持冻结。GIN/F16已见旧pool，因此这是selector适应迁移，不是wholemodel新样本确认。
3. 所有阈值/早停/校准只用训练内部分。不得用outer/inner/CAL的分数分位数重新调自适应quota；本主方法固定11/9，无quota扫描。
4. 保存HGB及每split中间结果后一次评价。inner/CAL若历史已被反复看过，仍只能作后验迁移验证，不能改称全新heldout。
5. 需要推理现有KIN时明确是frozen forward，不训练generator；实际source/ID/内存和输出检查后进行，不在本文档交付时执行。

若想排除训练内KIN过拟合影响，需要generator跨track交叉拟合；那会新增生成训练，超出本轮有限验证，不伪称本设计已经解决。

## V2：只有证据闭环后才做信息消融

固定池/划分/选择政策/训练预算，将条件HGB与candidate-geometry-only HGB作一项因素对照：删去past派生通道及其交叉量，其余候选几何、family等保持。past组包含v0、past speed/length、相对v0的align/along/lateral/first-stepalignment/dv0；不能删原past向量却保留其派生量冒充无历史。

这是“显式条件通道”消融，候选本身已编码历史，不是完全无历史。两臂参数有效容量随输入改变，解释局限预先报告；不同时换listwise loss、树深、quota或候选生成器。

V2不由V0的探索阳性自动触发训练。若V0发现严重尾部或保护失败，先将问题定位到删掉的支撑，再单独设计集合效用目标；不假定更强网络必然修好。

## 验收与决策

- 原始数字可重现且完整源/ID证据齐全，才称“实现复核通过”。
- 同K20 coverage上升但minADE/FDE恶化：称覆盖—误差折中，不称全面优于native。
- 所有seed/group五motion必须≤1.01native；整体通过不能替代分组。
- 原项目变化ADE/FDE各≥5%改善、protected≤1%伤害、原成功保留、三seed及校正CI等门不变。新+4pp探索门不是原major门替代。
- K40仅为池参考；K20输出预算相等不等于计算预算相等——仍生成40条并运行HGB，报告实际延迟/内存/模型总参数。
- 如果只有stable改善，变化错误/支撑失败，则不得宣称复杂运动理解突破。
- 若严格门未过：保留互补证据，不部署，不自动再扫比例/boost权重/epoch。

## 资源与流程

所有新输出在新执行目录，旧source/models/results不改。当前任务+保留进程树按真实E与1.3E及动态系统余量管理；不沿用撤销的30min/2s/固定6GiB禁令，不OOM、不调OS或kill用户程序。先实际吞吐和磁盘审核才给ETA；需后台运行时按唯一启动、一次半程、一次完成流程。本文档未启动任务，无虚构ETA。
