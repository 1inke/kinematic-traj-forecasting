# Native K20精细模式与误差归因：执行合同

用户已明确执行这份设计。DESIGN_ONLY是原提案状态，不是本次执行禁令。仅CPU、冻结原Native缓存，不训练、不预测器forward、不KIN/B1/B2、不写回新预测，不删除任何内容、不动旧暂停任务。

P0复现9777唯一窗、Native good及旧L0；P1仅GT-free X输入，封存E3全部12口径；之后P2关联GT及P3连续误差/P4两个受限oracle/P5整轨迹20000描述CI/P6完成审核。失败保留版本和来源，不静默修锁后代码。

## 预先固定的实现

- floor继承0.006982410783134709。主history3=h0-h[-4]，要求均速>floor、最后3段向量和/段长和>=.5；稳健敏感性为最后4段中非低速位移逐坐标median，保留合向量coherence>=.5检查。低速/历史无效/未来方向无效分别存。
- early1..4、middle5..8、late9..12；entry与internal20°主、15/30敏感性。局部方向用末两段合位移（t1只有1段），均速有效门同floor；触发必须连续2或3个实际相邻有效帧且同一偏转侧，缺测不跳过。左右均有持续事件或有效方向累计绝对变化>=180°标complex；无可靠局部方向标undefined。
- L3匹配同时要求开始及内部trend（相对±.2且绝对变化>=floor）及早中晚低速bit。原旧L0使用原实现精确回归；不替换旧类别。
- GT边界：entry/internal/任一有效局部角距当前阈值<=1°；GT各阶段或全均速距floor<=5%；速度trend ratio距±.2<=.01且abs变化已跨floor，或abs变化距floor<=5%且ratio已跨.2。先边界/必要无效，再entry/onset/speed/同模式仍失败；onset不可定义但entry有效单列PARTIAL_VALID。各层无条件与共同有效分母另存。
- E4弧长shape固定33个q含0/1，静止路径不制造方向；所有20槽指标均保存，kJ/kA/kF/kMode（EMPTY=-1）分列。
- O-END线性t/12端点补偿，无界/.2/.4三档；幅度不合格保持原候选且标infeasible，绝不clip。均是全20各自操作，不与single-slot上界混用。
- O-TIME：固定空间折线与原进度q，取q4/q8为锚点；前两段弧长进度比在[.75,1.25]步长.025的21×21网格，末段比例由q12=1唯一确定，只有末段也满足[.75,1.25]才准入。每原时间增量保持同一段比，q0=0/q12=1，零原增量使严格单调不可行，保持原轨迹并标原因。identity始终明确加入；每槽只按ADE在全部可行网格取最小（tie identity优先，再固定网格顺序）。这是**此预登记有限动作类**的精确oracle，不声称连续单调函数的全局最优。未救回不能排空更宽动作类。中间副本只内存，不保存新预测路径。
- 两项收益并集/交集直接逐ID，原生成功损失另列；END保持/不保持与GT oracle set-level回退不能当部署安全。原五pooled motion与每窗motion均保存；不因oracle好而放宽1%。
- 统计同sequence|target整轨迹20000次default_rng20261027；train/val是opt子集不加独立总量，Native无seed伪重复。主hold/change原失败683分母；12敏感性都输出，不用hold调定义。

## 反方审查DA1（同助手、非外审）

Major：离散时间搜索只能证明找到的有限见证，不是所有时间修正的上界；报告明确动作类/身份可行和未决更宽类。Major：层级缺席顺序非因果，另存无条件每层/交集。Minor：历史方向与阈值边界可能造成大量PARTIAL/undefined，不删除以凑结论。旧B2保护/运动学失败不因oracle收益被洗掉。

## 图合同

Python已保存后端。五族图为quantitative grid/案例image-plate：交叉热图、共同有效漏斗、成功vs失败误差、GT oracle重叠、固定hash病例及反例。结论以真实计算为准，不预设阳性；全部实际数据聚合不抽点美化。183mm宽、5–7pt、editable SVG/PDF与600dpiPNG，读回后视觉QA。全部分母/无效/边界/GT特权标清，零样本保留NO_CASE，不期刊投稿认证。资源v3 E2GiB为inventory（SciPy256+输入512+有界oracle512+bootstrap256+绘图/IO512MiB），1.3仅预警，真实压力安全停；无OS/pagefile/用户kill。
