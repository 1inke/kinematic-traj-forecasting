# S0：原生全候选支撑安全机会筛查——完整结果

结论：**执行与审核通过，科学门未通过；停止当前注册安全子族，不启动S1。**

FIT变化组4725个窗口，native/safe4/safe20均为1937个完整合格窗口，净增0；要求至少237个。
整数判定为0 < 23625。这里的“完整合格”始终是同一条候选ADE≤0.2m且FDE≤0.4m，而非不同slot的两个最小值拼接。
[整数判定](<<WORKSPACE_ROOT>/NATIVE_SUPPORT_WIDE_REFINEMENT_DESIGN_20261015/S0_execution_20261015/DECISION.json>)、[完整结果](<<WORKSPACE_ROOT>/NATIVE_SUPPORT_WIDE_REFINEMENT_DESIGN_20261015/S0_execution_20261015/RESULT.json>)。

## 1. 运行和证据强度

唯一后台任务Trajectory-NativeSupport-S0-20261015，Python18068/service2688。开始2026-10-15T15:20:46.147188+09:00，结束2026-10-15T15:24:09.383285+09:00，
worker与exit双SUCCESS，实际203.236000秒（约3分23秒），不是原规划5.889578分钟。
唯一半程核验2026-10-15T15:24:16.1803706+09:00发现已全部完成；未等待原15:26:39.521预计截止，也没有第二次半程或完成PID查询。
4个科学job、8个主要NPZ/JSON输出、4个done合同、24679唯一ID；8个预检128窗口块不算科学job。
生产独立保存审核124.849342秒已包含在运行耗时内，不能再加一次。[审核](<<WORKSPACE_ROOT>/NATIVE_SUPPORT_WIDE_REFINEMENT_DESIGN_20261015/S0_execution_20261015/ARITHMETIC_AUDIT.json>)。

全12端口×48组的完整质量、原SciPy全K20 complete-link（meanpath.10m/endpoint.20m）、五项pooled运动学、safe ray、fallback分类、
固定16与零lost均PASS，最大算术差0。额外CPU补充16.965489秒逐项重算所有保存slot ADE/FDE/J/good及组计数/certificate，
1863来源、8主要输出和4done合同SHA初末PASS，最大差0；不是又一次全模型前向、全部梯度回放或外部审稿。
1863项来源保留全部A0的1823项旧digest，不声称覆盖所有历史文件。[补充审核](<<WORKSPACE_ROOT>/NATIVE_SUPPORT_WIDE_REFINEMENT_DESIGN_20261015/S0_execution_20261015/completion_review/REVIEW.json>)、[保存概要](<<WORKSPACE_ROOT>/NATIVE_SUPPORT_WIDE_REFINEMENT_DESIGN_20261015/S0_execution_20261015/completion_review/SUMMARY.json>)。

本实验无神经网络、optimizer、训练、solver、teacher或新数据。GT只用于固定方向特权机会构造，不是部署预测或S1监督teacher。
未触发S1的6次训练；正式新训练计数0。旧模型、decoder/scorer、缓存、结果和失败来源不改；baseline、历史cycle11与总目标不变。

## 2. 四个人口的变化组：唯一source gate失败

|人口|变化窗口n|native合格|safe4合格|safe20合格|safe20净增|A0放松all20球上界参照|
|---|---:|---:|---:|---:|---:|---:|
|FIT|4725|1937|1937|1937|0|3108|
|outer|253|110|110|110|0|166|
|inner|497|166|166|166|0|288|
|original_CAL|700|243|243|243|0|418|

A0的FIT all20放松位置球上界仍为3108（native1937），不是安全可实现收益。S0对注册的单GT球方向施加33个对应导数ray约束后没有新增完整覆盖。
outer/inner/CAL只是描述，不能替代FIT gate。既不换0.3/0.5标准，也不四舍五入救回门槛。
全部16组的safe4与safe20 rescue=0、lost=0；各20个slot“可救新窗口数”也全部为0。
[旧A0参照](<<WORKSPACE_ROOT>/NATIVE_SUPPORT_WIDE_REFINEMENT_DESIGN_20261015/S0_execution_20261015/../../GENERATION_FEASIBILITY_CONDITIONING_DESIGN_20261015/A0_execution_20261015/RESULT_ANALYSIS.md>)。

## 3. 为什么“安全通过”却“没有救回”：修正几乎被收缩掉

FIT变化组共94500条candidate，最终alpha=0为47755（50.534392%）；
实际非零修正46745条（49.465608%）。
alpha中位数0、75分位1.102867111e-7、90分位2.958716838e-7、
99分位9.171589324e-7、最大0.000006813079396。
候选最大逐点修正的全组均值1.923225779e-8m，全组最大也仅0.000001362615879m。

47109条candidate至少一个原生加速度向量精确为0；
4814条至少一个jerk向量精确为0，二者可能重叠，不可相加为互斥比例。
逐向量1.01倍原生预算在原生为零/近零时非常窄；一个candidate共用的alpha又由33约束最小值决定，
所以任一局部很小预算都可能将整个12步GT方向收缩。加速度约束绑定104883次、jerk约束8413次、
速度0次；这里统计**含并列约束频次**，总数可超过candidate数，不可当成互斥百分比。
完整33维频次在[保存概要](<<WORKSPACE_ROOT>/NATIVE_SUPPORT_WIDE_REFINEMENT_DESIGN_20261015/S0_execution_20261015/completion_review/SUMMARY.json>)与RESULT各certificate内。

数值fallback只有47/94500（0.049735%），不是“alpha归零”的同义词，也不是全部收缩的主要计数来源。
独立实现对fallback分类EXACT；没有加评价epsilon，没有更改1.01门、固定1−2^-20内部余量或候选级回退规则。
这些是注册映射限制的直接迹象，不证明其它安全方向、改变时间形状的生成或原组内五统计安全域没有空间。

反例必须保留：全FIT stable含少数最大0.187197812m的修正，FIToverall ADE/FDE改善约0.012692%/0.007368%；
因此不能说所有输出都bitwise等于native或所有平均误差完全不变。
但FIT变化组ADE/FDE改善仅0.000001035676656%/0.000001920925652%，没有形成哪怕一个新增完整好候选。
[独立补充](<<WORKSPACE_ROOT>/NATIVE_SUPPORT_WIDE_REFINEMENT_DESIGN_20261015/S0_execution_20261015/completion_review/SUMMARY.json>)。

### 所有16组的收缩与数值回退分母

|人口/组|candidate数|alpha=0|实际非零修正|fallback|alpha99分位|最大逐点修正m|
|---|---:|---:|---:|---:|---:|---:|
|FIT/overall|378180|201247|176933|201|0.000001505692021|0.1871978121|
|FIT/stable|215340|119726|95614|133|0.000001796852584|0.1871978121|
|FIT/transition|68340|33766|34574|21|0.000001324818042|0.000002752082327|
|FIT/change|94500|47755|46745|47|9.171589324e-7|0.000001362615879|
|outer/overall|20700|11226|9474|12|0.000001425182787|0.006660398388|
|outer/stable|12080|7023|5057|8|0.000001658130570|0.006660398388|
|outer/transition|3560|1660|1900|3|0.000001318520862|8.025503386e-7|
|outer/change|5060|2543|2517|1|9.615327821e-7|0.000001329515280|
|inner/overall|37120|19228|17892|8|0.000001707892548|0.006626666581|
|inner/stable|19560|10557|9003|5|0.000001991173624|0.006626666581|
|inner/transition|7620|3636|3984|1|0.000001362264233|0.000001628773547|
|inner/change|9940|5035|4905|2|0.000001019696271|9.171065775e-7|
|original_CAL/overall|57580|31920|25660|26|0.000001270901491|0.00001033694829|
|original_CAL/stable|30700|18197|12503|21|0.000001356205853|0.00001033694829|
|original_CAL/transition|12880|6364|6516|2|0.000001344053854|7.387906583e-7|
|original_CAL/change|14000|7359|6641|3|9.891540086e-7|0.000001016640087|

## 4. 全48组指标，不只展示change

ADE和FDE列是各自minK的均值；它们不一定对应同slot，不能代替完整覆盖。J=minK max(ADE/.2,FDE/.4)对应一个完整slot。
Q=ADE+.5FDE的slot与Jslot分别登记；早中晚均取各自同一完整候选，不分段拼oracle。
effective-good和distinct为GT-free全K20聚簇后统计，数量不是越多越好。下面为每个端口的完整概要，精确值和五项运动学见RESULT。

|人口/组|端口|n|合格|ADE|FDE|J|有效good簇均值|distinct均值|五运动学联合|
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|FIT/overall|native20|18909|10858|0.1729743827|0.2992296137|0.9777443986|0.7915807288|19.72209001|PASS|
|FIT/overall|safe4|18909|10858|0.1729743826|0.2992296131|0.9777443975|0.7915807288|19.72209001|PASS|
|FIT/overall|safe20|18909|10858|0.1729524288|0.2992075662|0.9776280663|0.7915807288|19.72209001|PASS|
|FIT/stable|native20|10767|7414|0.1332240691|0.2291789015|0.7525135754|0.9891334634|19.64641962|PASS|
|FIT/stable|safe4|10767|7414|0.1332240690|0.2291789013|0.7525135751|0.9891334634|19.64641962|PASS|
|FIT/stable|safe20|10767|7414|0.1331855162|0.2291401895|0.7523092884|0.9891334634|19.64641962|PASS|
|FIT/transition|native20|3417|1507|0.2099077287|0.3648045153|1.189969991|0.5700907229|19.89113257|PASS|
|FIT/transition|safe4|3417|1507|0.2099077284|0.3648045144|1.189969989|0.5700907229|19.89113257|PASS|
|FIT/transition|safe20|3417|1507|0.2099077239|0.3648045014|1.189969963|0.5700907229|19.89113257|PASS|
|FIT/change|native20|4725|1937|0.2368453635|0.4114341809|1.337508510|0.5015873016|19.77227513|PASS|
|FIT/change|safe4|4725|1937|0.2368453631|0.4114341794|1.337508507|0.5015873016|19.77227513|PASS|
|FIT/change|safe20|4725|1937|0.2368453610|0.4114341730|1.337508494|0.5015873016|19.77227513|PASS|
|outer/overall|native20|1035|625|0.1754775852|0.2966327369|0.9807671745|0.8144927536|19.69855072|PASS|
|outer/overall|safe4|1035|625|0.1754775849|0.2966327362|0.9807671730|0.8144927536|19.69855072|PASS|
|outer/overall|safe20|1035|625|0.1754691100|0.2966307944|0.9807247942|0.8144927536|19.69855072|PASS|
|outer/stable|native20|604|434|0.1348864208|0.2277001909|0.7526372640|1.006622517|19.62582781|PASS|
|outer/stable|safe4|604|434|0.1348864207|0.2277001908|0.7526372638|1.006622517|19.62582781|PASS|
|outer/stable|safe20|604|434|0.1348719008|0.2276968710|0.7525646606|1.006622517|19.62582781|PASS|
|outer/transition|native20|178|81|0.2085045388|0.3418610002|1.172400633|0.6011235955|19.91011236|PASS|
|outer/transition|safe4|178|81|0.2085045388|0.3418610001|1.172400633|0.6011235955|19.91011236|PASS|
|outer/transition|safe20|178|81|0.2085045334|0.3418609843|1.172400599|0.6011235955|19.91011236|PASS|
|outer/change|native20|253|110|0.2491466190|0.4293782977|1.390568401|0.5059288538|19.72332016|PASS|
|outer/change|safe4|253|110|0.2491466181|0.4293782952|1.390568396|0.5059288538|19.72332016|PASS|
|outer/change|safe20|253|110|0.2491466157|0.4293782883|1.390568381|0.5059288538|19.72332016|PASS|
|inner/overall|native20|1856|932|0.1933621567|0.3382179531|1.087725790|0.6648706897|19.84267241|PASS|
|inner/overall|safe4|1856|932|0.1933621564|0.3382179522|1.087725788|0.6648706897|19.84267241|PASS|
|inner/overall|safe20|1856|932|0.1933608289|0.3382177691|1.087719145|0.6648706897|19.84267241|PASS|
|inner/stable|native20|978|601|0.1536866739|0.2660315637|0.8652214888|0.8333333333|19.80572597|PASS|
|inner/stable|safe4|978|601|0.1536866739|0.2660315635|0.8652214885|0.8333333333|19.80572597|PASS|
|inner/stable|safe20|978|601|0.1536841575|0.2660312254|0.8652088993|0.8333333333|19.80572597|PASS|
|inner/transition|native20|381|165|0.2087770275|0.3656703311|1.175069917|0.5826771654|19.91338583|PASS|
|inner/transition|safe4|381|165|0.2087770273|0.3656703303|1.175069916|0.5826771654|19.91338583|PASS|
|inner/transition|safe20|381|165|0.2087770231|0.3656703173|1.175069890|0.5826771654|19.91338583|PASS|
|inner/change|native20|497|166|0.2596188096|0.4592218419|1.458613303|0.3963782696|19.86116700|PASS|
|inner/change|safe4|497|166|0.2596188086|0.4592218396|1.458613298|0.3963782696|19.86116700|PASS|
|inner/change|safe20|497|166|0.2596188061|0.4592218311|1.458613284|0.3963782696|19.86116700|PASS|
|original_CAL/overall|native20|2879|1490|0.1805755727|0.3032534137|1.012621098|0.7311566516|19.65578326|PASS|
|original_CAL/overall|safe4|2879|1490|0.1805755724|0.3032534128|1.012621097|0.7311566516|19.65578326|PASS|
|original_CAL/overall|safe20|2879|1490|0.1805755683|0.3032534008|1.012621073|0.7311566516|19.65578326|PASS|
|original_CAL/stable|native20|1535|994|0.1314107482|0.2137134740|0.7312347886|0.9648208469|19.52768730|PASS|
|original_CAL/stable|safe4|1535|994|0.1314107481|0.2137134737|0.7312347883|0.9648208469|19.52768730|PASS|
|original_CAL/stable|safe20|1535|994|0.1314107432|0.2137134595|0.7312347607|0.9648208469|19.52768730|PASS|
|original_CAL/transition|native20|644|253|0.2267221025|0.3767711639|1.263927263|0.4984472050|19.88975155|PASS|
|original_CAL/transition|safe4|644|253|0.2267221020|0.3767711620|1.263927260|0.4984472050|19.88975155|PASS|
|original_CAL/transition|safe20|644|253|0.2267220975|0.3767711483|1.263927233|0.4984472050|19.88975155|PASS|
|original_CAL/change|native20|700|243|0.2459322019|0.4319653801|1.398459406|0.4328571429|19.72142857|PASS|
|original_CAL/change|safe4|700|243|0.2459322014|0.4319653787|1.398459403|0.4328571429|19.72142857|PASS|
|original_CAL/change|safe20|700|243|0.2459321993|0.4319653731|1.398459392|0.4328571429|19.72142857|PASS|

全部48组均对dt.4的12v/11a/10j全候选pooled speed p95、accel mean/p95、jerk mean/p95逐项literal≤1.01native通过。
不是平均window-p95，也没有用加速度一项代替五项。安全通过不等于覆盖成功。
所有early/middle/late、Q/J槽位、每组5motion原值、alpha与33频次均保存在[RESULT.json](<<WORKSPACE_ROOT>/NATIVE_SUPPORT_WIDE_REFINEMENT_DESIGN_20261015/S0_execution_20261015/RESULT.json>)的records/metrics/certificate，
原坐标与每slot指标在4个NPZ内；没有伪造3seed重复或CI。

## 5. 资源与实现来源

生产peak private2.081611633GiB、RSS2.096004486GiB，
最低可用physical16.47989273GiB、commit23.78207397GiB。
补充CPU peak private0.8971481323GiB；不拿预检峰值替代生产峰值。
process6GiB、physical和commit各reserve6GiB保持；没有改OS/pagefile或关闭用户程序。[RESOURCE](<<WORKSPACE_ROOT>/NATIVE_SUPPORT_WIDE_REFINEMENT_DESIGN_20261015/S0_execution_20261015/RESOURCE.json>)。

第一次预检在端口标签native20/native接线处KeyError，未生产启动；仅修字段读取，失败代码和trace保存在新目录failures，
随后完整预检通过。生产本次worker stderr为空且双SUCCESS，不把旧预检失败混作生产或正式训练失败。
锁后生产科学源未改；补充与报告仅新增completion_review/分析文件，旧输出SHA与来源清单保留。
原自动worker初步报告已另存completion_review/INITIAL_WORKER_RESULT_ANALYSIS.md/json，不删证据。

## 6. 反方核查与后续决定

ARS同助手inline核查，不是外部或独立同行评审。本报告为AI辅助研究记录；仅使用本地注册协议和真实保存证据，没有额外文献或因果证明。

- 检查1（沿用设计、回看范围）：GT方向球+保守ray只是有限见证，不是整个安全集合最优；这是方法范围限制，不能补一个未注册solver。
- 检查2（证据链）：同时保留A0大放松机会、S0小的非零误差改善与stable大修正反例；数值fallback与zeroalpha分开计数。
  结论限定为本方向/半径/33逐导数映射。详见[ARS_DA2](<<WORKSPACE_ROOT>/NATIVE_SUPPORT_WIDE_REFINEMENT_DESIGN_20261015/S0_execution_20261015/completion_review/ARS_DA2.md>)。
- 检查3（最终压力测试）：本实验没有训练，不能归因为loss、P2、共享head干扰或scorer，更不能否定所有安全生成；
  停止的是这条注册配方的自动训练。详见[ARS_DA3](<<WORKSPACE_ROOT>/NATIVE_SUPPORT_WIDE_REFINEMENT_DESIGN_20261015/S0_execution_20261015/completion_review/ARS_DA3.md>)。

**执行STOP_SAFE_OPPORTUNITY_UNDER_REGISTERED_MAP：S1未预检、未启动、不是失败训练，也不凑六次run。**
不扫描球半径、margin、余量、方向、width/LR/epochs，不复活A0→A1/B、D0→U1、旧选择器、T1或暂停任务。
按指南可讨论的下一类问题是“能改变时间形状、而非沿一个逐点GT球方向共同缩放的结构级可行生成”，
但本报告不是这类方案的有效性证据或启动许可；需要新的精确设计和相应决定。
研究总目标仍未完成，baseline不升级，旧major5%与全组five-motion1%标准不变。

当前一次半程节点已消耗，原15:26:39.521完成ETA取消，不再为已完成18068安排PID/任务查询。
runtime与账本记为ANALYZED，active_worker=false；当前COUNT1跟进已执行完，无新决定时安静，不每日复查。
