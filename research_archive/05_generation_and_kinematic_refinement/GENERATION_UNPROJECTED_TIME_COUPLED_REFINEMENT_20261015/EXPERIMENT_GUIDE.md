# U0：移除ray后的全K20时间耦合修正

2026-10-15 人类明确要求直接执行两阶段研究：先验证新的时间耦合修正是否改善完整覆盖，再分析准确性与运动学代价。此文在预检/生产结果前登记；不是复活已停止S0/S1。

## 问题与因子

两臂都learned newhead、fullK20、native锚定，没有GT oracle、教师、逐时刻1.01约束、alpha、全路径统一缩放或失败回退。保留相同0.20(t/12)^2位置预算、参数25304、输入observed P296+native24、FIT-only共同统计、原K20 hardADE+.5FDE winner+.05*190pair squaredhinge(.3)、AdamW.001wd.0001/batch256/25epoch/paired seeds42,43,44。旧F16/decoder/scorer不训练，GT/groups/IDs/scene/logits不forward。

唯一结构因子是时间输出算子：
- free_position：r_t=.20/144 * t² * u_t。
- integrated_acceleration：r_t=.20/144 * sum_{j=1..t}[2(t-j)+1]u_j。

u_j为float32 head tanh后转换float64，逐二维控制向量除max(1,L2norm)。这只是共同控制幅度界，不是S0的依原生导数预算/全路径alpha。两臂都不事后按运动学裁剪。
第二式等于零初始修正位置/速度下、每.4s常加速度控制的精确双重积分；控制加速度范数上界固定2*.20/(144*.4²)=.0173611111m/s²，推导自原位置预算，不从质量选尺度。正权重和为t²，故||r_t||≤原b_t；不另做逐点位置投影。它不是Bernstein，亦非旧自由增量模型重启。相同参数不等于同可达集合/梯度条件数或同有效容量；时间耦合可能过度收缩，失败不能否定全部时间结构。

## 数据与预算

现有target-grouped FIT18909/outer1035/inner1856/originalCAL2879，24679uniqueIDs；outer固定final25唯一主人口，inner/CAL不选checkpoint或超参。FIT统计P2每window一次/native每candidate一次，float32feature、float64总体mean/std、std<1e-6→1。继承S0全部1863digest并扩入其完成产物/失败及新代码，不覆盖旧hash。
两臂同weights/scalers/kernelbank、zero-last初始native。仅时间算子选择不同。6fresh×25=150epochs/11100steps；全4split×6final=24科学保存job，另native4参考job不算训练。无scheduler、gradientclip、earlystop、teacher、事件监督、newloss/scorer/unfreeze/newdata。实际preflight planning>30min不启动，不减少预算凑门。

## 评价与第二阶段分支

覆盖=至少一条samecomplete candidate ADE<=.2且FDE<=.4。报告全部split/group/seed的rescue/lost/retained/ID列表、J=minmax、minADE/FDE、Q/J各自同slot早中晚、GTfree全K20原SciPycomplete-link.10/.20有效good与distinct、五pooledmotiondt.4、修正幅度和训练梯度/全150history。有效good不是最大化，scorer不进入。
主outerchange 12个比较：coupled-native、free-native、coupled-free各coverage/ADE/FDE/J。20000PCG64seed20261015 pairedtargettrack×3pairedseed交叉bootstrap，同一次draw用于全部端口，窗口加权；95%探索CI和Bonferroni12同时CI，不缩家族。
潜力/准确性信号（不是安全成功）：coupled-native outerchange覆盖均值>=5pp、同时CI下界>0、三个seed均正，ADE/FDE不害；仅FIT改善须报告未迁移。保留nativegood：每seed/group lost/nativegood<=1%并完整展示净增交换；overall/stable/transition ADE/FDE<=native1%害。原major changeADE/FDE各5%、native/control联合CI、三seed、保护与five-motion1%全部保持，不宣称unsafe潜力为baseline成功。
运动学全5项：speedp95 accelmean/p95 jerkmean/p95，pooled allK20，literal<=1.01native。它们本轮是原标准不变的评价门，允许观察失败，不作为生产中途fallback/改loss/重选epoch。覆盖可信改善但运动学失败→下一步才精确设计代价控制；无覆盖提升→查方向/幅度/时间结构/有限优化，不自动loss/scorer或局部扫参。两者皆过再新确认设计，当前不自动训练新保护项。三个seed、重复窗口/已见F16/适应性复用outer/CAL均探索，不SOTA/独立wholemodel泛化/意图证明。

## 真实预检与停止

实际GPU双方8FIT256梯度/paired全paramsbuffers/nativeidentity/finiteK20/位置界/前向GTfree；独立积分递推和全K20loss标量/坐标梯度、零/并列fixture、固定double参数FD1e-4/1e-5且活动面变动明示。双方513FIT+12inner两epoch连续中断ALLparamsbuffersoptimizerTorchCPUCUDAloaderNumPyPythonRNGhistory EXACT，source/output/resume错配拒绝。actual128保存质量/全聚簇/运动学/controls重构、源与ID审核；实测4warm32measure/both、完整保存评估/24聚簇/CI/audit/IO/SHA估时。
process6GiB/BOTHphysical+commit eachreserve6GiB/~17GB长期规划，不OS/pagefile/userkill。source/ID/resource/科学可辨识实质阻碍STOP留来源等人类，不能放宽标准或盲无限变体。通过才unique0triggerOSworker，ONEstartup/ONEhalf→一次actualcompletion，随后退出、不对话轮询；完整双SUCCESS+所有保存审核才分析。模型/模型前向均只新head，不能冒称旧整个网络又冻结重放。

## ARS同助手反方1（非外审）

本实验不重复A0知道GT即可更接近的事实；真正测试GTfree learnedmapping。旧Bernstein、freeincrements/timequery/velocitybasis失败保留，新的积分式也可能失效；不能凭平滑直觉宣称新颖或有效。对照因子是有限有界输出表示和优化几何，不pure safety-removal因果，更不是独立泛化。低加速度不确保完整路径正确；coverage上升可伴lost与jerk害，必须保留。新方案仅本次有限对照，不旧source改名重启；总目标/baseline/historycycle11不变。
