# K20形态/GT贡献离线审计

2026-10-27 human“评估并执行对应实验”授权。原文DESIGN_ONLY为提供的研究设计，本目录为实际有限执行。不训练/不改候选、不调用旧模型/修正器/KIN。E1单独进程只读取本目录隔离后的无GT输入；E1配置与所有模式结果锁后，E2才关联GT。原9777唯一窗口只计一次，seed42原生作为经SHA绑定的参考，不复制3个seed样本。

输出压缩NPZ和UTF8 CSV（当前环境无pyarrow，不安装依赖）；科研图用已保存Python后端。新文件只写本目录，旧来源/失败/模型/自动记录保留，不恢复任何旧任务。CPU前台有限进程运行完成并审核，不创建新的每日跟进。

## 当前冻结实现合同

- 原opt4096分train3103/val993，hold3825与inner1856；原sourceID、scene、sequence、target和group不重分。
- train历史7段速度的Q05与float32量化速度上界4×Q95(spacing)/.4取max为操作性slow floor；它不是实测传感器噪声估计。低速需mean速度和endpoint位移同时低于floor及floor×4.8；初末4段位移方向过小时undefined，不硬叫转弯。未来从历史末点p0起算12段，dt=.4继承已锁规则；原始视频时间戳未随cache保存，不冒称重做视频时间认证，start index歧义如实保留。
- 主coarse：low-motion/undefined/straight/left/right；speed trend以abs(delta)>=floor且ratio超过±.2分accelerate/constant/decelerate，形成辅助11类。turn主20°，固定15°/30°和floor×.5/2敏感性。
- phi=[相对历史朝向endpointXY,sin(turn),cos(turn),mean speed,delta speed,straightness,turn-valid]，undefined角sin/cos置0且保留mask，历史朝向未知时endpoint用norm/0并标记。train等窗等候选float64均值标准差，constant std置1；辅助complete-link距离阈值=全部train候选最近邻距离median，固定×.5/1/2。无GT/no固定簇数。
- 每coarse/辅助类与辅助簇medoid取同组phi距离平均最小，exact tie取原physical slot最小，近tie另记；非语义身份。主coarse是E2-A/B解释口径，joint与clustering敏感性不可择优。
- good literal同slot ADE<=.2 AND FDE<=.4。J/A/F所有exact与2e-12近tie赢家都存。纯算术20次单槽删除，保存严格delta与2e-12数值等价标签；good不加epsilon。oracle-removable必须非medoid且同模式>=2，且L=0及三delta<=2e-12；同时保存strictzero计数。此为单GT/单次删除，不实际删轨迹，不能相加后同时删。
- nearest-GT phi仅E2。GT undefined与阈值附近1°/速度相对边界5%另列，原始缺席/存在×good列联完整保留，不通过删边界制造结论。
- 20000 default_rng20261027 whole(sequence|target)重抽，population全track先抽后group mask，窗口加权分子/分母重算，invalid不补抽。描述95%CI，无显著性或2pp门；3scene不当可靠独立场景样本。按scene/sequence/track输出分布。
- E1案例每group固定hash最小3窗，不GT选；E2四类各固定hash最小2窗，标GT事后选例，不存在类明确空，不补造。
- resource v3沿最新human：1.3预警、真实压力持续才停，CPU E2GiB来自384MiB基础+512MiB单人口输入+512MiB数组+384MiBbootstrap+256MiB绘图/IO。无OS/pagefile/userkill/删除。

## DA1（同助手，非外审）

PASS有限描述性问题。Major：粗分型可人为造成集中，因此必须同时报连续量、11类/角度/低速/聚簇尺度敏感性；Q05是行为下分位不是传感器噪声，明确代理性质；最多3赢家的平凡删除结论必须单独解释。当前GT-free结构重复和GT oracle可移除不能升级为部署策略。原F16见数据/反复开发/相关窗口/时钟来源限制保留，不新模型、不自动下一方法。

## 图合同（quantitative grid + 案例image plate）

E1主图说明几何分散是否伴形态集中，直方/stack occupancy/转角速度hexbin/geom-vs-mode/无GT案例分别提供分布、组成、连续形态、口径对照和可见反例。E2主图对照缺席vs存在失败、名额vs命中、J差异、oracle可移除与结构、GT事后案例。仅真实全数据聚合，不抽点美化；Python SVG/PDF/PNG，editable text、183mm宽/可读字号，来源CSV/NPZ与n、整track95%CI注释；非期刊最终投稿认证。
