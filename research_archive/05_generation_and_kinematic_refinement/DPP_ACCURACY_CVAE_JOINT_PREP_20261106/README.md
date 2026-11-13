# DPP + ADE/FDE 与新CVAE联合训练：脚本准备包

**当前：SCRIPTS_ONLY_DO_NOT_START。用户明确“先写好实验脚本，暂且不跑”。没有训练、GPU实际预检、OS任务或自动跟进。**

## 两个方向

|目录|生成器|更新范围|配对臂|
|---|---|---|---|
|01_existing_joint|原冻结F16+h3，原DSF64→128→1280|只更新新DSF；旧权重/bool buffers不动|accuracy_only / accuracy_dpp|
|02_cvae_joint|新CVAE先验、训练后验、直接12步解码器与DSF|新CVAE和新DSF从零联合更新；观测上下文固定|accuracy_only / accuracy_dpp|

每臂42/43/44，共12新run，每run20轮/13580更新，总240轮/162960更新。原纯DPP六模型只读保留，不重训、不用失败模型warmstart。

第一方向保持冻结生成器，目标是“联合损失”，不是解冻旧生成器。第二方向则真正联合优化新先验/后验/解码器/DSF，不先预训练再永远冻住CVAE。推理不运行后验、不读取未来。

## 损失，固定后不按dev扫描

共同准确性：`Lacc = mean min_i(ADE_i + .5 FDE_i)`，同一完整候选，物理slot首个tie，不拼不同候选的ADE/FDE。

- 现有生成器：`L = Lacc - .1 ECkin/20`，对照将DPP系数设0。
- CVAE：`L = Lacc + Lposterior_recon + .01 KL(q||p) - .1 ECkin/20`，对照同结构/同KL/重构，仅DPP系数0。

EC与原StageA同核、同FIT-only qkin阈值；除以20后取值在[0,1]，DPP项贡献[-.1,0]，避免原裸EC数量级默认压过米制误差。`.1`和`.01`是此新设计的固定工程初值，不宣称最优/来自原论文，也不按dev调参。与旧纯`-EC`研究相比同时改变监督与EC缩放，因此旧→新比较不是严格只加一项；同包accuracy_only→accuracy_dpp才是对应单因素对照。

CVAE后验重构为固定sigma=.2m的Gaussian平方误差NLL（省略常数），KL为16维对角Gaussian KL总和；不是把ADE伪称严格ELBO。训练还包含部署时实际生成K20的同slot ADE/FDE，以减轻只训posterior、测试prior的断层；不能保证消除该断层。

## CVAE究竟替换了什么

保留相同已冻结的64维观测context缓存；它来自旧F16与h3.context观测投影。因此不是全新独立encoder，也不冒称完全抛弃每一行旧decoder代码。

新轨迹生成器**不再使用**旧h3的mode_mlp、slot模板、proposal_scale/delta、4/8/12锚点插值或score。新decoder输入context64+latent16，经128/128ReLU直接输出12×2局部位移并转换世界坐标。没有旧锚点可达盒或新增安全投影，运动学仍独立评价。

条件prior `p(z|c)`；训练posterior `q(z|c,future_local)`；logvar固定截断[-8,8]只为数值建模，不是评价容差。新DSF输出20×16标准化坐标偏移，加每seed固定20×16高斯基底，再经条件prior均值/方差到真实latent。零DSF偏移不是CVAE nativeidentity，更不是原模型identity；原模型DSF零偏移才保证原生路径。没有擅加原radius对照或scorer。CVAE模块参数与整个冻结观测编码链参数必须分开记录，未实测完整链不能声称<=305000或更轻。

## 数据、冻结与统计

FIT21719训练；原dev4753、change960仅在**两个方向全部12final SHA锁定后**新评价。缓存只有观测，不以dev/未来fit归一化。全部相同ID、同seed成对初值、epoch窗口顺序与CVAE重构噪声RNG可精确恢复。GT只进入FIT损失/训练posterior；`generate(bundle)`没有future/ID/group/map/logit参数。

原samecandidate覆盖、lost<=1%、protectedADE/FDE1%、每seed/split/group五pooled-motion1%、major changeADE/FDE各5%等不放宽。完整K20 complete-link(.10mean/.20endpoint)及独立pdist、五motion和接续、q/EC独立循环、救回lostIDs均保存；新CVAE没有训练scorer，不用全零分数的slot0冒Top1结果。

主devchange比较共5×6=固定family30（协议列全），20000整targettrack→pairedseed、窗口加权后seed均值，Bonferroni和fixedseed敏感性全部保存；不因某臂失败缩家族，不足20change tracks或>5%无效draw不补抽/不判阳性。旧纯DPP仅历史描述，cross-family有结构/容量/优化几何/可达集差异，不是纯生成器因果证明。三CVAE新generator种子仍共享同一个已训练观测编码器；原family只三DSF种子，不能称三独立wholemodel。

## 脚本入口与防误启动

- `run_series.py`：不带参数或`plan`只打印协议。
- 两子目录`run.py`：各方向入口，默认也只plan。
- `preflight_real.py`：写好未来GPU/真实513FIT两轮中断19/34更新精确恢复、分阶段梯度、128保存审核、坏合同拒绝、吞吐与资源估计流程；**本次不执行**。
- `joint_train.py`：完整Adam状态、所有CPU/CUDA/Python/NumPy/loader RNG、epoch ID order、cursor、history检查点与完成SHA-skip。
- `joint_evaluate.py`、`joint_statistics.py`：全12final后显式评价、26人口和family30/原保护门。
- `validate_scripts.py`：仅CPU合成前向/梯度/固定双direction双eps FD、NumPy质量/EC、ADE/FDE独立梯度、默认入口与hold拒绝检查；无optimizer/训练更新/真实数据标签。

即使传`--execute`，`control.json`当前false也会在读取训练数据/创建模型前拒绝。不存在Start-Process、任务注册、定时器或自动解锁。未来必须得到新启动指令，按实际前检结果建立新execution源锁后才能跑，不能把CODE_VALIDATION当正式GPU预检。

资源沿warning-first v3：1.3仅预警，无固定6GiB/30min硬停；真实持续系统压力检查点STOP，不锁内存、不改OS/pagefile、不kill用户软件。资源E/ETA须未来实际测量，本次无生产ETA。训练/评价不删除任何旧文件；已有部分结果不能覆盖，不完整保存人口需要新恢复namespace，不盲重训完成模型。

当前只准备科学脚本，不伪造真实raw-observation完整端到端成本模块；未来science完成状态明确标cost未测量，不能当部署PASS。旧DSF成本阶段未封存也不在本包悄悄修复。

## ARS同助手反证检查（非外审）

- DA1：加accuracy可能只修平均误差而仍丢旧成功，原lost/coverage门继续；两方向不叠成同一次结构+loss“唯一因果”。
- DA2：CVAE必须保留真实posterior重构与KL、GT-free推理和prior K20 accuracy，不把普通随机MLP称CVAE。保留共享旧观测表示/新参数可达性差异。
- DA3：脚本写好不等GPU/resume/物理与数据准入通过；CPU合成证明不冒真实模型结果，未测成本不PASS。旧dev反复探索/相关窗口/3seed局限保留。

原理来源：[DSF/DPP论文](https://arxiv.org/abs/1907.04967)、[CVAE原始论文](https://proceedings.neurips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf)。本包是用户提出的联合目标新变体，不声称复现原DSF冻结生成器训练流程或文献最优配置。
