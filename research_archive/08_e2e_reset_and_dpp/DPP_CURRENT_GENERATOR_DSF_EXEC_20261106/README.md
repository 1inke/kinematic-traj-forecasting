# 当前冻结生成器上的 DSF：均匀质量 vs 运动学质量

2026-11-06 human“继续推进”，承接“先用目前轨迹生成方法，不行才预训练CVAE”。实际当前接口已通过，因此不另训CVAE。本目录新有限配对实验；旧Stage A/所有旧模型结果失败日志和PAUSED记录不改不删。

## 唯一因素与范围

原fresh-B0 seed42 final8 EMA F16+h3295474参数全部冻结。新增同一DSF64→128ReLU→20×64/zero-last173440参数，偏移原mode query，仍原scale/模板/tanh/4-8-12线性轨迹。整体468914>旧305000目标如实披露。这是非高斯确定性control DSF，不声称原CVAE Gaussian prior/radius或原论文逐字复现。

只做uniform r1和kin r=sqrt(eps+(1-eps)qkin)，三组配对DSF seed42/43/44、各固定20轮batch32/13580updates，六run120epoch81480updates。原生成器只一份seed42，不把三DSF种子当三独立wholemodel。唯一loss=-mean ExpectedCardinality，原高斯轨迹L2核，继承StageA FIT-only阈/IQR/k/epsilon，不新增监督/teacher/assignment/正则/保护loss/投影或latent norm半径。Adam1e-4/noWD/clip5/final20/noSelector；20epochs与Adam1e-4来源原论文AppendixB.2，batch32及clip5是本现有接口固定实施选择。所有源和超参在首个科学训练前锁定。

只训练FIT21719的因果context，不使用GT作为DSF输入/目标。所有六final源/模型SHA锁定后才新评价dev4753标签，全部FIT/dev人口保留，不换pool/删困难场景。之前多轮dev已用于开发，始终探索不独立泛化。

地图代码和Y-net真实17图/H资产已具备，但0/7科学地图准入：0/1类规则、静态annotation权限/来源及独立米制valid域/配准尚缺；uni_examples全1不能当安全地图。因此本次scene/joint明确NOT_TRIGGERED，不声称双重可行性已验证，不给缺图qscene=1，不从地图有效子集事后筛人群。B0 Gaussian-radius对本生成器NOT_APPLICABLE。地图问题不阻断已完整定义且不依赖地图的uniform/kin配对。

## 前检与缓存边界

prepare一次真实冻结模型因果观测→context，每个compute32零control原路径/logit bitwise EXACT，所有参数/boolbuffer不变，ID/输入/NPZ读回SHA锁定；旧compute64保存路径仅原2e-5数值兼容，不把此容差加到质量或物理门。缓存只降低训练重复编码成本，不称真实rawstream端到端成本。

生产必须先两臂真实GPU8×32分阶段grad/paired全部初值，实际513FIT子集两epoch34updates中断19后的参数、Adam、CPU/CUDA/Python/NumPy/loaderRNG、订单游标/history全部EXACT；dev12仅观测前向不训练/不读GT。独立NumPy DPP loss全path梯度+全parameterchain，固定双direction eps1e-4/1e-5活动面；实际128保存原same-slot质量、fullK20 GTfree complete-link.10/.20+pdist、五pooled与junction双实现、原raw/logq/EC独立循环与NumPy decoder64重构到生产float32原2e-5路径容差。坏source/split/ID/stat/checkpoint/output必须拒绝，不把smoke算科学训练。

## 完成与统计

六run final20、各13580update，逐epoch原optimizer和全部RNG证据保留。ALL6_FINALS_LOCK先于dev标签评价。12模型人口+2只读native人口共14，每条保存K20/显式IDs/全controls/q与raw/logq/EC/质量/聚簇/五motion/junction及独立审核，不用Q与J不同slot拼假成功。每组/每seed rescue/lostID与反例完整保存。

主devchange为kin-uniform/kin-native/uniform-native × coverage/kin-validcoverage/违规/minADE/minFDE/effectivegood共family18；固定20000 default_rng20261106 sharedwholedevtargettrack再pairedDSFseed、窗权重再seedmean，普通95/Bonf18和fixedseed敏感性全列，每draw第二种multiplicity算法复算；少于20change tracks或invalid>5% UNDERPOWERED不补抽。主要定位是质量项机制探索，不自动升级baseline/宣称重大成功。原major5%ADEFDE/5pp覆盖、旧好lost<=1%、protectederrors1%、每seed每组原五motion literal1%和相关性审核不放宽。固定同一预训练模型及反复dev限制尤其重要。

只计一项六DSF-run研究，不是六wholemodel实验，不重置baseline/historycycle11/未完goal。完成自动写RESULT_ANALYSIS/全统计和保存审核；同助手ARS DA2/3非外审，不再自动新lambda/LRwidthEpoch/score/旧解冻/额外CVAE变体。

## 资源与后台

warning-first v3：E来自实峰+明确payload/allocator，1.3仅预警，持续当前physical/commit真实短缺才checkpointSTOP；无固定6GiB/30min/6hour门，不OS/pagefile/用户进程操作，不删任何文件腾空间。实际吞吐和完整保存/独立循环审核/CI/IO给ETA，非保证。新sole on-demand XML0 OS任务（无定时trigger）独立运行，自带训练→保存→独立审核→统计→实际50warm500数组全模型链成本→源终SHA后退出。成本明确不含raw录制文件读入和邻居检索，不缓存冒端到端，不以新DSF参数冒整模轻量。

来源：原用户Downloads/DPP_双重可行性质量评分_改进实验设计报告.md；https://arxiv.org/html/1907.04967；上一阶段锁定CALIBRATION/原C0身份合同。地图官方来源 https://github.com/HarshayuGirase/Human-Path-Prediction/issues/29 ，软件MIT LICENSE不自动当所有数据annotation许可/配准证书。
