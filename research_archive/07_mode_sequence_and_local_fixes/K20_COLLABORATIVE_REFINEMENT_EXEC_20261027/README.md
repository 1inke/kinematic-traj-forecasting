# K20候选间协同细化：一次固定机制验证

人类2026-10-27两次明确“按照他的这个方向，进行一次验证，你来执行”授权。依据Downloads/K20_候选间协同细化_最小验证实验与创新路线_20261027.md，补全以下固定选择，不是已通过的结果。旧STOP-gate D1保持暂停，旧PAUSED不动，任何文件/日志/失败不删除。

B0冻结native20；B1独立细化与B2跨候选细化各42/43/44三fresh8epoch，无KIN、无旧模型前向或训练。唯一结构因素是相同注意力的对角mask与全连接mask。共同263维=每candidate原生24坐标+该candidate五motion+历史42+原float32 last/mean tokens192。没有候选槽位embed、GT、group、ID、logit或全K摘要前向。全局train统计不是单窗候选归一化。

opt4096按实际509整track的SHA256 `collaborative-val-v1|track`排序前ceil20%=102track为小验证，其余407track训练；实际训练3103窗/change721，小验证993窗，20%是track比例而非window比例。数量由INPUT_AUDIT实际报告，原216是opt变化组track数，不能冒全opt分母。预先固定全部超参数和最终8轮，不搜索、不earlystop，小验证只固定终态描述。所有6终态SHA锁后才held GT/评价，hold3825 change为primary、inner1856 secondary。F16已见这些数据/反复开发/相关窗3seed探索限制保持，不独立wholemodel泛化。

两臂均263→32GELU，2头QKV/O attention+残差LayerNorm，一层32→64→32GELU FFN+LayerNorm。B1 self-only时Q/K结构性梯度0是预期，不假造全参数正梯度；B2全连接需非零离对角依赖和集合排列等变。offset32→6全零初值，gate32→1全零=>.5，初始native bitwiseidentity；gate/前层首次零梯度预期，后续分阶段连接。g不是概率/意图。

三锚点4/8/12与原点0残差，SciPy CubicSpline边界r'(0)=0,r''(12)=0，共同C2固定基，逐点残差范数≤.20m由基的绝对行和归一化及锚点unitdisk/tanh保证。没有S0 ray/alpha/projection/fallback；motion只共同soft训练项与严格原评价。

令j_k=softmax_tau(A_k/.2,F_k/.4)，tau=.1，s=-.1logsumexp(-j/.1)。Lset=mean((1+1[Jnative>1])*s)；Lkeep=mean(1[Jnative<=1]relu(Jnew_literal−Jnative_detach))，literal max ADE-first/min physical-first；Lmotion=mean(relu((mean[v,a,j]new−mean[v,a,j]native)/train_RMS_scales)^2)；Lsmall=mean(||residual||²)/.2²。固定total=Lset+Lkeep+.1Lmotion+.01Lsmall，native masks/参考只loss，无teacher/GTforward。复制候选可降低softmin但不增加literal覆盖，保存有效good/distinct/soft与literal及梯度/位移分布，不加DPP/diversity。

AdamW .001/wd.0001/betas.9,.999/eps1e-8/foreachFalse，clip5，batch32，shuffle seed配对，8fixedfinal。B0不训练，共6模型。主holdchange B2-B1与B2-B0两比较：各seed严格正、均>=2pp、整holdtrack后配对3seed的20000draw RNG20261027，Bonferroni2 97.5% CI和fixedseed敏感性下界>0；invalid>5%或change tracks<20不足，不补抽。保护hold+inner各seed各组native-success lost<=1%，overall/stable/transition ADE/FDE<=native1.01，全部五pooledmotion<=native1.01，innerchange不降。原major change ADE/FDE各5%与独立泛化/成本另审，不升baseline/historycycle11/goal。

实际GPU8/独立NumPy scalar及全参数chain/FD1e-4与1e-5固定夹具/真实513训练窗两epoch连续中断ALL状态RNG/history EXACT/坏source-ID-stat-checkpoint-output/resume拒绝/128真保存全quality原complete-link+pdist五motion/实际benchmark与1.3资源全PASS才sole OS0trigger worker。此后ONEstartup→ONEhalf COUNT1→实际completion，不daily/poll。完成同助手ARS DA2/3非外审；阴性停止此一次实现，不扫超参数/预算分配/新分支。

成本分别报告新head参数/50warm500模块CPU1thread GPU batch1/32测量；旧整模参数可只读state元数据认证，真实raw端到端若未测必须PENDING，不缓存冒充。无部署/轻量化成功假称。新artifact仅本目录，旧digest与结果/失败保留；可定位工程错保留来源后修，科学/source/真实资源问题停止报告。
