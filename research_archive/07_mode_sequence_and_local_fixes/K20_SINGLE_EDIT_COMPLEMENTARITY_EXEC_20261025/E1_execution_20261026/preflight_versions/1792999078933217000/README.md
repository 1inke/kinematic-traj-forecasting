# E1：一次编辑的Independent vs Set-ranking

2026-10-26 E0全4915200动作独立审核及全部原门通过后，进入本实现目录。当前代码准备/局部回归，不是完整预检PASS或正式训练启动。

固定324维、11488参数GELU、同多正例T/GT-free动作准入、同统计/初始化/order，唯一差异为I窗口平衡BCE vs S softplus(1+firstmax_negative-firstmax_positive)。零末层nativeidentity。仅FIT_opt4096训练，六fresh×8epoch；完整六终态锁定后才解锁hold/inner标签和最终评价。

本目录只新增，不删除或覆盖旧代码、权重、结果、日志与失败。E0原12/11/10修订与1.01 literal无epsilon保持；动作0STOP，其次r/l。

特征：past位置/速度/加速度42、原float32缓存recent96+mean8tokens96（先按旧float32 mean再转数值审核float64，保持原定义）、native均值轨迹24+endpoint总体协方差3+五motion5=32、replacement57、isSTOP1。STOP保留共享266，raw replacement57为0，统计后不重新归零。特征API没有GT/groups/ID/logits。

统计只用FIT_opt：窗口等权，窗内eligible含STOP等权，float64合并窗口均值与窗内方差，ddof0/std<1e-6置1；模型输入与统计buffers转float32。统计独立二遍审核，负方差只在预先浮点证书内处理，不能随意clip。完整统计尚待实现/真实验证，不能以局部fixtures冒充。

完整启动前仍必须：原输入身份/全4096共同stat与源锁、真实GPU8batch32/phasedgradient、独立标量与双eps参数FD且max活动面不切换、双方513opt子集2epoch所有params/buffers/optimizer/TorchCPUCUDAloaderNumPyPythonRNG/history EXACT、completed合同/坏source-split-stat-checkpoint-output读取拒绝、128真实保存全质量/聚簇/五motion、当前RAM/disk/实测4warm32吞吐。任何科学门失败停止，不偷偷替换eps/阈值/训练样本。

本阶段没有OSworker或生产ETA，尚不能运行正式六模型。不得读取hold/inner GT来选特征/损失或检查点；不访问outer/CAL新科学人口。模型/梯度预检只针对新编辑器，不冒称旧F16再次GPU回放。基线/historycycle11与原major/保留/运动学门不变。

同助手ARS DA1：oracle机会不等于可学，S责任集中不等于解决所有credit assignment；可学性必须由同容量I/S实际对照和留出结果判定，非外审。
