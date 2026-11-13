# 03｜I1：去掉候选竞争后的运动信息读出

状态：DESIGN_ONLY。两臂、三seed，共6个新训练。它是诊断，不是新的候选生成baseline。

## 1. 为什么不是重跑M1

M1用动态历史编码器训练goal生成，同时涉及多候选hardwinner、goal支撑和泛化。M1阴性无法单独回答“这段历史在现有训练预算下能否读出有用的未来修正信息”。

I1将未来轨迹作为统一监督目标，不选择winner，不加diversity，不设0.2m修正界。所有候选条件都接受同一真实未来监督，因此先拿掉候选责任分配和局部可达域两类障碍。它牺牲多模态生成任务的完整性，换取更窄的诊断；不能用I1的单点回归成功声称候选集合成功。

## 2. 唯一两臂因素：显式动态历史 vs 静态历史占位

局部观测q[0..7]，q7=0，dt=0.4：

\[
v_t=(q_t-q_{t-1})/0.4\;(t=1..7),\quad
a_t=(v_t-v_{t-1})/0.4\;(t=2..7).
\]

- **S静态臂**：hS=[v7重复7次，二维零加速度重复6次]，26维。
- **H历史臂**：hH=[v1..v7，a2..a7]，26维。

每slot输入x=[P2(96), native_j(24), h(26)]，146维。共享MLP146→128→64→24，GELU，共**28632参数**；两臂全部参数同seed严格配对，最后一层零初始化。无GRU、attention、slot embedding、MoE或新scorer。

只用FIT的真实H特征拟合一套共同统计，并原样用于S/H：float32特征、float64人口mean/std，std<1e-6置1。P2/native统计也相同。历史以window计一次、native以window×20计；全部分母与重复策略锁入manifest。没有outer/inner/CAL统计。

注意：P2和native可能已经编码完整历史。因此H不是“给模型新的传感器观测”，而是额外提供同一已观测历史的显式读出通道；S也不是一个没有历史信息的模型。阳性证明这一通道有用，不证明P2完全丢失信息；阴性也可能是冗余、当前读出器/优化预算不足。

## 3. 共同目标与预测

\[
\hat p_{ij}=p^0_{ij}+f_\theta(x_{ij}),\qquad
L_{readout}=\frac1{B\cdot20\cdot12}\sum_{i,j,t}\|\hat p_{ijt}-GT_{it}\|_2^2.
\]

head计算float32，路径加法/误差归约float64。没有tanh、积分控制、投影、步长界、native-success keep、物理正则、diversity或hardmin。GT仅用于FIT监督和事后评价，不进入forward。

该目标等价于预测GT−native的直接回归。不同native条件可能被映射到相似预测，这是诊断目标允许的结果；不能把重复/塌缩的20个回归输出宣称多模态候选成功。

seed42/43/44；25epochs；batch256 windows，全部K20；固定epoch25；AdamW lr.001、wd.0001、betas(.9,.999)、eps1e-8、foreach=false、无clip。两臂同样本顺序和更新次数；从fresh paired init开始，不复用旧head权重。

## 4. 主要读出指标与对照

先计算window级：

\[
MSE_i=\frac1{20\cdot12}\sum_{j,t}\|\hat p_{ijt}-GT_{it}\|_2^2.
\]

再在同group window上平均。**N仍是窗口数，不是N×20个独立样本**。本项primary不使用minK，避免用候选选择掩盖平均预测能力。

共同无训练参照：

- native20原路径（同样mean20 MSE）；
- CV路径pCV_t=0.4t·v7，复制到候选轴仅用于相同指标计算，不计为20个独立预测。

outer change两个共同primary比较：H−S MSE与H−CV MSE；同时报告相对收益、ADE/FDE、native参照、early t1..4 / middle5..8 / late9..12误差，以及20预测均值路径和输出分歧。

探索性附表必须包含FIT/outer/inner/CAL全部overall、stable、transition、change；native成功/失败分层；所有seed。仍报告完整候选coverage、有效簇和五motion，但这些只是指出读出是否可能塌缩/不安全，不替代primary或升级baseline。

## 5. 统计与唯一信息触发门

- 20,000次PCG64 seed20261016，within-split同ID配对、target-track×模型seed交叉bootstrap；沿用T2已审核的窗口加权与seed平均顺序。
- 两个primary构成固定family2，每个双侧97.5% percentile区间（Bonferroni同时覆盖口径）；不得事后缩成family1。3seed和相关窗口仍只提供探索证据。
- H−CV中CV无模型seed，三个seed共享同一CV输出，不能把它当三个随机baseline。
- outer change中H相对S的MSE每seed均下降、平均相对收益≥5%，且H−S与H−CV两个同时CI上界都<0；同时outer overall/stable/transition的H相对S MSE每seed点估计不恶化超过1%，才记`INFORMATION_READOUT_SIGNAL=true`。
- 5%是这个新诊断的事前最小效应门，不是修改原G1覆盖/ADE/FDE/安全门。该阳性仍不是证明泛化到新场景。

非阳性情况分别标记：FIT未拟合、FIT改善而outer未迁移、outer有改善但不胜CV、区间不确定、protected subgroup退化。不要把它们合并成“历史无信息”。不因阴性加宽、加epoch或换GRU求PASS。

## 6. 必须预检与交付

- 对齐原past/P2/native/GT/IDs、坐标/单位/时间轴；验证真实history只依赖已观察8点；常速fixture产生零加速度；S保持末速度、抹去变化顺序。
- 两臂同统计/参数/批次、initial native identity；实际GPU8 FIT batch256 phased合法梯度；MSE NumPy标量/坐标梯度与固定非零参数double FD，不声称旧L0梯度一致。
- Both真实513 FIT+12 inner子集两epoch连续/中断：参数、buffers、optimizer、Torch CPU/CUDA/loader/Python/NumPy RNG、history严格一致；不是完整FIT两轮。
- 六run各存checkpoint、run_manifest、statistics、split IDs、25行history、完整预测与window指标、source/output SHA。新目标必须明确标为DIRECT_READOUT_MSE，不混旧canonical L0。
- 独立CPU保存结果重算primary及全部探索指标、两个主CI、source和模型文件SHA；不是二次模型/梯度重放。

F0正负均不改变I1标签、输入、损失或抽样。I1不读F0求解轨迹，避免把oracle监督和表示改变混成同一因素。
