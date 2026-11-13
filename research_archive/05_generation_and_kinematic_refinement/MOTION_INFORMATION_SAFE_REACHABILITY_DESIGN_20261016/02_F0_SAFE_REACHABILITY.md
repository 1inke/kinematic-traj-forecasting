# 02｜F0：安全局部修正的有限存在性诊断

状态：DESIGN_ONLY；NONFORMAL；不训练，不生成教师库。

## 1. 假设及识别边界

问题不是“GT球投影能否过质量门”——A0已经回答；也不是“单射线逐点保护是否有用”——S0已经回答。本项问：

> 在U0的时间耦合控制域中，允许完整20条候选联合变化，能否找到至少一组通过原质量及五运动学审核的修正？

正例可以证明所测试窗口存在安全机会。求解失败、超时、p95审核失败，都不能证明安全机会不存在。下述per-window均值约束比原group-pooled门更严格，因此即使某子问题被数学证明不可行，也只能否定该子问题，不能否定原完整安全域。

## 2. 一次性样本和预算

仅从FIT change中抽取48窗口：32个native failure、16个native success。

1. 固定tag=`20261016-safe-reachability-v1`；先对canonical sequence+target-track key做SHA256排序，再在每track内按tag+sample-ID排序取第一条符合该成功/失败层的窗口。
2. 先取32个failure tracks，再从未选track取16个success tracks；身份/time歧义照原登记保留并报告，不能伪造独立性。
3. 只按以上规则取样，不按修正幅度、求解难度、场景或输出效果换例。不足规模则记录阻碍，不放回补样。
4. 16个native success固定identity，不优化；32个failure每个枚举全部20个目标slot，共640个SOCP子问题，不因早找到解而跳过其余slot。
5. 每个子问题最多2秒墙钟；整个F0计算+保存+完整复核预检预测须≤30分钟。2秒是计算停止条件，不是数学判定阈值。solver超时可留下经独立验证的可行primal，不可当最优解。

最多32个FIT救回仅相当于4725窗口的0.6773pp；本项不可能凭这套样本达到原+5pp门。条件抽样32例的成功率不能直接外推全部FIT/outer。

## 3. 变量、输出域与凸子问题

对某个窗口，变量为全20候选的控制向量u[j,t]∈R²，t=1..12：

\[
\|u_{jt}\|_2\le1,\quad
r_{jt}=\frac{0.20}{144}\sum_{q=1}^{t}[2(t-q)+1]u_{jq},\quad p_{jt}=p^0_{jt}+r_{jt}.
\]

此闭单位盘是原tanh/normalize输出域的闭包；不能默认盘边界上的每个解都由有限raw精确实现。积分式保证原逐时位置界b_t=0.20(t/12)²，不增加独立自由位置控制。

固定目标slot k后，解以下凸子问题：

\[
\min_u \sum_{j=1}^{20}\sum_{t=2}^{12}\|a_{jt}(p)\|_2
\]

约束：

- 该**同一slot k**满足mean_t||p_kt−GT_t||≤0.2，||p_k12−GT_12||≤0.4；
- window内ALL20 acceleration mean≤1.01×native window acceleration mean；
- window内ALL20 jerk mean≤1.01×native window jerk mean；
- 全部控制盘和位置界；初始位置/速度边界按原motion算子，不另造t0加速度。

a、jerk为路径的线性差分，因此范数和及质量约束可写为二阶锥约束。**三个p95没有被凸化，也没有用平均数冒充**，在保存解上按原定义精确审核。允许20条共同调整，不强迫“只有获救slot动”，也不用S0的每一步相对native向量范数门。

最小加速度只是固定搜索准则，不保证找到所有满足p95的可行解；失败后不改成多个目标权重搜索。

## 4. 从求解解到真实可实现见证

对每个返回的primal仅做一次事前固定数值实现：

1. 先验证finite与原始primal残差，保存未改primal；
2. 若某控制norm>1（即使仅solver容差），不能直接用它声称原域可行；
3. 对合法盘内primal统一乘固定(1−2^−20)，得到严格盘内u'，计算raw=atanh(u')后转为实际head的float32输出类型，再依次执行生产tanh/normalize、float64积分与原生路径相加；不搜索缩放量。不能只用float64 atanh回算冒充生产可实现性；
4. 以**这个实际重构路径**重新判断同slot质量、位置界和全部五个window pooled motion；所有科学门literal比较，不加epsilon；
5. 任何一道失败记为“该primal不是有效见证”，不回溯选阈值；保存原值、重构值、差异和失败类别。

缩小可能让恰好J=1的质量解失效，这是诚实的未找到，不可改成1+epsilon。独立算术的数值一致性容差与科学成功门必须分别保存。

每window最终仅取通过全部门的**最低原physical slot k**的解；若没有则identity。此规则依赖GT求解及审核，明确是oracle诊断，不可部署。无解窗口不删出分母。

## 5. 完整安全检查与归因

除48窗口外，FIT其余所有原生路径不动；构造一次完整18909人口的诊断portfolio。重算overall/stable/transition/change全部五pooled motion、same-path覆盖、rescue/lost、有效good簇。per-window p95安全不蕴含全population p95安全，因此必须实际重算完整FIT。

报告：

- 640次尝试的optimal/可行非最优/超时/solver infeasible/数值失败/三个p95拒绝等计数；solver标签不是通用不可行性证明；
- 每个获救窗口和目标slot、控制/路径、quality余量、五motion原值与修正值、全部20条修正幅度；
- 是否靠改变其他候选降低平均加速度腾出预算；逐候选Δaccel mean、改变slot数量、原簇变化，不隐去“候选集合内补偿”；
- 未获救窗口的质量失败、均值约束、p95失败、时间预算各自情况，允许重叠；
- 16个原成功窗口identity只证明oracle诊断人为保留成功，不证明可学习保留策略。

`SAFE_WITNESS_PRESENT=true`仅当至少1个非零救回窗口经独立重构通过、完整FIT四group五motion也全部通过、其余审核/source PASS。这个门只是“值得继续考察”的存在性门，不是足够大的总体收益或可学性证明。

零正例统一结论为`SAFE_OPPORTUNITY_UNRESOLVED`；本阶段不设置“原安全域为空”的出口。不触发solver/阈值/系数扫参。

## 6. 预检和必须保存的证据

将solver实现和版本锁入source；安装/依赖情况在执行时审计，本设计不假定已有可用solver。不在此轮下载或安装。

预检包括：积分算子对旧U0的固定controls一致、native identity、人工已知可行与不可行小例、同candidate门、防止独立ADE/FDE拼接、zero-native均值、严格literal边界、超时与未证实状态、完整population quantile反例，以及保存重构与source检查。

输出至少：`sample_registry.json`、`solver_contract.json`、每次`status/primal/controls/raw/reconstructed_paths/metrics`、`ALL_ATTEMPTS.jsonl`、`FULL_FIT_AUDIT.json`、`RESOURCE.json`、`SOURCE_MANIFEST.json`、`RESULT_ANALYSIS.md/json`。

F0没有训练seed、optimizer或训练RNG恢复，不能伪称完成三seed模型验证。只允许在完整源/输出SHA兼容时跳过已审核的分析记录。
