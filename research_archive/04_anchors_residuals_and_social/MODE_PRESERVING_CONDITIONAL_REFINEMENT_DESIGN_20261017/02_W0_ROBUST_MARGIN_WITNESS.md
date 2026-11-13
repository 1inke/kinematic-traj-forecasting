# 02｜W0：固定样本的安全质量余量见证

状态：DESIGN_ONLY；一次有限 SOCP 诊断，不训练，不产生 teacher。优先级是解释 F0 边界现象，不占用主实验的科学入口。

## 1. 精确问题与相对 F0 的改变

问题：在相同 native K20、相同 U0 控制域及原严格质量/物理标准下，能否找到**经实际生产精度重构后仍具有正质量余量**的完整候选？

F0 的 min sum acceleration 改为 max quality margin；同时将旧“求解后再统一缩小”改为“求解中明确控制内部余量，随后直接走真实生产映射”。这两项组成**数值稳健提取方案**，不是单因素训练比较；不能把结果差异纯归因于求解目标。最终质量、位置和物理门没有改变，旧 F0 的0通过不改写。

固定复用 F0 已锁的48窗口：32个native失败、16个native成功，逐ID/track/GT/native/candidate顺序核验。前32每个枚举原20 slot，共640题，不能只选4个近成功题、14个optimal窗口或换hash选例。16成功只做identity对照。

## 2. 求解方程

每题目标slot j，联合变量 u[k,t,xy]，k=1..20，t=1..12，及γ。原积分矩阵 K[t,q]=.20/144×[2(t−q)+1]（q≤t，否则0）。

```text
maximize γ
subject to
  0 ≤ γ ≤ 1
  norm2(u[k,t]) ≤ 1 − 2^(-20)                      # 数值内部域，不是评价epsilon
  p = native + K @ u
  ADE(p[j], GT) ≤ .20 × (1 − γ)
  FDE(p[j], GT) ≤ .40 × (1 − γ)
  mean_ALL20_acceleration(p) ≤ 1.01 × mean_ALL20_acceleration(native)
  mean_ALL20_jerk(p) ≤ 1.01 × mean_ALL20_jerk(native)
```

沿用 F0 相同 dt=.4、差分边界和 ALL20 分母。速度/加速度/jerk 的 p95 不伪装成凸约束；实际输出必须再检查全部五项。这里只覆盖比最终 pooled 门更强的 per-window mean 子域，不能证明整个科学安全域为空。

默认沿用已验证隔离环境 CVXPY1.7.3/Clarabel0.11.1、max_iter200、tol1e-9；环境不可用时先记录并准备同版本，不盲换求解器/容差扫参。无每题2秒截断，无30分钟否决。真实资源不足保留 partial，不能把资源停止标作 infeasible。

## 3. 重构与严格验收

1. 保存原始primal、γ、状态、全部连续约束残差。optimal/inaccurate 都只是求解状态，不是科学成功。
2. 非有限解拒绝；任一 u 分量绝对值≥1 无法有限 atanh，标记不可重构。**不事后clip，不再做旧固定shrink。**
3. 对可重构 u，raw=atanh(u) 转float32，经过原 U0 的 tanh→单位盘归一化→float64积分/native加法。保存原始u、raw、actual u、actual p和各差值。
4. 原primal对内部半径或单位盘有小越界时，完整记录，不能声称原凸解严格可行；只要上述实际映射有定义，仍可作为一个候选提案进行实际输出审核。是否接受只由真实映射输出的原科学门决定，不由原始norm四舍五入决定。
5. 同一目标slot实际ADE≤.2且FDE≤.4、所有实际位置界合格、该窗口ALL20五motion literal≤1.01native，才计 `STRICT_ACTUAL_WITNESS`。
6. actual_margin=min(1−ADE/.2,1−FDE/.4)，连续值完整报告。actual_margin≥1e-4另标 `ROBUST_MARGIN_WITNESS`；这是额外稳健标签（至少ADE20μm、FDE40μm余量），**不是原成功门的放宽，也不是新增轨迹精度要求替代旧门**。未达该标签但满足原门者仍如实计严格见证。

必须分别报告“primal内部约束通过”“实际映射严格成功”“有稳健余量”，不能混为同一 PASS。CPU独立重构与GPU实际生产映射都检查；边界正负差异列失败，不加入人工评价容差。

## 4. 固定输出与统计

逐640题保存 JSON+NPZ：ID、slot、native/GT、solver状态/耗时、γ、控制、actual完整K20、每项残差和拒绝原因。报告640题状态计数与32窗口计数；640不是640独立样本。

同一失败窗口若多个题通过，仅为描述性完整FIT oracle portfolio，按实际margin最大、原slot最小打破并列选择一个完整K20方案；所有未通过窗口和16成功窗口保持native。这个GT依赖选择只用于存在性展示，**不作为部署策略、训练标签、P1责任或超参数来源**。

在完整FIT18909上另审核：同路径coverage/rescue/lost、全K聚簇、四组五pooled motion。per-window quantile门不能保证pooled quantile，因此必须区分 `WINDOW_WITNESS_PRESENT` 与 `PORTFOLIO_GLOBAL_SAFE`；后者失败不能抹掉前者，也不能宣称可部署。

若没有见证：结论限于“当前有限样本、映射、求解提取仍无合格见证”。有见证：仅证明这些已知GT样本的可达性，不证明能从past预测或外推。

## 5. 与主线的关系及停止规则

- W0不训练G1，不用I1再次测试作为门，不自动延长样本、改solver或调内部半径。
- P1不依赖W0阳性，因为P1直接测真实生成器的学习/保留效果；它不是旧G1历史读出迁移实验。
- W0只能作为背景解释。如果W0正而P1负，支持“存在机会但当前学习方式没得到”，不能定位唯一表示根因；W0负而P1正则按实际P1输出复核，不能因oracle没找着而拒绝真实正例。
- 一次640题及保存审核结束即报告，不自动 W1/W2 数值扫参。
