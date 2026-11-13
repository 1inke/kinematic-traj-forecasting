# 实施与审核合同：成功保留参数更新仲裁

日期：2026-10-25；配套[科学设计](DESIGN.md)。状态：**DESIGN_ONLY**。

本文给出后续实现边界与伪代码，不包含可直接执行的runner。本次没有执行以下预检、训练或审核。

## 1. 建议目录与模块职责

```text
execution_<actual_date>/
  protocol.json                 # 将本设计精确转录、冻结；不得继承旧执行授权字段
  source_manifest.json          # 相关源、模型、输入、原done等逐项SHA
  identity_audit/               # FIT/inner ID、track、组、冻结成功标签
  protection_sets.py            # 仅FIT；由native和A0构造固定互斥组
  task_adapter.py               # 只读导入旧A必要类/函数，无旧顶层写入
  step_projection.py            # 全参数向量的两半空间投影；不读数据标签
  optimizer_step.py             # 原AdamW提议、仲裁、应用和完整状态保存
  preflight.py                  # 固定fixtures、真实GPU、resume、成本/资源
  g0_one_step.py                # 48个独立同起点反事实，不累积训练
  train.py                      # 仅G0通过后，U/N/P × 3seed固定续训
  evaluate_saved.py             # 保存完整路径及质量、运动、转移
  completion_review/            # 另一保存算术实现；不导入训练器
  failures/                     # 每次失败原代码、trace、局部输出及来源
  RESULT_ANALYSIS.md/json
```

所有目录名仅建议，不表示已经创建execution或锁定任何科学源。本设计文档不应覆盖任何原报告、模型、digest或失败记录。

## 2. 起点和数据准入

实施首先逐seed核对旧A：

1. checkpoint/resume/done/manifest/source SHA相容；epoch8、592 steps及完整原history一致。
2. 模型82,008参数、buffers、统计、参数顺序、AdamW参数组、一二阶矩、step和全部RNG可恢复；不存在默默默认初始化字段。
3. 用原保存输出明确核对FIT/inner ID、native、GT、组标签与同源缓存。不得只依数组行号猜身份。
4. 小批旧A冻结回放与原路径按原生产/审计合同核验，另验证统计buffers完全一致；有差异先定位，不从新模型输出重建旧A标签以掩盖差异。
5. 所有three-arm起点逐参数、buffer、optimizer、RNG完全相同；三seed本来不同，不要求跨seed参数相同。

原数据读取器可能带顶层建目录/写cache行为，使用只读隔离适配层。不要把整份旧脚本import后执行其main。无需实例化旧F16/decoder/scorer来证明SHA保护；不能因此宣称做了其全部GPU前向回放。

已归档的旧NPZ不是遗失数据。只在实际依赖缺失时定位档案manifest、估计恢复空间并恢复确切文件；原digest保持不变。不运行旧审核批任务探测全部缺失，也不重训重建旧输出。

## 3. 冻结保护标签与可微质量

### 标签

保存每个FIT ID对应的native_good、A0_good、四格组、原J/ADE/FDE及seed。冻结标签按独立float64保存算术确定，严格同一路径同时合格，无evaluation epsilon；并保存组ID SHA。

标签不是模型输入；`prediction_forward(features)`仅接受原210维输入。`build_constraints(pred, gt, fixed_group)`为训练监督接口，GT和组只到这里。inner标签只在结果分析生成。

### 训练R

按原训练坐标及dtype得到ADE/FDE，逐slot计算`max(ADE/.2,FDE/.4)`，再取最小slot。candidate并列选第一physical slot；组件并列通过stack max选择ADE分支。保存slot切换、组件激活及并列数量，不能忽略不可微面。

对非空组取均值；空组不添加限制。float64标签与float32训练R在边界上可能不同，单独记录，不动态重贴组标签。不能把label计算换成float32使身份“对齐”。

## 4. 实际AdamW增量的处理

核心伪代码（概念合同，不可直接作为生产实现）：

```python
theta0 = snapshot_trainable_parameters(model)
pred = model(features)                         # 一次原模型前向
task_loss = original_A_loss(pred, gt)           # 原标量，不加R
g_shared, g_rescued = constraint_gradients(pred, gt, frozen_FIT_groups)
g_task = task_gradient(task_loss)

# 精确采用原optimizer路径/顺序/clip，包括weight decay；不手写近似Adam。
assign_grad(model, g_task)
original_clip_grad_norm_(model, 5.0)
optimizer.step()                               # 推进任务梯度对应的m/v/step
theta_adam = snapshot_trainable_parameters(model)
d0 = flatten(to_double(theta_adam) - to_double(theta0))
restore_parameters_only(model, theta0)         # 不回滚optimizer状态

G = normalize_nonzero_rows([g_shared, g_rescued])
dP, certificate = project_two_halfspaces(d0, G)

if arm == 'U':
    write_parameters_exactly(theta_adam)        # U须bitwise复现原AdamW一步
elif arm == 'N':
    dN = zero_if_d0_zero_else_norm_ratio_times_d0(dP, d0)
    write_float32_parameters(theta0 + dN)
else:
    write_float32_parameters(theta0 + dP)

audit_actual_increment(theta0, model, certificate)
save_optimizer_and_all_RNG_and_step_diagnostics()
```

实现必须先把theta_adam与theta0分别转float64再相减；不得先float32相减才转换。伪代码的flatten仅表达向量化，不允许改变参数顺序。

三个autograd结果均在同一before-state计算，不累计彼此`.grad`；约束梯度不经过task clip，正比例L2行归一化用于数值条件，不改变半空间。不将g_shared/g_rescued加入task梯度。若存在随机层，三臂配对前向及诊断需显式控制RNG；诊断不得额外推进后续训练的RNG。

Adam状态始终从原task梯度更新，P/N实际路径与Adam提议不同是本算法的公开定义。参数写回不得重置或重新拟合矩状态。已发现NaN、非法源或solver错误时，该失败实验保留来源并停止，不继续使用已推进的optimizer。

## 5. 两约束投影：不安装求解器

令归一化约束矩阵G有0–2行。目标为到d0的欧式最近点，满足`Gd≤0`。

- 枚举活动集：空集、第一行、第二行、两行。
- 对活动集A解`(G_A G_A^T) λ = G_A d0`，候选`d=d0−G_A^T λ`。
- 检查全部原约束可行、活动乘子非负、互补条件；从有效候选中选平方距离最小者。
- 平行/反向/秩亏情形通过明确的SVD伪逆及KKT检查处理；相对奇异值阈值固定1e-12，不因失败调阈值。
- Gram和投影统一float64；归一化约束残差数值门固定`1e-10 × max(1, ||d0||₂)`，并保存未经阈值处理的原始残差。该门是求解精度，**不是轨迹评价epsilon**。
- 零向量数学上可行，但不能在solver失败时静默返回零假装求得最近点；找不到通过证书的最小解即STOP。
- 结果须核验投影范数不大于d0范数（在算术界内）；否则不得为了计算N而强行clip比例到1掩盖错误。

float32参数写回后，以实际`d_actual`重算Gd。令`e=d_actual−d_ideal`，单位行有`|g·e|≤||e||₂`。保存各行原始Gd、理想求解残差和实际舍入界；超界STOP。**即使在算术界内出现正Gd，也只能称数值近似满足，不能伪写实际严格非增。** 科学质量和运动指标仍按原literal阈值判定。

## 6. 上线前测试矩阵

### 6.1 纯数值与隔离fixtures

- 0/1/2约束；互相正交、平行同向、反向、近相关；零梯度；d0=0；已经可行点；投影仅落一面/同时落两面。
- 各fixture核对可行性、KKT、最近性、非负乘子、幂等投影；独立二维解析/穷举活动集实现核对，不以生产函数调用自身当独立证明。
- N与P在同一before-state的理想范数一致，N方向与d0一致；写回量差单独记算术界。
- U一步与原AdamW实际参数、optimizer状态bitwise一致。
- synthetic samecandidate disjoint-minima：minADE和minFDE来自不同slot时不得计good。
- 改inner GT/组标签不得改变FIT保护集合、任何训练batch或预测forward。

### 6.2 真实GPU与恢复

- 固定8个FIT256批：原A loss/scalar/all task gradients复现；两个保护R与NumPy算术校验；所有梯度有限，合法参与参数的连接性如实报告，不能把解析抵消当非零连接。
- 固定非零double隔离副本做两eps（1e-4、1e-5）中心有限差分，分别审计原task与保护R梯度，不穿过离散标签。固定seed20261025生成4个单位参数方向；各eps、各方向均需`abs(analytic−numeric)≤1e-6+1e-3×max(abs(analytic),abs(numeric))`，同时保存原始两值，不用极小数的百分比误差作唯一判断。winner/max活动面切换记录并按既定fixture准入STOP，不搜索eps直到PASS。不能声称验证了float32量化导数。
- G0独立同起点一步证书和实际R变化全部保存；不把一阶线性证书当有限步改善。
- U/N/P各做真实FIT513与inner12的**子集**连续/中断2epoch测试；params、全部buffers、optimizer、全部Torch CPU/CUDA/loader/NumPy/Python RNG、history、投影诊断EXACT。不是完整18,909窗两轮验证。
- analysis done SHA-skip与拒绝坏source/坏输出另测；这不是optimizer/RNG恢复测试的替代。
- 三臂实际保存128窗完整K20质量/原SciPy聚簇/五运动学读回独立重算；不能以smoke任务计入正式18输出。

### 6.3 资源与预算

先列input/cache/model/moments/多梯度/临时副本/输出/审核的内存inventory E；资源规则继承当前项目1.3余量及动态剩余、Windows retained peak。不得套旧固定6GiB限制或用预检峰替生产峰。

36实际batch/臂（4warm+32measure），加完整FIT/inner保存与quality耗时、初末SHA/IO估算。报告G0、G1、G2分别预算和整体ETA；如不符合用户当前短预算偏好，应缩减为交付G0结论并说明G1未执行，而不是静默降epoch/换seed/跳审核使其“够快”。

## 7. 必须保存的记录

每个run至少包含：

```text
checkpoint.pt / resume.pt
run_manifest.json / done.json
history.jsonl
step_diagnostics.parquet
FIT_predictions.npz / inner_predictions.npz
metrics.json / assignment_stats.json
```

manifest绑定：父checkpoint/resume/done SHA、当前全部相关源SHA、split与保护组ID SHA、特征统计、原loss完整定义、当前更新规则、参数展平顺序、dtype、solver合同、原592步和新增592步、loader切换seed、optimizer/RNG、所有输出SHA。

逐步诊断至少保存：批次ID hash、两组计数、task/R、梯度范数、raw点积与cosine、d0/dP/d_actual范数、N比例、活动集、乘子、KKT残差、cast误差界、零更新、原loss hardwinner与Jwinner及组件并列计数。G0另保存每个U/N/P实际一步后的两组R和task变化。G1不强制每步额外post-forward，但必须固定epoch末记录FIT/inner诊断且不能反向用于选模型。

不保存每step完整82,008维梯度以无界膨胀磁盘；保留固定hash选取的有限诊断批完整向量供重构，数量在上线前注册，其余保存标量与可验证来源。没有保存的全训练梯度不得声称完成独立二次回放。

## 8. G2独立保存算术与报告

- CPU审核不import训练main或调用模型，不跑optimizer。
- 完整18输出的ID、有限K20×12×2、质量同路径J/ADE/FDE、两参考的rescue/lost/retained、四格保护组全部核算。
- 全候选原SciPy complete-link，与另一距离实现核对；不能只验winner或抽128窗代替全量。
- 运动学先将坐标转float64再求差/除0.4；pooled分布完整重算，p95不可由子组均值相加。
- 五motion与所有科学阈值不新增epsilon；数值审核容差继承被比较生产字段的原合同并明确列出，不混用优化器KKT门。
- 两个主对比的同ID、同track、配对seed重抽确定性重构；native重复三份不增加独立样本。
- 对全部9个manifest、72个新增epoch行、5,328个训练step计数和相关输入/旧模型/输出SHA初末核验；不同记录频率不得伪造未保存的逐步字段。
- 新增总报告必须分别给出：执行完整性、G0进入门、G1局部机制门、完整科学安全门、成本预算门、全部未执行项。

最终报告用语示例：

> P相对U减少丢失，但未优于N，因此暂不能将收益归因于冲突方向处理。

> P在FIT的保护有效，inner无稳定覆盖改善，因此本轮没有支持其解决迁移问题。

> 局部覆盖/保留门通过，但平均加速度或总参数预算未通过，尚不可作为完成的轻量化互补生成方法。

严禁把“runner成功”“QP可行”“保护R均值下降”直接写为“理解运动”或“互补生成创新已完成”。完成有限系列后停止自动扩张，保留所有阴性、失败源和反例。
