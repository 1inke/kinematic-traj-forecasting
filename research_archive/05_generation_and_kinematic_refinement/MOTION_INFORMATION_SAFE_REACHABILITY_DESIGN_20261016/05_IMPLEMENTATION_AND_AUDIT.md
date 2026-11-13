# 05｜实现合同、审计与反方检查

状态：DESIGN_ONLY；下面是未来实现接口，不是已编写或验证的实验代码。

## 1. 建议模块边界

```text
execution_<date>/
  protocol.json / runtime_state.json / source_manifest.json
  data_contract.py       # 唯一ID/缓存/单位/split/raw history核验
  features.py            # P2、native24、H/S26、共同FIT统计
  head.py                # 相同146→128→64→24；不读GT
  output_maps.py         # I1直接回归 / G1原U0积分；显式区分
  objectives.py          # I1 mean20 MSE / G1原L0；不能默认混用
  f0_solver.py           # 只接收明确标为oracle的FIT诊断材料
  train.py               # 固定25epoch、fresh paired init、完整恢复
  evaluate_saved.py      # 仅保存输出质量/聚簇/motion/同ID统计
  audit_saved.py         # 独立算术，不调用训练/求解器
  resource_policy.json / resource_guard.py
  preflight/ / failures/ / runs/ / results/ / completion_review/
```

F0、I1、G1分别执行目录；不得在已有locked目录补改科学实现、写回旧manifest或覆盖旧模型。只读旧类时禁止执行会写旧目录的module顶层。

### forward与label隔离

```python
def make_history(past8, mode):
    v = diff(past8, time_axis) / 0.4       # 7×2
    a = diff(v, time_axis) / 0.4           # 6×2
    if mode == "static":
        return concat(repeat(v[-1], 7), zeros_like(a))
    if mode == "history":
        return concat(v, a)
    raise ValueError(mode)

def forward(p2, native20, past8, shared_stats, arm, task):
    h = make_history(past8, arm)           # 没有GT/group/ID/logits
    x = concat_per_slot(p2, native20, h)
    raw = head(normalize(x, shared_stats))
    if task == "I1_DIRECT_READOUT":
        return native20.float64() + raw.float64().reshape_as(native20)
    if task == "G1_BOUNDED_GENERATION":
        return native20.float64() + exact_u0_integrated_map(raw)
    raise ValueError(task)
```

这是说明接口的伪代码，不是可直接运行实现。目标/指标选择必须由manifest里的task显式分派，不允许缺字段时落入旧L0或旧output map。

## 2. 运行合同最小字段

每个训练checkpoint与run_manifest双向绑定：

- stage、arm、seed、source manifest SHA、原数据/ID/split SHA；
- P2/raw past/native坐标单位、时间轴、dt与slot顺序；
- 输入146维、参数28632、共同statistics文件SHA与FIT ID来源；
- 完整objective字段及分母：I1 B×20×12；G1 winner定义和190pair；
- output map及float32/float64转换、初始化SHA、optimizer全部参数；
- 25epoch history、loader顺序/状态、全部Torch CPU/CUDA、Python、NumPy RNG；
- 资源估计E、1.3E、current/retained peak、系统最低余量；
- 每split输出和指标SHA、执行代码SHA、完成标记。

F0另存solver版本/设置、48样本选择过程、640尝试、时间预算与primal，不写虚假的optimizer或模型seed。

## 3. 执行前验证

1. **身份和信息边界**：24679 unique IDs、全部split、每条cache alignment；更改GT/group/ID元数据不能改变forward。原场景标签只用于报告。
2. **统计**：真实FIT-only、chunk累加检查、常数维置1、共同stats两臂相同；不能在H/S各拟一套后仍声称所有buffers相同。
3. **模型**：实际生产GPU/批次、paired参数/buffers、zero identity、finite合法梯度；I1按MSE独立验证，G1按旧U0 L0及积分函数验证。
4. **数值**：固定非零fixture、NumPy独立标量/路径/梯度，double有限差分只是隔离参数数学检查，不是float32量化证明。硬min/normalize活动面切换如实记录，不能搜eps或放宽阈值求PASS。
5. **恢复**：两臂真实513FIT+12inner子集两epoch连续/中断全部参数、buffers、optimizer、所有RNG及history EXACT；已完成source/output SHA兼容才skip；坏source/output必须实际拒绝。
6. **保存生产分支**：实际128窗口保存→读回→全质量/聚簇/motion审核；标为smoke，不混正式人口。
7. **预算**：GPU每臂4warm+32measure的FIT256与完整保存评估/审计吞吐；F0使用注册样本的固定前若干问题测耗时，不据效果换样本。预检F0解不计独立重复，生产要么SHA兼容复用并明确计数，要么完整重做但不能声称更多独立样本。
8. **资源**：从输入、激活、optimizer、输出、求解器workspace和审计缓存明列E；检查retained峰值，保留失败。预检值不替代生产生命周期峰值。

若预计>30分钟，本阶段记BUDGET_NOT_LAUNCHED，不拆成多个隐性长任务或降规格自动继续。数值/读取等常规实现错误在新版本保留来源后自行修复并完整回归；科学域、阈值和预算不能借修复改变。

## 4. 保存与完成清单

I1/G1每阶段：6个训练×25epoch=150行history，6个模型；每模型4人口(FIT/outer/inner/CAL)，共24个科学预测job。大FIT按有界chunk保存，所有chunk显式ID及SHA；native/CV参照文件另列，不重复算作新模型。

正式完成必须：worker与exit双SUCCESS、全部6run/24job、source/output/模型SHA、资源合同、生产保存算术和独立completion review均通过。150行history不等于全部科学完成。

独立CPU审核至少包括：

- I1：24job的window mean20 MSE、所有组/seed及CV、primary family2的全部重抽样重建；
- G1：24job同candidate ADE/FDE/J、rescue/lost/retained、原SciPy全K20完整聚簇、五pooled motion、积分controls/位置界、assignment、primary family8；
- F0：每次保存primal、生产精度控制重构、未解决/失败状态、固定slot选择、48窗口及完整FIT portfolio的literal门；
- 全部有限值/分母/原生零值/ID以及初末source与既有digest保持；
- 每组同时列点估计、逐seed、分母、CI口径、反例，禁止只输出最好的change结果。

保存算术复核不是所有模型/梯度再运行，也不是外部独立评审。不得删除原失败、修改旧报告、把source lock换成最新hash掩盖变更。

## 5. 统计实现合同

按window加权，不把20候选、12时间点或同模型输出当独立重复。交叉bootstrap是同时抽模型seed和target-track，在同一draw内保持臂与native/CV、同ID配对；一个track的所有窗口一起重抽。模型seed抽样与track抽样独立，track抽样在各seed间共享，不能先把三seed当三个数据集再混抽窗口。

与T2已审计算法保持相同的窗口加权→seed平均顺序。原percentile实现、RNG和quantile方法在执行协议锁定；有效draw数、空组draw、绝对/相对指标分母均保存。相对收益定义为1−mean(H)/mean(S)，不能与mean(1−H/S)混用；逐seed收益另算。

小样本、3seed、窗口相关、F16已见、outer/CAL适应性复用仍然存在。I1的family2与G1的family8只覆盖各自事前比较，不声称跨历次实验整体控制了选择偏差，也不把区间跨0称等价。

## 6. 三轮反方检查（同助手inline，非外审）

### DA1：问题定义

**反驳：这不过是再做一次历史特征小改。**

回应：I1移除hardwinner/多样性/局部输出域，用统一监督分离信息可读出性；F0独立问安全机会。但I1仍只是指定MLP与预算的试验，不能证明互信息是否为0。若无法读出就停止本系列，不连续加特征/GRU/宽度。

### DA2：方法识别

**反驳：F0的per-window均值约束与p95后筛仍可能错过很多原pooled安全解。**

成立。因此只允许正例存在性结论；超时、infeasible标签、p95失败和零rescue均不构成完整域不可行性证明。完整FIT portfolio复核防止per-window p95与global p95混淆。GT辅助正例只指导是否值得做G1，不当部署策略或教师。

**反驳：I1的MSE目标容易得到条件均值，不能代表多模态预测。**

成立。因此I1不升baseline；必须经G1 fullK20同路径覆盖、有效good与motion验证。I1阳性G1阴性也不能唯一归因loss，因为两个阶段的输出域和目标共同变化。

### DA3：综合与停止

**反驳：F0只有一个正例就触发G1过弱，而且复用outer选择方向会乐观。**

成立。触发门只为有限6run提供存在性和信息线索，不保证收益规模；G1沿用严格全门、保留复用/适应性限制。最多12run、每阶段≤30分钟、G1不触发就结束，不允许结果不好继续第四局部变体。最终结论必须包含全部负例和未执行项。

## 7. 最终报告结构

`SERIES_REPORT.md/json`须列：完成状态表、历史证据、新问题、F0可行/未确定边界、I1两主比较、G1触发与全门表、资源/失败记录、反例与局限、下一大分支及为何不继续其他分支。设计交付不创建此结果报告的假数据，也不提前填写PASS。
