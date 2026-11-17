# 冻结缓存上的候选交互探索：完成，NO_SET_GAIN

**Execution COMPLETE / scientific verdict `NO_SET_GAIN`。** 六组×三个修正模块 seed，共18次训练、360轮、30,600次更新；固定最终轮评价。正式程序耗时954.69秒，独立核验通过。B0保持冻结，baseline不升级，不进入端到端B阶段。

[结果解读（中文）](RESULT_INTERPRETATION_ZH.md) · [完整报告](REPORT.md) · [精确结果](snapshot/RESULT.json) · [状态](STATUS.json) · [原设计](snapshot/design_original/EXP_SET_INTERACT_PROBE_20261117.md) · [实现与用户确认事项](snapshot/IMPLEMENTATION_NOTES.md)

## 核心结果

A1只看自身，A2全连接无偏置，A3固定终点近邻稀疏图加关系偏置；A4/A6/A5分别在这三种结构上增加再分配损失。所有结构使用严格逐候选特征，A1/A4不含集合均值。六组注册参数均为7019，但实际使用并获得梯度的参数不同，详见报告。

| 固定比较 | minFDE相对改善 | 判定 |
|---|---:|---|
| Q1：A1 对冻结B0 | +1.2895% | 代理有效性通过 |
| Q2：A3 对A1 | −0.0357% | 未通过 |
| Q2：A2 对A1 | +0.0408% | 未通过 |
| Q3：A5 对A4 | −0.1966% | 未通过 |
| Q4：A5 对A6 | −0.0960% | 未通过 |
| Q5：A4 对A1 | −0.3057% | 未通过 |

额外有效收益门槛为相对minFDE至少0.5%，同时相对minADE改善至少−0.2%。所有组均通过平均保护门，失败来自没有有效额外收益。`NO_EFFECTIVE_ARM`不是保护失败。A5 Top1 FDE恶化1.9276%，接近2%上限。A2数值最好不代表通过门槛；这次也不是所有修正相对B0都无效。

B0平均有效候选19.746/20，但未覆盖率26.846%；0.2m终点近距重复很少。该诊断限制了“挪走重复候选就能补覆盖”的解释，但不能证明失败原因或排除其他冗余。新损失下赢家FDE门与minFDE未覆盖率不同，分别报告。

## 归档与复算

- `snapshot/code/`和`tests/`（位于snapshot内）：冻结科研实现、原8项测试、配置、缓存校验、GPU精确恢复与独立核验记录。`standalone/`保留Windows独立启动入口、日志和服务父链。
- `snapshot/A0/`和`runs/`（位于snapshot内）：19份不含坐标的`window_metrics.npz`；18次完整训练历史、各20个断点的散列回执、逐seed结果、原科学输出和完成凭据。
- `dependencies/`与`COPY_PROVENANCE.json`：实际封存的文本依赖和逐字节副本映射。`OMITTED_ARTIFACTS.json`列出保留本机的输入、缓存、完整预测、模型和断点二进制；原始坐标、权重、凭据和.memory不上传。
- `tools/reanalyze.py`：可移植NumPy-only工具，从19份逐窗派生指标分别重建各分组的均值/p90/方向有效比例/赢家频率，以及Q1–Q5、保护门、机制判定和事后描述性差分；检查同seed各臂数据顺序及公开文件散列。

```bash
python -m pip install -r requirements-analysis.txt
python -I -B tools/reanalyze.py
```

公开复算不需要GPU、PyTorch、原始数据、模型权重或网络，不会启动训练。它不能从省略的坐标重算轨迹误差/聚类/标签、重放模型或验证省略断点二进制；这些已在本机独立核验，其原始回执在snapshot中保留。公开复算与本机几何核验的能力不同。

原训练与核验脚本保持原字节和历史路径，仅是可审核的源快照；完整重训还需配置匹配的旧NEXT_PHASE代码/数据/底座、环境和新输出目录。直接运行snapshot下的START不会得到可移植重训，不能把公开复算入口当作训练发行版。依赖文件重命名副本通过COPY_PROVENANCE映射回原路径。

## 实验范围与状态说明

只用一个B0@64 seed42底座和C0 fit21719/dev4753，三个seed是修正模块初始化，不是三个独立底座。dev反复用于开发，无新ETH或独立测试评价；无增强、有界局部修正、20轮固定预算。结论仅用于短期探索方向选择，不能证明所有端到端集合交互都无效。

原`IMPLEMENTATION_NOTES`、解读和完成凭据中的“尚未对外发布”是当时记录；用户随后明确要求同步GitHub，当前发布状态以本目录STATUS.json为准。保护失败按用户确认单独标记，机制原分类不替换。没有新增训练、延长预算、改变阈值、启动B阶段或恢复旧五折。

前次研究：[机动模板oracle](../../16_kine_anchor_oracle/EXP_KINE_ANCHOR_ORACLE_20261117/) · [减速诊断](../../15_decel_diag/EXP_DECEL_DIAG_20261117/)
