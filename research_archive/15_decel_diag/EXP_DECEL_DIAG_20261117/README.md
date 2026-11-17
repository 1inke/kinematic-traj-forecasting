# 减速诊断：实验完成，存在候选选择问题

**Execution COMPLETE / 主判定 B / B1、B2 均未满足，保留 `B_UNRESOLVED`。**

2026-11-17 完成并归档。全程读取已有预测，零训练、零模型推理、无GPU；正式汇总2.516秒，代码准备和独立复核另计。纳入ETH测试364窗的B0、宽速度增强和T1（seed42），以及C0 dev同4753窗的B0 seed42–46，共8组；五个seed没有叠成五倍独立样本。

[结果解读（中文）](RESULT_INTERPRETATION_ZH.md) · [完整指标表](DECEL_DIAG.md) · [完整JSON](snapshot/DECEL_DIAG.json) · [机器状态](STATUS.json) · [原设计](snapshot/design_original/EXP_DECEL_DIAG_20261117.md) · [运行前定义补充](snapshot/IMPLEMENTATION_NOTES.md)

| ETH减速89窗 | 原B0 | 宽速度增强 |
|---|---:|---:|
| 存在纵向误差≤0.4m候选 | 88.8% | 78.7% |
| Top1纵向过冲>0.4m | 70.8% | 75.3% |
| 每窗减速候选数 | 12.53 | 8.08 |
| 最优候选分数名次中位数 | 8 | 10 |

宽增强中存在纵向可达候选的70窗，仍有48窗（68.6%）Top1过冲。满足事前B条件，不满足A；B1/B2细分均未通过，并非“都不明显”。增强后减速候选减少约35.5%，最优候选纵向与横向绝对误差同时增大，说明生成也有退步；不能把未达A阈值解释成生成没有问题。

C0五个seed也都满足B主类；42/43/44/46满足B2的操作性条件，45保留B_UNRESOLVED。因此选择现象不只出现在ETH，但不能证明所有场景普遍成立。ETH校准偏差使B2不成立；历史趋势分层也不支持本次B1条件，不能声称历史本质不可预测或打分完全没用历史。当前baseline不变，下一步优先探索评分与概率校准，同时检查减速覆盖与横向误差。

## 文件与复核范围

- `snapshot/`：实际设计、执行代码、7个合成测试、固定规则、结果、Windows后台入口和封存证据，逐字保留。
- `snapshot/populations/<dataset>/<model>/decel_window_metrics.npz`：8份逐窗口派生指标，包含纵横误差、速度比、候选概率、机动和历史趋势标签，无原始世界坐标或完整12步坐标预测。
- `dependencies/`：实际使用的机动标签函数、五折读取函数及历史封存清单；文件映射见 `COPY_PROVENANCE.json`。
- `ARTIFACT_MANIFEST.json`：公开归档文件SHA256；`OMITTED_ARTIFACTS.json`记录本机保留的输入及其散列。

发布前本机重新核验31个源文件和10个科学输出，逐窗口对原始保存预测、全部20候选、t6/t12独立重算；分层样本数、校准、主要误差/比例及A/B判定通过。原始完成证据与本次发布前复核分别保存。

公开工具可以从派生数组重新汇总全部机动×历史分层、主/敏感性方向规则、A/B/B1/B2判定和同窗条件选择统计，并检查所含字段的一致性。它使用冻结的聚合代码；不能替代缺少原始坐标时的独立坐标重算，也不能重新推导历史或机动标签。标签使用未来仅作评价，从未进入模型。

```bash
python -m pip install -r requirements-analysis.txt
python -B tools/reanalyze.py
python -B -m unittest discover -s snapshot/tests -v
```

仅需Python 3.10+和NumPy，不需GPU、PyTorch、模型、原始数据或网络。工具只核验，不重训或更改结论。

`snapshot`保留的绝对路径和任务XML仅作本机执行来源。原始CPU分析要重新配置数据、完整预测和输出路径；不要将历史Windows启动入口误当任意机器的一键运行工具。主方向采用末3步平均位移长度>.03m，旧总位移规则另作敏感性分析；ETH减速主组均89窗。小样本、单seedETH及已反复查看的测试数据仅支持探索，不作显著性或因果结论。未恢复旧五折、未运行COMBO或任何后续训练。

前次速度增强见 [Phase14](../../14_eth_speed/EXP_ETH_SPEED_20261116/)；旧五折仍为 [未完成](../../11_fivefold_b0_incomplete/FIVEFOLD_B0_20261115/)。
