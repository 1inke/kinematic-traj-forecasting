# 按情境调整范围与 Energy Score

P1、P2 已完成。P1 使用五个 B0@64 seed。P2 比较 40 个模型及参照实例，包含四种 Energy Score（ES）。结果限于 C0 的 21,719 个 fit 窗口和 4,753 个 dev 窗口。

| 分析 | 主要结果 |
| --- | --- |
| P1 观测分组 | 五个 seed 都在 fit 选择将近静止候选缩到 0.9 倍，其他组保持原宽度。总体 minFDE 改善 0.32%，近静止改善 4.06%，转向难组无变化。排除 zara02 后，近静止仍改善 2.63%。 |
| P1 上限 | 使用未来动作时总体 minFDE 改善 3.41%。直接按 dev 真值逐窗选择时改善 50.42%。后两级都使用未来信息。 |
| P2 主分析 | 轨迹加权 ES 在五个 seed 均选择近静止 0.6 倍，其他组仍为 1.0。近静止散布中位数从 1.97 m 降到 1.18 m，该组 ES 下降 16.52%。总体 minFDE 改善 0.57%，排除 zara02 后恶化 0.07%。 |
| P2 排序 | 模型族 ES 与 minFDE 的总体 Spearman 为 0.921，转向难组为 0.525。剔除 zara02 后，E3_set 和 E3_set_nobias 的总体 ES 仍比 B0 低超过 2 倍 B0 的 seed 标准差。 |

E2、E3、E4 达到预设门槛。E4 对应轨迹加权版本。按等权 ES 选择倍数时，总体 minFDE 恶化 3.47%。主分析的近静止组总体 ES 降低，未来起步组 ES 上升 8.83%，终点 HPD90 覆盖率下降 6.11 个百分点。现有结果用于判断后续训练需要保护的指标。

## 结果和复核

- [实验报告](snapshot/REPORT.md) · [判定结果](snapshot/RESULT.json) · [执行口径](snapshot/PROTOCOL.md) · [原设计](snapshot/DESIGN.md)
- [P1 的缩放曲线](snapshot/P1_alpha_curves.json) · [三级上限和各向异性结果](snapshot/P1_levels.json)
- [P2 模型排序和分解](snapshot/P2_ranking.json) · [按四种 ES 选择的倍数](snapshot/P2_es_alpha.json)
- [状态](STATUS.json) · [复制校验](COPY_PROVENANCE.json) · [本机资产索引](OMITTED_ARTIFACTS.json)

在 Python 3 和 NumPy 环境运行以下命令，可重算四种 ES 的逐窗分解、分组汇总、模型族秩相关和七项判定。P1 的各项变化及配对轨迹区间保留在汇总中。原 C0 坐标、封存的完整预测和模型权重留在本机。P3 文献工作按设计另行处理。

```bash
python -I -B tools/reanalyze.py
```
