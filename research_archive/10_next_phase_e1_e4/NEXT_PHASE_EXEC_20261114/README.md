# NEXT_PHASE E1–E4：代码、可复算分析数据与最终结果

本实验于 **2026-11-15 13:32:29（Asia/Shanghai）完整结束**，执行状态 `SUCCESS`、退出码 `0`。新训练47次，其中E2/E3/E4共27个方法模型；统一预算 `E*=64`。九种方法均未通过预设的完整成功规则，保留B0，不升级baseline。

这是2026-11-14设计包的实际执行归档。原设计中 `DESIGN_ONLY` 是执行前状态；实际训练、恢复和最终成功证据见下方。T3使用用户确认的“逐步位移旋转、再累加”，保留每步速度大小。

## 先读结果

- [完整中文解读](RESULTS_REVIEW_ZH.md)：收益、代价、区间、成本与下一步建议。
- [原始最终结果报告](snapshot/RESULT_ANALYSIS.md) / [完整数值与区间](snapshot/RESULT_ANALYSIS.json)。
- [执行完成记录](snapshot/standalone_recovery_20261114/RESULT.json) / [真实退出记录](snapshot/standalone_recovery_20261114/WORKER_EXIT.json)。
- [逐方法对比](snapshot/analysis/)、[完整原始观测管线耗时](snapshot/cost/)。
- [E1预算选择](snapshot/E1/BUDGET_DECISION.json)、[E1噪声与重复性](snapshot/E1/E1A_NULL.json)、[CALIB训练曲线](snapshot/E1/figures/)。

| 方法 | 整体覆盖差（pp） | change覆盖差（pp） | 完整成功规则 |
|---|---:|---:|---|
| E2_natural | +0.2595 | +1.2153 | 未通过，运动学代价 |
| E2_clamped | +0.2104 | +0.9722 | 未通过，运动学代价 |
| E2_clamped_launch | −0.0701 | +0.7292 | 未通过，运动学代价 |
| E3_self | +2.7912 | +3.0556 | 未通过，stable运动学伤害 |
| E3_set | +2.5738 | +3.9931 | 未通过，交互证据不足及运动学伤害 |
| E3_set_nobias | +2.9595 | +4.4444 | 未通过，运动学代价 |
| T1：退火软WTA | +2.6720 | +4.9306 | 未通过，运动学代价 |
| T2：change加权 | −2.0408 | +1.2153 | 未通过，多群体覆盖退化 |
| T3：转向增强 | −0.0982 | +0.3472 | 未通过，收益未达筛选规则 |

所有差值相对同64轮B0，按seed42/43/44配对后取均值。B0整体覆盖56.6099%、change覆盖35.6944%。本轮是重复使用开发集的探索性研究，未运行官方测试集、五折或因素组合。程序成功与科学成功是不同的判定。

## 文件结构与证据状态

| 路径 | 内容 |
|---|---|
| `snapshot/` | 实际实验源码、原设计、合同、逐seed/epoch统计、评估与机制诊断、最终报告；复制的文件保持原字节 |
| `snapshot/standalone_recovery_20261114/` | 独立Windows后台执行与完整断点恢复代码、预检、服务父链、完成与退出记录 |
| `dependencies/` | 原source seal绑定的项目/ASTRA源码与配置，以及原聚簇评价实现 |
| `analysis_inputs/` | 从保存输出导出的紧凑逐窗口统计，含ID、分组、覆盖、最小ADE/FDE、同候选good标记、Q/J分段误差及Top1结果；dev另含有效好簇数 |
| `tools/reanalyze.py` | 无GPU、无模型、无原始数据集的统计复核入口 |
| `tools/export_analysis_inputs.py` | 从原始本地实验重新构建本发布包的工具，仅读保存输出 |
| `COPY_PROVENANCE.json` | 原文件到本归档的路径映射、SHA256及原封存校验状态 |
| `ARTIFACT_MANIFEST.json` | 本发布包全部负载文件的相对路径、大小、SHA256；校验清单本身不自包含 |
| `PUBLICATION_VALIDATION.json` | 发布前复算验证结果 |
| `PUBLICATION_ENVIRONMENT.json` | 导出时环境版本；原实验环境证据另见封存预检 |

**当前执行状态以recovery目录为准。** `snapshot/CURRENT.json`、`FAILURE.json`和根级旧退出记录是初版worker失败现场，按原字节保留；不要用它们覆盖最终recovery SUCCESS。最初问题是完成记账时`seal`名称遮蔽；实际训练使用恢复adapter，未改变原`Trainer.step`。Windows脚本保留历史机器路径，属于原执行证据。

## 在新机器复算统计

从本仓库根目录运行，使用Python3.10或3.11及下方锁定的NumPy版本。只读取本包文件，不要求Torch、GPU、原始ETH/UCY数据或权重。

```bash
python -m pip install -r research_archive/10_next_phase_e1_e4/NEXT_PHASE_EXEC_20261114/requirements-analysis.txt
python research_archive/10_next_phase_e1_e4/NEXT_PHASE_EXEC_20261114/tools/reanalyze.py --verify
```

`--verify`执行发布清单SHA校验、紧凑输入与全部population聚合值校验、E1四档预算CALIB结果与两套五seed噪声标尺复核、11项对比的fit/dev逐组逐seed覆盖/ADE/FDE/挽救丢失复核，以及全部11×4组×4指标的20000次整track+配对seed bootstrap，核对point、普通95%区间和family176调整区间，绝对误差容限`2e-12`。每项bootstrap另有8次直接展开窗口的独立算式检查。

需要另存复算结果时，指定一个尚不存在的输出文件：

```bash
python research_archive/10_next_phase_e1_e4/NEXT_PHASE_EXEC_20261114/tools/reanalyze.py --verify --output /path/to/new_validation.json
```

较快地只核对点估计与挽救/丢失，可省略`--verify`。这样不会重做bootstrap或完整包清单校验。

## 中间数据的保留范围

紧凑数据由保存的`window_metrics.npz`与`predictions.npz`导出，不重新前向、不重训、不降低浮点精度。ID及分组保持原顺序；dev的complete-link聚簇计数用原实现重算，并与封存的四组聚合结果逐项一致检查。`analysis_inputs/INDEX.json`记录每份输入对应的原窗口指标、原预测文件SHA。

本仓库保存全部实际评估的E1/E2/E3/E4 population，以及E1使用的历史OLD42参考。FIT/CALIB的有效簇和运动学聚合值保留于原`metrics.json`，而逐窗口有效簇数只在dev紧凑输入中导出，因为bootstrap仅作用于dev。

本包没有上传训练权重/优化器/逐轮恢复断点（原目录约59GiB），原始标签与处理后的坐标输入，完整候选轨迹/第一遍轨迹数组（原数组共约4.2GB），运行缓存或本地记忆。权重及原数组仍在实验原机器，哈希与来源信息保留。

因此，本包可独立复核主要统计结果；完整候选几何、运动学重新计算、重新计时和重训还需要原始数组、原始数据或模型权重。已有运动学双实现审核、尾部指标、逐群体summary及几何诊断原件均保留，复算脚本不会声称从紧凑输入重做这些审核。

## 训练源码与复现边界

`snapshot/`和`dependencies/`用于检查实际科学实现及恢复机制。原源码和历史封存文件中的绝对路径保持不变，原SHA不改写；`COPY_PROVENANCE.json`提供本包对应位置。它们不是自动适配新机器的一键训练发行版。重训需按该映射配置原项目和ASTRA依赖路径、准备具有原合同哈希的数据与必要的历史参考，再使用recovery入口；不要直接使用存在旧完成记账故障的初版worker。

`tools/export_analysis_inputs.py --source <原NEXT_PHASE_EXEC目录>`在原环境中可重建此包的原件与紧凑输入；导出还需要SciPy（原环境1.12.0）进行原complete-link聚簇。分析依赖文件固定原NumPy1.25.2；推理/训练环境信息见`PUBLICATION_ENVIRONMENT.json`与原预检。任何新环境的训练都应作为新运行封存，不沿用旧成功收据。

## 解释限制

每方法仅3个训练seed；相邻窗口共享目标轨迹，不能将全部窗口当独立样本。bootstrap按整条目标轨迹再按配对seed重采样，20000次、固定seed20261114；176项调整区间用于描述，原成功规则不会由区间结果替换。全部方法的change调整区间均包含零；少数整体指标有正向信号，也不解除运动学约束。没有事后删seed、放宽阈值或升级baseline。
