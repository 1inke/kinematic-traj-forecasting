# 本机解码器后续实验

本阶段归档 2026 年 11 月 24 日完成的六项实验。各项实验沿用自己的数据划分、对照和评估口径。

| 实验 | 评估范围 | 主要结果 |
| --- | --- | --- |
| [LAYER-POOL](layer_pool/results/REPORT.md) | QRANK 五折，34 组冻结重放 | 40 条跨层候选扩大了覆盖。固定 20 条的 16+4 覆盖方案未达到预设门槛，判定为 `NO_FIXED_K_SIGNAL` |
| [ES-Calib-1L](es_calib/RESULT_SUMMARY.json) | UNIV，seed 42 | minFDE 低于 B0，Top-1 FDE 上升。收益缺少同结构的无 ES 对照 |
| [Decoupled-Calib-1L](decoupled_calib/RESULT_SUMMARY.json) | UNIV，seed 42 | 保存了几何与评分分离的对照、温度统计和各项概率指标 |
| [MCUS 单层探针](mcus/RESULT_SUMMARY.json) | ETH、UNIV，seed 42 | 相对 B0，UNIV minFDE 降低 7.19%，ETH 上升 7.22% |
| [MCUS-V2](mcus_v2/RESULT_SUMMARY.json) | ETH、UNIV，seed 42 | ETH minFDE 从 V1 的 0.6405 m 降至 0.5985 m，接近 B0 的 0.5974 m。V2 同时修改门控和校准 |
| [MCUS 配对消融](mcus_ablation/SUMMARY.json) | ETH、UNIV，各 2 个种子，3 臂共 12 组 | 在有重叠偏置时，可学习门相对固定门使 UNIV minFDE 降低 1.51%，使 ETH 上升 5.52% |

MCUS 的单种子结果用于探索。配对消融按相同训练预算比较，并从训练折中保留同一组 HOTEL 录制作开发评估。固定门与可学习门的可用残差幅度上限不同，结果反映这两种完整设置的差异。

## 代码与复核

各目录保留原研究脚本、结果回执和无坐标逐窗指标。单层探针的逐窗指标由本机保存预测和对齐的测试标签计算，来源散列见 [PROBE_METRIC_SOURCES.json](PROBE_METRIC_SOURCES.json)。原始轨迹、完整预测及模型权重由本机研究目录保管。

训练与冻结重放入口按原工作区布局保存，需要 [Phase 26 的基线源码](../26_advisor_3dir/EXP_ADVISOR_3DIR_20261121/code/sealed/)、[Phase 27 的 QRANK 源码](../27_qrank_mldec/EXP_QRANK_MLDEC_L4_20261122/) 和原实验的本机数据。归档复核复用 [Phase 29](../29_refine_pred/) 的轨迹簇和机动分组。在本目录用安装了 NumPy 的 Python 运行 `python -I -B tools/reanalyze.py`，可重算 LAYER-POOL 判定、MCUS 配对汇总与区间，以及单层探针的主要误差和评分指标。
