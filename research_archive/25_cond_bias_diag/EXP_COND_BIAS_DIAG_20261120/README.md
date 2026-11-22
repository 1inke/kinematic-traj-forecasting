# EXP-COND-BIAS-DIAG

2026-11-20 完成。实验复用五个 B0@64 种子和两个 E1B 留出种子的预测，分析 C0 dev 上的条件偏差。观测方向有效的窗口为 3,550 个。几何干预对所有候选施加共同的逐步平移，概率干预在候选坐标固定时调整权重。

**只有 P1 达标。** 第 12 步的概率加权中心沿观测方向的偏差，低速组比高速组高 0.368–0.419 m。五个 B0 种子和两个 E1B 留出种子的配对区间均排除零。按观测特征估计的线性平移与概率倾斜都未通过原定收益和保护判据。B0 继续作为基线。

- [实验设计](snapshot/DESIGN.md) · [执行口径](snapshot/PROTOCOL.md) · [执行状态](STATUS.json)
- [结果与五项判据](snapshot/RESULT.json) · [预检](snapshot/PREFLIGHT.json) · [完成凭据](snapshot/DONE.json)
- [几何修正表](snapshot/geom_correction.json.gz) · [概率修正表](snapshot/prob_correction.json.gz) · [描述性分组](snapshot/descriptive_outcome_groups.json)

核心代码与七组逐种子偏差表保存在 `snapshot/`。两份较大的修正表以 gzip 无损保存，解压后的 SHA-256 与原完成凭据一致。干预代码引用 [Phase 24 的数据与评分代码](../../24_adapt_width_centered/EXP_ADAPT_WIDTH_CENTERED_20261120/snapshot/) 及早期基线输入，原始预测和模型权重由本机实验目录保管。

## 公开复核

在本目录运行 `python -I -B tools/reanalyze.py`。该工具仅需 Python 3.10 或更高版本，从已保存的分组统计和配对区间重复执行原判据。原始坐标、预测和轨迹重抽样属于原实验的输入层，复核从已保存统计开始。
