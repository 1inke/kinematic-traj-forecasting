# EXP-ADAPT-WIDTH-CENTERED

2026-11-20 完成。五个 B0@64 seed 和两个 E1B 留出 seed 都完成了 C0 dev 评估。粗分组 minFDE 总体改善 1.16%，转向难组退步 2.69%。C1H 和 C5 达标，C1、C2、C3、C4、C6 未达标。

- [结果解读](RESULT_INTERPRETATION_ZH.md) · [执行状态](STATUS.json) · [预设判据结果](snapshot/RESULT.json)
- [实验设计](snapshot/DESIGN.md) · [执行口径](snapshot/PROTOCOL.md) · [原始报告](snapshot/REPORT.md)
- [预检结果](snapshot/PREFLIGHT.json) · [倍率选择](snapshot/selections.json) · [完整配对和逐录制留一结果](snapshot/dev_effects.json.gz) · [均值偏差、速度与平坦度](snapshot/descriptors.json)
- [封存清单](snapshot/DONE.json) · [文件来源](COPY_PROVENANCE.json) · [本机输入索引](OMITTED_ARTIFACTS.json)

`snapshot/` 保留代码和原始输出。七个 `curves_*.json` 保留选倍率时的分组曲线和 dev 曲线及其配对区间。配对结果以 gzip 无损保存，解压后 SHA-256 与原封存的 `dev_effects.json` 完全一致。原始报告中同名的链接对应解压后的文件。

## 公开复核

从本目录执行以下命令。工具只需要 Python 标准库，根据已保存的分组曲线重选倍率，根据配对统计重算协议判据，并核对 ES 分解和 minFDE 相对效应。

```sh
python -I -B tools/reanalyze.py
```

公开数据保留完整分组统计和区间，缺少重抽样所需的逐窗统计及坐标，复核范围限于上述聚合层。原始数据、模型权重和完整轨迹预测留在本机，索引提供散列。CPU 执行端还依赖 [Phase 23 的数据和评分代码](../../23_adapt_es_precheck/EXP_ADAPT_ES_PRECHECK_20261120/snapshot/) 及其早期依赖。两次留出推理依赖 [Phase 21 的模型读取代码](../../21_width_cause_diag/EXP_WIDTH_CAUSE_DIAG_20261120/snapshot/) 和本机权重。

