# 候选集合校准与二次修正机制

2026-11-19 的三项 C0 诊断已完成。三项共用 4,753 个 dev 窗口。需要拟合量化参照或训练轻量修正器时使用 21,719 个 fit 窗口。

| 实验 | 完成情况 | 主要发现 |
| --- | --- | --- |
| EXP-SET-CALIB-DIAG | 5 组 B0、3 组 E3_self 保存预测 | B0 等权终点散布的平均 β 为 1.000，总体 90% HPD 覆盖 92.3%。近静止覆盖 97.8%，minFDE 比 fit 分箱量化参照高 8.4%。预设结论为 `RANGE_OK_PLACEMENT_BAD`。 |
| EXP-REFINE-MECH-DIAG | 3 组 E3_self 全量回放 | B0、第一遍和第二遍的平均 minFDE 为 0.3145、0.4206 和 0.3074 m。匀速直行改善 4.6%–8.8%，静止和纯转向均受损。 |
| EXP-REFINE-ABL | 5 臂 × 3 seed × 20 轮 | 只保留 mode 或只保留自身几何时，minFDE 比 A1 高 0.98% 和 1.21%。中心信息额外改善 0.02%。低速限幅后静止 minFDE 改善 0.12%，低于 5% 门槛。 |

等权 90% HPD 覆盖对带宽敏感。冻结消融按确认的输入维数改变首层参数量，输入内容与参数量的各自贡献仍待区分。二次修正确立了联合训练后两遍之间的依赖。各预算独立训练的 EXP-REFINE-BUDGET 状态为 `NOT_STARTED`。

## 实验记录

- [SET-CALIB-DIAG 报告](snapshot/SET_CALIB_DIAG/REPORT.md)、[数值](snapshot/SET_CALIB_DIAG/RESULT.json)、[执行口径](snapshot/SET_CALIB_DIAG/PROTOCOL.md)、[图表](snapshot/SET_CALIB_DIAG/figures/)。
- [REFINE-MECH-DIAG 报告](snapshot/REFINE/MECH_DIAG/REPORT.md)、[数值](snapshot/REFINE/MECH_DIAG/RESULT.json)、[逐窗指标](snapshot/REFINE/MECH_DIAG/windows/)。
- [REFINE-ABL 报告](snapshot/REFINE/ABL/REPORT.md)、[数值](snapshot/REFINE/ABL/RESULT.json)、[各组训练和逐窗指标](snapshot/REFINE/ABL/runs/)。
- [二次修正原设计](snapshot/REFINE/DESIGN.md)、[两项执行口径](snapshot/REFINE/PROTOCOL.md)、[归档状态](STATUS.json)。

## 复核

原始执行代码随实验记录保留。归档包含 26 份逐窗派生数组、校准图、300 轮训练凭据，以及已有输入和权重的散列。原 C0 数据、完整预测坐标、冻结缓存和可恢复训练的权重保存在本机。Phase 18 的封存评分与训练实现参见 [解码器信息探针归档](../../18_dec_info_probe/EXP_DEC_INFO_PROBE_20261117/)。

安装 [复核依赖](requirements-analysis.txt) 后，在本目录运行

```bash
python -I -B tools/reanalyze.py
```

复核工具检查归档散列，并用已保留的逐窗指标复算散布–误差斜率、覆盖、量化参照误差、二次修正的机制标记以及冻结消融的配对汇总。它也核对 300 轮顺序记录。原始锚点回放、概率 ECE、带宽拟合、K-means 和重训需要本机原始输入。相应输入与执行凭据见 [来源清单](COPY_PROVENANCE.json)、[本机资产索引](OMITTED_ARTIFACTS.json) 和 [预发布检查](PREPUBLICATION_SOURCE_CHECK.json)。
