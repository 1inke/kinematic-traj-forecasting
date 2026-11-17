# ETH/HOTEL 零训练诊断结果

训练建议：`ELIGIBLE_EXPLORATORY`。这是探索筛选，不能据此认定速度是唯一原因。

| 场景 | 条件 | 样本数 | minADE | minFDE | 覆盖率 | ADE 改善 | FDE 改善 |
|---|---|---:|---:|---:|---:|---:|---:|
| eth | identity | 364 | 0.446220 | 0.655116 | 16.21% | 0.00% | 0.00% |
| eth | high_motion | 364 | 0.408781 | 0.607877 | 16.21% | 8.39% | 7.21% |
| eth | high_scene | 364 | 0.408240 | 0.606445 | 16.21% | 8.51% | 7.43% |
| eth | reverse_motion | 364 | 0.776695 | 1.151718 | 16.21% | -74.06% | -75.80% |
| eth | fixed075_motion | 364 | 0.401589 | 0.602164 | 12.09% | 10.00% | 8.08% |
| eth | T1_saved_descriptive_only | 364 | 0.476119 | 0.712731 | 21.98% | -6.70% | -8.79% |
| hotel | identity | 1197 | 0.142783 | 0.223330 | 67.92% | 0.00% | 0.00% |
| hotel | high_motion | 1197 | 0.143272 | 0.224262 | 68.50% | -0.34% | -0.42% |
| hotel | high_scene | 1197 | 0.143888 | 0.225427 | 68.42% | -0.77% | -0.94% |
| hotel | reverse_motion | 1197 | 0.145208 | 0.229672 | 67.34% | -1.70% | -2.84% |
| hotel | fixed075_motion | 1197 | 0.147925 | 0.235426 | 68.34% | -3.60% | -5.42% |

## 固定条件检查

- pipeline_audit_passed: PASS
- baseline_replay_passed: PASS
- low_speed_identity_passed: PASS
- high_speed_n_at_least_30: PASS
- overall_ADE_gain_at_least_5pct: PASS
- overall_FDE_gain_at_least_2pct: PASS
- high_ADE_gain_at_least_10pct: PASS
- forward_beats_reverse_by_2pp: PASS

速度分界只由相应折的 95% 训练子集确定；机动类型使用未来标签，仅用于评价分组。
整体改善不等于每种机动都改善；小于 20 窗的分组只作描述。
场景整体缩放同时改变邻居距离，因此不能单独归因于速度。
本地视频、提取脚本和标签链的一致性不构成上游物理标定的独立验证。
ETH/HOTEL 均已被用于开发分析，本结果不作为未查看过的独立测试。
DECISION.json 只给出是否值得做一次增强训练的建议；没有自动启动训练。
