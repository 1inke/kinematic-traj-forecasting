# E0：cal_v11 干净人口上的 oracle 互补机会重估 — 结果分析

日期：2026-10-20｜实验 ID：E0_CALV11_ORACLE_OPPORTUNITY_20261020｜性质：零训练诊断，K0-C 为 ORACLE_DIAGNOSTIC_ONLY（使用 GT，不可部署）
运行：54.8 分钟，CPU；全部输入前后 SHA-256 不变（`inputs_unchanged=True`）。

## 判定：Gate ① **PASS**——干净条件下互补机会存在且未缩水

### 主结果（change 组完整成功数，ADE≤0.2 且 FDE≤0.4 同一条路径）

| seed | 人口 | native K20 | K0-C oracle | Δ | 相对幅度 | W 联合门 |
|---|---|---:|---:|---:|---:|:---:|
| 42 | **clean（cal_v11）** | 146/443 | 182 | +36 | **+8.1pp** | True |
| 43 | **clean** | 146/443 | 180 | +34 | **+7.7pp** | True |
| 44 | **clean** | 146/443 | 186 | +40 | **+9.0pp** | True |
| 42 | dropped（暴露对照） | 97/257 | 117 | +20 | +7.8pp | True |
| 43 | dropped | 97/257 | 117 | +20 | +7.8pp | True |
| 44 | dropped | 97/257 | 116 | +19 | +7.4pp | True |

### Track 簇 bootstrap（50 个 change track，PCG64(20261020)，20,000 draws）

Δcoverage（K0-C − native，change 组）：

| 口径 | 均值 | 95% CI |
|---|---:|---|
| 三 seed 平均 | **+8.27pp** | **[+6.47, +10.15]** |
| seed42 | +8.10pp | [+5.29, +11.07] |
| seed43 | +7.66pp | [+4.53, +10.92] |
| seed44 | +9.04pp | [+5.81, +12.83] |

CI 不跨 0，且下界（+6.5pp）高于 D-0 在 outer 上估计的 MDE（4.6–6.8pp）的可比量级——本人口的效应在当前功效之上可判别。旧 CAL 评测的 BLOCKED_IDENTITY 限制在此被 v1.1 的真实 track 标签解除。

### 质量与结构

- 变化组 ADE 同时改善（0.2549→0.2336–0.2378，约 −6~8%），FDE 改善（0.4488→0.3854–0.3946，约 −12~14%）。
- **lost=0（全部 6 个 cell）**：oracle 的单调约束在干净人口上无一原成功丢失。
- rescue 36/34/40（clean），swaps 平均仅 0.33–0.50 次/窗：外科式稀疏替换，适合学进单模型。

## 三条推断

1. **机会不是上游数据接触的产物**。clean 与 dropped 的相对增益几乎相同（+8.3pp vs +7.7pp）；若机会来自训练暴露，dropped 应显著高于 clean。
2. **机会在更难的人口上不打折**。clean 的 native 基数更低（33.0% vs 37.7%），绝对增量反而更大（+36.7 vs +19.7）。
3. **outer 与 CAL 同构**。outer（未除污）+6.3~9.1pp、CAL 全量 +8.0pp、cal_v11 干净 +8.3pp——三个口径一致，机会幅度稳定在 +8pp 附近。

## 边界与未证明

- K0-C 是 GT 使用的诊断上界：E0 只证明"机会存在"，不证明任何无 GT 方法能拿到它（这正是 E1 的问题）。
- cal_v11 只解决 CAL 方向的污染；outer 的 C6 暴露仍存在，本实验未使用 outer 做判定。
- KIN 教师缓存来自被审计指认为"配方用当前 FIT"的管线；cal_v11 与 FIT 已 track+时间互斥，故对本实验结论无影响，但后续论文级主张仍需干净重训的教师。
- 单一场景族（students003 为主的 CAL），未证明跨场景。

## 证据入口

[RESULT.json](RESULT.json)（含逐 seed 质量、rescue/lost、swap 统计、SHA 记录）、[run_e0.py](run_e0.py)、k0c 逐窗索引 `k0c_indices_*.npy`、配对差值 `delta_change_*.npy`。划分依据：`qoder_work/v1_1/SPLIT_V1_1.json`（FROZEN_BY_AUTHOR，CAL 范围）。
