# E1：学生蒸馏五臂筛选与三 seed 确认 — 结果分析

日期：2026-10-20｜目录：`qoder_work/E1_STUDENT_DISTILL_SCREENING_20261020/`｜性质：筛选+确认，仅消耗 inner；outer / cal_v11 未动
学生：kdec DirectPositionDecoder（SharedTrunk MLP + K20 query，57,432 参数，单次前向）；8 epochs / batch 256 / AdamW 1e-3 / FIT 训练；教师为冻结 native/KIN 的 FIT 缓存 + FIT GT（合法训练监督）；λ=1.0 固定。
运行：每臂约 15s GPU；输入前后 SHA-256 不变（三份 RESULT 均 `inputs_unchanged=True`）。

## 判定：确认阶段 **FAIL**——"保留＋补全优于普通蒸馏"未复现

inner change 组完整成功数（N=497；native 参考 = 166）：

| seed | A 基线 | B 普通蒸馏 | C 仅保留 | D 仅补全 | E 保留+补全 |
|---|---:|---:|---:|---:|---:|
| 42 | 140 | 145 | 138 | 142 | **161** |
| 43 | 137 | 133 | — | — | 129 |
| 44 | 148 | 137 | — | — | 138 |
| 均值 | **141.7** | 138.3 | — | — | **142.7** |

关键差值（E−B）：+16 / −4 / +1，符号跨 seed 不一致，均值 +4.3 窗（+0.87pp）。
E−A：+21 / −8 / −10（均值 +1.0 窗）。B−A：+5 / −4 / −11（均值 −3.3）。

按预登记停止规则：**E 相对 B 的优势在 seed43/44 不复现 → 机制主张不成立；且普通蒸馏 B 本身也不稳定优于基线 A → 在此学生规模下"教师监督有无"都不是一阶因素。**

## 筛选期（seed42）曾满足全部通过条件——这正是三 seed 确认存在的理由

seed42 单点：E 取得最高 change good（161）、最高 rescue（45）、最低 lost（50）、最低 ADE/FDE，且 E>B、D.rescue≥C.rescue、C.lost≤D.lost 三条全过。seed43/44 全部反转。**单 seed 筛选的 +3.2pp 落在 seed 噪声内**，与 D-0 的 MDE 警告一致。

## 机制层面的两个观察（解释，不是翻案）

1. **补全信号在训练集上太稀疏**：FIT 的 8,051 个 native 失败窗中，只有约 26% 存在任何 union 好候选可教（平均 0.26 条/失败窗）。E0 证明的 +8pp 机会集中在 change/难窗，而 FIT 训练分布中这类可教样本占比很小——信号稀疏可能是不可学的一阶原因。
2. **学生容量与 native 差距大**：inner change 上 native=166，最好的学生臂=161（仅 seed42），多数臂低 20+。57K 参数 MLP 学生连"追上 native"都不稳定，"保留 native 成功模式"的保留目标本身超出了该容量。

## 结论与停止

- Gate ①（机会存在）：**PASS**，且干净、CI 不跨 0（见 E0 报告）。
- Gate ②③（可学习性 / 机制归因，筛选规模）：**FAIL**。按用户既定纪律——"E≤B 即不能主张新机制"、"失败要能被归因"——**此路线在当前规模停止，不进入 λ 扫描 / 加容量 / 加长训练的小变体循环**。
- 若未来重启，合理的唯一新问题是预登记一个**容量与信号密度假设**（更大学生 + 显式过采样可教窗），那是新实验，不是本次筛选的延续。

## 证据入口

`RESULT.json`（seed42 全五臂）、`RESULT_seed43.json`、`RESULT_seed44.json`、`run_e1.py`、学生 checkpoint `student_{A..E}_seed42.pt` 等、运行日志 `run_e1*.log`。设计与判定规则：[DESIGN.md](DESIGN.md)。
