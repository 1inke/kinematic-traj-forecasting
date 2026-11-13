# E1：学生蒸馏五臂筛选 · 设计草案（DRAFT，未冻结，执行取决于 E0 结果）

日期：2026-10-20｜前置门：E0（cal_v11 干净人口 oracle 机会重估）必须显示机会仍存在

## 要回答的问题

固定学生结构与预算，仅改变训练监督：
**"保留＋补全"分解监督（E）是否优于普通集合蒸馏（B）？**

关键比较不是 E vs A，而是 **E vs B**；辅以 C（仅保留）、D（仅补全）做机制归因。

## 学生（所有臂共用，固定）

- 结构：`kdec` 的 `DirectPositionDecoder`（SharedTrunk MLP + K=20 可学 query，zero-init head），57,432 参数，共享编码器一次前向输出 K20。推理无教师、无选择器。
- 训练预算：8 epochs、batch 256、AdamW lr=1e-3、FIT 18,909 窗（与既有 kdec 配方一致；单 run ≈5–25s，CUDA）。
- 基础损失：WTA（winner_loss）+ diversity_hinge（λ=0.05），所有臂相同。

## 教师池（零训练，仅用 FIT 的 GT 构造——训练期监督，合法）

- native K20：冻结缓存 `eth_train.npz` 的 FIT 行（F16 native，25 epoch canonical）。
- KIN K20：`UNION_SELECTOR_K40_TO_K20_20261018/results/cache_pred/kin_FIT_seed{42,43,44}.npy`（与学生 seed 匹配）。
- 好候选：ADE≤0.2 且 FDE≤0.4（与评价同一 J 准则）。

## 五臂定义

| 臂 | 监督 | 作用 |
|---|---|---|
| A | WTA + diversity | 公平基线 |
| B | A + 对 union40 全部好候选的单向 Chamfer 覆盖 | 普通集合蒸馏对照 |
| C | A + 仅覆盖"native 成功窗中的 native 好候选" | 保留信号 |
| D | A + 仅覆盖"native 失败窗中的 union 好候选" | 补全信号 |
| E | A + C 项 + D 项（独立权重 λ_R、λ_N） | 保留＋补全 |

Chamfer 单向：每条教师好路径拉最近的学生槽位（min over student slots）。
满足三条实现约束：不做槽位编号匹配；不平均互补路径；不冻结学生任何槽位。

## 评估与判定（筛选级，inner 划分）

- 评估集：**inner**（1856 窗；对 FIT 训练的学生 track 互斥，C2 已证）。不消耗 outer / cal_v11。
- 种子：seed 42 单 seed（筛选）；晋级后 42/43/44 三 seed 确认。
- 指标：change 组 coverage（good_count）、minADE/minFDE、rescue/lost、运动学 5 项（MOTION_FACTOR=1.01）。
- 通过条件（预登记，筛选级点估计）：
  1. E 相对 B 的 inner change coverage 点估计 > 0；
  2. E 相对 A：change coverage ↑ 且 change minADE/minFDE 恶化 ≤3%、运动学 PASS；
  3. 归因一致性：D 的 rescue ≥ C 的 rescue；C 的 lost ≤ D 的 lost。
- 停止条件：E ≤ B（点估计）→ 机制不成立，记录负结果，不进入三 seed；E 违反护栏 → 按证伪停。

## 成本

教师构造 <1 min；5 臂 × 1 seed ≈ 2 min GPU；评估分钟级。三 seed 确认共 15 run < 15 min。
真正成本在判定纪律，不在算力。

## 已知风险

- kdec 学生容量远小于 F16 native（历史 KIN coverage 0.565 vs native 0.60+）：筛选回答的是"机制是否可学"，不是"超越 native"；最终主张需换到 canonical 学生或证明小模型已够。
- inner 的 change 窗 497 个，单 seed 点估计噪声大：筛选只判方向与护栏，显著性留给三 seed + D-0 口径 MDE。
- 教师 KIN 缓存来自曾被指接触 CAL 的配方：E0 在 cal_v11 上重估的就是这个前提；若 E0 FAIL，本设计不启动。
