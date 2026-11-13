# 无好覆盖 ∩ 不可预见 — 估算设计

状态：可复现估算（事后诊断，非部署特征）。  
范围：ETH outer，主报告 seed42（N=1035）；若 KIN 缓存齐则附 seed43/44。  
数据：`visual_review_overnight/data/arrays.npz`（past/future/native 等）+ `UNION_SELECTOR_K40_TO_K20_.../kin_outer_seed{seed}.npy`。  
合同阈值来自 `00_contract.md` / `k0/config.py`：ADE≤0.2、FDE≤0.4 为「好候选」主标准。

## 1. 目标量

对每个窗口 i：

1. **无好候选覆盖** `no_cover(S_i)`：集合 S_i 中不存在候选 j 使  
   `ADE_ij ≤ τ_A` 且 `FDE_ij ≤ τ_F`（等价 J≤1，主标准 τ_A=0.2、τ_F=0.4）。
2. **急变事件** `sudden_i`：仅用 **GT 未来**相对历史末刻运动学外推的偏离定义（事后标签）。
3. **可预兆** `cue_i`：仅用 **观测窗 past**（及由 past 导出的速度/转角），判断历史末段是否已出现与急变同号的趋势。
4. **不可预见** `unfore_i := sudden_i ∧ ¬cue_i`。

主统计：

| 量 | 定义 |
|---|---|
| N_nc | ∑ no_cover |
| N_uf∩nc | ∑ (no_cover ∧ unfore) |
| 占无覆盖 | N_uf∩nc / N_nc |
| 占全测 | N_uf∩nc / N |

并拆分无覆盖子集：`not_sudden` / `sudden∧cue` / `sudden∧¬cue`。

## 2. 集合 S（双口径 soft 报告）

| 口径 | S | 含义 |
|---|---|---|
| **native_K20** | 原生 20 条 | 相对原生池无覆盖 |
| **K40_union** | native∥KIN（40） | 相对过夜并集池无覆盖（与 K0 池可达上界同物理池） |

敏感性：主门 (0.2, 0.4)；备用 (0.3, 0.5)（合同 §3 固定敏感性，不取代主门）。

不把 k1_S / deploy 选后的 K20 当作「池无覆盖」主口径（那是选择器丢覆盖）；可选附列仅作对照。

## 3. 急变事件（GT 仅用于事件定义）

记号：`dt=0.4s`，`past` 末点为原点，`past[-1]=(0,0)`。

- 历史末速度：`v0 = (past[-1] − past[-2]) / dt`（若 ‖v0‖<ε，用更早非零段或标 `low_speed`）。
- 恒速外推：`cv[t] = v0 · (t+1)·dt`，t=0…11。
- 位移方向单位：`û = v0 / ‖v0‖`（低速则用 past 首末位移）。

逐窗特征（相对 CV）：

1. **max_lat**：`max_t | (gt[t] × û)_z |`（横向偏离，米）
2. **max_dev**：`max_t ‖gt[t] − cv[t]‖`
3. **heading_delta_deg**：历史末航向与「GT 前半段平均速度」夹角（度）；辅以「GT 末段航向」夹角取较大者
4. **speed_ratio**：`mean‖v_gt‖ / max(‖v0‖, ε)`

**sudden**（主阈值 T1）当且仅当：

`max_lat ≥ L` **或** `heading_delta_deg ≥ Θ`

| 组 | L (m) | Θ (°) | 备注 |
|---|---|---|---|
| **T1 主** | 0.40 | 45 | 与可视化「突然大拐」同量级 |
| T2 松 | 0.30 | 35 | 更敏感 |
| T3 紧 | 0.60 | 60 | 更保守 |

低速窗（‖v0‖ < 0.15 m/s）：仅当 `max_dev ≥ L` 或末点位移相对 CV 足够大时标 sudden；单独计数 `n_low_speed`。

## 4. 可预兆（仅 past）

ETH 本估算 **纯运动学**（无邻居/地图；合同数据未进场景图）。

由 past 速度序列 `v_h[k] = (past[k+1]−past[k])/dt`，k=0…6：

- `hist_turn_deg`：末 3 段航向角变化之和（有符号）
- `hist_turn_abs`：|hist_turn_deg|
- `fut_turn_sign`：由 `gt` 相对 `û` 的横向符号（或 GT 航向变化符号）——**仅用于与 hist 同号比对**；判别式本身不把 GT 形状喂给「覆盖」定义

**cue / 可预兆**（主）：

`hist_turn_abs ≥ H` **且** `sign(hist_turn) == fut_turn_sign`（同号）

| 组 | H (°) |
|---|---|
| 随 T1 | 12 |
| 随 T2 | 8 |
| 随 T3 | 18 |

备选（写入结果但不改主定义）：历史末加速度方向与 fut 横向同号。无同号大幅转角 → ¬cue。

## 5. 不可预见与解释

`unforeseeable = sudden ∧ ¬cue`

含义：**未来相对恒速外推已构成急变，但观测末段没有同号转弯预兆** → 开环、仅自车历史时信息论上难「单条贴 GT」；集合覆盖仍可能偶然命中，但「无好覆盖」中这一块不宜归咎为选择器/距离键。

`sudden ∧ cue`：历史有预兆仍无覆盖 → 更偏向生成器/池多样性不足。  
`¬sudden`：无大急变仍无覆盖 → 误差超门、多模态未盖中等其它原因。

## 6. 实现与产物

- 脚本：`estimate_unforeseeable_no_cover.py`（本 EXEC 根目录）
- 设计：本文 `UNFORESEEABLE_NO_COVER_ESTIMATE_DESIGN.md`
- 输出：
  - `UNFORESEEABLE_NO_COVER_ESTIMATE_seed42.json`
  - `UNFORESEEABLE_NO_COVER_ESTIMATE_seed42.md`
- 只读旧 P1 / MODE_PRESERVING；只写本 EXEC 新文件。
- 低占用：纯 numpy，单 seed 一次扫 N=1035。

## 7. 局限（必须披露）

1. 「不可预见」是 **操作定义**，不是因果证明；阈值敏感，故报 T1–T3。
2. 无地图/邻居：路口规则、避让等社会预兆未计入 → 可能 **高估** 不可预见。
3. CV 外推对已在弯中的目标偏严；已用 hist_turn 同号纠正一部分。
4. 好覆盖用 ADE/FDE 硬门，与 W 的 change 组门槛不同；本估算回答的是「池里有没有好候选」，不是「是否达 W」。
5. K40 仍可能全体 miss 急变模态；oracle 达 W 只说明 **平均** 可救，不保证每个 unfore 窗可救。
