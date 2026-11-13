# 安全修复协同归因审计结果

日期：2026-10-18。状态：**ANALYZED**。执行目录：`SAFE_REPAIR_COOPERATION_AUDIT_EXEC_20261018`。

约束：冻结W0输出的离线反事实；无SOCP重求解、无Torch训练、无λ扫描、无P2、无GT-as-teacher。

## 一句话（推断，非事实本身）

主窗口内同时存在「T可独立保留」与「T失败需非目标」两类保存方案 → 保留异质性，不强行宣布唯一结构根因。

## 完成性

- 640题索引；18份actual路径全部SHA核对通过；缺失0。
- 反事实记录 **756/756**；目标标量 **288/288**；FIT portfolio **4/4**。
- 双实现CPU复核 mismatches=0。
- 墙钟约 **52.5s**（Asia/Tokyo 本地）。

## FACTS：主13 chosen 窗口运动学

| 观测 | 计数 /13 |
|---|---:|
| J与T均 window_motion PASS | 6 |
| J PASS 且 T FAIL | 7 |
| 其中T含均值accel/jerk超预算 | 7 |
| 其中T仅p95类失败 | 0 |
| 至少一个ADD_k运动PASS | 9 |
| 全部ADD运动FAIL | 4 |
| 全部DROP运动PASS | 13 |

逐案事实见 `case_summaries.json` / `cases/*.json`。

### 同窗口敏感性（secondary）

- secondary 合格方案数：4；其中 T motion PASS：2
- primary依赖但同窗另一方案T可通过：无

### 负例 sample22057/slot7

```json
{
  "J_window_motion_pass": false,
  "T_window_motion_pass": false,
  "O_window_motion_pass": true,
  "J_target_good": true,
  "T_target_good": true,
  "J_set_good": true,
  "T_set_good": true,
  "O_set_good": false,
  "N_set_good": false,
  "n_DROP_motion_fail": 19,
  "n_ADD_motion_pass": 0,
  "all_ADD_motion_fail": true,
  "all_DROP_motion_pass": false,
  "T_fails_mean_accel": true,
  "T_fails_mean_jerk": false,
  "T_fails_any_p95": true,
  "failing_motion_keys_T": [
    "accel_mean",
    "accel_p95"
  ],
  "failing_motion_keys_J": [
    "accel_p95"
  ]
}
```

## FACTS：完整FIT四portfolio

| portfolio | overall coverage | rescue | lost | overall motion | 13窗合并 motion |
|---|---:|---:|---:|---|---|
| FIT_N | 0.574224 | 0 | 0 | PASS | PASS |
| FIT_J | 0.574911 | 13 | 0 | PASS | PASS |
| FIT_T | 0.574911 | 13 | 0 | PASS | PASS |
| FIT_O | 0.574224 | 0 | 0 | PASS | PASS |

说明：lost=0 与 change覆盖+13 对 FIT_J/FIT_T 是构造预期；本轮区分力在安全性/支撑/目标兼容，不是再次庆祝救回。

## FACTS：P1 A/B/C/D 标量兼容（仅NumPy）

| cell | config | n | ΔL>0 | ΔL<0 | ≈0 |
|---|---|---:|---:|---:|---:|
| C | J | 13 | 1 | 12 | 0 |
| C | T | 13 | 1 | 12 | 0 |
| D | J | 13 | 2 | 11 | 0 |
| D | T | 13 | 2 | 11 | 0 |

ΔL>0 表示该保存方案相对native在当前目标下无净激励；不能推出不存在其他下降方向。

## 推断与边界

- **lean**：`heterogeneous`
- 主窗口内同时存在「T可独立保留」与「T失败需非目标」两类保存方案 → 保留异质性，不强行宣布唯一结构根因。
- 不自动训练集合模型；不启动P2；不扫伙伴/λ。
- 13窗来自32失败中的GT已知成功提取，非随机总体。

## 三次反方（同助手）

- DA1：T失败不排除另一条target-only安全解存在；结论仅限「这条保存方案」。
- DA2：目标路径逐元素不变量已断言；安全/支撑/损失分列；p95不按候选相加。
- DA3：18解全纳入、负例保留、chosen不重选；无native-good保护实证不冒称。

本有限审计完成即结束。
