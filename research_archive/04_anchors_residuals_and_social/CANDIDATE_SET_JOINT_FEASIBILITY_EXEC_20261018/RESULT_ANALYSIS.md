# K0 联合可达性 RESULT_ANALYSIS

日期：2026-10-19T00:15:17+09:00（Asia/Tokyo）
标签：`ORACLE_DIAGNOSTIC_ONLY`。本阶段未训练 generator / K1 / K2。

## 1. 执行状态

- freeze_inputs: **OK**
- original_CAL: **BLOCKED_NO_KIN_CACHE**
- solver budgets (outer): `{"time_limit_s": 600, "node_limit": null, "registered_before_outer": true, "preflight_wall_s": 0.2754533290863037, "preflight_status": "NUMERIC_INFEASIBILITY_UNVERIFIED", "note": "Time limit is stop mechanism, not infeasibility evidence."}`
- GO_K1 登记: **False** — outer_incomplete — 

## 2. 三字段（每 seed × split）

| seed | split | pool_joint_witness | policy_oracle_witness | relaxation_certificate_status |
|---:|---|---|---|---|
| 44 | FIT_cal | True | True | R_INCUMBENT_VERIFIED_W |

## 3. Outer 对照（好窗口数 / change ADE / change FDE / W）

## 4. 分支解释

{
  "GO_K1": false,
  "reason": "outer_incomplete",
  "detail": "need 3 seeds"
}

## 5. 未做之事

- 未训练 K1 / K2 / generator
- 未扫配额 / boost / λ / 网络 / epoch
- 未改写旧 P1 / MODE_PRESERVING / COMPLEMENTARY_POOL_VALIDATION 结果
- K0 oracle 产物不进入 K1 teacher 白名单


---


---

## 附录 A · 副 split 补全（过夜续跑）

- 写入时间：2026-10-19T00:53:16+09:00（Asia/Tokyo）
- original_CAL：仍 **BLOCKED_NO_KIN_CACHE**（本夜不重开）

| seed | split | pool_joint | policy_oracle | cert | change_good(k0c) | change ADE/FDE | W_pass(k0c) |
|---:|---|---|---|---|---:|---|---|
| 42 | FIT_cal | True | True | R_INCUMBENT_VERIFIED_W | 301 | 0.26637986205528813/0.4398269810005079 | True |
| 43 | FIT_cal | True | True | R_INCUMBENT_VERIFIED_W | 296 | 0.26599937244064875/0.4375511866744952 | True |
| 44 | FIT_cal | True | True | R_INCUMBENT_VERIFIED_W | 295 | 0.26532538562449326/0.44103460377198983 | True |
| 42 | inner | True | True | R_INCUMBENT_VERIFIED_W | 207 | 0.24309235705623497/0.41013691038776123 | True |
| 43 | inner | True | True | R_INCUMBENT_VERIFIED_W | 211 | 0.24128042167741406/0.4070558760399247 | True |
| 44 | inner | True | True | R_INCUMBENT_VERIFIED_W | 203 | 0.24246535786111656/0.41215192498579645 | True |
| 42 | outer | True | True | R_INCUMBENT_VERIFIED_W | 126 | 0.23664530545170753/0.3907952146171078 | True |
| 43 | outer | True | True | R_INCUMBENT_VERIFIED_W | 133 | 0.23243657541691962/0.3845334811735341 | True |
| 44 | outer | True | True | R_INCUMBENT_VERIFIED_W | 127 | 0.23109289210337167/0.3915086382901264 | True |
