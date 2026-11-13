# original_CAL 迁移补评

- time: 2026-10-19T05:36:14+09:00 (Asia/Tokyo)
- dir: `<WORKSPACE_ROOT>\ORIGINAL_CAL_TRANSFER_EVAL_20261019`
- KIN cache: present `(2879,20,12,2)` per seed 42/43/44
- **BREAKTHROUGH=false**（本任务为收尾补评，非突破门槛）

## 身份 / track CI

- track_ci_status: **BLOCKED_IDENTITY**
- n_unique_track_key=1（期望 ≫1 才可做 track CI）
- 说明：original_CAL rows sit past n_id=25912; loader fills sequence='?', ped=-1. Cannot forge track/hierarchical CI. Point metrics still valid under GT.
- **未伪造** track / hierarchical bootstrap CI。

## 主表（点估计；W 同 00 合同）

| seed | arm | W | change_good | change_cov | change_ADE | change_FDE | overall_ADE | motion_change | swaps |
|---:|---|---|---:|---:|---:|---:|---:|---|---:|
| 42 | native20 | False | 243 | 0.3471 | 0.24593 | 0.43197 | 0.18058 | PASS | — |
| 42 | v1_cond_boost_11_9 | False | 245 | 0.3500 | 0.33150 | 0.63767 | 0.21396 | PASS | — |
| 42 | K1_S | False | 248 | 0.3543 | 0.25652 | 0.45096 | 0.18243 | PASS | 3.22 |
| 42 | K0_C | True | 299 | 0.4271 | 0.23019 | 0.38454 | 0.16356 | PASS | 0.39 |
| 43 | native20 | False | 243 | 0.3471 | 0.24593 | 0.43197 | 0.18058 | PASS | — |
| 43 | v1_cond_boost_11_9 | False | 247 | 0.3529 | 0.32894 | 0.62901 | 0.21291 | FAIL | — |
| 43 | K1_S | False | 252 | 0.3600 | 0.25439 | 0.44678 | 0.18079 | PASS | 2.25 |
| 43 | K0_C | True | 297 | 0.4243 | 0.23075 | 0.38334 | 0.16303 | PASS | 0.48 |
| 44 | native20 | False | 243 | 0.3471 | 0.24593 | 0.43197 | 0.18058 | PASS | — |
| 44 | v1_cond_boost_11_9 | False | 252 | 0.3600 | 0.32805 | 0.63051 | 0.21267 | FAIL | — |
| 44 | K1_S | False | 245 | 0.3500 | 0.25644 | 0.44757 | 0.18213 | PASS | 2.61 |
| 44 | K0_C | True | 302 | 0.4314 | 0.22735 | 0.37543 | 0.16309 | PASS | 0.41 |

## 读法

- **native20 / V1 / K1-S**：可部署或历史选择器对照（无 GT 选择）。
- **K0-C**：`ORACLE_DIAGNOSTIC_ONLY`，用 GT 的贪心策略上界；不可当作迁移成功。
- 若 K0-C 达 W 而 K1-S/V1 未达：迁移上仍是「池可达、选择器未达」，与 outer 同型。
- track CI 缺失 ⇒ **不能**声称 CAL 上的稳健区间结论；仅报告点估计。

## W 摘要

- native20: per-seed W = False, False, False
- v1_cond_boost_11_9: per-seed W = False, False, False
- K1_S: per-seed W = False, False, False
- K0_C: per-seed W = True, True, True

## 产物

- `<WORKSPACE_ROOT>\ORIGINAL_CAL_TRANSFER_EVAL_20261019\RESULT_ANALYSIS.json`
- `<WORKSPACE_ROOT>\ORIGINAL_CAL_TRANSFER_EVAL_20261019\RESULT_ANALYSIS.md`
- `<WORKSPACE_ROOT>\ORIGINAL_CAL_TRANSFER_EVAL_20261019\STATUS.json`
- indices under `<WORKSPACE_ROOT>\ORIGINAL_CAL_TRANSFER_EVAL_20261019\indices`
