# BRANCH_DECISION (after K1)

- time: 2026-10-19T03:09:37+09:00 (Asia/Tokyo)
- exec: `<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018`
- K1 artifacts: `<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\k1_exec`

## Gates

| item | value |
|---|---|
| GO_K1 | true (K0 outer 3-seed pool + K0-C reached W) |
| GO_K2 | **false** |
| start K2 | **no** |
| BREAKTHROUGH | **false** |

Main fail clauses (from `k1_exec/GO_K2_REGISTER.json`): all seeds miss change ADE/FDE >=5% improve; transition protect fails; bootstrap S-native change coverage lo<=0 and error hi>=0; secondary inner/FIT_cal fail; original_CAL still BLOCKED_NO_KIN_CACHE.

## Design branch: C

> Strategy oracle succeeds; learner fails (`04` section 5C)

Evidence:
1. Pool+policy reachable: K0-C outer change_good ~126/133/127, W=true.
2. Learned arms miss joint gate: I and S both W_pass=false.
3. Local mechanism: S beats I on change error (Bonferroni CI upper <0) but coverage unstable; vs native still harms precision -> not deployable.
4. Behavior: S ~2.5 swaps (~30% zero); I ~9 swaps. S pred_gate rejects >> I (over-conservative, not over-swapping).

Contract action: **end this I/S contrast**; no tree-depth/state/label-weight chase; no K2; no generator retrain; no K0 oracle teacher.

## Gap to K0-C

| seed | native cg | S | I | K0-C | S-native ADE | K0-C-native ADE |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 110 | 113 | 107 | 126 | +0.0037 | -0.0125 |
| 43 | 110 | 113 | 120 | 133 | +0.0094 | -0.0167 |
| 44 | 110 | 111 | 107 | 127 | +0.0069 | -0.0181 |

Set-marginal supervision reduced I/V1-style error collapse, but did **not** learn K0-C style swaps that raise coverage and cut error. Bottleneck is GT-free pred gates / ranking / deploy-state, not empty pool.

## Outer reject totals (3 seeds)

| arm | motion | pred_gate | no_candidate |
|---|---:|---:|---:|
| I | 67708 | 382677 | 5 |
| S | 7254 | 903463 | 175 |

## Overnight follow-up

- Done: K0 secondary, formal K1, RESULT_ANALYSIS, GO_K2=false, this branch note.
- Will **not** start K2 / new run_k1 / hyperparam scan.
- Optional read-only FP/FN later; default standby until user returns.

See: `k1_exec/RESULT_ANALYSIS.md`, `BRANCH_C_DIAGNOSTIC.json`.

## Index-set overlap (outer; native slots assumed 0..19)

| seed | mean |S\cap native| | mean |K0C\cap native| | mean |S\cap K0C| | frac S==native | frac K0C==native | frac S==K0C |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 17.36 | 19.63 | 17.09 | 0.303 | 0.625 | 0.183 |
| 43 | 17.55 | 19.57 | 17.36 | 0.282 | 0.567 | 0.181 |
| 44 | 17.38 | 19.62 | 17.15 | 0.290 | 0.621 | 0.176 |

K0-C stays near-native with sparse surgical swaps; S also sparse but **different** sets (~18% identical to K0-C). I diverges hard (~10 native kept).
