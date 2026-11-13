# DEPLOY_S_RANK_MOTION_ONLY — RESULT_ANALYSIS

- time: 2026-10-19T04:45:08+09:00 (Asia/Tokyo)
- dir: `<WORKSPACE_ROOT>\DEPLOY_S_RANK_MOTION_ONLY_20261019`
- factor: remove pred_gate; keep true kinematics ≤1.01 native; S (B) / I (C) rank
- **BREAKTHROUGH=false**

## Outer summary

| seed | arm | W | change_good | change_cov | change_ADE | change_FDE | ADE↑% | FDE↑% | swaps |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | native | False | 110 | 0.4348 | 0.24915 | 0.42938 | 0.00 | 0.00 | — |
| 42 | A_K1S | False | 113 | 0.4466 | 0.25287 | 0.45105 | -1.50 | -5.05 | 2.641545893719807 |
| 42 | B_S_mot | False | 89 | 0.3518 | 0.31186 | 0.58500 | -25.17 | -36.24 | 18.393236714975846 |
| 42 | C_I_mot | False | 90 | 0.3557 | 0.31560 | 0.58710 | -26.67 | -36.73 | 18.556521739130435 |
| 42 | K0C | True | 126 | 0.4980 | 0.23665 | 0.39080 | 5.02 | 8.99 | — |
| 43 | native | False | 110 | 0.4348 | 0.24915 | 0.42938 | 0.00 | 0.00 | — |
| 43 | A_K1S | False | 113 | 0.4466 | 0.25852 | 0.46613 | -3.76 | -8.56 | 2.4521739130434783 |
| 43 | B_S_mot | False | 103 | 0.4071 | 0.29161 | 0.54175 | -17.04 | -26.17 | 17.978743961352656 |
| 43 | C_I_mot | False | 103 | 0.4071 | 0.29526 | 0.54864 | -18.51 | -27.78 | 18.16135265700483 |
| 43 | K0C | True | 133 | 0.5257 | 0.23244 | 0.38453 | 6.71 | 10.44 | — |
| 44 | native | False | 110 | 0.4348 | 0.24915 | 0.42938 | 0.00 | 0.00 | — |
| 44 | A_K1S | False | 111 | 0.4387 | 0.25603 | 0.45709 | -2.76 | -6.45 | 2.6164251207729468 |
| 44 | B_S_mot | False | 97 | 0.3834 | 0.30117 | 0.56380 | -20.88 | -31.31 | 17.971980676328503 |
| 44 | C_I_mot | False | 93 | 0.3676 | 0.30691 | 0.57566 | -23.19 | -34.07 | 18.11207729468599 |
| 44 | K0C | True | 127 | 0.5020 | 0.23109 | 0.39151 | 7.25 | 8.82 | — |

## B vs A (treatment vs locked K1-S)

- seed42: Δchange_good=-24, Δchange_ADE=+0.05899, Δchange_FDE=+0.13395, W A=False→B=False, swaps 2.641545893719807→18.393236714975846
- seed43: Δchange_good=-10, Δchange_ADE=+0.03308, Δchange_FDE=+0.07563, W A=False→B=False, swaps 2.4521739130434783→17.978743961352656
- seed44: Δchange_good=-14, Δchange_ADE=+0.04514, Δchange_FDE=+0.10670, W A=False→B=False, swaps 2.6164251207729468→17.971980676328503

## Judgment

Removing pred_gate alone does **not** clear W on all outer seeds (or fails protect/error clauses). S rank without the gate is insufficient.

### Next scientific question (standby — no further weak variants)

Given motion-feasible actions exist and K0-C accepts many of them, but neither pred_gate-on (A) nor pred_gate-off+S-rank (B) reaches W: **what stop / accept rule is learnable without GT?** Candidate design: *calibrated stop* — predict whether one more motion-PASS swap improves set margin with a holdout-calibrated threshold (or pairwise preference over motion-PASS actions), not another ON/OFF of the same gate.
Do **not** open another threshold/rank-key tweak as a third/fourth weak variant.

## Artifacts
- `BREAKTHROUGH.json`, `RESULT_ANALYSIS.json`, `results/seed{42,43,44}/`
- indices under `indices/`
