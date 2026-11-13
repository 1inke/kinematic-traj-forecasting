# DEPLOY_S_RANK_MOTION_ONLY — DESIGN

- time_zone: Asia/Tokyo
- dir: `<WORKSPACE_ROOT>\DEPLOY_S_RANK_MOTION_ONLY_20261019`
- prior diagnostic: `K0C_LEARNER_GAP_DIAGNOSTIC_20261019` (S pred_gate misreject ≈82% on K0-C accepted actions; do not redo)

## Scientific question

Given frozen K1 **S** arm (113-d features, five-output HGB, same action space), is the **pred_gate** the binding constraint that blocks reaching W?

## Single factor

| arm | kinematics (true window) | pred_gate L/∆A/∆F/∆C|∆J | rank key |
|-----|--------------------------|-------------------------|----------|
| **A** (locked baseline) | ≤1.01× native | **ON** (K1 deploy) | S: (∆C, ∆J, ∆A+0.5∆F, −rem, −add) |
| **B** (treatment) | ≤1.01× native | **OFF** | same S rank key |
| **C** (optional cheap) | ≤1.01× native | **OFF** | **I** arm same key |

- A: read existing `k1_exec` outer S indices / metrics — **no retrain, no redeploy**
- B/C: load frozen `S_*` / `I_*` joblibs from `k1_exec/seed{42,43,44}/models`; max 20 swaps; enumerate legal native→KIN pairs only
- Evaluate **outer** seeds 42/43/44 only

## Forbidden

Retrain HGB; change HGB hyperparams or labels; train generator; use K0 teacher; sweep thresholds/quotas.

## Breakthrough (hard)

`BREAKTHROUGH=true` iff outer **all three seeds** pass screening **W** (same `check_W` as K0/K1: change cov floor, change ADE/FDE each ≥5% vs native, protect groups, lost≤1%, motion PASS).

If false: standby; state the next scientific design question (e.g. calibrated stop); **do not** chain further weak variants.

## Resources

THREADS=2, Idle priority, affinity low cores; quiet overnight.
