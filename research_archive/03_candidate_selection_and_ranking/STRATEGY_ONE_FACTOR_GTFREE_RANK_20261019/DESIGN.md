# DESIGN ? STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019

- time zone: Asia/Tokyo
- parent: after K0 GO_K1=true, K1 GO_K2=false, branch **C** (oracle ok / learner fail)
- contract: **no** HGB / tree / label-weight scan; **no** K2; **no** generator retrain; **no** K0 teacher in deploy

## One factor

Remove the learned pred-gate (L??A??F??C). Keep the **same K0-C action space** on the frozen native20+KIN20 pool. Rank / accept with **GT-free geometry + real kinematics only**.

Question: can this get closer to K0-C?s joint W on outer (3 seeds)?

## Fixed (not the factor)

- Pool: frozen native20 ? KIN20 (same caches as K0/K1)
- Splits: outer (primary), plus read-only reuse of existing K0-C / K1-S / V1 indices
- Action space: start native `{0..19}`; ?20 native?KIN replacements; inserted KIN never removed; removed native never restored
- Window motion gate: literal ratios ? **1.01** vs that window?s native20 (same `motion_stats` / `motion_gate` as K0)

## Factor (locked score ? do not sweep)

At each step, among legal pairs `(r,l)` that **PASS** the real motion gate, pick the max of:

```
score(r,l) = (
  min_j?S\{r}  D(l,j),                          # larger = better hole-fill
  -?endpoint(l) ? mean_endpoints(S\{r})?,       # closer to remaining mean
  -r, -l                                        # ascending slots
)
```

where `D = max(mean_path_dist/0.10, endpoint_dist/0.20)` (same as K1 `pairwise_D_matrix`).

Stop when no motion-passing legal pair remains, or 20 swaps done.

**Forbidden in deploy choice:** future GT, group labels, K0 oracle traces/indices as teacher, HGB / any learned prediction.

## Controls

| arm | source |
|---|---|
| native20 | frozen |
| V1 `cond_boost_11_9` | existing bundle indices |
| K1-S | `k1_exec/seed*/predictions/outer/S_indices.npy` |
| K0-C | `results/seed*/outer/k0c_indices.npy` (oracle diagnostic only) |
| **GTF** (this) | new selector |

## Evaluation

Reuse K0 `evaluate_set` + `check_W`. Report outer?{42,43,44}: change good / cov / minADE / minFDE / motion / W_pass / mean swaps.

## BREAKTHROUGH

`true` only if **all three** outer seeds: `W_pass=true` **and** change ADE & FDE each ?5% better than native (same hard clauses as W). Else `false`. At most one follow-up (e.g. rank key = relative set-NN distance) allowed; still no generator / HGB sweep.
