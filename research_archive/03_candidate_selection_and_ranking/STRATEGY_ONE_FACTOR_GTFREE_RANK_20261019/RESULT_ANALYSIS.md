# RESULT_ANALYSIS ? STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019

- time: 2026-10-19T03:16:56+09:00 (Asia/Tokyo)
- dir: `<WORKSPACE_ROOT>\STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019`
- factor: GT-free min_D hole-fill rank + real motion gate; **no** HGB / GT / K0 teacher

## Outer 3-seed table

| seed | arm | change_good | change_minADE | change_minFDE | W_pass | n_swaps |
|---:|---|---:|---:|---:|---|---:|
| 42 | native20 | 110 | 0.24914661896382423 | 0.4293782977069047 | False | None |
| 42 | v1_cond_boost_11_9 | 116 | 0.3209116139319278 | 0.634290020443257 | False | None |
| 42 | K1_S | 113 | 0.2528738472886832 | 0.451053818699378 | False | 2.641545893719807 |
| 42 | K0_C | 126 | 0.23664530545170753 | 0.3907952146171078 | True | 0.3748792270531401 |
| 42 | GTF | 97 | 0.3071944175635802 | 0.5736161783002811 | False | 18.306280193236717 |
| 43 | native20 | 110 | 0.24914661896382423 | 0.4293782977069047 | False | None |
| 43 | v1_cond_boost_11_9 | 119 | 0.317906626661751 | 0.6195238372737318 | False | None |
| 43 | K1_S | 113 | 0.2585209695297367 | 0.4661263007724296 | False | 2.4521739130434783 |
| 43 | K0_C | 133 | 0.23243657541691962 | 0.3845334811735341 | True | 0.43285024154589374 |
| 43 | GTF | 108 | 0.2880181995569887 | 0.5247657442582376 | False | 17.9256038647343 |
| 44 | native20 | 110 | 0.24914661896382423 | 0.4293782977069047 | False | None |
| 44 | v1_cond_boost_11_9 | 110 | 0.3166486933781599 | 0.623940046619975 | False | None |
| 44 | K1_S | 111 | 0.2560301308706257 | 0.4570929162633646 | False | 2.6164251207729468 |
| 44 | K0_C | 127 | 0.23109289210337167 | 0.3915086382901264 | True | 0.37874396135265703 |
| 44 | GTF | 91 | 0.3054489410239148 | 0.575319043760213 | False | 17.829951690821257 |

## vs native (GTF)

- seed42: ?change_good=-13, ADE%=23.30, FDE%=33.59, W=False, swaps=18.306
- seed43: ?change_good=-2, ADE%=15.60, FDE%=22.22, W=False, swaps=17.926
- seed44: ?change_good=-19, ADE%=22.60, FDE%=33.99, W=False, swaps=17.830

## BREAKTHROUGH

**BREAKTHROUGH = False**

```json
{
  "BREAKTHROUGH": false,
  "clauses": {
    "42": {
      "W_pass": false,
      "change_ade5": false,
      "change_fde5": false,
      "change_good": 97,
      "change_minADE": 0.3071944175635802,
      "change_minFDE": 0.5736161783002811,
      "native_ade": 0.24914661896382423,
      "native_fde": 0.4293782977069047
    },
    "43": {
      "W_pass": false,
      "change_ade5": false,
      "change_fde5": false,
      "change_good": 108,
      "change_minADE": 0.2880181995569887,
      "change_minFDE": 0.5247657442582376,
      "native_ade": 0.24914661896382423,
      "native_fde": 0.4293782977069047
    },
    "44": {
      "W_pass": false,
      "change_ade5": false,
      "change_fde5": false,
      "change_good": 91,
      "change_minADE": 0.3054489410239148,
      "change_minFDE": 0.575319043760213,
      "native_ade": 0.24914661896382423,
      "native_fde": 0.4293782977069047
    }
  }
}
```

## Notes

- K0-C is oracle diagnostic only (uses GT in selection).
- K1-S uses learned set-marginal pred-gate; GTF removes that gate.
- No generator retrain; no HGB hyperparam scan.

## Follow-up GTF_REL

See `<WORKSPACE_ROOT>\STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019\followup_rel/RESULT_ANALYSIS.md`. BREAKTHROUGH=False.


---

## Overnight close-out (2026-10-19T03:22:37+09:00 Asia/Tokyo)

| arm | seed42 cg | seed43 cg | seed44 cg | typical swaps | W any seed? |
|---|---:|---:|---:|---:|---|
| native20 | 110 | 110 | 110 | 0 | no |
| GTF (abs min_D) | 97 | 108 | 91 | ~18 | no |
| GTF_REL (rel>0) | 95 | 98 | 100 | ~5 | no |
| K1-S | 113 | 113 | 111 | ~2.5 | no |
| K0-C oracle | 126 | 133 | 127 | ~0.4 | **yes all** |

**BREAKTHROUGH=false** (primary and sole follow-up).

Mechanism: dropping the learned pred-gate without a GT-free **quality** stop (only motion + geometry diversity) either over-swaps (GTF) or still fails to match K0-C?s sparse surgical swaps (GTF_REL). Geometry/physics alone in this action space did **not** approach K0-C W.

No further direction in this budget. Standby.
