# RESULT_ANALYSIS ? FOLLOWUP GTF_REL (relative set-NN)

- time: 2026-10-19T03:20:41+09:00 (Asia/Tokyo)
- dir: `<WORKSPACE_ROOT>\STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019\followup_rel`
- factor: rel=min_D(l)-min_D(r)>0 + motion; no HGB/GT/teacher

| seed | arm | change_good | change_minADE | change_minFDE | W_pass | n_swaps |
|---:|---|---:|---:|---:|---|---:|
| 42 | native20 | 110 | 0.24914661896382423 | 0.4293782977069047 | False | None |
| 42 | GTF_REL | 95 | 0.2623128135830642 | 0.46512247976556215 | False | 5.406763285024154 |
| 42 | GTF | 97 | 0.3071944175635802 | 0.5736161783002811 | False | 18.306280193236717 |
| 42 | K1_S | 113 | 0.2528738472886832 | 0.451053818699378 | False | 2.641545893719807 |
| 42 | K0_C | 126 | 0.23664530545170753 | 0.3907952146171078 | True | 0.3748792270531401 |
| 42 | v1_cond_boost_11_9 | 116 | 0.3209116139319278 | 0.634290020443257 | False | None |
| 43 | native20 | 110 | 0.24914661896382423 | 0.4293782977069047 | False | None |
| 43 | GTF_REL | 98 | 0.26295056069620953 | 0.4659291551515457 | False | 5.0 |
| 43 | GTF | 108 | 0.2880181995569887 | 0.5247657442582376 | False | 17.9256038647343 |
| 43 | K1_S | 113 | 0.2585209695297367 | 0.4661263007724296 | False | 2.4521739130434783 |
| 43 | K0_C | 133 | 0.23243657541691962 | 0.3845334811735341 | True | 0.43285024154589374 |
| 43 | v1_cond_boost_11_9 | 119 | 0.317906626661751 | 0.6195238372737318 | False | None |
| 44 | native20 | 110 | 0.24914661896382423 | 0.4293782977069047 | False | None |
| 44 | GTF_REL | 100 | 0.25500831684207836 | 0.45312234153011055 | False | 4.841545893719807 |
| 44 | GTF | 91 | 0.3054489410239148 | 0.575319043760213 | False | 17.829951690821257 |
| 44 | K1_S | 111 | 0.2560301308706257 | 0.4570929162633646 | False | 2.6164251207729468 |
| 44 | K0_C | 127 | 0.23109289210337167 | 0.3915086382901264 | True | 0.37874396135265703 |
| 44 | v1_cond_boost_11_9 | 110 | 0.3166486933781599 | 0.623940046619975 | False | None |

**BREAKTHROUGH = False**

```json
{
  "BREAKTHROUGH": false,
  "clauses": {
    "42": {
      "W_pass": false,
      "change_ade5": false,
      "change_fde5": false,
      "change_good": 95,
      "change_minADE": 0.2623128135830642,
      "change_minFDE": 0.46512247976556215
    },
    "43": {
      "W_pass": false,
      "change_ade5": false,
      "change_fde5": false,
      "change_good": 98,
      "change_minADE": 0.26295056069620953,
      "change_minFDE": 0.4659291551515457
    },
    "44": {
      "W_pass": false,
      "change_ade5": false,
      "change_fde5": false,
      "change_good": 100,
      "change_minADE": 0.25500831684207836,
      "change_minFDE": 0.45312234153011055
    }
  },
  "arm": "GTF_REL"
}
```