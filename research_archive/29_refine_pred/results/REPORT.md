# EXP-REFINE-PRED

**Decision: STOP**. C1=False, C2=True, C3=False.

Frozen QRANK models, 34 replays, zero training. Positive minFDE changes denote higher error.

| Fold | Scope | E0 deceleration change (95% CI) | E0 steady change | QFL deceleration change | E0 longitudinal ratio, decel / steady |
| --- | --- | --- | --- | --- | --- |
| ETH | exploratory | +40.67% [+19.74%, +64.47%], 92 windows / 14 tracks | -12.12% | +26.28% | 1.091 / 1.013 |
| ZARA01 | exploratory | +19.87% [+9.97%, +30.09%], 196 windows / 32 tracks | -18.92% | +10.94% | 0.917 / 0.830 |
| HOTEL | confirmation | +0.96% [-19.94%, +12.19%], 46 windows / 13 tracks | -10.78% | +4.61% | 0.975 / 0.862 |
| ZARA02 | confirmation | +2.13% [-4.41%, +8.56%], 387 windows / 46 tracks | -13.31% | +2.36% | 0.868 / 0.751 |
| UNIV | confirmation | -0.22% [-2.29%, +1.95%], 4407 windows / 468 tracks | -13.08% | +1.09% | 0.830 / 0.779 |

## Observation proxy AUROC (E0)

| Proxy | ETH | ZARA01 | HOTEL | ZARA02 | UNIV | Pass |
| --- | --- | --- | --- | --- | --- | --- |
| X1 | 0.664 | 0.469 | 0.804 | 0.709 | 0.544 | False |
| X2 | 0.448 | 0.570 | 0.301 | 0.306 | 0.451 | False |
| X3 | 0.507 | 0.554 | 0.752 | 0.697 | 0.559 | True |
| X4 | 0.419 | 0.538 | 0.281 | 0.574 | 0.539 | False |

The JSON also reports secondary proxies, all maneuver labels, seed consistency, speed-matched strata, dependence on the largest contributing tracks, UNIV recording sensitivity, and ZARA02 without seed 44.

PREFLIGHT.json records differences between the printed exploratory AUROCs and recomputation from archive 28.

Changes compare jointly trained layer outputs. They describe associations within frozen checkpoints.
