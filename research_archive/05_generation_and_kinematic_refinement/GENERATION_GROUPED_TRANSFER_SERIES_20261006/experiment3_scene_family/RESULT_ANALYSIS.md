# Experiment 3 — Scene-family transfer analysis

## Verdict

The experiment completed successfully, but it does not upgrade the baseline or establish a universal shared-head failure. Its strongest result is evidence for a zara-family distribution/exposure shift: training with zara windows (random arm) transfers better to held-out zara windows than training with zero zara windows (grouped arm). The result is exploratory because the held family contains three recordings, F16 is already seen, and grouped versus random changes finite-sample composition and optimization exposure.

## What was tested

- Target family: zara; 1,561 outer windows, 309 tracks, 3 recordings.
- Random training exposed 4,154 zara windows; grouped training exposed 0.
- Both arms used 16,337 training windows, identical loss-group budgets, common statistics from 12,183 windows, the same raw state/head/projection/objective, three seeds, and frozen F16/decoder/scorer.
- All six trainings reached 80 epochs. The result contains 216 saved metric rows, 180 exploratory target-track bootstrap summaries, and complete frozen/source SHA audits.

## Primary outer result

On held-out zara, final complete-candidate coverage (ADE <= .2 and FDE <= .4) averaged across seeds:

| arm | overall | change |
|---|---:|---:|
| native | 66.62% | 38.86% |
| random (zara exposed) | 68.76% | 38.07% |
| grouped (zara unseen in training) | 66.58% | 33.96% |

Random therefore rescues more complete candidates and loses fewer existing good candidates. Grouped has a much weaker rescue/lost balance on the held family. Target-track exploratory bootstrap gives grouped minus random: +0.0119 m ADE (95% CI +0.0085,+0.0157), +0.0260 m FDE (+0.0177,+0.0347), +0.0629 J (+0.0434,+0.0844), and -2.18 percentage points coverage (-3.67,-0.69). These intervals are exploratory, not independent scene-level confirmation.

## CAL cross-check and motion

On the original CAL split, the difference is not the same: grouped final overall averages 1,487 good windows, random 1,484, and native 1,490; for CAL change, grouped averages 231.7, random 219.7, and native 243. Thus zara exposure improves zara transfer but does not produce a universal gain on CAL. This pattern favors distribution shift over a simple claim that random overlap always makes the model better.

The motion gate is not uniformly satisfied. On zara outer change, random speed-p95 is about +1.47% to +2.48% versus native and grouped is +0.77% to +2.03%; several outer groups therefore fail the preregistered 1% protection gate. Mean acceleration is near +0.8%, but passing an average does not pass all five motion checks.

## Interpretation

The result supports the branch “scene/family distribution shift or exposure mismatch is important.” It does not identify a unique root cause. Random and grouped differ in same-sequence and same-target exposure by design, and grouped training also changes the finite composition of examples even with matched loss-group counts. Therefore the result is not proof of leakage, not proof that the shared head is intrinsically incapable, and not evidence for immediately adding MoE, a scorer, or a new loss.

The appropriate next question is a preregistered distribution-shift analysis of state/context/kinematic feature distributions and matched recording/time blocks. Any future structural intervention should wait until that analysis separates exposure shift from composition and finite-optimization effects.

## Audit status and limitations

Worker and exit both succeeded at 18:38:25.8694 HK; stderr was empty; all six histories contain 80 epochs. `RESULT.json`, `ARITHMETIC_AUDIT.json`, `FROZEN_AUDIT.json`, `PRESERVATION_AUDIT.json`, 18 model hashes, 114 output hashes, 589 current sources, and 34,351 inherited files passed their respective checks. The first preflight's stale common-count assertion (12,544 versus the registered 12,183) and the stale prelaunch train-count display (16,104 versus 16,337) were preserved and documented; only the implementation assertion/report supplement was corrected.

This is a new-head-only transfer study on an already-seen dataset, not whole-model unseen-scene generalization, SOTA evidence, intent inference, or a causal proof about the decoder/scorer.
