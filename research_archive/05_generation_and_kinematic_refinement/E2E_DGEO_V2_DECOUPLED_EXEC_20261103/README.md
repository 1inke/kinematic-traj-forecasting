# DGeo V2: finite 2x2 seed42 pilots
User authorization: 2026-11-03 execute supplied NEXT_STAGE_DGEO_V2_EXPERIMENT_DESIGN.md.
This new namespace only; no historical source, model, output, failure or PAUSED task modification/deletion.

Four arms (A=causal Maneuver-12D; B=metric Taylor residual):
- DK-CenterDiff: A0 B0,303970 parameters.
- DK-ManeuverDiff: A1 B0,304802.
- DK-KinTaylorDiff: A0 B1,303970.
- DK-FullDiff: A1 B1,304802.

Each whole-model fresh seed42, fixed8 final EMA,2720 scientific steps; total32epochs10880updates.
C0 fit21719/dev4753; original64D/clip5/optimizer/loss/coverage/augmentation/EMA and all original RNG/batchIDs retained.
Only focal coordination changes. Kinematic residual maps all decoder agent slots through the unchanged shared head; only focal trajectories are scientifically evaluated, as before.
Common coordinator tensors paired across all4; geo_proj introduced after common constructor.
New module fork seed20261103+42, not V1 seed20261102+42. V1 descriptive reference not exact single-factor causal repeat.
No score residual, U, new loss, teacher, test/outer/CAL, routing or automatic extra seeds.
Run complete evidence requires worker+exit SUCCESS,4models,8full populations dual audit and SHA, factorial/common-support/unconditional tails/dual screen review.
No subjective root cause conclusions. Canonical prelaunch alignment and DA1: AUTHORIZATION_AND_ALIGNMENT.md.
45min planning based on actual measurements, not silent timeout or a reason to skip audits. Warning-first v3 unchanged.

