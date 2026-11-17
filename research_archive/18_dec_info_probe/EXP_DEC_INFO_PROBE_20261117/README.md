# Decoder information probe — COMPLETE / NO_INFO_GAIN

This study tests whether a small refiner can improve on self-only refinement by reading additional frozen encoder tokens, and whether candidate-path risk is useful for choosing neighbors. All 18 runs finished (360 epochs, 30,600 updates). The formal run took about 18 minutes and 34 seconds. It used C0 fit/dev and one frozen B0 seed42, with three seeds for each refiner. No test set was used.

[中文结果解读](RESULT_INTERPRETATION_ZH.md) · [Original report](REPORT.md) · [Machine-readable results](snapshot/RESULT.json) · [Status](STATUS.json) · [Protocol and implementation](snapshot/IMPLEMENTATION_NOTES.md)

The A1 self-only control exactly reproduced all three prior seed minFDE values, averaging a 1.2895% gain over B0. It narrowly beat the widened A1L control on the prespecified minFDE reference rule. C1 (target history), C2 (dense neighbors), C3 (path-risk neighbors) and C4 (distance neighbors) all worsened mean minFDE versus A1. C3 improved 0.161% versus C4, below the 0.5% threshold, and worsened Top1 FDE 3.89% versus B0, beyond the 2% limit.

For 78.19% of proximity-defined social windows, C3 selected different sets of neighbors across candidate paths. The mean numbers read were 2.719 for C3 and 2.960 for C4. The comparison changes filtering, neighbor count and relationship features together. C3 did not improve the social subset versus C1; its social-subset minFDE fell slightly when neighbor groups were permuted at fixed neighbor count. The mechanism check therefore failed. The negative verdict applies to this frozen interface on reused development data. It does not establish that decoder information or neighbor reasoning is generally unhelpful. B0 remains the baseline; no end-to-end B stage started.

## Reanalysis

```bash
python -m pip install -r requirements-analysis.txt
python -I -B tools/reanalyze.py
```

The NumPy-only tool verifies the artifact manifest, reconstructs the 28 saved-metric evaluation populations and both D0 summaries, checks the matched-count permutation and paired order receipts, and reproduces the fixed comparisons, decision and protection results. Evaluation populations share the same dev windows; their counts are not counts of independent samples.

The snapshot preserves the original design, code, tests, training histories, epoch receipts, and [local independent verification](snapshot/INDEPENDENT_VERIFICATION.json). The local verification recomputed metrics from complete saved trajectories. Public reanalysis starts from derived metrics and cannot repeat raw-coordinate checks, run inference or resume training with the omitted assets. The preserved historical scripts refer to the original setup; use the reanalysis tool for the portable entry point.

[Copy provenance](COPY_PROVENANCE.json) · [Omitted local assets](OMITTED_ARTIFACTS.json) · [Artifact checksums](ARTIFACT_MANIFEST.json) · [Previous set interaction study](../../17_set_interact_probe/EXP_SET_INTERACT_PROBE_20261117/)
