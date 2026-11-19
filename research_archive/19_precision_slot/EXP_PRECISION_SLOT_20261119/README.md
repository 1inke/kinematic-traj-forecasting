# Precision slot experiments, 2026-11-19

The three studies are complete. A constant-velocity (CV) trajectory improved candidate precision under a fixed 20-candidate budget. The learned CV slot with E3_self fell short of the fixed 32-epoch screening rule. Restoring exact CV only for near-stationary histories recovered precision, while the resulting model still failed the maneuver and motion checks against E3_self.

| Study | Result |
|---|---|
| [PREC-DIAG](snapshot/PREC_DIAG/RESULT_INTERPRETATION_ZH.md) | Offline CV replacement improved B0 minADE/minFDE by 3.25%/2.92% across five seeds. Deletion disproportionately harmed turning cases. |
| [PREC-PILOT](snapshot/PREC_PILOT/RESULT_INTERPRETATION_ZH.md) | Two new 32-epoch seed42 runs with matched saved controls. The combination improved 1.54%/0.50% over E3_self against the 1.5% joint screen. Change, deceleration/stop and motion checks failed. |
| [PREC-HISTORY-DIAG](snapshot/PREC_HISTORY_DIAG/RESULT_INTERPRETATION_ZH.md) | Restoring exact CV in 1,109 near-stationary observation windows improved Prior_self minFDE by 1.12% with zero joint coverage exchanges. The repaired model did not pass protection against E3_self. |

All evaluations use the reused C0 development set. These are exploratory studies. The conditional 64-epoch, five-fold and transfer stages were not reached.

[Machine-readable status](STATUS.json) · [Original source and copy records](COPY_PROVENANCE.json) · [Omitted local assets](OMITTED_ARTIFACTS.json) · [Artifact checksums](ARTIFACT_MANIFEST.json)

To repeat aggregate analysis with Python and NumPy:

```sh
python -m pip install -r requirements-analysis.txt
python -I -B tools/reanalyze.py
```

The tool aggregates the 18 compact arrays of derived metrics, verifies gains and coverage exchange, reconstructs the pilot and history protection decisions from saved motion summaries, and checks the saved epoch order hashes. The three snapshot directories also retain the original protocols, scripts, results and independent local verification receipts. Motion, history classification and geometric measurements require the local inputs to recompute from scratch. Raw data, complete trajectory predictions, caches and model weights remain local. Original scripts are preserved with their historical paths for provenance; the portable entry point is the aggregate analysis tool.

[Previous decoder information probe](../../18_dec_info_probe/EXP_DEC_INFO_PROBE_20261117/)
