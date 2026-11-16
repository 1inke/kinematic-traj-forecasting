# Kinematic Trajectory Forecasting (`f16-research`)

Versioned research codebase, protocol governance, independent audit suite, and phase-organized experimental archive for multimodal kinematic pedestrian trajectory forecasting.

> **Repository Sync Policy**: Tracked code, configuration manifests, documentation, and metric archives mirror the GitHub repository (`1inke/kinematic-traj-forecasting`), excluding raw datasets and binary model weights. Recent archives additionally retain compact, checksum-verified per-window analysis arrays for independent CPU aggregate reanalysis.

## Latest completed study: EXPLORE_FAST — calibration not passed (2026-11-16)

**本轮实验已完成：8/8 次训练和评价，约 80 分钟。16/32 轮均未通过预设预算校准，未选出正式预算，未升级 baseline。** Execution completed successfully; this is not a method-confirmation success. The 32-epoch E3_self change-group coverage gain missed its fixed threshold by one window (21 net windows versus 22 required).

[Code, results and CPU reanalysis](research_archive/12_explore_fast/EXPLORE_FAST_20261116/) · [Results analysis (中文)](research_archive/12_explore_fast/EXPLORE_FAST_20261116/RESULT_ANALYSIS.md) · [Original budget decision](research_archive/12_explore_fast/EXPLORE_FAST_20261116/snapshot/EXPLORATION_BUDGET.json)

T1 shows a stronger change-group coverage signal and improved aggregate motion statistics at 32 epochs; E3_self gains mainly on stable windows. Both methods have only seed42, and their change-group Top1 good-trajectory rates decrease despite improved best-of-20 coverage. Reused development data do not establish stable rankings or independent generalization. No automatic 64/128 training or old five-fold resume occurred.

## Paused study: FIVEFOLD B0 — INCOMPLETE (2026-11-16)

**未执行完全：预算比较 10/10 完成；最终评测仅 2/15 完成，另有 2 次中途暂停、11 次未开始。** The study was paused by user request to prioritize short exploratory experiments. No final five-fold three-seed mean/SD or study-completion seal exists, and the baseline has not been upgraded.

[Incomplete study, code and CPU reanalysis](research_archive/11_fivefold_b0_incomplete/FIVEFOLD_B0_20261115/) · [All planned run statuses](research_archive/11_fivefold_b0_incomplete/FIVEFOLD_B0_20261115/RUN_STATUS.csv) · [Partial conclusions (中文)](research_archive/11_fivefold_b0_incomplete/FIVEFOLD_B0_20261115/reference_analysis/PARTIAL_CONCLUSION.md)

The archive preserves budget selection (E*=128), two completed ETH final evaluations, partial-checkpoint metadata, per-window metric arrays, recording-isolation audits and execution/recovery history. A separately labeled single-seed reference reuses 95%-training budget models; it does not fill missing 100%-training final runs. The package discloses earlier C0 development-data overlap with four benchmark test folds. Weights, raw datasets and full trajectory predictions remain local. The completed EXPLORE_FAST study is archived separately in Phase12.

## Earlier completed experiment: NEXT_PHASE E1–E4 (2026-11-15)

[Code, analysis inputs and reproducibility instructions](research_archive/10_next_phase_e1_e4/NEXT_PHASE_EXEC_20261114/) · [Results interpretation (中文)](research_archive/10_next_phase_e1_e4/NEXT_PHASE_EXEC_20261114/RESULTS_REVIEW_ZH.md) · [Original final report](research_archive/10_next_phase_e1_e4/NEXT_PHASE_EXEC_20261114/snapshot/RESULT_ANALYSIS.md)

The finite study completed successfully at 13:32:29 Asia/Shanghai: 47 new training runs, including 27 E2/E3/E4 method models, fixed 64-epoch method budgets, paired evaluation, geometry diagnostics, latency measurements and final bootstrap analysis. **None of the nine methods passed the full prespecified success screen; B0 remains the baseline.** E3_self shows promising overall coverage/error improvements with a stable-group kinematic tradeoff. No official test, five-fold study, factor combination or baseline upgrade was performed.

The publication includes compact saved-output statistics and a NumPy-only verifier that reconstructs all 11 comparisons and all saved 20,000-draw bootstrap interval endpoints without model weights or raw datasets.

---

## Repository Structure

| Path | Description |
| :--- | :--- |
| `src/f16/` | Maintained core Python package (`eval`, `data`, `preprocessing`, `hybrid`, `scene_prior`, `guardrails`, `governance`, `provenance`, `statistics`, `checkpoint`, `resource`, `paths`, `io`, `notifications`) |
| `config/` | Protocol contracts (`protocol_v1.json`, `data_manifest.v1.json`), design drafts (`A_scene_prior.draft.json`, `B_joint_hybrid.draft.json`), and path template (`paths.example.yaml`) |
| `audit/` | Independent evaluation verification (`verify.py`, `statistics_verify.py`), input digest locking (`freeze_inputs.py`), and source security scanner (`security_scan.py`) |
| `tests/` | Automated unit & contract test suite (`test_protocol.py`, `test_guardrails.py`, `test_frozen_audit.py`, `test_provenance.py`) |
| `experiments/` | Active formal protocol benchmark workspace (`2026-10-20_D0_protocol_calibration/`) |
| `reviews/` | Governance verification snapshots (`governance_20261020/`) |
| `archive/` | Historical migration notes (`MIGRATION_PLAN.md`), directory index (`INDEX.csv`, `build_index.py`), and legacy root readme snapshot |
| `docs/` | Organized architectural design specs (`docs/architecture/`), multi-series synthesis reviews (`docs/synthesis/`), and detailed study design notes |
| `research_archive/` | Chronological/thematic archive of experimental series (`01_` through `12_`), preserving scripts, configs, negative & positive reports, aggregate metrics, compact reanalysis inputs, the incomplete Phase11 snapshot, and completed Phase12 calibration |
| `RESEARCH_INDEX.md` | Complete mapping from every legacy experiment folder to its phased path under `research_archive/` |

---

## Phased Research Archive (`research_archive/`)

1. `01_early_encoders_and_f16/` — Early F16 encoders, baseline sweeps, and attention variants (`AUTONOMOUS_EXPERIMENTS`, `EXP_*`)
2. `02_cross_dataset_and_efficiency/` — Cross-scene/dataset generalization (ETH-UCY, SDD, AV2) & runtime efficiency studies
3. `03_candidate_selection_and_ranking/` — Selector design, rankability probes, and local fine reranking
4. `04_anchors_residuals_and_social/` — Anchor corrections, residual heads, social stimulus verifiers, and interaction diagnostics
5. `05_generation_and_kinematic_refinement/` — Kinematic decoding, mode-preserving refinement, and generation cycles 1–5+
6. `06_protocol_audit_and_governance/` — Leakage audits, track-overlap checks, and arithmetic verification stages
7. `07_mode_sequence_and_local_fixes/` — Late observability probes, endpoint discrete set utility, and Q-mode sequences
8. `08_e2e_reset_and_dpp/` — End-to-end decoupled geometric optimization (`E2E_DGEO`/`UD`) and DPP conditional VAE feasibility studies
9. `09_misc_studies/` — Supporting utilities and exploratory notes
10. `10_next_phase_e1_e4/` — Completed E1 calibration, continuous-curve decoders, candidate refinement, training factors, compact reanalysis inputs and negative-result evidence
11. `11_fivefold_b0_incomplete/` — **PAUSED / INCOMPLETE** B0 five-fold study: 10/10 budget runs, 2/15 final runs, partial-checkpoint records and CPU aggregate reanalysis
12. `12_explore_fast/` — **Execution COMPLETE / calibration NOT_CALIBRATED**: eight short-budget B0/T1/E3_self runs, fixed 16/32-epoch decisions, analysis and compact CPU reanalysis inputs

---

## Quick Verification

```bash
pip install -e ".[test]"
pytest -q
python audit/security_scan.py
```

See [BASELINE.md](BASELINE.md), [PROTOCOL.md](PROTOCOL.md), [DESIGN_AB.md](DESIGN_AB.md), and [GEMINI.md](GEMINI.md) for scientific governance, anti-leakage rules, and evaluation priorities ($J$, Complete Good Candidate Coverage $ADE \le 0.2\text{m}, FDE \le 0.4\text{m}$, Net Rescue, Effective Good Clusters, Kinematic Plausibility).
