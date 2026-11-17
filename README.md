# Kinematic Trajectory Forecasting (`f16-research`)

Versioned research codebase, protocol governance, independent audit suite, and phase-organized experimental archive for multimodal kinematic pedestrian trajectory forecasting.

> **Repository Sync Policy**: Tracked code, configuration manifests, documentation, and metric archives mirror the GitHub repository (`1inke/kinematic-traj-forecasting`), excluding raw datasets and binary model weights. Recent archives additionally retain compact, checksum-verified per-window analysis arrays for independent CPU aggregate reanalysis.

## Latest completed study: frozen-cache set interaction probe — NO_SET_GAIN (2026-11-17)

**18次探索训练全部完成并独立核验；未支持候选交互加再分配方案，baseline不变。** Self-only refinement improves minFDE by 1.29% over frozen B0, but generic interaction adds only 0.041%, below the fixed 0.5% threshold. Sparse interaction and reallocation comparisons fail. All mean protection checks pass; no end-to-end B stage follows.

[Code, per-window metrics and CPU reanalysis](research_archive/17_set_interact_probe/EXP_SET_INTERACT_PROBE_20261117/) · [结果解读](research_archive/17_set_interact_probe/EXP_SET_INTERACT_PROBE_20261117/RESULT_INTERPRETATION_ZH.md) · [Status](research_archive/17_set_interact_probe/EXP_SET_INTERACT_PROBE_20261117/STATUS.json)

Six arms × three refiner seeds, 20 epochs each, one frozen B0 seed42, C0 fit/dev only; 954.69 seconds. The negative result concerns this frozen, bounded local-refinement proxy, not all possible end-to-end interaction models. Exact source/evidence, 19 derived per-window metric arrays and NumPy-only aggregate reanalysis are archived; no raw coordinates, full predictions or weights are published.

## Earlier completed study: kinematic anchor oracle — NOT_SUPPORTED (2026-11-17)

**零训练模板检验已完成：150次聚类，固定判定NOT_SUPPORTED，baseline不变。** K improves ETH ADE/FDE by 27.73%/21.07% versus P, but other-fold FDE, five-fold validation errors and pooled deceleration/stop constraints fail. C1 and C6 pass; C2–C5 fail. This rejects advancement of the tested fixed template design, not all kinematic decoders.

[Code, templates and CPU reanalysis](research_archive/16_kine_anchor_oracle/EXP_KINE_ANCHOR_ORACLE_20261117/) · [结果解读](research_archive/16_kine_anchor_oracle/EXP_KINE_ANCHOR_ORACLE_20261117/RESULT_INTERPRETATION_ZH.md) · [Status](research_archive/16_kine_anchor_oracle/EXP_KINE_ANCHOR_ORACLE_20261117/STATUS.json)

Five folds, four main arms, six fixed sensitivity variants and three clustering seeds; no network training or inference. Exact source/results, 21 compact template/metric arrays and independent local geometry-verification evidence are archived. Sensitivity outcomes do not replace the main configuration. No follow-up training or old five-fold resume occurred.

## Earlier completed study: deceleration diagnosis — selection issue, subtype unresolved (2026-11-17)

**零训练、零推理诊断已完成：ETH主判B（选择问题），B1/B2尚未满足，baseline不变。** Eight saved-output populations include three ETH models and five C0 seeds. In the ETH wide-speed model's 89 valid-heading deceleration windows, 78.7% have a longitudinally reachable candidate but 75.3% of Top1 trajectories overshoot; 48/70 reachable cases still select an overshooting Top1. Candidate counts and both error components also deteriorate after augmentation.

[Code, per-window results and CPU reanalysis](research_archive/15_decel_diag/EXP_DECEL_DIAG_20261117/) · [结果解读](research_archive/15_decel_diag/EXP_DECEL_DIAG_20261117/RESULT_INTERPRETATION_ZH.md) · [Status](research_archive/15_decel_diag/EXP_DECEL_DIAG_20261117/STATUS.json)

All five C0 seeds also meet the B main criterion; four meet the exploratory B2 conditions. This does not establish intrinsic unpredictability or a universal causal mechanism. No follow-up training, baseline upgrade or old five-fold resume occurred.

## Earlier completed study: ETH speed diagnostics and augmentation — promising with tradeoffs (2026-11-16)

**零训练诊断和一次条件增强训练均完成。ETH minADE/minFDE 改善11.40%/6.93%，但覆盖、Top1及减速/变化组存在代价，未升级baseline。** The fixed single-seed exploratory criterion passes; it does not establish broad generalization or a complete motion-state solution.

[Code, outputs and CPU reanalysis](research_archive/14_eth_speed/EXP_ETH_SPEED_20261116/) · [Detailed analysis (中文)](research_archive/14_eth_speed/EXP_ETH_SPEED_20261116/RESULT_ANALYSIS_ZH.md) · [Publication status](research_archive/14_eth_speed/EXP_ETH_SPEED_20261116/STATUS.json)

High-speed windows account for about 83% of aggregate ADE improvement, while deceleration-group FDE worsens 34.74%, change-group FDE worsens 15.54%, and overall Top1 FDE increases 8.52%. Joint coverage falls from 59 to 55 windows (13 rescued, 17 lost). Zero-training normalization already achieves similar aggregate errors. One ETH E64 seed42 run reuses the matched B0; all 64 epoch batch orders match. Raw coordinate-processing checks pass within the existing pipeline, without independent upstream physical calibration. Old five-fold training remains paused, and COMBO/extra seeds were not run.

## Earlier completed study: ETH-T1 — no help on primary errors (2026-11-16)

**实验一已完成：ETH / T1 / 64轮 / seed42，复用同设置B0，约68分钟。主指标判定为无帮助，实验二暂不运行。** The single-seed exploratory result worsens ETH minADE20 by 6.70% and minFDE20 by 8.79%, despite a 5.77 percentage-point gain in joint coverage. This is a negative scientific result from a successful execution, not a program failure or proof that all seeds/combinations fail.

[Experiment code, outputs and CPU reanalysis](research_archive/13_eth_t1/EXP_ETH_T1_20261116/) · [Detailed analysis and experiment-two decision (中文)](research_archive/13_eth_t1/EXP_ETH_T1_20261116/RESULT_ANALYSIS_ZH.md) · [Original primary result](research_archive/13_eth_t1/EXP_ETH_T1_20261116/snapshot/ETH_T1_RESULT.json)

Change-group minADE/minFDE worsen by 11.61%/27.54%; error growth among 274 jointly uncovered windows outweighs gains in newly covered cases. Observed speed shift is a diagnostic clue, not an established cause. The current C0 combo experiment remains shelved because it does not directly resolve this ETH failure; it was not executed. No baseline upgrade, extra seed or old five-fold resume occurred.

## Earlier completed study: EXPLORE_FAST — calibration not passed (2026-11-16)

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
| `research_archive/` | Chronological/thematic archive of experimental series (`01_` through `17_`), preserving scripts, configs, negative & positive reports, compact reanalysis inputs, incomplete Phase11, Phase12 calibration, Phase13 ETH-T1 and Phase14 speed diagnostics/augmentation with tradeoffs, Phase15 deceleration diagnosis, Phase16 kinematic-template oracle and Phase17 frozen-cache interaction probe |
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
13. `13_eth_t1/` — **Execution COMPLETE / primary verdict NO_HELP (single seed)**: ETH T1@64 versus reused B0@64, diagnostic error decomposition, observed-speed summaries and CPU reanalysis; combo remains shelved
14. `14_eth_speed/` — **Execution COMPLETE / PROMISING_SINGLE_SEED with tradeoffs**: ETH/HOTEL zero-training speed diagnostics, one conditional ETH wide-speed B0@64, paired error/coverage/maneuver analysis and CPU reanalysis; no baseline upgrade
15. `15_decel_diag/` — **Execution COMPLETE / B, subtype B_UNRESOLVED**: zero-training/inference saved-output candidate-generation/selection diagnosis across three ETH models and five C0 seeds, with per-window CPU reanalysis
16. `16_kine_anchor_oracle/` — **Execution COMPLETE / NOT_SUPPORTED**: five-fold zero-network-training kinematic-template oracle comparison, fixed sensitivity analysis, 150 clustering runs and per-seed CPU reanalysis
17. `17_set_interact_probe/` — **Execution COMPLETE / NO_SET_GAIN**: frozen B0 cache, six arms × three refiner seeds, complete training histories, fixed mechanism comparisons and CPU reanalysis

---

## Quick Verification

```bash
pip install -e ".[test]"
pytest -q
python audit/security_scan.py
```

See [BASELINE.md](BASELINE.md), [PROTOCOL.md](PROTOCOL.md), [DESIGN_AB.md](DESIGN_AB.md), and [GEMINI.md](GEMINI.md) for scientific governance, anti-leakage rules, and evaluation priorities ($J$, Complete Good Candidate Coverage $ADE \le 0.2\text{m}, FDE \le 0.4\text{m}$, Net Rescue, Effective Good Clusters, Kinematic Plausibility).
