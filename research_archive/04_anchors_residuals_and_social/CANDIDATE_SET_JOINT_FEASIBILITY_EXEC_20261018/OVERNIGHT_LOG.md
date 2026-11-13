# OVERNIGHT_LOG


- [2026-10-19T00:04:55+09:00] executor resume: no python; seed44 FIT_cal incomplete; launching serial quiet chain

- [2026-10-19T00:05:31+09:00] overnight_serial_quiet start
- [2026-10-19T00:05:31+09:00] backed up GO_K1/RESULT_ANALYSIS -> <WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\logs\backup_before_k0_fitcal
- [2026-10-19T00:05:31+09:00] START K0_seed44_FIT_cal: <WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\.venv-k0\Scripts\python.exe -u <WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\run_k0.py --seeds 44 --splits FIT_cal
- [2026-10-19T00:15:17+09:00] END K0_seed44_FIT_cal rc=0
- [2026-10-19T00:15:17+09:00] restored GO_K1_REGISTER from backup
- [2026-10-19T00:15:17+09:00] GO_K1 preserved=true; secondary_splits updated
- [2026-10-19T00:15:17+09:00] merged K0 secondary n_results=9
- [2026-10-19T00:15:17+09:00] START K1_formal_I_vs_S: <WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\.venv-k0\Scripts\python.exe -u <WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\run_k1.py

- [2026-10-19T00:51:42+09:00] executor: found dual run_k1 (44080 idle 4.8MB + 31072 working ~938MB); killed 44080; set 31072 Idle+affinity=0x3 (2 cores)

- [2026-10-19T00:52:43+09:00] executor: K1 died mid seed42 FIT_cal; patched Idle+skip-fit-resume; restarting overnight_serial_quiet (append log)

- [2026-10-19T00:53:02+09:00] executor relaunch overnight_serial_quiet pid=18360 Idle+affinity=0x3; K1 will resume seed42 FIT_cal then seed43/44
- [2026-10-19T00:53:16+09:00] overnight_serial_quiet start
- [2026-10-19T00:53:16+09:00] FIT_cal seed44 already complete; merge only
- [2026-10-19T00:53:16+09:00] GO_K1 preserved=true; secondary_splits updated
- [2026-10-19T00:53:16+09:00] merged K0 secondary n_results=9
- [2026-10-19T00:53:16+09:00] START K1_formal_I_vs_S: <WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\.venv-k0\Scripts\python.exe -u <WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\run_k1.py

- [2026-10-19T00:53:20+09:00] executor bat-relaunch; python_pids=22892,12144,27144,38064

- [2026-10-19T00:53:29+09:00] note: venv python stubs (~5MB) parent the real System Python workers; do NOT kill stubs. Single chain OK. Monitoring.

- [2026-10-19T00:58:39+09:00] monitor: alive; log:     select 200/2958 swaps_mean=8.88 |     select 400/2958 swaps_mean=8.86 |     select 600/2958 swaps_mean=10.01

- [2026-10-19T01:08:51+09:00] monitor: n_py=4 cpu=91% |     select 2000/2958 swaps_mean=10.06 |     select 2200/2958 swaps_mean=10.12

- [2026-10-19T01:22:03+09:00] monitor: n_py=4 cpu=80% |   fitted S/dA seed43 const=False mean=-0.00234084 |   fitted S/dF seed43 const=False mean=-0.00583574 |   fitted S/dJ seed43 const=False mean=-0.0140207

- [2026-10-19T01:35:33+09:00] monitor: n_py=4 cpu=94% |     select 600/1856 swaps_mean=8.67 |     select 800/1856 swaps_mean=8.97

- [2026-10-19T01:54:33+09:00] monitor: n_py=4 Aff=3 Idle cpu~15% |     select 1200/2958 swaps_mean=8.68 |     select 1400/2958 swaps_mean=8.61 |     select 1600/2958 swaps_mean=8.68

- [2026-10-19T02:08:00+09:00] monitor: n_py=4 Aff=3 Idle | seed43 FIT_cal S ~2600/2958 |     select 2200/2958 swaps_mean=2.72 |     select 2400/2958 swaps_mean=2.72 |     select 2600/2958 swaps_mean=2.63 |     select 2800/2958 swaps_mean=2.61

- [2026-10-19T02:21:24+09:00] monitor: n_py=4 Aff=3 Idle | seed44 inner I started after outer I/S done

- [2026-10-19T02:31:57+09:00] parent: monitor executor interrupted again; K1 still alive seed44 inner S; reattached monitor only (no second job)

- [2026-10-19T02:33:16+09:00] monitor: n_py=4 Aff=3 Idle | seed44 inner S ~800/1856 progressing

- [2026-10-19T02:45:31+09:00] monitor: n_py=4 Aff=3 Idle WS_MB=943.7 CPU=11714.3 |     select 1200/1856 swaps_mean=9.56 |     select 1400/1856 swaps_mean=9.58 |     select 1600/1856 swaps_mean=9.57 |     select 1800/1856 swaps_mean=9.50 |   select arm=S |     select 200/1856 swaps_mean=1.88 |     select 400/1856 swaps_mean=1.95 |     select 600/1856 swaps_mean=3.35 |     select 800/1856 swaps_mean=3.13 |     select 1000/1856 swaps_mean=3.00 |     select 1200/1856 swaps_mean=2.79 |     select 1400/1856 swaps_mean=2.66 |     select 1600/1856 swaps_mean=2.57 |     select 1800/1856 swaps_mean=2.44 | [seed44] deploy FIT_cal N=2958 |   select arm=I |     select 200/2958 swaps_mean=8.03 |     select 400/2958 swaps_mean=7.88 |     select 600/2958 swaps_mean=8.67 |     select 800/2958 swaps_mean=9.22 |     select 1000/2958 swaps_mean=9.49 |     select 1200/2958 swaps_mean=9.29 |     select 1400/2958 swaps_mean=9.18 |     select 1600/2958 swaps_mean=9.17 |     select 1800/2958 swaps_mean=9.18

- [2026-10-19T02:57:51+09:00] monitor: n_py=4 Aff=3 Idle WS_MB=949.8 |     select 1200/2958 swaps_mean=3.60 |     select 1400/2958 swaps_mean=3.34 |     select 1600/2958 swaps_mean=3.21 |     select 1800/2958 swaps_mean=3.14 |     select 2000/2958 swaps_mean=2.98 |     select 2200/2958 swaps_mean=2.88 |     select 2400/2958 swaps_mean=2.88 |     select 2600/2958 swaps_mean=2.79
- [2026-10-19T02:58:51+09:00] END K1_formal_I_vs_S rc=0
- [2026-10-19T02:58:51+09:00] K1 done GO_K2=False
- [2026-10-19T02:58:51+09:00] overnight_serial_quiet finished

- [2026-10-19T03:06:37+09:00] monitor: K1 process ended; n_py=0

- [2026-10-19T03:09:37+09:00] K1 DONE GO_K2=False; wrote BRANCH_DECISION (C); no K2; Idle standby
- [2026-10-19T03:09:37+09:00] BREAKTHROUGH=false; S>I on change-error CI but harms native; large gap to K0-C

- [2026-10-19T03:10:18+09:00] wrote index-overlap diagnostic; no further heavy jobs; standby

- [2026-10-19T03:13:15+09:00] next direction: STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019 (GT-free min_D rank; no HGB/K2/generator); see <WORKSPACE_ROOT>\STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019

- [2026-10-19T03:19:18+09:00] GTF absolute min_D DONE BREAKTHROUGH=false (over-swap~18); start followup GTF_REL (relative nn, rel>0)

- [2026-10-19T03:22:37+09:00] GTF+GTF_REL both BREAKTHROUGH=false; STATUS at <WORKSPACE_ROOT>\STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019/STATUS.json; standby (no more follow-ups this budget)

- 2026-10-19T03:26:44+09:00: K0C_LEARNER_GAP_DIAGNOSTIC_20261019 done; S_pred_gate_misreject=0.8200; GTF_misrank=0.9935; BREAKTHROUGH=false; standby.

- 2026-10-19T03:30:57+09:00: original_CAL KIN cache attempt status=OK; see K0C_LEARNER_GAP_DIAGNOSTIC_20261019/CAL_CACHE_STATUS.json

- 2026-10-19T04:45:08+09:00: DEPLOY_S_RANK_MOTION_ONLY_20261019 done; BREAKTHROUGH=false; B_all_W=False; see <WORKSPACE_ROOT>\DEPLOY_S_RANK_MOTION_ONLY_20261019

- 2026-10-19T05:36:14+09:00: original_CAL transfer eval DONE BREAKTHROUGH=false; track_ci=BLOCKED_IDENTITY; see <WORKSPACE_ROOT>\ORIGINAL_CAL_TRANSFER_EVAL_20261019

- 2026-10-19T05:36:53+09:00: original_CAL transfer eval DONE BREAKTHROUGH=false; track_ci=BLOCKED_IDENTITY; K0-C W=True×3 (oracle); V1/K1-S W=False×3; see <WORKSPACE_ROOT>\ORIGINAL_CAL_TRANSFER_EVAL_20261019
- 2026-10-19T05:36:53+09:00: wrote OVERNIGHT_DIGEST_20261019.md; BREAKTHROUGH=false; standby (no new experiment directions)
