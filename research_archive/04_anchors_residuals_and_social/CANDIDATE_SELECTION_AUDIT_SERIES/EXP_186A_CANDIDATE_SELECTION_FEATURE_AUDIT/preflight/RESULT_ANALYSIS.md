# EXP-186A Candidate Selection Feature Audit

Decision: **PREFLIGHT_OK**

- Source/train-selected rule: R3_H3_MBR
- Raw-signal features: ['native_logit', 'native_probability', 'native_rank', 'speed_mismatch', 'heading_mismatch', 'mean_jerk', 'path_length_ratio', 'endpoint_speed_ratio', 'correction_t4', 'correction_t8', 'correction_t12', 'correction_total', 'endpoint_centrality', 'trajectory_centrality', 'h3_centrality', 'score_weighted_centrality', 'nearest_candidate_distance', 'endpoint_local_density', 'trajectory_local_density']
- Outer-test residual-signal features: none
- Source/train rule features: none
- Top1 J gain: -1.24%
- Positive folds: 3/5
- Mean regret gain: -2.15%
- P95 regret change: -0.35%

F16, H3, Joint-V2, native logits, and all K20 trajectories remained frozen.
