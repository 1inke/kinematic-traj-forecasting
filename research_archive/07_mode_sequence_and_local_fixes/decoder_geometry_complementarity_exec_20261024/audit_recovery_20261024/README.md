# Q0 independent-audit recovery, no Q0 reproduction

Parent worker11116 failed2026-10-24 12:26:05.472746 during independent motion audit; all62295 paired rows and12 saved primary artifacts already complete. Parent source/code/failure/logs/Parquet/CSV/NPZ immutable. Parent FAILURE remains FAILURE.

Root cause: independent auditor allocated float64 velocity but computed `p[:,:,0]/.4` in float32 before assignment. The producer correctly casts paths before division. Cast the first observed-origin-to-prediction velocity operands to float64 before division. Subsequent differences were already double. No scientific tolerance, dt, percentile, quality, clustering or map-admission changes.

One explicit AST assignment patch only; runtime compiles immutable old auditor with this correction and redirects its final REVIEW write into this child. Reads all completed parent outputs; no call to produce(), model, Torch, optimizer or solver. Full independent audit is resumed from immutable saved data, not optimizer/RNG resume and not another full Q0 production.

Actual regression includes the original failed FIT event motion population, old float32 discrepancy and corrected exact2e-12 checks; source and completed outputs checked before/after. Same3GiB inventory/1.3dynamic remaining + Windows retained peak policy. Map gate remains STOP; no Q1/Q2 or new formal training.
