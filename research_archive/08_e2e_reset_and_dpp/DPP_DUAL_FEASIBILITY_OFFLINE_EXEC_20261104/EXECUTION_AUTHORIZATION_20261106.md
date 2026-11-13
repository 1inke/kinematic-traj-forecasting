# 2026-11-06 execution authorization

Latest direct human request: 启动实验. This supersedes the 2026-11-04 script-only hold for the already prepared Stage A only.

Run fixed-K20 saved-data kinematic quality and DPP diagnostics, after real-data preflight. No model forward, optimizer, new training, new candidates or external data. Missing admitted metric maps and original DSF Gaussian-latent interface remain unresolved; do not launch Stage B or fabricate scene/joint quality.

The original CODE_VALIDATION.json and all sources/results/failures remain historical evidence. The original code package was SHA-verified before activation; 15 synthetic tests were rerun while execution remained disabled (synthetic_tests/1793947213500028200.json). Do not rerun the historical denial test against an enabled live control: that test assumes a disabled control.

Original execution_control.json: status SCRIPTS_ONLY_HUMAN_DO_NOT_START; allow_experiment_start=false; active_worker=false; background_task_created=false; scientific_models_started=0; real_data_experiments_started=0; reason 2026-11-04 latest user: continue writing scripts, do not start after scripts are ready; synthetic_tests_allowed=true.

Only additional execution infrastructure is introduced before the production source lock. Original scientific Python files, equations, calibration choices and tolerances are unchanged. CPU estimate 2 GiB: Python/NumPy/SciPy 384 MiB, input and metric population 512 MiB, current/next diagnostics and compressed IO 512 MiB, independent loops and allocator reserve 640 MiB. Resource v3 startup checks real physical and commit availability; 1.3 is warning only, not a mechanical hard limit. No memory locking, OS/pagefile changes, deletions or unrelated process actions.

ARS DA1/2/3 are same-assistant counterevidence checks, not external review. Scoring fixed candidates cannot improve their oracle errors or coverage. These five sources are one-seed model variants, not five independently trained DPP arms.
