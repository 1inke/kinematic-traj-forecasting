# T2 production stopped before training

The sole halfway check at 2026-10-16 17:50:04.523 HK found worker and exit FAILURE, finished 17:44:00.569583 HK. All three C histories are zero. Sixteen registered jobs are reused A/native references, not new C scientific outputs.

The data-loading line correctly loaded statistics_velocity.npz, but the following production run-manifest construction still attempted to hash statistics_position.npz (train.py line 23). This failed before seed/model/optimizer initialization or any training step. It is a file-reference implementation error, not a negative model result or memory shortage. The previous numerical/GPU/subset-resume preflight did not execute this real production manifest branch; an AST completed-branch check was insufficient.

Preserve worker_stderr.log, FAILURE.json, failures/1792143840539557100, all locked code and parent evidence. No final scientific/source/resource audit was completed. No C performance result or replacement ETA exists. No task restart, retraining, source edit or resource inflation was performed after discovery.

STOP_WAIT_HUMAN. A proposed recovery must use a new versioned folder, correct only the registered velocity-statistics manifest reference, actually execute and assert production manifest construction, and rerun full preflight/current-source/resource/budget checks before any sole launch. T1 historical resource deviation remains preserved; D/T3 remains not triggered. Old paused monitors remain untouched. The single-count halfway occurrence is consumed and no completion node is scheduled.
