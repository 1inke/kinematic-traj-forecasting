"""Build FIT_train action rows for I/S. No K0 oracle / outer GT teacher."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .config import STATES, N_ACTIONS_PER_STATE, FEATURE_DIM, OUTPUT_NAMES, N_TRAIN_WINDOWS
from .sampling import select_train_windows, build_all_states_for_window
from .features113 import build_pool_features_22, pairwise_D_matrix, action_feature_113
from .labels import labels_IS, set_min_metrics


def build_training_matrix(
    sample_id_full,
    past_tr, gt_tr, native_tr, kin_tr, sample_id_tr,
    out_dir: Path, seed: int,
    max_windows=None,
):
    """
    All *_tr arrays are FIT_train-aligned (same length / order).
    Selection uses sample_id hash; same IDs across seeds.
    Returns X, YI, YS, diag.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / f"train_matrix_seed{seed}_done.json"
    X_path = out_dir / f"X_seed{seed}.npy"
    YI_path = out_dir / f"YI_seed{seed}.npy"
    YS_path = out_dir / f"YS_seed{seed}.npy"
    meta_path = out_dir / f"rows_seed{seed}.npz"
    if marker.exists() and X_path.exists() and YI_path.exists() and YS_path.exists():
        print(f"  resume train matrix seed{seed}", flush=True)
        return np.load(X_path), np.load(YI_path), np.load(YS_path), json.loads(marker.read_text())

    # local indices 0..n_tr-1 as "fit_train_ix"
    local_ix = np.arange(len(sample_id_tr), dtype=np.int64)
    chosen_local, sel_info = select_train_windows(sample_id_tr, local_ix, n=N_TRAIN_WINDOWS)
    if max_windows is not None:
        chosen_local = chosen_local[:max_windows]
        sel_info["n_taken"] = int(len(chosen_local))
        sel_info["truncated"] = True

    n_rows = len(chosen_local) * len(STATES) * N_ACTIONS_PER_STATE
    X = np.zeros((n_rows, FEATURE_DIM), dtype=np.float64)
    YI = np.zeros((n_rows, 5), dtype=np.float64)
    YS = np.zeros((n_rows, 5), dtype=np.float64)
    row_sid = np.zeros(n_rows, dtype=np.int64)
    row_state = np.zeros(n_rows, dtype=np.int8)
    row_rem = np.zeros(n_rows, dtype=np.int16)
    row_add = np.zeros(n_rows, dtype=np.int16)
    state_to_i = {s: i for i, s in enumerate(STATES)}

    cursor = 0
    for wi, li in enumerate(chosen_local):
        sid = int(sample_id_tr[li])
        past_i = past_tr[li:li+1]
        nat_i = native_tr[li:li+1]
        kin_i = kin_tr[li:li+1]
        pool = np.concatenate([native_tr[li], kin_tr[li]], axis=0)
        feat40 = build_pool_features_22(past_i, nat_i, kin_i)[0]
        D40 = pairwise_D_matrix(pool)
        gt_i = gt_tr[li]
        nat_cov = set_min_metrics(native_tr[li], gt_i)[3]

        states = build_all_states_for_window(sid)
        for st in STATES:
            sel = states[st]["selected"]
            pairs = states[st]["pairs"]
            for r, a in pairs:
                X[cursor] = action_feature_113(feat40, D40, sel, r, a)
                I, S = labels_IS(pool, gt_i, sel, r, a, nat_cov)
                YI[cursor] = I
                YS[cursor] = S
                row_sid[cursor] = sid
                row_state[cursor] = state_to_i[st]
                row_rem[cursor] = r
                row_add[cursor] = a
                cursor += 1
        if (wi + 1) % 64 == 0 or (wi + 1) == len(chosen_local):
            print(f"  build rows {wi+1}/{len(chosen_local)} cursor={cursor}", flush=True)

    assert cursor == n_rows, (cursor, n_rows)
    np.save(X_path, X)
    np.save(YI_path, YI)
    np.save(YS_path, YS)
    np.savez_compressed(
        meta_path,
        sample_id=row_sid, state=row_state, rem=row_rem, add=row_add,
        chosen_local=chosen_local.astype(np.int64),
    )
    diag = {"selection": sel_info, "n_rows": n_rows, "outputs": {}}
    for arm, Y in (("I", YI), ("S", YS)):
        diag["outputs"][arm] = {}
        for j, name in enumerate(OUTPUT_NAMES):
            y = Y[:, j]
            diag["outputs"][arm][name] = dict(
                mean=float(y.mean()), std=float(y.std()),
                min=float(y.min()), max=float(y.max()),
                n_pos=int((y > 0).sum()), n_zero=int((y == 0).sum()),
                n_unique=int(len(np.unique(y))),
                is_constant=bool(len(np.unique(y)) <= 1),
            )
    for st, si in state_to_i.items():
        m = row_state == si
        diag[f"state_{st}"] = dict(
            n=int(m.sum()),
            I_L_pos=int((YI[m, 4] > 0).sum()),
            S_L_pos=int((YS[m, 4] > 0).sum()),
        )
    marker.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    return X, YI, YS, diag
