"""Identity and cohort sampling for Q0 Action Identifiability Diagnostic.
Strictly follows 01_Q0_实验设计.md Section 2 & 3.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

EXEC_DIR = Path(__file__).resolve().parent
ROOT = EXEC_DIR.parent

CACHE_PATH = ROOT / "REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz"
SPLITS_PATH = ROOT / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz"
EXPECTED_CAL_SHA = "2241ddb39e50ea22100c4065a60d9914fe052de0838467172dc4d7973f466c49"


def array_sha(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    return h.hexdigest()


def load_identities():
    with np.load(CACHE_PATH, allow_pickle=False) as z:
        sample_id = z["sample_id"].copy()
        change_group = z["change_group"].copy()

    sp = np.load(SPLITS_PATH, allow_pickle=False)
    fit_ix = sp["grouped_train_index"].copy()
    seq = sp["sequence"]
    ped = sp["pedestrian_id"].astype(np.int64)
    scene = sp["scene_family"]
    loss_group = sp["loss_change_group"]

    track_key = np.array([f"{s}|{p}" for s, p in zip(seq, ped)], dtype=object)

    return {
        "sample_id": sample_id,
        "change_group": change_group,
        "fit_ix": fit_ix,
        "track_key": track_key,
        "scene_family": scene,
        "loss_change_group": loss_group,
    }


def reconstruct_fit_cal_split(fit_ix: np.ndarray, track_key: np.ndarray):
    tracks = track_key[fit_ix]
    uniq = np.unique(tracks)
    assert len(uniq) == 689, f"Expected 689 tracks in FIT, got {len(uniq)}"

    rng = np.random.default_rng(20261018)
    rng.shuffle(uniq)

    n_cal = max(1, int(round(len(uniq) * 0.15)))
    assert n_cal == 103, f"Expected 103 cal tracks, got {n_cal}"

    cal_tracks_set = set(uniq[:n_cal].tolist())
    cal_arr = np.array(sorted(cal_tracks_set), dtype=object).astype(str)
    cal_sha = array_sha(cal_arr)
    assert cal_sha == EXPECTED_CAL_SHA, f"cal_track_list_sha mismatch: {cal_sha} vs {EXPECTED_CAL_SHA}"

    is_cal = np.array([t in cal_tracks_set for t in tracks], dtype=bool)
    fit_train_idx = fit_ix[~is_cal]
    fit_cal_idx = fit_ix[is_cal]

    assert len(fit_train_idx) == 15951, f"Expected 15951 train windows, got {len(fit_train_idx)}"
    assert len(fit_cal_idx) == 2958, f"Expected 2958 cal windows, got {len(fit_cal_idx)}"

    return fit_train_idx, fit_cal_idx, cal_tracks_set


def draw_cohort(cohort_name: str, candidate_sids: list, sample_to_idx: dict, track_key: np.ndarray, max_w: int = 128):
    by_track: dict[str, list] = {}
    for sid in candidate_sids:
        tk = str(track_key[sample_to_idx[sid]])
        by_track.setdefault(tk, []).append(sid)

    sorted_tracks = sorted(
        by_track.keys(),
        key=lambda tk: (hashlib.sha256(f"q0-track-v1|{cohort_name}|{tk}".encode()).hexdigest(), tk)
    )

    for tk in sorted_tracks:
        by_track[tk].sort(
            key=lambda sid: (hashlib.sha256(f"q0-window-v1|{cohort_name}|{sid}".encode()).hexdigest(), str(sid))
        )

    selected = []
    round_idx = 0
    while len(selected) < max_w:
        added_this_round = 0
        for tk in sorted_tracks:
            if round_idx < len(by_track[tk]):
                selected.append(by_track[tk][round_idx])
                added_this_round += 1
                if len(selected) == max_w:
                    break
        if added_this_round == 0:
            break
        round_idx += 1

    sel_tracks = {str(track_key[sample_to_idx[sid]]) for sid in selected}
    return selected, len(sel_tracks)


def main():
    ident = load_identities()
    sample_id = ident["sample_id"]
    track_key = ident["track_key"]
    fit_ix = ident["fit_ix"]
    sample_to_idx = {sid: i for i, sid in enumerate(sample_id)}

    fit_train_idx, fit_cal_idx, cal_tracks_set = reconstruct_fit_cal_split(fit_ix, track_key)
    fit_train_samples = set(sample_id[fit_train_idx])
    fit_cal_samples = set(sample_id[fit_cal_idx])

    # Verify identical seen samples across seed 42, 43, 44
    seen_ids_by_seed = {}
    for s in (42, 43, 44):
        rows_p = ROOT / f"CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018/k1_exec/seed{s}/train_data/rows_seed{s}.npz"
        rows = np.load(rows_p, allow_pickle=True)
        seen_ids_by_seed[s] = np.unique(rows["sample_id"])

    assert np.array_equal(seen_ids_by_seed[42], seen_ids_by_seed[43]), "seed42 vs seed43 training IDs differ"
    assert np.array_equal(seen_ids_by_seed[42], seen_ids_by_seed[44]), "seed42 vs seed44 training IDs differ"

    seen_samples = set(seen_ids_by_seed[42])
    assert len(seen_samples) == 4096, f"Expected 4096 seen samples, got {len(seen_samples)}"
    assert seen_samples.issubset(fit_train_samples), "Seen samples not a subset of FIT_train"

    # T_seen tracks
    t_seen = {str(track_key[sample_to_idx[sid]]) for sid in seen_samples}

    # C_same candidate pool
    same_samples = [
        sid for sid in fit_train_samples
        if sid not in seen_samples and str(track_key[sample_to_idx[sid]]) in t_seen
    ]

    # Partition FIT_cal tracks into calA (51 tracks) and calB (52 tracks)
    cal_tracks_sorted = sorted(
        list(cal_tracks_set),
        key=lambda tk: (hashlib.sha256(f"q0-cal-track-v1|{tk}".encode()).hexdigest(), str(tk))
    )
    n_a = len(cal_tracks_sorted) // 2
    cal_a_tracks = set(cal_tracks_sorted[:n_a])
    cal_b_tracks = set(cal_tracks_sorted[n_a:])
    assert len(cal_a_tracks) == 51 and len(cal_b_tracks) == 52

    cal_a_samples = [sid for sid in fit_cal_samples if str(track_key[sample_to_idx[sid]]) in cal_a_tracks]
    cal_b_samples = [sid for sid in fit_cal_samples if str(track_key[sample_to_idx[sid]]) in cal_b_tracks]

    pools = {
        "C_seen": list(seen_samples),
        "C_same": same_samples,
        "C_calA": cal_a_samples,
        "C_calB": cal_b_samples,
    }

    cohort_selected = {}
    cohort_stats = {}

    for c_name, pool in pools.items():
        sel_sids, n_tracks = draw_cohort(c_name, pool, sample_to_idx, track_key, max_w=128)
        assert len(sel_sids) >= 64, f"{c_name} has fewer than 64 windows: {len(sel_sids)}"
        assert n_tracks >= 30, f"{c_name} has fewer than 30 tracks: {n_tracks}"

        indices = np.array([sample_to_idx[sid] for sid in sel_sids], dtype=np.int64)
        tracks_in_sel = [str(track_key[idx]) for idx in indices]
        groups_in_sel = [int(ident["change_group"][idx]) for idx in indices]

        cohort_selected[c_name] = {
            "sample_ids": np.array(sel_sids, dtype=np.int64),
            "indices": indices,
            "track_keys": np.array(tracks_in_sel, dtype=object),
        }

        cohort_stats[c_name] = {
            "pool_size": len(pool),
            "selected_windows": len(sel_sids),
            "selected_tracks": n_tracks,
            "unique_tracks_ratio": float(n_tracks / len(sel_sids)),
            "change_group_counts": {str(g): int(np.sum(np.array(groups_in_sel) == g)) for g in range(4)},
        }

    # Save COHORT_IDS.npz
    np.savez_compressed(
        EXEC_DIR / "COHORT_IDS.npz",
        C_seen_ids=cohort_selected["C_seen"]["sample_ids"],
        C_seen_idx=cohort_selected["C_seen"]["indices"],
        C_seen_tracks=cohort_selected["C_seen"]["track_keys"],
        C_same_ids=cohort_selected["C_same"]["sample_ids"],
        C_same_idx=cohort_selected["C_same"]["indices"],
        C_same_tracks=cohort_selected["C_same"]["track_keys"],
        C_calA_ids=cohort_selected["C_calA"]["sample_ids"],
        C_calA_idx=cohort_selected["C_calA"]["indices"],
        C_calA_tracks=cohort_selected["C_calA"]["track_keys"],
        C_calB_ids=cohort_selected["C_calB"]["sample_ids"],
        C_calB_idx=cohort_selected["C_calB"]["indices"],
        C_calB_tracks=cohort_selected["C_calB"]["track_keys"],
    )

    audit_report = {
        "tag": "q0_identity_and_cohort_audit",
        "fit_train_total_windows": len(fit_train_idx),
        "fit_cal_total_windows": len(fit_cal_idx),
        "cal_tracks_sha": EXPECTED_CAL_SHA,
        "cal_tracks_count": len(cal_tracks_set),
        "cal_a_tracks_count": len(cal_a_tracks),
        "cal_b_tracks_count": len(cal_b_tracks),
        "seen_samples_count": len(seen_samples),
        "t_seen_tracks_count": len(t_seen),
        "cohorts": cohort_stats,
    }

    with (EXEC_DIR / "IDENTITY_AUDIT.json").open("w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    print("Cohorts constructed successfully:")
    for k, st in cohort_stats.items():
        print(f"  {k}: {st['selected_windows']} windows across {st['selected_tracks']} tracks (pool={st['pool_size']})")


if __name__ == "__main__":
    main()
