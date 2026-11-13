"""Load frozen pools / splits. No generator training."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import (
    CACHE, LABELS, SPLITS, ENCODER, SEL, CPV, CLAMP, SIZES, SEEDS, ROOT,
)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def array_sha(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    return h.hexdigest()


def load_bundle():
    with np.load(CACHE, allow_pickle=False) as z:
        sample_id = z["sample_id"].copy()
        native = z["candidate"][..., :24].reshape(-1, 20, 12, 2).astype(np.float64)
        groups = z["change_group"].copy()
    with np.load(LABELS, allow_pickle=False) as z:
        assert np.array_equal(z["sample_id"], sample_id), "LABELS sample_id mismatch"
        gt = z["future"].astype(np.float64)
    with np.load(ENCODER, allow_pickle=False) as z:
        assert np.array_equal(z["sample_id"], sample_id), "ENCODER sample_id mismatch"
        past = z["past"].astype(np.float64)
    assert np.allclose(past[:, -1], 0.0), "past origin not zero"
    sp = np.load(SPLITS, allow_pickle=False)
    assert np.array_equal(sp["sample_id"], sample_id[:25912]), "SPLITS id mismatch"
    split = dict(
        FIT=sp["grouped_train_index"].copy(),
        outer=sp["outer_eval_index"].copy(),
        inner=sp["inner_cal_index"].copy(),
        original_CAL=np.arange(25912, 28791),
    )
    for k, n in SIZES.items():
        assert len(split[k]) == n, f"{k} size {len(split[k])} != {n}"
    n_id = 25912
    seq = np.array(["?"] * len(sample_id), dtype=object)
    ped = np.full(len(sample_id), -1, dtype=np.int64)
    scene = np.array(["?"] * len(sample_id), dtype=object)
    seq[:n_id] = sp["sequence"]
    ped[:n_id] = sp["pedestrian_id"].astype(np.int64)
    scene[:n_id] = sp["scene_family"]
    track_key = np.array([f"{s}|{p}" for s, p in zip(seq, ped)], dtype=object)
    return dict(
        sample_id=sample_id, past=past, gt=gt, native=native, groups=groups,
        split=split, sequence=seq, pedestrian_id=ped, scene_family=scene,
        track_key=track_key,
    )


def load_fit_cal_split():
    p = CPV / "results" / "v1" / "FIT_CAL_SPLIT.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    # Reconstruct from same recipe as V1
    data = None  # caller may pass
    return rec, p


def make_fit_cal_indices(data, rec=None):
    """Reuse V1 FIT_train/FIT_cal by replaying saved recipe."""
    if rec is None:
        rec, _ = load_fit_cal_split()
    fit_ix = data["split"]["FIT"]
    tracks = data["track_key"][fit_ix]
    uniq = np.unique(tracks)
    rng = np.random.default_rng(int(rec["seed"]))
    rng.shuffle(uniq)
    n_cal = max(1, int(round(len(uniq) * float(rec["frac"]))))
    cal_tracks = set(uniq[:n_cal].tolist())
    is_cal = np.array([t in cal_tracks for t in tracks], dtype=bool)
    train_local = np.where(~is_cal)[0]
    cal_local = np.where(is_cal)[0]
    fit_train = fit_ix[train_local]
    fit_cal = fit_ix[cal_local]
    assert len(fit_train) == int(rec["n_windows_train"]), (
        f"FIT_train {len(fit_train)} != {rec['n_windows_train']}"
    )
    assert len(fit_cal) == int(rec["n_windows_cal"]), (
        f"FIT_cal {len(fit_cal)} != {rec['n_windows_cal']}"
    )
    return fit_train, fit_cal, rec


def load_kin(split_name: str, seed: int, data, fit_train=None, fit_cal=None):
    """Load KIN paths aligned to split indices. No training."""
    art = CPV / "artifacts" / "v1"
    if split_name == "outer":
        cache = SEL / "results" / "cache_pred" / f"kin_outer_seed{seed}.npy"
        assert cache.exists(), f"missing {cache}"
        kin = np.load(cache)
        assert kin.shape[0] == len(data["split"]["outer"])
        return kin.astype(np.float64), str(cache)
    if split_name == "inner":
        cache = art / f"kin_inner_seed{seed}.npy"
        assert cache.exists(), f"missing {cache}"
        kin = np.load(cache)
        assert kin.shape[0] == len(data["split"]["inner"])
        return kin.astype(np.float64), str(cache)
    if split_name == "FIT":
        cache = SEL / "results" / "cache_pred" / f"kin_FIT_seed{seed}.npy"
        assert cache.exists(), f"missing {cache}"
        kin = np.load(cache)
        assert kin.shape[0] == len(data["split"]["FIT"])
        return kin.astype(np.float64), str(cache)
    if split_name in ("FIT_train", "FIT_cal"):
        kin_fit, src = load_kin("FIT", seed, data)
        fit_ix = data["split"]["FIT"]
        pos = {int(g): i for i, g in enumerate(fit_ix)}
        ix = fit_train if split_name == "FIT_train" else fit_cal
        rows = [pos[int(g)] for g in ix]
        return kin_fit[rows], src
    if split_name == "original_CAL":
        # May be BLOCKED if no cache
        candidates = [
            art / f"kin_original_CAL_seed{seed}.npy",
            SEL / "results" / "cache_pred" / f"kin_original_CAL_seed{seed}.npy",
            SEL / "results" / "cache_pred" / f"kin_CAL_seed{seed}.npy",
        ]
        for c in candidates:
            if c.exists():
                kin = np.load(c)
                return kin.astype(np.float64), str(c)
        return None, "BLOCKED_NO_KIN_CACHE"
    raise ValueError(split_name)


def load_v1_selected_indices(split_name: str, seed: int):
    """V1 cond_boost_11_9 selected pool indices from bundle."""
    p = CPV / "artifacts" / "v1" / f"{split_name}_seed{seed}_bundle.npz"
    if not p.exists():
        return None, str(p)
    with np.load(p, allow_pickle=False) as z:
        idx = z["selected_pool_indices"].astype(np.int64)
        ids = z["sample_id"].copy()
    return idx, str(p)
