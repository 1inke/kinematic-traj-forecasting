import os, sys, hashlib
from pathlib import Path
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]

# Frozen sources
CACHE = ROOT / 'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz'
LABELS = ROOT / 'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz'
META = ROOT / 'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz'
SPLITS = ROOT / 'GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz'
ENCODER = ROOT / 'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz'
H0_LABELS = ROOT / 'FAR_GOAL_SHORT_PATH_DESIGN_20261019/execution_20261019/H0_labels.npz'
STATS = ROOT / 'FAR_GOAL_SHORT_PATH_DESIGN_20261019/execution_20261019/shared_statistics.npz'

def load_data():
    """
    Loads all frozen inputs, labels, metadata, splits, and long horizons.
    """
    with np.load(CACHE, allow_pickle=False) as z:
        sample_id = z['sample_id'].copy()
        native = z['candidate'][..., :24].reshape(-1, 20, 12, 2).astype(np.float64)
        groups = z['change_group'].copy()
        logits = z['logits'].copy()

    with np.load(LABELS, allow_pickle=False) as z:
        assert np.array_equal(z['sample_id'], sample_id)
        gt = z['future'].astype(np.float64)

    with np.load(META, allow_pickle=False) as z:
        mid = z['sample_id']
        order = np.argsort(mid)
        loc = order[np.searchsorted(mid[order], sample_id)]
        assert np.array_equal(mid[loc], sample_id)
        seq = z['sequence'][loc]
        ped = z['pedestrian_id'][loc]
        track = np.array([f"{s}::{int(p)}" for s, p in zip(seq, ped)])
        scene = z['scene_family'][loc].copy()

    with np.load(ENCODER, allow_pickle=False) as z:
        assert np.array_equal(z['sample_id'], sample_id)
        p2 = z['tokens'].mean(1).astype(np.float32)
        past = z['past'].astype(np.float64) # (28791, 8, 2)
        v0 = (past[:, -1] - past[:, -2]) / 0.4 # (28791, 2)

    with np.load(SPLITS, allow_pickle=False) as sp:
        assert np.array_equal(sp['sample_id'], sample_id[:25912])
        split_dict = dict(
            FIT=sp['grouped_train_index'],
            outer=sp['outer_eval_index'],
            inner=sp['inner_cal_index'],
            original_CAL=np.arange(25912, 28791)
        )

    with np.load(H0_LABELS, allow_pickle=False) as hz:
        assert np.array_equal(hz['sample_id'], sample_id)
        y20 = hz['y20'].astype(np.float64)
        m20 = hz['m20'].astype(bool)

    stats = {}
    with np.load(STATS, allow_pickle=False) as sz:
        for k in sz.files:
            stats[k] = sz[k].copy()

    data = dict(
        sample_id=sample_id,
        ids=sample_id,
        past=past,
        native=native,
        gt=gt,
        groups=groups,
        logits=logits,
        track=track,
        scene=scene,
        p2=p2,
        v0=v0,
        y20=y20,
        m20=m20,
        split=split_dict,
        stats=stats
    )
    return data
