import os, sys, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz'
LABELS = ROOT / 'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz'
META = ROOT / 'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz'
SPLITS = ROOT / 'GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz'
ENCODER = ROOT / 'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz'

GROUPS = ('overall', 'stable', 'transition', 'change')
SIZES = dict(FIT=18909, outer=1035, inner=1856, original_CAL=2879)

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load_dataset():
    """
    Loads all frozen features, native trajectories, ground-truth labels, and metadata.
    Guarantees exact identity alignment and deterministic split indices.
    """
    with np.load(CACHE, allow_pickle=False) as z:
        sample_id = z['sample_id'].copy()
        native = z['candidate'][..., :24].reshape(-1, 20, 12, 2).astype(np.float64)
        groups = z['change_group'].copy()
        logits = z['logits'].copy()

    with np.load(LABELS, allow_pickle=False) as z:
        assert np.array_equal(z['sample_id'], sample_id), "Sample ID mismatch in labels"
        gt = z['future'].astype(np.float64)

    with np.load(META, allow_pickle=False) as z:
        mid = z['sample_id']
        order = np.argsort(mid)
        loc = order[np.searchsorted(mid[order], sample_id)]
        assert np.array_equal(mid[loc], sample_id), "Sample ID mismatch in metadata"
        seq = z['sequence'][loc]
        ped = z['pedestrian_id'][loc]
        track = np.array([f"{s}::{int(p)}" for s, p in zip(seq, ped)])
        scene = z['scene_family'][loc].copy()

    with np.load(ENCODER, allow_pickle=False) as z:
        assert np.array_equal(z['sample_id'], sample_id), "Sample ID mismatch in encoder"
        p2 = z['tokens'].mean(1).astype(np.float32)

    with np.load(SPLITS, allow_pickle=False) as sp:
        assert np.array_equal(sp['sample_id'], sample_id[:25912]), "Sample ID mismatch in splits"
        split_dict = dict(
            FIT=sp['grouped_train_index'],
            outer=sp['outer_eval_index'],
            inner=sp['inner_cal_index'],
            original_CAL=np.arange(25912, 28791)
        )

    assert len(sample_id) == 28791 and len(np.unique(sample_id)) == 28791
    assert p2.shape == (28791, 96) and native.shape == (28791, 20, 12, 2)
    assert gt.shape == (28791, 12, 2)

    for name, idx in split_dict.items():
        assert len(idx) == SIZES[name], f"Size mismatch for split {name}"
        assert len(np.unique(idx)) == len(idx), f"Duplicate indices in split {name}"

    data = dict(
        sample_id=sample_id,
        ids=sample_id,
        native=native,
        gt=gt,
        groups=groups,
        logits=logits,
        track=track,
        scene=scene,
        p2=p2,
        split=split_dict
    )
    return data

def compute_fit_statistics(data, representation='position'):
    """
    Computes input mean and standard deviation strictly on the FIT split.
    Feature: p2 (96) + candidate native (24) = 120.
    """
    fit_idx = data['split']['FIT']
    p2_fit = data['p2'][fit_idx].astype(np.float64)
    if representation == 'position':
        feat_fit = data['native'][fit_idx].reshape(-1, 24).astype(np.float64)
    elif representation == 'velocity':
        # diff across time: v[:, 0] = p[:, 0], v[:, t] = (p[:, t] - p[:, t-1]) / 0.4
        p = data['native'][fit_idx]
        v = np.diff(p, axis=2, prepend=np.zeros_like(p[:, :, :1])) / 0.4
        feat_fit = v.reshape(-1, 24).astype(np.float64)
    else:
        raise ValueError(f"Unknown representation: {representation}")

    # p2 is repeated across 20 slots, or normalized per slot:
    # In standard implementation (common.stats):
    # s1[:96] accumulated over B samples; s1[96:] accumulated over B*20 slots
    s1 = np.zeros(120, dtype=np.float64)
    s2 = np.zeros(120, dtype=np.float64)
    n = np.zeros(120, dtype=np.float64)

    batch_size = 256
    for i in range(0, len(fit_idx), batch_size):
        b_idx = fit_idx[i:i + batch_size]
        p_b = data['p2'][b_idx].astype(np.float64)
        if representation == 'position':
            v_b = data['native'][b_idx].reshape(-1, 24).astype(np.float64)
        else:
            p_traj = data['native'][b_idx]
            v_diff = np.diff(p_traj, axis=2, prepend=np.zeros_like(p_traj[:, :, :1])) / 0.4
            v_b = v_diff.reshape(-1, 24).astype(np.float64)

        s1[:96] += p_b.sum(0)
        s2[:96] += (p_b * p_b).sum(0)
        n[:96] += len(p_b)

        s1[96:] += v_b.sum(0)
        s2[96:] += (v_b * v_b).sum(0)
        n[96:] += len(v_b)

    mean = s1 / n
    variance = np.maximum(s2 / n - mean * mean, 0)
    std = np.sqrt(variance)
    std = np.where(std < 1e-6, 1.0, std)

    return dict(
        mean=mean,
        std=std,
        training_ids=data['sample_id'][fit_idx],
        counts=n
    )
