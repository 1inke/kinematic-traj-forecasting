"""processed_data_diverse fold arrays -> the data dict used by NEXT_PHASE train_core.Trainer/infer.

Raw layout (as read by FRONT_ENCODER_SERIES/train_front_variant.TrajectoryDataset):
  data [N, 9, 20, 6]: agent 0 is the focal target; coordinates are the LAST two features
  (TRAJ_SCALE = 1, metres); scene id of a sample is data[:, 0, 0, 0].
  num  [N]: valid agents.
Output dict (same keys as C0):  past [N,9,8,2] float32, future [N,12,2] float32 (focal only),
  num_valid [N] int64 (min(num,9)), ids [N] int64 (row index), scene [N] int64,
  group [N] int8 (0 stable / 1 transition / 2 change, C0 formula; evaluation only).
"""
from __future__ import annotations

import numpy as np

from fivefold_common import MAX_AGENTS, OBS, PRED, VAL_FRACTION, fold_files


def change_groups(past_focal, future_focal):
    """Same formula/thresholds as NEXT_PHASE causal_dataset.groups (world float64)."""
    obs = past_focal[:, -1].astype(np.float64) - past_focal[:, -2]
    fut = (future_focal[:, -1].astype(np.float64) - past_focal[:, -1]) / 12
    a, b = np.linalg.norm(obs, axis=-1), np.linalg.norm(fut, axis=-1)
    prod = a * b
    cos = np.where(prod > 1e-6, np.sum(obs * fut, axis=-1) / np.maximum(prod, 1e-6), 1)
    turn = np.clip((1 - np.clip(cos, -1, 1)) / 2, 0, 1)
    speed = np.clip(np.abs(np.log((b + .05) / (a + .05))) / np.log(2), 0, 1)
    score = np.maximum(turn, speed)
    return np.where(score < .25, 0, np.where(score >= .5, 2, 1)).astype(np.int8), score


def to_dict(raw, num):
    raw = np.asarray(raw)
    if raw.ndim != 4 or raw.shape[1:] != (MAX_AGENTS, OBS + PRED, 6):
        raise ValueError(f"UNEXPECTED_SHAPE {raw.shape}")
    xy = np.asarray(raw[..., -2:], dtype=np.float32)
    past = np.ascontiguousarray(xy[:, :, :OBS])
    future = np.ascontiguousarray(xy[:, 0, OBS:OBS + PRED])
    if not (np.isfinite(past[:, 0]).all() and np.isfinite(future).all()):
        raise ValueError("NONFINITE_FOCAL")
    num_valid = np.minimum(np.asarray(num, dtype=np.int64), MAX_AGENTS)
    group, score = change_groups(past[:, 0], future)
    return dict(past=past, future=future, num_valid=num_valid, ids=np.arange(len(raw), dtype=np.int64),
                scene=np.asarray(raw[:, 0, 0, 0], dtype=np.int64), group=group, change_score=score)


def scene_tail_split(scene, fraction=VAL_FRACTION):
    """Exact replica of train_front_variant --validation-split-mode scene_tail:
    per scene (np.unique order), the last round(len*fraction) row indices (Python round),
    clipped to [1, len-1], become validation."""
    scene = np.asarray(scene)
    parts = []
    for s in np.unique(scene):
        cand = np.flatnonzero(scene == s)
        if len(cand) < 2:
            raise ValueError(f"scene {int(s)} has fewer than two training samples")
        n = max(1, min(len(cand) - 1, int(round(len(cand) * fraction))))
        parts.append(cand[-n:])
    val = np.sort(np.concatenate(parts))
    mask = np.zeros(len(scene), bool); mask[val] = True
    return np.flatnonzero(~mask), val


def load_split(data_dir, fold, split):
    """split: 'train' or 'test'. Test is only ever loaded by the final phase."""
    files = fold_files(data_dir, fold)
    raw = np.load(files[f"data_{split}"], mmap_mode="r")
    num = np.load(files[f"num_{split}"], mmap_mode="r")
    if len(raw) != len(num):
        raise ValueError("DATA_NUM_LENGTH_MISMATCH")
    return to_dict(raw, num)


def subset(data, index):
    index = np.asarray(index)
    return {k: v[index] for k, v in data.items()}
