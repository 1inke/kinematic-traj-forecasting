"""EXP-KINE-ANCHOR-ORACLE: data loading, groups and heading frame (NumPy only).

Functions marked COPIED are verbatim logic from the archived code so that windows, groups and
labels are identical to EXP-156 / FIVEFOLD / EXP-ETH-SPEED / EXP-DECEL-DIAG:
  research_archive/15_decel_diag/.../dependencies/16_fivefold_data.py
  research_archive/15_decel_diag/.../dependencies/17_fivefold_common.py
  research_archive/15_decel_diag/.../dependencies/00_eth_speed_math.py
  FRONT_ENCODER_SERIES/train_front_variant.normalize_to_focal_heading (torch -> numpy)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
OBS, PRED, MAX_AGENTS, K = 8, 12, 9, 20
DT = 0.4
VAL_FRACTION = 0.05
MANEUVERS = ('stationary', 'start', 'stop', 'turn_and_speed', 'turn',
             'acceleration', 'deceleration', 'steady')
SPEED_NAMES = ('low', 'middle', 'high')
CHANGE_NAMES = ('stable', 'transition', 'change')


# ----------------------------------------------------------------------------- data (COPIED)
def fold_files(data_dir, fold):
    d = Path(data_dir)
    return {k: d / f"{fold}_{k}.npy" for k in ("data_train", "num_train", "data_test", "num_test")}


def change_groups(past_focal, future_focal):
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
    files = fold_files(data_dir, fold)
    raw = np.load(files[f"data_{split}"], mmap_mode="r")
    num = np.load(files[f"num_{split}"], mmap_mode="r")
    if len(raw) != len(num):
        raise ValueError("DATA_NUM_LENGTH_MISMATCH")
    return to_dict(raw, num)


def subset(data, index):
    index = np.asarray(index)
    return {k: v[index] for k, v in data.items()}


# ----------------------------------------------------------------------------- labels (COPIED)
def observed_speed(past, dt=DT):
    p = np.asarray(past, np.float64)
    if p.ndim == 4:
        p = p[:, 0]
    if p.shape[1:] != (8, 2) or not np.isfinite(p).all() or dt <= 0:
        raise ValueError('BAD_OBSERVATIONS')
    return np.linalg.norm(np.diff(p, axis=1), axis=-1).mean(1) / dt


def fit_profile(past):
    v = observed_speed(past)
    p50, p90 = np.quantile(v, [.5, .9])
    if p90 <= .1:
        raise ValueError('TRAINING_SPEED_RANGE_TOO_SMALL')
    return dict(n=len(v), p50=float(p50), p90=float(p90),
                p95=float(np.quantile(v, .95)), maximum=float(v.max()))


def speed_bins(past, profile):
    return np.searchsorted([profile['p50'], profile['p90']], observed_speed(past), side='right')


def maneuver_labels(past, future):
    """Evaluation only. Labels use future."""
    p = np.asarray(past, np.float64)[:, 0]
    f = np.asarray(future, np.float64)
    obs = observed_speed(p)
    v = np.diff(np.concatenate([p[:, -1:], f], axis=1), axis=1) / DT
    fut = np.linalg.norm(v, axis=-1).mean(1)
    a = (p[:, -1] - p[:, -4]) / (3 * DT)
    b = (f[:, -1] - p[:, -1]) / (12 * DT)
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    angle = np.degrees(np.arccos(np.clip((a*b).sum(1) / np.maximum(norm, 1e-12), -1, 1)))
    angle[norm <= 1e-8] = 0
    ratio = fut / np.maximum(obs, .1)
    turn = angle >= 30
    acc, dec = ratio >= 1.25, ratio <= .8
    labels = np.full(len(p), 'steady', dtype='<U20')
    labels[dec] = 'deceleration'; labels[acc] = 'acceleration'
    labels[turn] = 'turn'; labels[turn & (acc | dec)] = 'turn_and_speed'
    labels[(obs >= .1) & (fut < .1)] = 'stop'
    labels[(obs < .1) & (fut >= .1)] = 'start'
    labels[(obs < .1) & (fut < .1)] = 'stationary'
    return labels


# ----------------------------------------------------------------------------- focal frame
HEADING_LAG = 3          # identical to normalize_to_focal_heading
HEADING_MIN_DISP = 0.03  # metres over the 3-step lag; below -> default heading (1, 0)


def focal_frame(past, future):
    """Heading-normalised focal future, exactly like B0 (normalize_to_focal_heading).

    Returns future_local [N,12,2] (metres, origin = last observed focal point, x = heading),
    recent speed v_rec [N] (m/s, |p[-1]-p[-4]| / (3*DT)), heading_valid [N] bool.
    """
    p = np.asarray(past, np.float64)
    if p.ndim == 4:
        p = p[:, 0]
    f = np.asarray(future, np.float64)
    disp = p[:, -1] - p[:, -1 - HEADING_LAG]
    norm = np.linalg.norm(disp, axis=-1)
    valid = norm > HEADING_MIN_DISP
    heading = np.where(valid[:, None], disp / np.maximum(norm, 1e-6)[:, None], np.array([1.0, 0.0]))
    c, s = heading[:, 0], heading[:, 1]
    rel = f - p[:, -1:, :]
    x = rel[..., 0] * c[:, None] + rel[..., 1] * s[:, None]
    y = -rel[..., 0] * s[:, None] + rel[..., 1] * c[:, None]
    return np.stack((x, y), axis=-1), norm / (HEADING_LAG * DT), valid
