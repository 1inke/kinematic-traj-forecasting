"""Official training labels only. Target eligibility and neighbor input are separate."""
from collections import defaultdict
import hashlib
import numpy as np
from common import DT

RECORDINGS = {
    'biwi_hotel': 1, 'crowds_zara01': 3, 'crowds_zara02': 4,
    'crowds_zara03': 3, 'students001': 2, 'students003': 2,
    'uni_examples': 2,
}


def load_recording(path):
    table = np.genfromtxt(path, dtype=str)
    time = table[:, 0].astype(np.float32).astype(np.int64)
    target = table[:, 1].astype(np.float32).astype(np.int64)
    xy = table[:, [13, 15]].astype(np.float32)
    if not np.isfinite(xy).all():
        raise ValueError('NONFINITE_RAW_POSITION')
    frames = defaultdict(dict)
    tracks = defaultdict(dict)
    for t, ped, point in zip(time, target, xy):
        if int(ped) in frames[int(t)]:
            raise ValueError('DUPLICATE_RECORDING_TIME_ID')
        frames[int(t)][int(ped)] = point.copy()
        tracks[int(ped)][int(t)] = point.copy()
    return dict(frames), dict(tracks), int(time.min()), int(time.max())


def windows(tracks):
    for ped in sorted(tracks):
        track = tracks[ped]
        for start in sorted(track):
            if all(start + j in track for j in range(20)):
                yield ped, start, np.stack([track[start + j] for j in range(20)])


def past_input(frames, target, t0):
    # Requiring all eight OBSERVED samples is causal. No lookup after t0.
    past_frames = [frames[t] for t in range(t0 - 7, t0 + 1)]
    visible = set(past_frames[0])
    for frame in past_frames[1:]:
        visible.intersection_update(frame)
    if target not in visible:
        raise ValueError('TARGET_OBSERVATION_MISSING')
    focal = frames[t0][target]
    neighbors = sorted(visible - {target},
                       key=lambda ped: (float(np.linalg.norm(frames[t0][ped] - focal)), ped))
    # Closest eight full-observation neighbors, no future-life criterion or GT gate.
    chosen = [target] + neighbors[:8]
    past = np.zeros((9, 8, 2), dtype=np.float32)
    ids = np.full(9, -1, dtype=np.int64)
    for i, ped in enumerate(chosen):
        past[i] = np.stack([f[ped] for f in past_frames])
        ids[i] = ped
    return past, ids, len(chosen), len(neighbors)


def path_hash(scene, path):
    return hashlib.sha256(bytes([scene]) + np.asarray(path, dtype='<f4').tobytes()).hexdigest()


def groups(past, future):
    # Same fixed historical score/thresholds, evaluated in float64 world coordinates.
    obs = past[:, -1].astype(np.float64) - past[:, -2]
    fut = (future[:, -1].astype(np.float64) - past[:, -1]) / 12
    a = np.linalg.norm(obs, axis=-1)
    b = np.linalg.norm(fut, axis=-1)
    prod = a * b
    cos = np.where(prod > 1e-6, np.sum(obs * fut, axis=-1) / np.maximum(prod, 1e-6), 1)
    turn = np.clip((1 - np.clip(cos, -1, 1)) / 2, 0, 1)
    speed = np.clip(np.abs(np.log((b + .05) / (a + .05))) / np.log(2), 0, 1)
    score = np.maximum(turn, speed)
    return np.where(score < .25, 0, np.where(score >= .5, 2, 1)), score


def motion(past, future):
    """Future-only 12v/11a/10j plus explicit past/future junction."""
    past, future = np.asarray(past, np.float64), np.asarray(future, np.float64)
    v = np.diff(np.concatenate([past[..., -1:, :], future], axis=-2), axis=-2) / DT
    a = np.diff(v, axis=-2) / DT
    j = np.diff(a, axis=-2) / DT
    pv = (past[..., -1, :] - past[..., -2, :]) / DT
    pa = (past[..., -1, :] - 2 * past[..., -2, :] + past[..., -3, :]) / DT**2
    ja = (v[..., 0, :] - pv) / DT
    jj = (ja - pa) / DT
    return {'speed': np.linalg.norm(v, axis=-1),
            'acceleration': np.linalg.norm(a, axis=-1),
            'jerk': np.linalg.norm(j, axis=-1),
            'junction_acceleration': np.linalg.norm(ja, axis=-1)[..., None],
            'junction_jerk': np.linalg.norm(jj, axis=-1)[..., None]}
