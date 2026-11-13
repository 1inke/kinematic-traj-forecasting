"""Feature extraction and frozen model adapter for Q0 Action Identifiability Diagnostic.
Pure functional extraction; strictly isolated from original code execution.
"""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np

EXEC_DIR = Path(__file__).resolve().parent
ROOT = EXEC_DIR.parent

DT = 0.4
FEATURE_DIM = 113
N_NAT = 20
N_KIN = 20
PRED_L_MAX = 0.01
PRED_DA_MIN = 0.0
PRED_DF_MIN = 0.0


def build_cond_features_22(past: np.ndarray, paths: np.ndarray, family: np.ndarray) -> np.ndarray:
    """Computes 22-d candidate condition features for a batch of candidates."""
    B, K, T, _ = paths.shape
    past = past.astype(np.float64)
    paths = paths.astype(np.float64)
    v0 = (past[:, -1] - past[:, -2]) / DT
    past_speed = np.linalg.norm(v0, axis=-1, keepdims=True)
    past_len = np.linalg.norm(np.diff(past, axis=1), axis=-1).sum(-1, keepdims=True)
    ep = paths[:, :, -1, :]
    ep_n = np.linalg.norm(ep, axis=-1, keepdims=True)
    v0n = v0 / (np.linalg.norm(v0, axis=-1, keepdims=True) + 1e-6)
    ep_u = ep / (ep_n + 1e-6)
    align = (ep_u * v0n[:, None, :]).sum(-1, keepdims=True)
    v = np.diff(paths, axis=2, prepend=np.zeros_like(paths[:, :, :1])) / DT
    speed = np.linalg.norm(v, axis=-1)
    a = np.diff(v, axis=2) / DT
    accel = np.linalg.norm(a, axis=-1)
    v1 = v[:, :, 0, :]
    v1_align = (v1 * v0n[:, None, :]).sum(-1, keepdims=True) / (
        np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-6
    )
    dv0 = np.linalg.norm(v1 - v0[:, None, :], axis=-1, keepdims=True)
    perp = np.stack([-v0n[:, 1], v0n[:, 0]], axis=-1)
    lat = (ep * perp[:, None, :]).sum(-1, keepdims=True)
    along = (ep * v0n[:, None, :]).sum(-1, keepdims=True)
    mean_ep = ep.mean(1, keepdims=True)
    to_mean = np.linalg.norm(ep - mean_ep, axis=-1, keepdims=True)
    plen = np.linalg.norm(np.diff(paths, axis=2), axis=-1).sum(-1, keepdims=True)
    fam_f = family.astype(np.float64)[..., None]
    ps = np.broadcast_to(past_speed[:, None, :], (B, K, 1)).copy()
    pl = np.broadcast_to(past_len[:, None, :], (B, K, 1)).copy()
    d = np.linalg.norm(ep[:, :, None, :] - ep[:, None, :, :], axis=-1)
    eye = np.eye(K, dtype=bool)
    d[:, eye] = np.inf
    nn = d.min(axis=-1, keepdims=True)
    return np.concatenate([
        ep, ep_n, align, lat, along,
        speed.mean(-1, keepdims=True), speed.max(-1, keepdims=True), speed[:, :, -1:],
        accel.mean(-1, keepdims=True), accel.max(-1, keepdims=True),
        v1_align, dv0, plen, to_mean, nn, fam_f, ps, pl,
        np.broadcast_to(v0[:, None, :], (B, K, 2)).copy(),
    ], axis=-1)


def build_pool_features_22(past: np.ndarray, native: np.ndarray, kin: np.ndarray) -> np.ndarray:
    """Builds 22-d features for pool of 40 candidates."""
    B = native.shape[0]
    fam_n = np.zeros((B, native.shape[1]))
    fam_k = np.ones((B, kin.shape[1]))
    fn = build_cond_features_22(past, native, fam_n)
    fk = build_cond_features_22(past, kin, fam_k)
    ep_n = native[:, :, -1, :].astype(np.float64)
    ep_k = kin[:, :, -1, :].astype(np.float64)
    d = np.linalg.norm(ep_n[:, :, None, :] - ep_k[:, None, :, :], axis=-1)
    fn = np.concatenate([fn, d.min(axis=2, keepdims=True)], -1)
    fk = np.concatenate([fk, d.min(axis=1)[..., None]], -1)
    return np.concatenate([fn, fk], axis=1)


def pairwise_D_matrix(pool40_i: np.ndarray) -> np.ndarray:
    """Pairwise D metric between candidates: D = max(meanpath/0.10, endpoint/0.20)."""
    p = pool40_i.astype(np.float64)
    d = np.linalg.norm(p[:, None, :, :] - p[None, :, :, :], axis=-1)
    meanpath = d.mean(-1)
    endpoint = d[:, :, -1]
    return np.maximum(meanpath / 0.10, endpoint / 0.20)


def action_feature_113(feat40: np.ndarray, D40: np.ndarray, selected: list | np.ndarray, rem: int, add: int) -> np.ndarray:
    """Constructs 113-d action feature for swapping `rem` with `add`."""
    sel = np.asarray(selected, dtype=np.int64)
    fr = feat40[rem]
    fl = feat40[add]
    fs = feat40[sel]
    mean = fs.mean(0)
    amin = fs.min(0)
    amax = fs.max(0)
    native_ratio = float(np.mean(sel < N_NAT))
    others = sel[sel != rem]
    if others.size == 0:
        d_r = 0.0
        d_l = 0.0
    else:
        d_r = float(D40[rem, others].min())
        d_l = float(D40[add, others].min())
    out = np.concatenate([
        fr, fl, mean, amin, amax,
        np.array([native_ratio, d_r, d_l], dtype=np.float64),
    ])
    assert out.shape == (FEATURE_DIM,), f"Feature dimension mismatch: {out.shape}"
    return out


def batch_action_features(feat40: np.ndarray, D40: np.ndarray, selected: list | np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.zeros((0, FEATURE_DIM), dtype=np.float64)
    return np.stack([action_feature_113(feat40, D40, selected, r, a) for r, a in pairs], axis=0)


def load_frozen_models(seed: int) -> dict[str, joblib.Memory]:
    targets = ["dA", "dF", "dJ", "dC", "L"]
    models = {}
    for tgt in targets:
        p = ROOT / f"CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018/k1_exec/seed{seed}/models/S_{tgt}_seed{seed}.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Model missing: {p}")
        models[tgt] = joblib.load(p)
    return models


def predict_actions(models: dict, X: np.ndarray) -> np.ndarray:
    """Runs prediction for [dA, dF, dJ, dC, L].
    Returns shape (N, 5).
    """
    if X.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float64)
    preds = [
        models["dA"].predict(X),
        models["dF"].predict(X),
        models["dJ"].predict(X),
        models["dC"].predict(X),
        models["L"].predict(X),
    ]
    return np.column_stack(preds)
