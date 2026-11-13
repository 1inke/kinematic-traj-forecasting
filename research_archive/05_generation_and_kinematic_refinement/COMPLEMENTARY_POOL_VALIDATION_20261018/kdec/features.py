"""条件特征（与 UNION_SELECTOR cond_ranker.build_cond_features 对齐）。"""
from __future__ import annotations
import numpy as np

DT = 0.4

FEATURE_NAMES = [
    "ep_x", "ep_y", "ep_n", "align", "lat", "along",
    "speed_mean", "speed_max", "speed_last",
    "accel_mean", "accel_max",
    "v1_align", "dv0", "plen", "to_mean", "nn_within",
    "family", "past_speed", "past_len", "v0_x", "v0_y",
    "cross_family_min_ep",
]


def build_cond_features(past, paths, family):
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


def build_pool_features(past, native, kin):
    B = native.shape[0]
    fam_n = np.zeros((B, native.shape[1]))
    fam_k = np.ones((B, kin.shape[1]))
    fn = build_cond_features(past, native, fam_n)
    fk = build_cond_features(past, kin, fam_k)
    ep_n = native[:, :, -1, :].astype(np.float64)
    ep_k = kin[:, :, -1, :].astype(np.float64)
    d = np.linalg.norm(ep_n[:, :, None, :] - ep_k[:, None, :, :], axis=-1)
    fn = np.concatenate([fn, d.min(axis=2, keepdims=True)], -1)
    fk = np.concatenate([fk, d.min(axis=1)[..., None]], -1)
    return fn, fk


def geometry_only_features(native, kin):
    """V2 用：去掉 past 派生通道及其交叉量。"""
    # keep: ep, ep_n, speed/accel stats, plen, nn_within, family, cross_family — drop align/lat/along/v1_align/dv0/past_*/v0
    B = native.shape[0]
    out = []
    for paths, fam_val in ((native, 0.0), (kin, 1.0)):
        paths = paths.astype(np.float64)
        ep = paths[:, :, -1, :]
        ep_n = np.linalg.norm(ep, axis=-1, keepdims=True)
        v = np.diff(paths, axis=2, prepend=np.zeros_like(paths[:, :, :1])) / DT
        speed = np.linalg.norm(v, axis=-1)
        a = np.diff(v, axis=2) / DT
        accel = np.linalg.norm(a, axis=-1)
        plen = np.linalg.norm(np.diff(paths, axis=2), axis=-1).sum(-1, keepdims=True)
        d = np.linalg.norm(ep[:, :, None, :] - ep[:, None, :, :], axis=-1)
        K = paths.shape[1]
        eye = np.eye(K, dtype=bool)
        d[:, eye] = np.inf
        nn = d.min(axis=-1, keepdims=True)
        fam = np.full((B, K, 1), fam_val, dtype=np.float64)
        feat = np.concatenate([
            ep, ep_n,
            speed.mean(-1, keepdims=True), speed.max(-1, keepdims=True), speed[:, :, -1:],
            accel.mean(-1, keepdims=True), accel.max(-1, keepdims=True),
            plen, nn, fam,
        ], -1)
        out.append(feat)
    fn, fk = out
    ep_n = native[:, :, -1, :].astype(np.float64)
    ep_k = kin[:, :, -1, :].astype(np.float64)
    d = np.linalg.norm(ep_n[:, :, None, :] - ep_k[:, None, :, :], axis=-1)
    fn = np.concatenate([fn, d.min(axis=2, keepdims=True)], -1)
    fk = np.concatenate([fk, d.min(axis=1)[..., None]], -1)
    return fn, fk
