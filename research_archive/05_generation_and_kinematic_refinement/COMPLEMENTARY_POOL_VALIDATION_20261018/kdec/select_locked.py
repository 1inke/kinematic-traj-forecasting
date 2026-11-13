"""锁定主方法 cond_boost_11_9 与参照；稳定并列规则。"""
from __future__ import annotations
import numpy as np

N_NAT = 11
N_KIN = 9
BOOST = 0.35


def stable_topk(scores: np.ndarray, k: int) -> np.ndarray:
    """分数降序；并列按物理 slot 升序。"""
    slots = np.arange(len(scores), dtype=np.int64)
    order = np.lexsort((slots, -scores.astype(np.float64)))
    return order[:k].astype(np.int64)


def gather(pool: np.ndarray, idx: np.ndarray) -> np.ndarray:
    B, K = idx.shape
    out = np.empty((B, K) + pool.shape[2:], dtype=pool.dtype)
    for b in range(B):
        out[b] = pool[b, idx[b]]
    return out


def select_cond_boost_11_9(native, kin, sn, sk, boost=BOOST, n_nat=N_NAT, n_kin=N_KIN):
    """返回 paths, pool_indices(0..39), idx_n, idx_k, sk_boost."""
    B = native.shape[0]
    ep_n = native[:, :, -1, :].astype(np.float64)
    ep_k = kin[:, :, -1, :].astype(np.float64)
    idx_n = np.zeros((B, n_nat), dtype=np.int64)
    idx_k = np.zeros((B, n_kin), dtype=np.int64)
    sk_boost = np.zeros_like(sk, dtype=np.float64)
    for b in range(B):
        idx_n[b] = stable_topk(sn[b], n_nat)
        dmin = np.linalg.norm(
            ep_k[b][:, None, :] - ep_n[b, idx_n[b]][None, :, :], axis=-1
        ).min(axis=1)
        sk_boost[b] = sk[b] * (boost + dmin)
        idx_k[b] = stable_topk(sk_boost[b], n_kin)
    paths = np.concatenate([gather(native, idx_n), gather(kin, idx_k)], axis=1)
    pool_idx = np.concatenate([idx_n, idx_k + native.shape[1]], axis=1)
    return paths, pool_idx, idx_n, idx_k, sk_boost


def select_cond_top_11_9(native, kin, sn, sk, n_nat=N_NAT, n_kin=N_KIN):
    B = native.shape[0]
    idx_n = np.zeros((B, n_nat), dtype=np.int64)
    idx_k = np.zeros((B, n_kin), dtype=np.int64)
    for b in range(B):
        idx_n[b] = stable_topk(sn[b], n_nat)
        idx_k[b] = stable_topk(sk[b], n_kin)
    paths = np.concatenate([gather(native, idx_n), gather(kin, idx_k)], axis=1)
    pool_idx = np.concatenate([idx_n, idx_k + native.shape[1]], axis=1)
    return paths, pool_idx, idx_n, idx_k


def select_cond_boost_natunique(native, kin, sn, sk, boost=BOOST, n_nat=N_NAT, n_kin=N_KIN, uniq_eps=0.15):
    """自改进：native 分数 × (eps + 到其他 native 的最小端点距)，优先保留独特支撑。"""
    B = native.shape[0]
    ep_n = native[:, :, -1, :].astype(np.float64)
    ep_k = kin[:, :, -1, :].astype(np.float64)
    idx_n = np.zeros((B, n_nat), dtype=np.int64)
    idx_k = np.zeros((B, n_kin), dtype=np.int64)
    sn_u = sn.astype(np.float64).copy()
    for b in range(B):
        d = np.linalg.norm(ep_n[b][:, None, :] - ep_n[b][None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        sn_u[b] = sn[b] * (uniq_eps + d.min(axis=1))
        idx_n[b] = stable_topk(sn_u[b], n_nat)
        dmin = np.linalg.norm(
            ep_k[b][:, None, :] - ep_n[b, idx_n[b]][None, :, :], axis=-1
        ).min(axis=1)
        sk_b = sk[b] * (boost + dmin)
        idx_k[b] = stable_topk(sk_b, n_kin)
    paths = np.concatenate([gather(native, idx_n), gather(kin, idx_k)], axis=1)
    pool_idx = np.concatenate([idx_n, idx_k + native.shape[1]], axis=1)
    return paths, pool_idx, idx_n, idx_k, sn_u


def select_cond_boost_kinsafe(native, kin, sn, sk, boost=BOOST, n_nat=N_NAT, n_kin=N_KIN, speed_factor=1.01):
    """自改进：先按路径均速相对 native 池剔除过快 KIN，再 boost 选。"""
    B = native.shape[0]
    # per-path mean speed
    def mean_speed(p):
        v = np.diff(p.astype(np.float64), axis=2, prepend=np.zeros_like(p[:, :, :1])) / 0.4
        return np.linalg.norm(v, axis=-1).mean(-1)

    sn_spd = mean_speed(native)
    sk_spd = mean_speed(kin)
    ep_n = native[:, :, -1, :].astype(np.float64)
    ep_k = kin[:, :, -1, :].astype(np.float64)
    idx_n = np.zeros((B, n_nat), dtype=np.int64)
    idx_k = np.zeros((B, n_kin), dtype=np.int64)
    for b in range(B):
        thr = float(np.quantile(sn_spd[b], 0.95)) * speed_factor
        idx_n[b] = stable_topk(sn[b], n_nat)
        sk_b = sk[b].astype(np.float64).copy()
        sk_b[sk_spd[b] > thr + 1e-12] = -1e9
        dmin = np.linalg.norm(
            ep_k[b][:, None, :] - ep_n[b, idx_n[b]][None, :, :], axis=-1
        ).min(axis=1)
        sk_b = sk_b * (boost + dmin)
        # if too few valid, fall back
        if (sk_b > -1e8).sum() < n_kin:
            sk_b = sk[b] * (boost + dmin)
        idx_k[b] = stable_topk(sk_b, n_kin)
    paths = np.concatenate([gather(native, idx_n), gather(kin, idx_k)], axis=1)
    pool_idx = np.concatenate([idx_n, idx_k + native.shape[1]], axis=1)
    return paths, pool_idx, idx_n, idx_k
