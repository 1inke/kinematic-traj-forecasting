"""GT-free and oracle selectors: K40 pool -> K20."""
from __future__ import annotations
import numpy as np

DT = 0.4


def endpoints(paths: np.ndarray) -> np.ndarray:
    return paths[:, :, -1, :].astype(np.float64)


def path_speed_mean(paths: np.ndarray) -> np.ndarray:
    v = np.diff(paths.astype(np.float64), axis=2, prepend=np.zeros_like(paths[:, :, :1])) / DT
    return np.linalg.norm(v, axis=-1).mean(-1)


def gather(pool: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """pool (B,M,T,2), idx (B,K) -> (B,K,T,2)"""
    B, K = idx.shape
    out = np.empty((B, K) + pool.shape[2:], dtype=pool.dtype)
    for b in range(B):
        out[b] = pool[b, idx[b]]
    return out


def _fps_one(ep: np.ndarray, k: int, start: int | None = None) -> np.ndarray:
    """ep (M,2) -> indices length k."""
    M = ep.shape[0]
    k = min(k, M)
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    if start is None:
        c = ep.mean(0)
        start = int(np.linalg.norm(ep - c, axis=1).argmax())
    chosen = [start]
    dmin = np.linalg.norm(ep - ep[start], axis=1)
    for _ in range(1, k):
        nxt = int(dmin.argmax())
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(ep - ep[nxt], axis=1))
    return np.asarray(chosen, dtype=np.int64)


def fps_batch(ep: np.ndarray, k: int) -> np.ndarray:
    """ep (B,M,2) -> (B,k)"""
    B, M, _ = ep.shape
    k = min(k, M)
    out = np.zeros((B, k), dtype=np.int64)
    for b in range(B):
        out[b] = _fps_one(ep[b], k)
    return out


def select_native(native, kin):
    return native.copy()


def select_kin(native, kin):
    return kin.copy()


def select_union(native, kin):
    return np.concatenate([native, kin], axis=1)


def select_strat_fps(native, kin, n_nat: int, n_kin: int):
    assert n_nat + n_kin == 20
    ep_n = endpoints(native)
    ep_k = endpoints(kin)
    ix_n = fps_batch(ep_n, n_nat)
    ix_k = fps_batch(ep_k, n_kin)
    part_n = gather(native, ix_n)
    part_k = gather(kin, ix_k)
    return np.concatenate([part_n, part_k], axis=1)


def select_global_fps(native, kin, k: int = 20):
    pool = np.concatenate([native, kin], axis=1)
    ix = fps_batch(endpoints(pool), k)
    return gather(pool, ix)


def select_swap_redundant(native, kin, n_swap: int = 6):
    """Start from native; repeatedly replace most-redundant native with farthest KIN."""
    B = native.shape[0]
    ep_n = endpoints(native)
    ep_k = endpoints(kin)
    # working set = native indices 0..19; we rebuild paths
    out = native.copy()
    used_kin = np.zeros((B, kin.shape[1]), dtype=bool)
    for _ in range(n_swap):
        ep_s = endpoints(out)  # (B,20,2)
        # redundancy of each native slot: distance to nearest other in set
        # (B,20,20)
        d = np.linalg.norm(ep_s[:, :, None, :] - ep_s[:, None, :, :], axis=-1)
        eye = np.eye(20, dtype=bool)
        d[:, eye] = np.inf
        nearest = d.min(axis=-1)  # (B,20) small = redundant
        drop = nearest.argmin(axis=1)  # (B,)
        # farthest KIN from current set (min dist to set), unused
        d_k = np.linalg.norm(ep_k[:, :, None, :] - ep_s[:, None, :, :], axis=-1)  # B,20kin,20set
        mind = d_k.min(axis=-1)  # B,20kin
        mind = np.where(used_kin, -1.0, mind)
        add = mind.argmax(axis=1)
        for b in range(B):
            a = int(add[b])
            if used_kin[b, a] or mind[b, a] < 0:
                continue
            out[b, int(drop[b])] = kin[b, a]
            used_kin[b, a] = True
    return out


def select_motion_strat(native, kin, n_nat: int = 12, n_kin: int = 8, speed_q: float = 0.98):
    """Drop high-speed outliers within family (relative to family p98), then strat FPS."""
    sn = path_speed_mean(native)
    sk = path_speed_mean(kin)
    # per-sample thresholds
    thr_n = np.quantile(sn, speed_q, axis=1, keepdims=True)
    thr_k = np.quantile(sk, speed_q, axis=1, keepdims=True)
    # mask: keep if speed <= thr (always keep at least something via FPS on all if too few)
    # Simpler: downweight high speed by shrinking endpoints toward 0 for FPS? 
    # Instead: run FPS only among kept; if fewer than needed, fill from remainder.
    B = native.shape[0]
    out_n = np.zeros((B, n_nat) + native.shape[2:], dtype=native.dtype)
    out_k = np.zeros((B, n_kin) + kin.shape[2:], dtype=kin.dtype)
    ep_n = endpoints(native)
    ep_k = endpoints(kin)
    for b in range(B):
        keep_n = np.where(sn[b] <= thr_n[b, 0] + 1e-12)[0]
        keep_k = np.where(sk[b] <= thr_k[b, 0] + 1e-12)[0]
        if len(keep_n) < n_nat:
            keep_n = np.arange(native.shape[1])
        if len(keep_k) < n_kin:
            keep_k = np.arange(kin.shape[1])
        ix_n = _fps_one(ep_n[b, keep_n], n_nat)
        ix_k = _fps_one(ep_k[b, keep_k], n_kin)
        out_n[b] = native[b, keep_n[ix_n]]
        out_k[b] = kin[b, keep_k[ix_k]]
    return np.concatenate([out_n, out_k], axis=1)


def select_oracle_cover(native, kin, gt, quality=(0.2, 0.4)):
    """If any cand in pool is good, ensure one good in K20; fill with FPS. Cov == union."""
    from .metrics import errors
    pool = np.concatenate([native, kin], axis=1)
    ade, fde, joint, good, _, _ = errors(pool, gt)
    ep = endpoints(pool)
    B, M = good.shape
    idx = np.zeros((B, 20), dtype=np.int64)
    for b in range(B):
        goods = np.where(good[b])[0]
        if len(goods):
            # pick good with best joint
            j = joint[b, goods]
            first = int(goods[int(j.argmin())])
            rest_pool = [i for i in range(M) if i != first]
            if len(rest_pool) >= 19:
                sub_ep = ep[b, rest_pool]
                sub_ix = _fps_one(sub_ep, 19)
                chosen = [first] + [rest_pool[i] for i in sub_ix]
            else:
                chosen = [first] + rest_pool
                while len(chosen) < 20:
                    chosen.append(first)
            idx[b] = np.asarray(chosen[:20], dtype=np.int64)
        else:
            idx[b] = _fps_one(ep[b], 20)
    return gather(pool, idx)


def select_oracle_minade_top20(native, kin, gt):
    from .metrics import errors
    pool = np.concatenate([native, kin], axis=1)
    ade, _, _, _, _, _ = errors(pool, gt)
    ix = np.argsort(ade, axis=1)[:, :20]
    return gather(pool, ix)


def cand_features(paths: np.ndarray, family: np.ndarray) -> np.ndarray:
    """paths (B,K,T,2), family (B,K) 0=native 1=kin -> (B,K,F)"""
    ep = endpoints(paths)
    ep_n = np.linalg.norm(ep, axis=-1, keepdims=True)
    v = np.diff(paths.astype(np.float64), axis=2, prepend=np.zeros_like(paths[:, :, :1])) / DT
    speed = np.linalg.norm(v, axis=-1)
    a = np.diff(v, axis=2) / DT
    accel = np.linalg.norm(a, axis=-1)
    j = np.diff(a, axis=2) / DT
    jerk = np.linalg.norm(j, axis=-1)
    plen = np.linalg.norm(np.diff(paths.astype(np.float64), axis=2), axis=-1).sum(-1, keepdims=True)
    fam = family.astype(np.float64)[..., None]
    # curvature proxy: mean turning
    vxy = v
    # cos between consecutive velocity vectors
    v0 = vxy[:, :, :-1]
    v1 = vxy[:, :, 1:]
    n0 = np.linalg.norm(v0, axis=-1) + 1e-6
    n1 = np.linalg.norm(v1, axis=-1) + 1e-6
    cos = ((v0 * v1).sum(-1) / (n0 * n1)).mean(-1, keepdims=True)
    return np.concatenate([
        ep,
        ep_n,
        speed.mean(-1, keepdims=True),
        speed.max(-1, keepdims=True),
        speed[:, :, -1:],  # final speed scalar last step — wait speed is (B,K,T)
        accel.mean(-1, keepdims=True),
        accel.max(-1, keepdims=True),
        jerk.mean(-1, keepdims=True),
        plen,
        cos,
        fam,
    ], axis=-1)


def fit_logistic_ranker(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, steps: int = 800, lr: float = 0.08):
    """X (N,F), y (N,) in {0,1}. Class-balanced logistic. Returns w (F,), b."""
    mu = X.mean(0)
    sig = X.std(0) + 1e-6
    Xs = (X - mu) / sig
    F = Xs.shape[1]
    w = np.zeros(F, dtype=np.float64)
    b = 0.0
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    # weight positives higher
    sw = np.where(y > 0.5, 0.5 * len(y) / pos, 0.5 * len(y) / neg)
    for _ in range(steps):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = (p - y) * sw
        w -= lr * (Xs.T @ g / sw.sum() + l2 * w)
        b -= lr * float(g.sum() / sw.sum())
    return dict(w=w, b=b, mu=mu, sig=sig)


def score_logistic(feat: np.ndarray, model: dict) -> np.ndarray:
    if model.get("type") == "mlp":
        from .mlp_ranker import score_mlp
        return score_mlp(feat, model)
    if model.get("type") == "cond_mlp":
        from .cond_ranker import score_cond
        return score_cond(feat, model)
    Xs = (feat - model["mu"]) / model["sig"]
    z = Xs @ model["w"] + model["b"]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def pool_features(native: np.ndarray, kin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (feat_native, feat_kin) with cross-family min endpoint distance feature."""
    B = native.shape[0]
    ep_n = endpoints(native)
    ep_k = endpoints(kin)
    # min dist native->kin set, kin->native set
    # (B,20,20)
    d_nk = np.linalg.norm(ep_n[:, :, None, :] - ep_k[:, None, :, :], axis=-1)
    min_n_to_k = d_nk.min(axis=2, keepdims=True)
    min_k_to_n = d_nk.min(axis=1, keepdims=True).transpose(0, 2, 1)  # wrong
    min_k_to_n = d_nk.min(axis=1)[:, :, None]  # (B,20kin,1) — d_nk is (B,n,k) so min over n -> (B,k)
    min_k_to_n = d_nk.min(axis=1)[..., None]
    fn = cand_features(native, np.zeros((B, native.shape[1])))
    fk = cand_features(kin, np.ones((B, kin.shape[1])))
    fn = np.concatenate([fn, min_n_to_k], axis=-1)
    fk = np.concatenate([fk, min_k_to_n], axis=-1)
    return fn, fk


def select_fit_rank_strat(native, kin, model, n_nat: int = 12, n_kin: int = 8, top_m: int = 16):
    """Score pool with FIT logistic; take top_m per family by score then FPS to n_*."""
    B = native.shape[0]
    fn, fk = pool_features(native, kin)
    sn = score_logistic(fn.reshape(-1, fn.shape[-1]), model).reshape(B, -1)
    sk = score_logistic(fk.reshape(-1, fk.shape[-1]), model).reshape(B, -1)
    ep_n = endpoints(native)
    ep_k = endpoints(kin)
    out_n = np.zeros((B, n_nat) + native.shape[2:], dtype=native.dtype)
    out_k = np.zeros((B, n_kin) + kin.shape[2:], dtype=kin.dtype)
    for b in range(B):
        topn = np.argsort(-sn[b])[: min(top_m, native.shape[1])]
        topk = np.argsort(-sk[b])[: min(top_m, kin.shape[1])]
        ix_n = _fps_one(ep_n[b, topn], n_nat)
        ix_k = _fps_one(ep_k[b, topk], n_kin)
        out_n[b] = native[b, topn[ix_n]]
        out_k[b] = kin[b, topk[ix_k]]
    return np.concatenate([out_n, out_k], axis=1)




def select_score_quota(native, kin, model, k: int = 20, min_nat: int = 8, min_kin: int = 4):
    """Greedy by score with family minimum quotas."""
    pool = np.concatenate([native, kin], axis=1)
    B = native.shape[0]
    Kn = native.shape[1]
    fn, fk = pool_features(native, kin)
    feat = np.concatenate([fn, fk], axis=1)
    F = feat.shape[-1]
    s = score_logistic(feat.reshape(-1, F), model).reshape(B, -1)
    idx = np.zeros((B, k), dtype=np.int64)
    for b in range(B):
        order = list(np.argsort(-s[b]))
        chosen = []
        n_n = n_k = 0
        # first pass: fill quotas with best of each family
        for i in order:
            if len(chosen) >= k:
                break
            is_n = i < Kn
            if is_n and n_n < min_nat:
                chosen.append(i); n_n += 1
            elif (not is_n) and n_k < min_kin:
                chosen.append(i); n_k += 1
        # second: fill rest by score
        for i in order:
            if len(chosen) >= k:
                break
            if i not in chosen:
                chosen.append(i)
                if i < Kn: n_n += 1
                else: n_k += 1
        idx[b] = np.asarray(chosen[:k], dtype=np.int64)
    return gather(pool, idx)


GT_FREE_NAMES = [
    "strat_fps_10_10",
    "strat_fps_12_8",
    "strat_fps_15_5",
    "global_fps_20",
    "swap_redund_4",
    "swap_redund_6",
    "swap_redund_8",
    "motion_strat_12_8",
    "fit_rank_12_8",
    "score_top_12_8",
    "score_top_15_5",
    "score_top_16_4",
    "score_top_18_2",
    "score_top_11_9",
    "score_top_13_7",
    "score_top_14_6",
    "score_top_10_10",
    "score_global_top20",
    "mmr_score_20",
    "dup_swap_4",
    "dup_swap_6",
    "score_mmr_nat18_kin2",
    "score_quota_8_4",
    "score_quota_10_5",
    "score_quota_6_6",
]



def select_score_swap(native, kin, sn, sk, n_swap: int = 6):
    """Replace lowest-scored native slots with high-score complementary KIN."""
    B = native.shape[0]
    out = native.copy()
    ep_n0 = endpoints(native)
    ep_k = endpoints(kin)
    for b in range(B):
        drops = list(np.argsort(sn[b])[:n_swap])
        remain = [i for i in range(20) if i not in drops]
        ep_remain = ep_n0[b, remain]
        # rank KIN by score * dist to remaining natives
        dist = np.linalg.norm(ep_k[b][:, None, :] - ep_remain[None, :, :], axis=-1).min(axis=1)
        util = sk[b] * (0.25 + dist)
        adds = list(np.argsort(-util)[:n_swap])
        for di, ai in zip(drops, adds):
            out[b, di] = kin[b, ai]
    return out


def select_from_scores(native, kin, sn, sk, n_nat: int, n_kin: int):
    ix_n = np.argsort(-sn, axis=1)[:, :n_nat]
    ix_k = np.argsort(-sk, axis=1)[:, :n_kin]
    return np.concatenate([gather(native, ix_n), gather(kin, ix_k)], axis=1)


def select_global_from_scores(native, kin, sn, sk, k: int = 20):
    pool = np.concatenate([native, kin], axis=1)
    s = np.concatenate([sn, sk], axis=1)
    ix = np.argsort(-s, axis=1)[:, :k]
    return gather(pool, ix)


def select_quota_from_scores(native, kin, sn, sk, k=20, min_nat=8, min_kin=4):
    pool = np.concatenate([native, kin], axis=1)
    B = native.shape[0]
    Kn = native.shape[1]
    s = np.concatenate([sn, sk], axis=1)
    idx = np.zeros((B, k), dtype=np.int64)
    for b in range(B):
        order = list(np.argsort(-s[b]))
        chosen = []
        n_n = n_k = 0
        for i in order:
            if len(chosen) >= k:
                break
            is_n = i < Kn
            if is_n and n_n < min_nat:
                chosen.append(i); n_n += 1
            elif (not is_n) and n_k < min_kin:
                chosen.append(i); n_k += 1
        for i in order:
            if len(chosen) >= k:
                break
            if i not in chosen:
                chosen.append(i)
        idx[b] = np.asarray(chosen[:k], dtype=np.int64)
    return gather(pool, idx)


def select_score_top_family(native, kin, model, n_nat: int, n_kin: int):
    B = native.shape[0]
    fn, fk = pool_features(native, kin)
    F = fn.shape[-1]
    sn = score_logistic(fn.reshape(-1, F), model).reshape(B, -1)
    sk = score_logistic(fk.reshape(-1, F), model).reshape(B, -1)
    ix_n = np.argsort(-sn, axis=1)[:, :n_nat]
    ix_k = np.argsort(-sk, axis=1)[:, :n_kin]
    return np.concatenate([gather(native, ix_n), gather(kin, ix_k)], axis=1)


def select_score_global_top(native, kin, model, k: int = 20):
    pool = np.concatenate([native, kin], axis=1)
    fn, fk = pool_features(native, kin)
    feat = np.concatenate([fn, fk], axis=1)
    F = feat.shape[-1]
    s = score_logistic(feat.reshape(-1, F), model).reshape(pool.shape[0], -1)
    ix = np.argsort(-s, axis=1)[:, :k]
    return gather(pool, ix)


def select_mmr(native, kin, model, k: int = 20, lam: float = 0.7):
    """MMR on endpoints with quality = logistic score."""
    pool = np.concatenate([native, kin], axis=1)
    fn, fk = pool_features(native, kin)
    feat = np.concatenate([fn, fk], axis=1)
    F = feat.shape[-1]
    s = score_logistic(feat.reshape(-1, F), model).reshape(pool.shape[0], -1)
    ep = endpoints(pool)
    B, M, _ = ep.shape
    idx = np.zeros((B, k), dtype=np.int64)
    for b in range(B):
        chosen = []
        remain = list(range(M))
        # init: highest score
        first = int(np.argmax(s[b]))
        chosen.append(first)
        remain.remove(first)
        for _ in range(1, k):
            best_i, best_v = remain[0], -1e18
            for i in remain:
                # sim = max neg dist to chosen (= closeness)
                d = np.linalg.norm(ep[b, i] - ep[b, chosen], axis=-1).min()
                # prefer far: use -exp(-d) or just d as diversity
                div = d
                v = lam * s[b, i] + (1 - lam) * div
                if v > best_v:
                    best_v, best_i = v, i
            chosen.append(best_i)
            remain.remove(best_i)
        idx[b] = np.asarray(chosen, dtype=np.int64)
    return gather(pool, idx)


def select_dup_swap(native, kin, model, n_swap: int = 4, dup_eps: float = 0.15):
    """Only replace a native if a near-duplicate native remains; add high-score far KIN."""
    B = native.shape[0]
    fn, fk = pool_features(native, kin)
    F = fn.shape[-1]
    sn = score_logistic(fn.reshape(-1, F), model).reshape(B, -1)
    sk = score_logistic(fk.reshape(-1, F), model).reshape(B, -1)
    ep_n = endpoints(native)
    ep_k = endpoints(kin)
    out = native.copy()
    used_kin = np.zeros((B, kin.shape[1]), dtype=bool)
    active = np.ones((B, 20), dtype=bool)  # which native slots still original? always 20 slots
    for _ in range(n_swap):
        ep_s = endpoints(out)
        for b in range(B):
            # find drop candidate: lowest score among those that have a neighbor within dup_eps
            d = np.linalg.norm(ep_s[b][:, None, :] - ep_s[b][None, :, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            has_dup = d.min(axis=1) <= dup_eps
            if not has_dup.any():
                continue
            scores = sn[b].copy()
            # map: out slots may already be KIN — use low priority to drop KIN again? skip if already swapped
            # track via used count: compare to original native endpoint
            is_orig = np.linalg.norm(out[b, :, -1] - native[b, :, -1], axis=-1) < 1e-9
            cand_drop = np.where(has_dup & is_orig)[0]
            if len(cand_drop) == 0:
                continue
            # drop lowest score among dup-able originals
            drop = int(cand_drop[np.argmin(scores[cand_drop])])
            # KIN: high score * distance to set, unused
            dist = np.linalg.norm(ep_k[b][:, None, :] - ep_s[b][None, :, :], axis=-1).min(axis=1)
            util = sk[b] * (0.2 + dist)
            util = np.where(used_kin[b], -1e9, util)
            add = int(util.argmax())
            if util[add] < -1e8:
                continue
            out[b, drop] = kin[b, add]
            used_kin[b, add] = True
    return out


def select_score_mmr_family(native, kin, model, n_nat=18, n_kin=2, lam=0.75):
    """Top/MMR within native for n_nat, MMR within kin for n_kin with high native retention."""
    B = native.shape[0]
    fn, fk = pool_features(native, kin)
    F = fn.shape[-1]
    sn = score_logistic(fn.reshape(-1, F), model).reshape(B, -1)
    sk = score_logistic(fk.reshape(-1, F), model).reshape(B, -1)
    ep_n = endpoints(native)
    ep_k = endpoints(kin)
    # native: top by score then light MMR fill — simpler: just top n_nat by score (preserve quality)
    ix_n = np.argsort(-sn, axis=1)[:, :n_nat]
    # kin: MMR among kin relative to selected native endpoints
    out_k = np.zeros((B, n_kin) + kin.shape[2:], dtype=kin.dtype)
    for b in range(B):
        chosen_ep = ep_n[b, ix_n[b]]
        remain = list(range(kin.shape[1]))
        picked = []
        for _ in range(n_kin):
            best_i, best_v = remain[0], -1e18
            for i in remain:
                dist = np.linalg.norm(ep_k[b, i] - chosen_ep, axis=-1).min() if len(chosen_ep) else 0.0
                if picked:
                    dist = min(dist, np.linalg.norm(ep_k[b, i] - ep_k[b, picked], axis=-1).min())
                v = lam * sk[b, i] + (1 - lam) * dist
                if v > best_v:
                    best_v, best_i = v, i
            picked.append(best_i)
            remain.remove(best_i)
            chosen_ep = np.concatenate([chosen_ep, ep_k[b, best_i][None]], 0)
        out_k[b] = kin[b, picked]
    return np.concatenate([gather(native, ix_n), out_k], axis=1)


def run_named(name: str, native, kin, gt=None, ranker=None):
    if name == "native":
        return select_native(native, kin)
    if name == "kin":
        return select_kin(native, kin)
    if name == "union40":
        return select_union(native, kin)
    if name == "oracle_cover":
        return select_oracle_cover(native, kin, gt)
    if name == "oracle_minade_top20":
        return select_oracle_minade_top20(native, kin, gt)
    if name == "strat_fps_10_10":
        return select_strat_fps(native, kin, 10, 10)
    if name == "strat_fps_12_8":
        return select_strat_fps(native, kin, 12, 8)
    if name == "strat_fps_15_5":
        return select_strat_fps(native, kin, 15, 5)
    if name == "global_fps_20":
        return select_global_fps(native, kin, 20)
    if name == "swap_redund_4":
        return select_swap_redundant(native, kin, 4)
    if name == "swap_redund_6":
        return select_swap_redundant(native, kin, 6)
    if name == "swap_redund_8":
        return select_swap_redundant(native, kin, 8)
    if name == "motion_strat_12_8":
        return select_motion_strat(native, kin, 12, 8)
    if name == "fit_rank_12_8":
        assert ranker is not None
        return select_fit_rank_strat(native, kin, ranker, 12, 8)
    if name == "score_top_12_8":
        return select_score_top_family(native, kin, ranker, 12, 8)
    if name == "score_top_15_5":
        return select_score_top_family(native, kin, ranker, 15, 5)
    if name == "score_top_16_4":
        return select_score_top_family(native, kin, ranker, 16, 4)
    if name == "score_top_18_2":
        return select_score_top_family(native, kin, ranker, 18, 2)
    if name == "score_global_top20":
        return select_score_global_top(native, kin, ranker, 20)
    if name == "mmr_score_20":
        return select_mmr(native, kin, ranker, 20, lam=0.7)
    if name == "dup_swap_4":
        return select_dup_swap(native, kin, ranker, 4)
    if name == "dup_swap_6":
        return select_dup_swap(native, kin, ranker, 6)
    if name == "score_mmr_nat18_kin2":
        return select_score_mmr_family(native, kin, ranker, 18, 2)
    if name == "score_top_11_9":
        return select_score_top_family(native, kin, ranker, 11, 9)
    if name == "score_top_13_7":
        return select_score_top_family(native, kin, ranker, 13, 7)
    if name == "score_top_14_6":
        return select_score_top_family(native, kin, ranker, 14, 6)
    if name == "score_top_10_10":
        return select_score_top_family(native, kin, ranker, 10, 10)
    if name == "score_quota_8_4":
        return select_score_quota(native, kin, ranker, 20, 8, 4)
    if name == "score_quota_10_5":
        return select_score_quota(native, kin, ranker, 20, 10, 5)
    if name == "score_quota_6_6":
        return select_score_quota(native, kin, ranker, 20, 6, 6)
    raise ValueError(name)
