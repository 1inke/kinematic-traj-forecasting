"""Per-window prefix metrics under conf and coverage orders (NumPy). Keys match archive 27 qrank_eval.py
(conf_minFDE{k}, riskp_minFDE{k}) so archive 27/29 analysis code can be reused unchanged."""
import numpy as np


def softmax(z):
    z = np.asarray(z, np.float64); z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def risk_order(end, w):          # copy of archive 27 order_numpy.risk_order
    end = np.asarray(end, np.float64); w = np.asarray(w, np.float64)
    N, K, _ = end.shape
    D = np.linalg.norm(end[:, :, None] - end[:, None], axis=-1)
    best = np.full((N, K), np.inf); taken = np.zeros((N, K), bool); rows = np.arange(N)
    tie = np.arange(K) * 1e-12; out = np.empty((N, K), int)
    for r in range(K):
        obj = (w[:, None, :] * np.minimum(best[:, None, :], D)).sum(-1) + tie
        obj[taken] = np.inf
        c = obj.argmin(1); out[:, r] = c; taken[rows, c] = True
        best = np.minimum(best, D[rows, c])
    return out


def conf_order(logits):
    return np.argsort(-np.asarray(logits, np.float64), axis=1, kind="stable")


def prefix_metrics(paths, logits, future, ks=(1, 3, 5, 20)):
    """paths (N,K,T,2) world metres, logits (N,K), future (N,T,2)."""
    err = np.linalg.norm(paths - future[:, None], axis=-1)
    ade, fde = err.mean(2), err[:, :, -1]
    orders = dict(conf=conf_order(logits), riskp=risk_order(paths[:, :, -1], softmax(logits)))
    out = {}
    for name, o in orders.items():
        for k in ks:
            out[f"{name}_minADE{k}"] = np.take_along_axis(ade, o[:, :k], 1).min(1)
            out[f"{name}_minFDE{k}"] = np.take_along_axis(fde, o[:, :k], 1).min(1)
    out["minFDE"] = fde.min(1)
    end = paths[:, :, -1]
    i, j = np.triu_indices(paths.shape[1], 1)
    out["spread_m"] = np.linalg.norm(end[:, i] - end[:, j], axis=-1).mean(1)          # MC-3 (collapse monitor)
    return out
