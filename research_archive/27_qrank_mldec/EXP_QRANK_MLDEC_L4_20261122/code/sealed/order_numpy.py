"""NumPy reference for the GT-free orderings used between decoder layers.
The torch versions in ml_rerank_ref.py must give identical orders (preflight PF-R1)."""
import numpy as np

def softmax(z, axis=-1):
    z = np.asarray(z, np.float64); z = z - z.max(axis, keepdims=True)
    e = np.exp(z); return e / e.sum(axis, keepdims=True)

def risk_order(end, w):
    """Greedy facility location (k-median) over ALL K candidates.
    end: (N,K,2) predicted endpoints (GT-free); w: (N,K) weights summing to 1.
    Step r picks the candidate s that minimises sum_j w_j * min(best_j, D[s,j]).
    Step 1 therefore picks the weighted medoid. Ties -> lowest index."""
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

def e1_threshold(past):
    """past: (N,8,2) metres. Final-step EMTA threshold, metres."""
    v = np.linalg.norm(np.diff(past, axis=1), axis=-1).mean(1) / .4
    return np.clip(.10 * v * 4.8, .5, 1.5)

def kinematic_excess(paths, past):
    """paths (N,K,12,2), past (N,8,2), metres, dt = 0.4 s. GT-free.
    Returns mean over the 12 future steps of relu(|acc| - a_ref), a_ref = max observed |acc| + 1.0 m/s^2."""
    v_obs = np.diff(past, axis=1) / .4
    a_ref = np.linalg.norm(np.diff(v_obs, axis=1) / .4, axis=-1).max(1) + 1.0
    full = np.concatenate([np.broadcast_to(past[:, None, -2:], (*paths.shape[:2], 2, 2)), paths], 2)
    acc = np.diff(np.diff(full, axis=2) / .4, axis=2) / .4
    return np.maximum(np.linalg.norm(acc, axis=-1) - a_ref[:, None, None], 0).mean(-1)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    end = rng.normal(size=(500, 20, 2)); w = softmax(rng.normal(size=(500, 20)))
    o = risk_order(end, w)
    assert (np.sort(o, 1) == np.arange(20)).all(), "not a permutation"
    # independent check of step 1 = weighted medoid, step 2 = best marginal
    D = np.linalg.norm(end[:, :, None] - end[:, None], axis=-1)
    assert (o[:, 0] == (D * w[:, None]).sum(-1).argmin(1)).all()
    n = np.arange(500); b1 = D[n, o[:, 0]]
    obj2 = (w[:, None] * np.minimum(b1[:, None], D)).sum(-1); obj2[n, o[:, 0]] = np.inf
    assert (o[:, 1] == obj2.argmin(1)).all()
    # identical candidates: tie -> lowest index
    same = np.zeros((1, 20, 2)); assert risk_order(same, np.full((1, 20), .05))[0, 0] == 0
    # kinematics: constant-velocity continuation has zero excess
    past = np.cumsum(np.full((1, 8, 2), .4), 1)
    fut = past[:, -1:] + np.cumsum(np.full((1, 12, 2), .4), 1)
    assert kinematic_excess(fut[:, None], past)[0, 0] == 0
    jump = fut.copy(); jump[0, 0] += 3.0
    assert kinematic_excess(jump[:, None], past)[0, 0] > 0
    assert np.allclose(e1_threshold(np.cumsum(np.full((3, 8, 2), [[[0.]]] * 1), 1)), .5)
    print("order_numpy self-test PASS")
