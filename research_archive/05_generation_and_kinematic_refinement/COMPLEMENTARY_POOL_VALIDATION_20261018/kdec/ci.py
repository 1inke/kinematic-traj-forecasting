"""按 track 聚簇 + 3seed 层级抽样的探索性 CI。"""
from __future__ import annotations
import numpy as np


def hierarchical_ci(per_seed_window_deltas, track_ids, n_draws=100000, seed=20261018,
                    alpha_family8=0.05):
    """
    per_seed_window_deltas: list of length S, each (N_windows,) delta (method - native)
    track_ids: (N_windows,) same across seeds (outer alignment)
    同一 draw：有放回抽 3 个 seed，并按原 track 数有放回抽 track，
    按每 track 抽取次数重复其全部窗口，再算窗口加权差值的 seed 平均。
    返回 family8 Bonferroni 百分位区间。
    """
    S = len(per_seed_window_deltas)
    N = len(track_ids)
    assert all(len(d) == N for d in per_seed_window_deltas)
    # unique tracks
    uniq, inv = np.unique(track_ids, return_inverse=True)
    T = len(uniq)
    # windows per track
    track_windows = [np.where(inv == t)[0] for t in range(T)]
    rng = np.random.default_rng(seed)
    # prebind deltas as (S,N)
    D = np.stack(per_seed_window_deltas, axis=0)
    lo_q = alpha_family8 / 8 / 2  # .003125 for alpha=.05
    hi_q = 1.0 - lo_q
    means = []
    invalid = 0
    for _ in range(n_draws):
        seeds = rng.integers(0, S, size=S)  # with replacement draw S seeds (design: 3)
        # draw T tracks with replacement
        tr_draw = rng.integers(0, T, size=T)
        # accumulate
        vals = []
        for s in seeds:
            parts = []
            for t in tr_draw:
                parts.append(D[s, track_windows[t]])
            if not parts:
                continue
            w = np.concatenate(parts)
            if len(w) == 0:
                continue
            vals.append(w.mean())
        if len(vals) < S:
            invalid += 1
            continue
        means.append(float(np.mean(vals)))
    means = np.asarray(means, dtype=np.float64)
    if len(means) == 0:
        return dict(valid=0, invalid=invalid, ci=None, mean=None)
    return dict(
        valid=int(len(means)),
        invalid=int(invalid),
        mean=float(means.mean()),
        ci_family8=[float(np.quantile(means, lo_q)), float(np.quantile(means, hi_q))],
        ci_95=[float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        note="exploratory; outer participated in historical method selection",
    )
