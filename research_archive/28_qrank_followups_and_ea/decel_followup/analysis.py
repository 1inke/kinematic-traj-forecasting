"""NumPy-only analysis of sealed QRANK decoder-layer diagnostics."""

import numpy as np

LAYERS = ("first", "final")
HORIZONS = (6, 12)
SELECTORS = ("best", "top1")
BLOCK_SIZE = 15


def pair_delta(first, final, mask):
    delta = (final - first).mean(axis=0)[mask]
    _, blocks = np.unique((np.arange(len(mask)) // BLOCK_SIZE)[mask], return_inverse=True)
    totals = np.bincount(blocks, weights=delta)
    counts = np.bincount(blocks)
    rng = np.random.default_rng(20261123)
    sampled = rng.integers(len(totals), size=(2000, len(totals)))
    draws = totals[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return dict(delta=float(delta.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(),
                by_seed=(final[:, mask] - first[:, mask]).mean(axis=1).tolist(),
                blocks=len(totals))


def observation_signal(ref):
    fast = ref["fast"]
    scores = ref["history_ratio"][fast]
    truth = ref["fast_deceleration"][fast]
    signal = scores <= .85
    tp = int((signal & truth).sum())
    fp = int((signal & ~truth).sum())
    fn = int((~signal & truth).sum())
    tn = int((~signal & ~truth).sum())
    positive, negative = scores[truth], scores[~truth]
    cmp = positive[:, None] - negative[None, :]
    return dict(windows=int(fast.sum()), positives=tp + fn, tp=tp, fp=fp, fn=fn, tn=tn,
                base_rate=float(truth.mean()), precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn), specificity=tn / (tn + fp),
                auc=float(((cmp < 0) + .5 * (cmp == 0)).mean()),
                median_ratio_later_decel=float(np.median(positive)),
                median_ratio_other=float(np.median(negative)),
                by_history={name: dict(windows=int(mask.sum()), future_decel=int(truth[mask].sum()))
                            for name, mask in (("decel", signal),
                                               ("steady", (scores > .85) & (scores < 1.15)),
                                               ("accel", scores >= 1.15))})


def summarize(rows):
    ref = rows[0]
    shared = ("ids", "direction_valid", "fast", "fast_deceleration", "fast_steady", "history_ratio")
    if not all(np.array_equal(row[k], ref[k]) for row in rows for k in shared):
        raise ValueError("The replay cohorts differ across seeds")
    groups = {
        "fast_deceleration": ref["fast_deceleration"],
        "fast_steady": ref["fast_steady"],
        "fast_other": ref["fast"] & ~ref["fast_deceleration"],
        "fast_overall": ref["fast"],
    }
    report = {}
    for group, mask in groups.items():
        valid = mask & ref["direction_valid"]
        if not valid.any():
            continue
        horizons = {}
        for t in HORIZONS:
            horizon = {}

            def series(layer, key):
                return np.stack([row[f"{layer}/{key}_t{t}"] for row in rows])

            for selection in SELECTORS:
                stats = {}
                for metric in ("fde", "longitudinal", "absolute_longitudinal", "absolute_lateral",
                               "longitudinal_squared", "lateral_squared", "overshoot"):
                    first = series("first", f"{selection}_{metric}").astype(np.float64)
                    final = series("final", f"{selection}_{metric}").astype(np.float64)
                    stats[metric] = dict(first=float(first[:, valid].mean()), final=float(final[:, valid].mean()),
                                         **pair_delta(first, final, valid))
                horizon[selection] = stats
            for metric in ("all_overshoot", "longitudinal_reachable"):
                first = series("first", metric).astype(np.float64)
                final = series("final", metric).astype(np.float64)
                horizon[metric] = dict(first=float(first[:, valid].mean()), final=float(final[:, valid].mean()),
                                       **pair_delta(first, final, valid))
            best_same = series("first", "best_index") == series("final", "best_index")
            horizon["best_slot_switch_rate"] = float((~best_same[:, valid]).mean())
            horizons[str(t)] = horizon
        report[group] = dict(windows=int(mask.sum()), valid_directions=int(valid.sum()), horizons=horizons)
    return dict(groups=report, observation=observation_signal(ref))
