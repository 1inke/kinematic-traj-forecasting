"""Deterministic geometry-only compression of scored endpoint pools."""
import numpy as np

NMS_THRESHOLDS = (.1, .2, .3, .5)


def nms(distances, weights, threshold, count=20):
    n, total = weights.shape
    order = np.argsort(-weights, axis=1, kind="stable")
    row = np.arange(n)
    available = np.ones((n, total), bool)
    unused = available.copy()
    selected = np.empty((n, count), int)
    for j in range(count):
        # Once no candidate passes NMS, fill in descending weight order.
        keep = np.where(available.any(1)[:, None], available, unused)
        idx = order[row, keep[row[:, None], order].argmax(1)]
        selected[:, j] = idx
        unused[row, idx] = False
        available &= distances[row, idx] >= threshold
        available[row, idx] = False
    return selected


def medoids(distances, weights, count=20, limit=24):
    """Greedy weighted-distance initialization and alternate medoid updates."""
    n, total = weights.shape
    row = np.arange(n)
    idx = np.empty((n, count), dtype=np.int64)
    taken = np.zeros((n, total), bool)
    current = np.full((n, total), np.inf)
    for j in range(count):
        if j == 0:
            best = (distances @ weights[:, :, None]).squeeze(2).argmin(1)
        else:
            best = np.where(taken, -1., current * weights).argmax(1)
        idx[:, j] = best
        taken[row, best] = True
        current = np.minimum(current, distances[row, best])
    initial = (weights * current).sum(1)
    active = np.arange(n)
    for _ in range(limit):
        if not len(active):
            break
        active_dist, active_w = distances[active], weights[active]
        selected = idx[active]
        loc = np.arange(len(active))
        assign = active_dist[loc[:, None], selected].argmin(1)
        # Give each medoid membership in its own cluster in exact distance ties.
        assign[loc[:, None], selected] = np.arange(count)
        members = assign[:, :, None] == np.arange(count)
        cost = active_dist @ (members * active_w[:, :, None])
        cost[~members] = np.inf
        old_cost = cost[loc[:, None], selected, np.arange(count)]
        best = cost.argmin(1)
        best_cost = cost[loc[:, None], best, np.arange(count)]
        changed = best_cost + 1e-13 < old_cost
        idx[active] = np.where(changed, best, selected)
        active = active[changed.any(1)]
    final = (distances[row[:, None], idx].min(1) * weights).sum(1)
    assert (final <= initial + 1e-12).all()
    assert (np.diff(np.sort(idx, axis=1), axis=1) > 0).all()
    return idx, initial, final, len(active)

