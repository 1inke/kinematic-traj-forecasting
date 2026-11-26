"""Track-cluster statistics and registered C1–C3 decisions."""

import numpy as np


EXPLORATORY = ("eth", "zara01")
CONFIRMATION = ("hotel", "zara02", "univ")
ARMS = ("O-conf+E0", "O-conf+QFL")
LABELS = ("deceleration", "steady", "acceleration", "turn", "turn_and_speed", "stop", "start", "stationary")
PROXIES = {"X1": 1, "X2": -1, "X3": 1, "X4": -1, "X5": 1, "X6": 1}
N_BOOT = 4000
BOOT_SEED = 20261124


def cluster_draws(tracks):
    units, lookup = np.unique(tracks, return_inverse=True)
    rng = np.random.default_rng(BOOT_SEED)
    selected = rng.integers(len(units), size=(N_BOOT, len(units)), dtype=np.int32)
    counts = np.zeros((N_BOOT, len(units)), dtype=np.int32)
    np.add.at(counts, (np.arange(N_BOOT)[:, None], selected), 1)
    return units, lookup, counts


def interval(draws, level):
    values = draws[np.isfinite(draws)]
    if not len(values):
        return None
    return np.quantile(values, [(1 - level) / 2, (1 + level) / 2]).tolist()


def ratio(numerator, denominator):
    a, b = np.asarray(numerator), np.asarray(denominator)
    return np.divide(a, b, out=np.full(np.broadcast_shapes(a.shape, b.shape), np.nan), where=b != 0)


def summarize_group(base, delta, tracks, mask):
    n = int(mask.sum())
    if not n:
        return dict(windows=0, tracks=0, small=True, relative=None, ci95=None)
    units, ix, draws = cluster_draws(tracks[mask])
    d = np.bincount(ix, weights=delta[mask])
    b = np.bincount(ix, weights=base[mask])
    abs_top = np.argsort(-np.abs(d), kind="stable")[:10]
    top_signed_ratio = float(d[abs_top].sum() / d.sum()) if d.sum() else None
    top_absolute_ratio = float(np.abs(d[abs_top]).sum() / np.abs(d).sum()) if np.abs(d).sum() else None
    return dict(windows=n, tracks=len(units), small=n < 30 or len(units) < 10, baseline_m=float(base[mask].mean()),
                mean_delta_m=float(delta[mask].mean()), relative=float(d.sum() / b.sum()),
                ci95=interval(ratio(draws @ d, draws @ b), .95), top10_share_signed=top_signed_ratio,
                top10_share_absolute=top_absolute_ratio, largest_positive_track_ids=units[np.argsort(-d, kind="stable")[:10]].tolist())


def tertiles(values):
    valid = np.isfinite(values)
    low, high = np.quantile(values[valid], [1 / 3, 2 / 3])
    below, above = ((values < low, values > high) if low == high else (values <= low, values >= high))
    return dict(low=below & valid, mid=valid & ~below & ~above, high=above & valid), [float(low), float(high)]


def auc(score, harm, tracks, level):
    if not harm.any() or harm.all():
        return dict(auc=None, ci=None, level=level)
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    ties = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]
    _, lookup, counts = cluster_draws(tracks[order])
    truth = harm[order]

    def weighted(weights):
        pos = np.add.reduceat(weights * truth, ties, axis=1)
        neg = np.add.reduceat(weights * ~truth, ties, axis=1)
        wins = (pos * (np.cumsum(neg, axis=1) - .5 * neg)).sum(axis=1)
        return ratio(wins, pos.sum(axis=1) * neg.sum(axis=1))

    point = float(weighted(np.ones((1, len(order)), dtype=np.int32))[0])
    draws = np.concatenate([weighted(c[:, lookup]) for c in np.array_split(counts, N_BOOT // 40)])
    return dict(auc=point, ci=interval(draws, level), level=level, valid_draws=int(np.isfinite(draws).sum()))


def average_ranks(values):
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    start = np.r_[0, np.flatnonzero(ranked[1:] != ranked[:-1]) + 1]
    count = np.diff(np.r_[start, len(order)])
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.repeat(start + (count - 1) / 2, count)
    return ranks


def spearman(x, y):
    a, b = average_ranks(x), average_ranks(y)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None


def high_minus_low(base, delta, tracks, masks, level):
    mask = masks["high"] | masks["low"]
    if not (masks["high"].any() and masks["low"].any()):
        return dict(delta=None, ci=None, level=level)
    units, codes, draws = cluster_draws(tracks[mask])
    high = masks["high"][mask]
    totals = np.column_stack([np.bincount(codes[m], weights=x[mask][m], minlength=len(units))
                              for m in (high, ~high) for x in (delta, base)])
    sums = totals.sum(axis=0)
    sample = draws @ totals
    return dict(delta=float(sums[0] / sums[1] - sums[2] / sums[3]),
                ci=interval(ratio(sample[:, 0], sample[:, 1]) - ratio(sample[:, 2], sample[:, 3]), level), level=level)


def coverage(data, moved, mask):
    out = {}
    for axis in ("long", "lat"):
        first, final = data[f"first_spread_{axis}"], data[f"final_spread_{axis}"]
        valid = mask & data["valid"] & (first > 1e-12)
        ratios = final[valid] / first[valid]
        out[axis] = dict(windows=int(valid.sum()), ratio_median=float(np.median(ratios)) if len(ratios) else None)
    support = mask & data["valid"]
    out["direction_valid_windows"] = int(support.sum())
    out["direction_valid_tracks"] = int(len(np.unique(data["tracks"][support])))
    out["movement_median_m"] = float(np.median(moved[mask])) if mask.any() else None
    if support.any():
        out["best_error_squared_delta_m2"] = {
            axis: float((data[f"final_best_error_{axis}_sq"] - data[f"first_best_error_{axis}_sq"])[support].mean())
            for axis in ("long", "lat")
        }
        out["best_slot_switch_rate"] = float(data["best_slot_switch"][support].mean())
    return out


def aggregate(rows):
    keep = ("ids", "tracks", "recording", "valid", "observed_steady", "X1", "X2", "X3", "X4")
    first = rows[0]
    for row in rows[1:]:
        for key in keep:
            np.testing.assert_array_equal(first[key], row[key])
    base_by_seed = np.stack([row["first_minFDE"] for row in rows])
    final_by_seed = np.stack([row["final_minFDE"] for row in rows])
    base = base_by_seed.mean(axis=0)
    delta = (final_by_seed - base_by_seed).mean(axis=0)
    data = {k: first[k] for k in keep}
    for k in ("first_spread_long", "first_spread_lat", "final_spread_long", "final_spread_lat",
              "first_entropy", "movement", "best_slot_switch",
              "first_best_error_long_sq", "first_best_error_lat_sq", "final_best_error_long_sq", "final_best_error_lat_sq"):
        data[k] = np.mean([row[k] for row in rows], axis=0)
    return data, base, delta, base_by_seed, final_by_seed


def fold_report(rows, labels, fold, seeds):
    data, base, delta, base_by_seed, final_by_seed = aggregate(rows)
    tracks = data["tracks"]
    harm = delta > .10
    groups = {label: labels == label for label in LABELS}
    full = np.ones(len(base), dtype=bool)
    group_masks = dict(groups, overall=full, observed_fast=data["X1"] > 1.3, observed_steady=data["observed_steady"],
                       observed_low_speed=data["X1"] <= 1.3, observed_decel=data["X2"] <= .85,
                       observed_not_decel=data["X2"] > .85)
    cohorts = {}
    for label, mask in group_masks.items():
        report = summarize_group(base, delta, tracks, mask)
        if mask.any():
            report["coverage"] = coverage(data, data["movement"], mask)
            report["by_seed_relative"] = {
                str(seed): float((f[mask] - b[mask]).sum() / b[mask].sum())
                for seed, b, f in zip(seeds, base_by_seed, final_by_seed)
            }
            report["by_seed_same_sign"] = all(np.sign(x) == np.sign(report["relative"])
                                              for x in report["by_seed_relative"].values())
        cohorts[label] = report
    proxies = dict(X1=data["X1"], X2=data["X2"], X3=data["X3"], X4=data["X4"],
                   X5=np.where(data["valid"], data["first_spread_long"], np.nan), X6=data["first_entropy"])
    table = {}
    for key, values in proxies.items():
        level = .9875 if key in ("X1", "X2", "X3", "X4") else .95
        supported = np.isfinite(values)
        masks, cutoffs = tertiles(values)
        stats = auc(PROXIES[key] * values[supported], harm[supported], tracks[supported], level)
        table[key] = dict(**stats, direction=PROXIES[key], windows=int(supported.sum()),
                          spearman=spearman(values[supported], delta[supported]),
                          tertile_edges=cutoffs, tertiles={name: summarize_group(base, delta, tracks, mask)
                                                          for name, mask in masks.items()},
                          high_minus_low=high_minus_low(base, delta, tracks, masks, level))
    oracle = auc(groups["deceleration"].astype(int), harm, tracks, .95)
    speed_masks, _ = tertiles(data["X1"])
    within_speed = {level: {label: summarize_group(base, delta, tracks, speed_mask & groups[label])
                            for label in ("deceleration", "steady")} for level, speed_mask in speed_masks.items()}
    sensitivity = {}
    if fold == "univ":
        for recording in np.unique(data["recording"]):
            inside = data["recording"] == recording
            sensitivity[recording] = dict(
                leave_out={key: summarize_group(base, delta, tracks, mask & ~inside)
                           for key, mask in group_masks.items()},
                within_auc={key: auc(PROXIES[key] * values[inside & np.isfinite(values)],
                                     harm[inside & np.isfinite(values)], tracks[inside & np.isfinite(values)], .9875)
                            for key, values in proxies.items() if key in ("X1", "X2", "X3", "X4")},
            )
    elif fold in ("hotel", "zara02"):
        for name, mask in group_masks.items():
            if mask.any():
                drop = cohorts[name]["largest_positive_track_ids"]
                sensitivity[name] = summarize_group(base, delta, tracks, mask & ~np.isin(tracks, drop))
    return dict(groups=cohorts, proxies=table, future_deceleration_auc=oracle, speed_matched=within_speed,
                sensitivity=sensitivity, harm_rate=float(harm.mean()),
                harmful_labels={key: int((harm & mask).sum()) for key, mask in groups.items()})


def decide(reports):
    c1, c3 = {}, {}
    primary, check = ARMS
    for fold in CONFIRMATION:
        decel = reports[fold][primary]["groups"]["deceleration"]
        steady = reports[fold][primary]["groups"]["steady"]
        qfl = reports[fold][check]["groups"]["deceleration"]
        c1[fold] = bool(not any(g["small"] for g in (decel, steady, qfl))
                        and decel["relative"] > 0 and decel["ci95"][0] > 0
                        and steady["relative"] < 0 and qfl["relative"] > 0)
        d, s = [g.get("coverage", {}) for g in (decel, steady)]
        c3[fold] = bool(all(g.get("direction_valid_windows", 0) >= 30 and g.get("direction_valid_tracks", 0) >= 10
                            for g in (d, s)) and d["long"]["ratio_median"] is not None
                        and s["long"]["ratio_median"] is not None
                        and d["long"]["ratio_median"] < .90
                        and s["long"]["ratio_median"] - d["long"]["ratio_median"] >= .05)

    def qualified(stats):
        return stats["auc"] is not None and stats["auc"] >= .60 and stats["ci"] is not None and stats["ci"][0] > .50

    signals = {}
    for key in ("X1", "X2", "X3", "X4"):
        good = [fold for fold in CONFIRMATION if qualified(reports[fold][primary]["proxies"][key])]
        exploratory = {fold: reports[fold][primary]["proxies"][key]["auc"] for fold in EXPLORATORY}
        consistent = all(value is not None and value >= .50 for value in exploratory.values())
        signals[key] = dict(confirming_folds=good, exploratory_auc=exploratory, passed=len(good) >= 2 and consistent)
    first = sum(c1.values()) >= 2
    second = any(entry["passed"] for entry in signals.values())
    third = sum(c3.values()) >= 2
    outcome = ("STOP" if not first else "GATED_REFINE" if second else
               "PRESERVE_LONGITUDINAL" if third else "REDIAGNOSE")
    return dict(outcome=outcome, C1=first, C2=second, C3=third, c1_by_fold=c1, c2_by_proxy=signals, c3_by_fold=c3)
