"""Curve summaries, paired selection effects and recording sensitivities."""
import numpy as np

from centered_inputs import OBS
from centered_metrics import FIELDS, from_paths
from centered_scaling_ref import center_scale
from adapt_math import ES_NAMES, Summary, median_draws
from coupling_analysis import TrackBootstrap, population


def scale_key(a):
    return ",".join(str(z) for z in a) if isinstance(a, tuple) else str(a)


def trim(sets):
    return {name: members for name, members in sets.items() if members.any()}


def curve_tables(grid, groups, track, intervals):
    """Bootstrap curves against identity on dev; selection curves use full-sample means."""
    before = grid[1.] if 1. in grid else grid[1., 1.]
    rows = {}
    for group, mask in trim(groups).items():
        helper = TrackBootstrap(track[mask]) if intervals else None
        info = helper.population if intervals else population(track[mask])[0]
        values = {}
        seq = list(grid)
        unit = np.ones(mask.sum())
        base_spread = float(np.median(before["spread"][mask]))
        base_draw = median_draws(helper, before["spread"][mask]) if intervals else None
        for batch in range(0, len(seq), 8):
            points = seq[batch:batch+8]
            metrics = {}
            for a in points:
                field = grid[a]
                key = scale_key(a)
                row = {}
                if intervals:
                    for k in FIELDS:
                        if k != "spread":
                            v = field[k][mask].astype(float)
                            metrics[key, k] = v, unit
                            metrics[key, f"delta_{k}"] = v - before[k][mask], unit
                    current = median_draws(helper, field["spread"][mask])
                    est = float(np.median(field["spread"][mask]))
                    row["spread"] = dict(estimate=est, ci95=np.quantile(current, [.025, .975]).tolist())
                    row["delta_spread"] = dict(estimate=est-base_spread,
                                               ci95=np.quantile(current-base_draw, [.025, .975]).tolist())
                else:
                    row = {k: float(np.median(field[k][mask]) if k == "spread" else field[k][mask].mean())
                           for k in FIELDS}
                values[key] = row
            if intervals:
                for (key, k), summary in helper.summarize(metrics).items():
                    values[key][k] = summary
        rows[group] = dict(population=info, scales=values)
    return rows


def pick(grid, labels, metric):
    """Exact ties go to identity first, then lower numeric multipliers."""
    def tie(a):
        parts = a if isinstance(a, tuple) else (a,)
        return (*(abs(round(v-1, 10)) for v in parts), *parts)

    choices = {}
    for g in np.unique(labels):
        mask = labels == g
        means = {a: float(f[metric][mask].mean()) for a, f in grid.items()}
        best = min(means, key=lambda a: (means[a], *tie(a)))
        choices[str(g)] = dict(scale=best, n=int(mask.sum()), value=means[best],
                               id_value=means[(1., 1.) if isinstance(best, tuple) else 1.])
    return choices


def use(grid, labels, choices):
    a0 = next(iter(grid))
    fields = {key: np.empty(len(labels), dtype=a.dtype) for key, a in grid[a0].items()}
    for g in np.unique(labels):
        mask = labels == g
        # Fall back to one if a group is absent in the selection data.
        scale = choices.get(g, {"scale": (1., 1.) if isinstance(a0, tuple) else 1.})["scale"]
        for name, values in grid[scale].items():
            fields[name][mask] = values[mask]
    return fields


def eq_grid(paths, data, w, h, draws, iso_values, base):
    equal = np.full(w.shape, 1 / w.shape[1])
    return {float(a): from_paths(center_scale(paths, equal, a), data["target_rel"], w, h, draws, center_w=w)
            if a != 1. else base for a in iso_values}


def settings(iso, aniso, iso_eq, select):
    cases = {
        "isotropic_coarse_FDE": ("iso", "obs", "minFDE"),
        "isotropic_fine_FDE": ("iso", "bin", "minFDE"),
        "oracle_future_FDE": ("iso", "label", "minFDE"),
        "anisotropic_coarse_FDE": ("aniso", "obs", "minFDE"),
        "isotropic_coarse_ADE": ("iso", "obs", "minADE"),
        "equal_center_coarse_FDE": ("equal_center", "obs", "minFDE"),
        "equal_center_coarse_ADE": ("equal_center", "obs", "minADE"),
    }
    for e in ES_NAMES:
        cases[f"equal_center_coarse_{e}_ES"] = "equal_center", "obs", f"{e}_ES"
        for scope, g in (("coarse", "obs"), ("fine", "bin")):
            cases[f"{scope}_{e}_ES"] = "iso", g, f"{e}_ES"
    grids = dict(iso=iso, aniso=aniso, equal_center=iso_eq)
    return {name: dict(grid=kind, labels=k, metric=metric, by_group=pick(grids[kind], select[k], metric))
            for name, (kind, k, metric) in cases.items()}


def row_diffs(before, after, mask):
    x, y = before["minFDE"][mask], after["minFDE"][mask]
    return float((x-y).sum()), float(x.sum())


def recording_sensitivity(comparisons, masks, dev):
    out = {name: {} for name in comparisons}
    for group, scope in masks.items():
        for name in comparisons:
            out[name][group] = dict(leave_one={}, summed_gain_by_recording_m={})
        for clip in np.unique(dev["scene"]):
            left = dev["scene"] != clip
            keep = scope & left
            dropped = scope & ~left
            if not keep.any():
                for name, (before, after) in comparisons.items():
                    entry = out[name][group]
                    entry["summed_gain_by_recording_m"][clip] = row_diffs(before, after, dropped)[0]
                    entry["leave_one"][clip] = dict(population=population(dev["track"][keep])[0])
                continue
            helper = TrackBootstrap(dev["track"][keep])
            units = np.ones(keep.sum())
            data = {}
            for name, (before, after) in comparisons.items():
                gain, _ = row_diffs(before, after, dropped)
                out[name][group]["summed_gain_by_recording_m"][clip] = gain
                x, y = before["minFDE"][keep], after["minFDE"][keep]
                data[name, "before"] = x, units
                data[name, "after"] = y, units
                data[name, "delta"] = y-x, units
                data[name, "relative_gain"] = x-y, x
            packed = helper.summarize(data)
            for name in comparisons:
                out[name][group]["leave_one"][clip] = dict(
                    population=helper.population, **{key: packed[name, key]
                    for key in ("before", "after", "delta", "relative_gain")})
        for name, (before, after) in comparisons.items():
            gain, _ = row_diffs(before, after, scope)
            entry = out[name][group]
            by_scene = entry["summed_gain_by_recording_m"]
            largest = max(by_scene, key=by_scene.get) if by_scene else None
            entry["maximum_contribution_share"] = by_scene[largest] / gain if gain > 0 and largest else None
            entry["maximum_contributor"] = largest
            entry["summed_net_gain_m"] = gain
    return out


def dev_effects(compare, masks, d):
    summary = Summary(masks, d["track"], d["scene"] != "crowds_zara02")
    sensitivity = recording_sensitivity(compare, trim(masks), d)
    return {name: dict(groups=summary.table(after, before), recordings=sensitivity[name])
            for name, (before, after) in compare.items()}


def descriptive(iso, aniso, dev, masks, select_d, select_masks, select_iso, select_aniso):
    checked = [.6, .8, 1.2, 1.4]
    flatness = {g: {str(a): float(iso[a]["minFDE"][m].mean() / iso[1.]["minFDE"][m].mean())
                    for a in checked} for g, m in trim({name: masks[name] for name in OBS}).items()}

    def tables(grid, groups):
        return {g: {scale_key(a): float(f["fast_steps"][m].mean()) for a, f in grid.items()}
                for g, m in trim(groups).items()}

    return dict(flatness=flatness,
                mean_endpoint_bias_dev={g: float(iso[1.]["mean_bias"][m].mean()) for g, m in trim(masks).items()},
                mean_endpoint_bias_selection={g: float(select_iso[1.]["mean_bias"][m].mean())
                                              for g, m in trim(select_masks).items()},
                fast_steps=dict(dev_iso=tables(iso, masks), dev_aniso=tables(aniso, {g: masks[g] for g in OBS}),
                                selection_iso=tables(select_iso, select_masks),
                                selection_aniso=tables(select_aniso, {g: select_masks[g] for g in OBS})))

