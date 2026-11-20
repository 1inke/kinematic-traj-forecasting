"""Summaries and registered thresholds for the width diagnosis."""

import numpy as np

from width_model import TERMS, WEIGHTS

TERM_NAMES = (*TERMS, "total")
ALPHAS = (.5, .6, .7, .8, .9, 1., 1.1)
C_ALPHAS = (.25, .5, .75, 1.)
THRESHOLDS = ("1/20", "1/40", "1/80")
GROUPS = ("remain", "start", "steady", "other")


def rate(n, d):
    return float(n/d) if d else None


def masks(g):
    return {"stationary": np.isin(g, ("remain", "start")), **{name: g == name for name in GROUPS}}


def errors(paths, truth):
    d = np.linalg.norm(paths.astype(np.float64) - truth[:, None], axis=-1)
    spread2 = np.square(paths[:, :, -1].astype(np.float64) - paths[:, :, -1].mean(1, dtype=np.float64)[:, None])
    return np.stack((d.mean(-1).min(1), d[:, :, -1].min(1), np.sqrt(spread2.sum(2).mean(1))), axis=1)


def metric_table(values, g, nearest_reference=None):
    out = {}
    for name, m in masks(g).items():
        if m.any():
            row = dict(zip(("minADE", "minFDE", "mean_spread"), np.mean(values[m], axis=0).tolist()))
            row.update(n=int(m.sum()), median_spread=float(np.median(values[m, 2])))
            if nearest_reference is not None:
                row["reference_minFDE"] = float(nearest_reference[m].mean())
                row["gap_to_reference_m"] = row["minFDE"] - row["reference_minFDE"]
            out[name] = row
    return out


def gradient_summary(a, winners_rate):
    T, G = a["S"], a["anchor_grad_norm"]
    winner = np.take_along_axis(G, a["winner"][:, None, None].repeat(G.shape[1], 1), 2)[:, :, 0]
    loser_norm = G.sum(2) - winner
    loser_s = T - a["winner_S"]
    out, templates, diversity = {}, {}, {}
    for name, m in masks(a["group"]).items():
        local = T[m]
        totals = G[m].sum((0, 2))
        ls = loser_norm[m].sum(0)
        summary = {}
        for i, key in enumerate(TERM_NAMES):
            summary[key] = dict(mean_S=float(local[:, i].mean()), sum_S=float(local[:, i].sum()),
                                 mean_abs_S=float(np.abs(local[:, i]).mean()),
                                 mean_winner_S=float(a["winner_S"][m, i].mean()),
                                 mean_loser_S=float(loser_s[m, i].mean()),
                                 pooled_loser_gradient_fraction=rate(ls[i], totals[i]))
        out[name] = dict(n=int(m.sum()), weighted=summary,
                         effective_coverage_candidates_median=float(np.median(a["effective_candidates"][m])))
        templates[name] = dict(zip(TERM_NAMES, a["template_radial"][m].sum(0).astype(float)))
        wider = np.maximum(-local[:, :6], 0).sum(0)
        diversity[name] = dict(n=int(m.sum()), active_pairs=float(a["active_pairs"][m].mean()),
                               widening_share=rate(wider[3], wider.sum()),
                               widening_magnitude_by_term=dict(zip(TERMS, wider.astype(float))))
    stationary = masks(a["group"])["stationary"]
    sat_table = {}
    for label, slice_ in {**{k: winners_rate < 1/int(k[2:]) for k in THRESHOLDS},
                           "ge_1/40": winners_rate >= 1/40, "all": np.ones(20, bool)}.items():
        s = a["saturated"][stationary][:, slice_].sum()
        i = a["inward_saturated"][stationary][:, slice_].sum()
        o = a["outward_saturated"][stationary][:, slice_].sum()
        n = int(stationary.sum())*int(slice_.sum())*6
        sat_table[label] = dict(components=n, saturation=rate(s, n), inward_all=rate(i, n),
                                outward_all=rate(o, n), inward_given_saturation=rate(i, s))
    return out, templates, diversity, sat_table


def loss_table(values, groups):
    out = {}
    for alpha, v in values.items():
        array = np.concatenate(v).astype(np.float64)
        result = {}
        for name, mask in masks(groups).items():
            if not mask.any():
                continue
            means = array[mask].mean(0)
            weighted = means * np.asarray(WEIGHTS)
            chosen = [0, 2, 4, 5]
            result[name] = dict(n=int(mask.sum()), raw=dict(zip(TERMS, means.tolist())),
                                weighted=dict(zip(TERMS, weighted.tolist())),
                                total=float(weighted.sum()), no_diversity=float(weighted.sum()-weighted[3]),
                                geometric_unweighted=float(means[chosen].sum()),
                                geometric_weighted=float(weighted[chosen].sum()))
        out[str(alpha)] = result
    return out


def frame(data, groups, covariances=None):
    past, truth = [np.asarray(x, np.float64) for x in (data["past"][:, 0], data["future"])]
    last = past[:, -1]
    disp = last - past[:, -4]
    length = np.linalg.norm(disp, axis=1)
    default = length <= .03
    heading = np.empty_like(disp)
    heading[default] = (1, 0)
    heading[~default] = disp[~default] / length[~default, None]
    stationary = masks(groups)["stationary"]

    def local(x):
        cos, sin = heading.T
        dx, dy = x.T
        return np.stack((cos*dx + sin*dy, -sin*dx + cos*dy), axis=1)

    endpoint = local(truth[:, -1] - last)
    points = endpoint[groups == "start"]
    length_start = np.linalg.norm(points, axis=1)
    nonzero = length_start > 1e-12
    unit = points[nonzero] / length_start[nonzero, None]
    rho = float(np.linalg.norm(unit.mean(0))) if len(unit) else None
    cov = np.cov(endpoint[stationary], rowvar=False, bias=True)
    residual = local(truth[:, -1] - last - 12*(last-past[:, -2]))
    residual_cov = np.cov(residual[stationary], rowvar=False, bias=True)
    data = dict(stationary_windows=int(stationary.sum()), default_heading=float(default[stationary].mean()),
                small_displacement_heading=float(((length > .03)&(length <= .12))[stationary].mean()),
                start_windows=len(points), zero_net_start=int((~nonzero).sum()), start_resultant_length=rho,
                angle_histogram_45_degree=np.histogram(np.arctan2(unit[:,1], unit[:,0]),
                                                       bins=np.linspace(-np.pi, np.pi, 9))[0].tolist(),
                truth_anisotropy=anisotropy(cov), truth_cv_residual_anisotropy=anisotropy(residual_cov))
    if covariances is not None:
        data["model_anisotropy"] = covariances
    return data


def anisotropy(cov):
    e = np.linalg.eigvalsh(cov)
    return rate(e[-1], e[0]) if e[0] > 1e-12 else None


def decide(seeds, A, B, C, D, F):
    per = {}
    for seed in seeds:
        key = str(seed)
        a, b, c, d = (part[key] for part in (A, B, C, D))
        remain = a["groups"]["remain"]
        loser = remain["weighted"]["total"]["pooled_loser_gradient_fraction"]
        losses = {float(x): y["stationary"]["total"] for x, y in b["fit"].items()}
        geom = [b["fit"]["0.8"][g]["geometric_unweighted"] -
                b["fit"]["1.0"][g]["geometric_unweighted"] for g in ("start", "remain")]
        primary = c["variants"]["1/40"]["0.5"]
        gain = primary["groups"]["stationary"]["relative_minFDE_gain"]
        start = primary["groups"]["start"]["relative_minFDE_gain"]
        mean_reachable = primary["reachable"]
        inward = a["saturation"]["1/40"]["inward_all"]
        active = d["stationary"]["fit_active_pairs_all_windows"]
        widening = d["stationary"]["widening_share"]
        optimum_shift = b["alpha_star"] - b["alpha_star_no_diversity"]
        per[key] = dict(
            M1=dict(passes=loser is not None and loser <= .10 and remain["effective_coverage_candidates_median"] <= 1.5,
                    loser_fraction=loser, coverage_effective_count=remain["effective_coverage_candidates_median"]),
            M2=dict(passes=all(losses[x] >= losses[1.] for x in (.7,.8,.9)),
                    loss_increases={str(x): losses[x]/losses[1.]-1 for x in (.7,.8,.9)}),
            M2_prime=dict(passes=any(losses[x] <= .99*losses[1.] for x in losses if x <= .9),
                          best_relative_reduction=1-min(v for x,v in losses.items() if x <= .9)/losses[1.]),
            M3=dict(passes=geom[0] > 0 and geom[1] <= 0, start_delta=geom[0], remain_delta=geom[1]),
            M4=dict(passes=gain >= .03 and start >= -.05, relative_gain=gain, start_relative_gain=start),
            M5=dict(passes=mean_reachable is not None and mean_reachable <= .50 and inward is not None and inward >= .30,
                    reachable=mean_reachable, inward_saturation=inward),
            M6=dict(passes=(active >= .05 and widening is not None and widening >= .30) or optimum_shift >= .1-1e-12,
                    active_pairs=active, widening_share=widening, alpha_star_decrease=optimum_shift))
    flags = {}
    for name in next(iter(per.values())):
        count = sum(x[name]["passes"] for x in per.values())
        directions = name != "M4" or all(x[name]["relative_gain"] > 0 for x in per.values())
        flags[name] = dict(active=count >= 4 and directions, seeds_passing=count)
        if name == "M4":
            flags[name]["all_five_improve"] = directions
    changes = {str(seed): F[f"E1B_E64_s{seed}"]["stationary"]["median_spread"] /
                          F[f"E1B_E8_s{seed}"]["stationary"]["median_spread"] - 1 for seed in (42,43)}
    flags["M7"] = dict(active=all(abs(x) <= .05 for x in changes.values()), relative_change=changes)
    return dict(markers=flags, per_seed=per)
