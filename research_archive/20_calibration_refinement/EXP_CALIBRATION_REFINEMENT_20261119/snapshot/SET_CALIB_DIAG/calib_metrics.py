"""Spread, cross-fitted HPD and reliability from existing predictions."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, logsumexp

SEED = 20261119
LEVELS = np.array([.50, .80, .90, .95])
H_GRID = np.array([.02, .03, .05, .075, .10, .15, .20, .30, .50])
K_GRID = np.array([.05, .10, .20, .30, .50])
WEIGHTS = ("equal", "weighted")
VARIANTS = ("global", "adaptive", "global_half", "global_double")
DRAW_COUNT = 1000


def moments(endpoints, targets, weights):
    center = np.einsum("nk,nkd->nd", weights, endpoints)
    radius2 = np.square(endpoints - center[:, None]).sum(-1)
    spread = np.sqrt(np.einsum("nk,nk->n", weights, radius2))
    return spread, targets - center


def line(s2, r2):
    valid = (s2 > 0) & (r2 > 0) & np.isfinite(s2) & np.isfinite(r2)
    x = np.log(s2[valid]) * .5
    y = np.log(r2[valid]) * .5
    if len(x) < 2 or np.ptp(x) == 0:
        raise ValueError("Insufficient spread variation for slope")
    xc = x - x.mean()
    beta = np.dot(xc, y - y.mean()) / np.dot(xc, xc)
    return float(beta), float(y.mean() - beta * x.mean())


def spread_skill(fit_s, spread, error, heading, masks):
    edges = np.quantile(fit_s, np.arange(.1, 1, .1))
    bins = np.searchsorted(edges, spread, side="right")
    squared_s, squared_r = spread ** 2, np.square(error).sum(1)
    cos, sin = heading.T
    dx, dy = error.T
    local = np.column_stack((cos * dx + sin * dy, -sin * dx + cos * dy))

    def means(indices):
        idx = bins[indices]
        count = np.bincount(idx, minlength=10)
        denom = np.where(count > 0, count, np.nan)
        s2 = np.bincount(idx, weights=squared_s[indices], minlength=10) / denom
        r2 = np.bincount(idx, weights=squared_r[indices], minlength=10) / denom
        return count, s2, r2

    count, s2, r2 = means(np.arange(len(spread)))
    beta, intercept = line(s2, r2)
    rows = []
    for b, n in enumerate(count):
        if n == 0:
            continue
        s, r = np.sqrt(s2[b]), np.sqrt(r2[b])
        bias = np.linalg.norm(local[bins == b].mean(0))
        rows.append(dict(bin=b + 1, n=int(n), S_m=float(s), R_m=float(r), B_m=float(bias),
                         R_over_S=float(r / s) if s else None, B_over_R=float(bias / r) if r else None))
    rng = np.random.default_rng(SEED)
    sample_betas = [line(*means(rng.integers(len(spread), size=len(spread)))[1:])[0]
                    for _ in range(1000)]
    groups = {name: dict(n=int(mask.sum()), mean_s_m=float(spread[mask].mean()),
                         mean_r_m=float(np.sqrt(squared_r[mask]).mean()))
              for name, mask in masks.items()}
    return dict(beta=beta, intercept=intercept, beta_ci95=np.quantile(sample_betas, [.025, .975]).tolist(),
                cuts_m=edges.tolist(), bins=rows, groups=groups,
                low_R_over_S=rows[0]["R_over_S"], high_R_over_S=rows[-1]["R_over_S"])


def log_density(d2, log_weights, h):
    h = np.broadcast_to(h, (len(d2),))
    return logsumexp(log_weights - .5 * d2 / h[:, None] ** 2, axis=1) - np.log(2 * np.pi * h ** 2)


def select_bandwidths(d2, log_weights, scale, half):
    all_h = {name: np.empty(len(d2)) for name in VARIANTS}
    selection = {}
    for evaluated in (0, 1):
        source, target = half != evaluated, half == evaluated
        selection["AB"[evaluated]] = {}
        for mode, grid, basis in (("global", H_GRID, np.ones(len(scale))),
                                  ("adaptive", K_GRID, scale)):
            means = [float(log_density(d2[source], log_weights[source], value * basis[source]).mean())
                     for value in grid]
            chosen = grid[np.argmax(means)]
            all_h[mode][target] = chosen * basis[target]
            selection["AB"[evaluated]][mode] = dict(chosen=float(chosen), grid=grid.tolist(),
                                                     mean_log_density=means)
    all_h["global_half"] = all_h["global"] * .5
    all_h["global_double"] = all_h["global"] * 2
    return all_h, selection


def hpd_mc(endpoints, weights, h, truth_log_density, uniforms, noise):
    """Draw from each mixture and integrate its highest-density region."""
    weights = weights / weights.sum(1, keepdims=True)
    count = uniforms.shape[1]
    discard = count // 10
    u, area = np.empty(len(h)), np.empty(len(h))
    for start in range(0, len(h), 32):
        block = slice(start, start + 32)
        centers, width, p = endpoints[block], h[block], weights[block]
        # Work in units of the local width, centered at the first kernel.
        scaled = (centers - centers[:, :1]) / width[:, None, None]
        cdf = np.cumsum(p, axis=1)
        cdf[:, -1] = 1.
        component = (uniforms[block, :, None] >= cdf[:, None]).sum(2)
        samples = scaled[np.arange(len(width))[:, None], component] + noise[block]
        dx = samples[:, :, None, 0] - scaled[:, None, :, 0]
        dy = samples[:, :, None, 1] - scaled[:, None, :, 1]
        np.square(dx, out=dx)
        np.square(dy, out=dy)
        dx += dy
        dx *= -.5
        np.exp(dx, out=dx)
        sampled_density = np.einsum("nmk,nk->nm", dx, p)
        assert (sampled_density > 0).all(), "Mixture sample density underflow"
        normalizer = 2 * np.pi * width ** 2
        sample_log = np.log(sampled_density) - np.log(normalizer[:, None])
        u[block] = (sample_log >= truth_log_density[block, None]).mean(1)
        threshold = np.partition(sampled_density, discard, axis=1)[:, discard, None]
        area[block] = normalizer * np.where(sampled_density >= threshold, 1 / sampled_density, 0).mean(1)
    assert np.isfinite(area).all() and (area > 0).all()
    return u, area


def coverage(endpoints, targets, log_weights, spread, half, masks, uniforms, noise, on_variant):
    scale = np.maximum(spread, 1e-12)
    d2 = np.square(endpoints - targets[:, None]).sum(2)
    bandwidths, fit = select_bandwidths(d2, log_weights, scale, half)
    result = dict(fitted_on_other_half=fit, adaptive_scale_floor_n=int((spread < 1e-12).sum()),
                  variants={})
    windows = {}
    p = np.exp(log_weights)
    for name, h in bandwidths.items():
        on_variant(name)
        log_truth = log_density(d2, log_weights, h)
        u, area = hpd_mc(endpoints, p, h, log_truth, uniforms, noise)
        result["variants"][name] = dict(
            mean_held_out_log_density=float(log_truth.mean()),
            width_m=[float(h.min()), float(h.max())],
            groups={g: dict(n=int(m.sum()), coverage=(u[m, None] <= LEVELS).mean(0).tolist(),
                            mean_HPD90_area_m2=float(area[m].mean())) for g, m in masks.items()})
        windows[f"u__{name}"], windows[f"area90__{name}"] = u, area
    return result, windows


def reliability(p, event, order):
    bucket = np.minimum(np.floor(p.ravel() * 10).astype(int), 9)
    n = np.bincount(bucket, minlength=10)
    total_p = np.bincount(bucket, weights=p.ravel(), minlength=10)
    total_y = np.bincount(bucket, weights=event.ravel(), minlength=10)
    rows = [dict(bin=b + 1, n=int(n[b]), mean_p=float(total_p[b] / n[b]),
                 event_rate=float(total_y[b] / n[b])) for b in range(10) if n[b]]
    return dict(ece=float(np.abs(total_p - total_y).sum() / p.size), bins=rows,
                rank_mean_p=np.take_along_axis(p, order, 1).mean(0).tolist(),
                rank_event_rate=np.take_along_axis(event, order, 1).mean(0).tolist())


def probability(logits, fde, good, half):
    logits = logits.astype(np.float64)
    logits -= logits.max(1, keepdims=True)
    winner = fde.argmin(1)
    tied = (fde == fde.min(1, keepdims=True)).sum(1) > 1
    order = np.argsort(-logits, axis=1, kind="stable")
    raw = log_softmax(logits, axis=1)
    scaled = np.empty_like(raw)
    temperatures = {}
    for evaluated in (0, 1):
        source, target = half != evaluated, half == evaluated
        z, choices = logits[source], winner[source]

        def nll(log_t):
            logp = log_softmax(z / np.exp(log_t), axis=1)
            return float(-logp[np.arange(len(z)), choices].mean())

        fit = minimize_scalar(nll, method="bounded", bounds=(-6, 6), options={"xatol": 1e-8})
        assert fit.success, fit.message
        log_t = min([-6., float(fit.x), 6.], key=nll)
        temp = float(np.exp(log_t))
        temperatures["AB"[evaluated]] = dict(T=temp, fit_nll=nll(log_t), at_boundary=abs(log_t) == 6)
        scaled[target] = log_softmax(logits[target] / temp, axis=1)
    assert np.array_equal(scaled.argmax(1), order[:, 0])
    event = np.arange(logits.shape[1])[None, :] == winner[:, None]
    return dict(cross_fitted_temperatures=temperatures, FDE_tie_window_n=int(tied.sum()),
                top1_winner_rate=float(event[np.arange(len(winner)), order[:, 0]].mean()),
                winner_nll=dict(raw=float(-raw[event].mean()), calibrated=float(-scaled[event].mean())),
                winner=dict(raw=reliability(np.exp(raw), event, order),
                            calibrated=reliability(np.exp(scaled), event, order)),
                joint=dict(raw=reliability(np.exp(raw), good, order),
                           calibrated=reliability(np.exp(scaled), good, order)))
