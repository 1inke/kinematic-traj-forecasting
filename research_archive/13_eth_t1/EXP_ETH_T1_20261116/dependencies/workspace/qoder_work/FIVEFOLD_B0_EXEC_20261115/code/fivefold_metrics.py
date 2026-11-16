"""Per-window and aggregate metrics for one population (numpy only).

Standard ETH/UCY convention (same as EXP-156 evaluate): focal agent, best-of-20,
minADE and minFDE minimised INDEPENDENTLY over the 20 candidates, mean over test windows;
fold average = equal weight over the five folds.

Also reported: same-path FDE (FDE of the ADE-best candidate), Top1 (argmax logit),
top-k by logit (k=1,3,5,10), project coverage (same slot ADE<=.2 and FDE<=.4) by
change group, constant-velocity reference, and DESCRIPTIVE score-calibration numbers
for the J-winner cell (prepares the later calibration study; no gate uses them).
"""
from __future__ import annotations

import numpy as np

TOPK = (1, 3, 5, 10)
GROUP_NAMES = {0: "stable", 1: "transition", 2: "change"}


def per_window(paths, logits, future, past_focal):
    paths = np.asarray(paths, np.float64); future = np.asarray(future, np.float64)
    logits = np.asarray(logits, np.float64)
    if paths.ndim != 4 or paths.shape[1:] != (20, 12, 2) or logits.shape != paths.shape[:2]:
        raise ValueError(f"BAD_POPULATION_SHAPE {paths.shape} {logits.shape}")
    e = np.linalg.norm(paths - future[:, None], axis=-1)            # [N,K,T]
    ade, fde = e.mean(-1), e[..., -1]
    n = np.arange(len(ade))
    best_ade = ade.argmin(1)
    top1 = logits.argmax(1)
    order = np.argsort(-logits, axis=1, kind="stable")
    good = (ade <= .2) & (fde <= .4)
    j_winner = np.maximum(ade / .2, fde / .4).argmin(1)
    z = logits - logits.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    p = np.exp(logp)
    onehot = np.zeros_like(p); onehot[n, j_winner] = 1
    v0 = past_focal[:, -1].astype(np.float64) - past_focal[:, -2]
    cv = past_focal[:, -1, None].astype(np.float64) + np.arange(1, 13)[None, :, None] * v0[:, None]
    cv_e = np.linalg.norm(cv - future, axis=-1)
    out = dict(minADE=ade.min(1), minFDE=fde.min(1), same_path_FDE=fde[n, best_ade],
               top1_ADE=ade[n, top1], top1_FDE=fde[n, top1],
               coverage=good.any(1), top1_good=good[n, top1], n_good=good.sum(1),
               nll_J=-logp[n, j_winner], brier_J=((p - onehot) ** 2).sum(1),
               top1_is_J=(top1 == j_winner), top1_prob=p.max(1),
               cv_ADE=cv_e.mean(1), cv_FDE=cv_e[:, -1])
    for k in TOPK:
        idx = order[:, :k]
        out[f"top{k}_minADE"] = np.take_along_axis(ade, idx, 1).min(1)
        out[f"top{k}_minFDE"] = np.take_along_axis(fde, idx, 1).min(1)
    return out


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total, err = len(conf), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            err += m.sum() / total * abs(conf[m].mean() - correct[m].mean())
    return float(err)


def aggregate(w, group):
    def block(m):
        r = {k: float(np.mean(v[m])) for k, v in w.items() if v.dtype != bool and k != "n_good"}
        r.update(coverage=float(w["coverage"][m].mean()), top1_good=float(w["top1_good"][m].mean()),
                 top1_is_J=float(w["top1_is_J"][m].mean()), n=int(m.sum()),
                 top1_conditional_recall=float(w["top1_good"][m & w["coverage"]].mean())
                 if (m & w["coverage"]).any() else None,
                 ece_top1_is_J=ece(w["top1_prob"][m], w["top1_is_J"][m].astype(float)))
        return r
    group = np.asarray(group)
    out = {"overall": block(np.ones(len(group), bool))}
    for g, name in GROUP_NAMES.items():
        if (group == g).any():
            out[name] = block(group == g)
    return out


def decide_budget(val_rows):
    """val_rows: {fold: {E: {'minADE':..,'minFDE':..}}} for E in (64,128). Returns decision dict."""
    from fivefold_common import BUDGET_FLAG_GAIN, BUDGET_MIN_GAIN, FOLDS
    missing = [f for f in FOLDS if f not in val_rows or 64 not in val_rows[f] or 128 not in val_rows[f]]
    if missing:
        raise ValueError(f"BUDGET_ROWS_MISSING {missing}")
    gains = {f: (val_rows[f][64]["minADE"] - val_rows[f][128]["minADE"]) / val_rows[f][64]["minADE"]
             for f in FOLDS}
    fde_delta = {f: val_rows[f][128]["minFDE"] - val_rows[f][64]["minFDE"] for f in FOLDS}
    mean_gain = float(np.mean(list(gains.values())))
    mean_fde_delta = float(np.mean(list(fde_delta.values())))
    e_star = 128 if (mean_gain >= BUDGET_MIN_GAIN and mean_fde_delta <= 0) else 64
    return dict(E_star=e_star, mean_relative_minADE_gain_64_to_128=mean_gain,
                mean_minFDE_delta_128_minus_64=mean_fde_delta,
                per_fold_relative_minADE_gain=gains, per_fold_minFDE_delta=fde_delta,
                possibly_not_saturated_at_128=bool(e_star == 128 and mean_gain >= BUDGET_FLAG_GAIN),
                rule="E*=128 iff equal-weight mean relative val minADE gain >=1% and mean val minFDE not worse")
