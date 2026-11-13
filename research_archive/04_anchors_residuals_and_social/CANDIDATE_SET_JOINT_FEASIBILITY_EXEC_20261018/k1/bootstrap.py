"""Family16 bootstrap: S-I and S-native × overall/change × cov/ADE/FDE/J."""
from __future__ import annotations
import numpy as np
from .config import BOOT_SEED, BOOT_DRAWS, BOOT_LO, BOOT_HI, SEEDS


def _group_mask(groups, name):
    if name == "overall":
        return np.ones(len(groups), dtype=bool)
    mapping = {"stable": 0, "transition": 1, "change": 2}
    return groups == mapping[name]


def per_window_metrics(paths, gt, quality=(0.2, 0.4)):
    d = paths.astype(np.float64) - gt[:, None].astype(np.float64)
    e = np.linalg.norm(d, axis=-1)
    ade = e.mean(-1)
    fde = e[:, :, -1]
    joint = np.maximum(ade / quality[0], fde / quality[1])
    good = (ade <= quality[0]) & (fde <= quality[1]) & (joint <= 1.0)
    return dict(
        covered=good.any(1),
        minADE=ade.min(1),
        minFDE=fde.min(1),
        minJ=joint.min(1),
    )


def aggregate(mask, m):
    if not mask.any():
        return None
    return dict(
        coverage=float(m["covered"][mask].mean()),
        minADE=float(m["minADE"][mask].mean()),
        minFDE=float(m["minFDE"][mask].mean()),
        mean_minJ=float(m["minJ"][mask].mean()),
        n=int(mask.sum()),
    )


def point_estimates(per_seed: dict):
    """
    per_seed[seed] = {
      'groups': (N,),
      'S': metrics dict, 'I': ..., 'native': ...
    }
    Returns nested point estimates averaged? Design: report per-seed and
    for bootstrap use track resampling across seeds.
    For point estimate of S-native etc we report each seed and mean of 3 seeds.
    """
    out = {}
    for contrast in ("S_minus_I", "S_minus_native"):
        out[contrast] = {}
        for g in ("overall", "change"):
            out[contrast][g] = {}
            for metric in ("coverage", "minADE", "minFDE", "mean_minJ"):
                vals = []
                for seed in SEEDS:
                    if seed not in per_seed:
                        continue
                    rec = per_seed[seed]
                    mask = _group_mask(rec["groups"], g)
                    a_name = "S"
                    b_name = "I" if contrast == "S_minus_I" else "native"
                    ag = aggregate(mask, rec[a_name])
                    bg = aggregate(mask, rec[b_name])
                    if ag is None or bg is None:
                        continue
                    vals.append(ag[metric] - bg[metric])
                out[contrast][g][metric] = dict(
                    per_seed={int(s): float(v) for s, v in zip(
                        [s for s in SEEDS if s in per_seed], vals
                    )} if len(vals) == len([s for s in SEEDS if s in per_seed]) else {},
                    mean=float(np.mean(vals)) if vals else None,
                    vals=vals,
                )
    return out


def family16_bootstrap(per_seed: dict, track_keys_by_seed: dict):
    """
    PCG64 seed 20261018, 100000 draws.
    Each draw: resample tracks with replacement (share track sample across seeds
    by using outer track universe — design: all seeds share track sampling).
    
    Practical approach matching design:
    - Collect unique tracks from seed42 outer (same windows across seeds? 
      Actually each seed has same outer IDs/windows; tracks align by sample index.
    - Resample window indices by track: group windows by track, resample tracks,
      take all windows of selected tracks.
    - Then resample 3 seeds with replacement.
    - Compute contrast on pooled resampled windows across the 3 drawn seeds.
    
    Simpler faithful version used here:
    - Outer windows are identical across seeds (same IDs).
    - Track keys identical across seeds for same row.
    - Draw: resample tracks -> window mask; draw 3 seeds with replacement;
      for each metric contrast, average the 3 seed values on the masked windows.
    """
    # Use first available seed for track structure
    seeds_present = [s for s in SEEDS if s in per_seed]
    if len(seeds_present) < 1:
        return dict(status="NO_SEEDS")
    ref = seeds_present[0]
    tracks = np.asarray(track_keys_by_seed[ref])
    uniq, inv = np.unique(tracks, return_inverse=True)
    n_tracks = len(uniq)
    n_win = len(tracks)
    groups = per_seed[ref]["groups"]

    rng = np.random.Generator(np.random.PCG64(BOOT_SEED))
    contrasts = [
        ("S_minus_I", "S", "I"),
        ("S_minus_native", "S", "native"),
    ]
    groups_need = ("overall", "change")
    metrics = ("coverage", "minADE", "minFDE", "mean_minJ")
    # store draws
    store = {
        f"{c}|{g}|{m}": []
        for c, _, _ in contrasts
        for g in groups_need
        for m in metrics
    }
    n_invalid_change = 0

    for d in range(BOOT_DRAWS):
        track_draw = rng.integers(0, n_tracks, size=n_tracks)
        # windows belonging to drawn tracks
        # map: for each drawn track id, include all windows with that track
        chosen_tracks = set(track_draw.tolist())
        # faster: count
        mask = np.isin(inv, track_draw)
        # actually with replacement of tracks: need to weight or expand.
        # Design: "有放回抽原track数" — bootstrap of tracks; standard is:
        # for each drawn track, include its windows (if track drawn k times,
        # include windows k times). For means this is equivalent to weighting.
        counts = np.bincount(track_draw, minlength=n_tracks)
        w = counts[inv].astype(np.float64)
        if w.sum() <= 0:
            continue
        seed_draw = rng.choice(seeds_present, size=3, replace=True)

        for cname, a, b in contrasts:
            for gname in groups_need:
                gmask = _group_mask(groups, gname)
                ww = w * gmask
                if gname == "change" and ww.sum() <= 0:
                    n_invalid_change += 1
                    for m in metrics:
                        store[f"{cname}|{gname}|{m}"].append(np.nan)
                    continue
                vals_s = []
                for sd in seed_draw:
                    rec = per_seed[sd]
                    ma = rec[a]
                    mb = rec[b]
                    # weighted means
                    def wmean(arr, weights):
                        s = weights.sum()
                        if s <= 0:
                            return np.nan
                        return float(np.dot(arr, weights) / s)
                    ca = wmean(ma["covered"].astype(np.float64), ww)
                    cb = wmean(mb["covered"].astype(np.float64), ww)
                    aa = wmean(ma["minADE"], ww)
                    ab = wmean(mb["minADE"], ww)
                    fa = wmean(ma["minFDE"], ww)
                    fb = wmean(mb["minFDE"], ww)
                    ja = wmean(ma["minJ"], ww)
                    jb = wmean(mb["minJ"], ww)
                    vals_s.append(dict(
                        coverage=ca - cb,
                        minADE=aa - ab,
                        minFDE=fa - fb,
                        mean_minJ=ja - jb,
                    ))
                # average across 3 drawn seeds
                for m in metrics:
                    store[f"{cname}|{gname}|{m}"].append(
                        float(np.nanmean([v[m] for v in vals_s]))
                    )
        if (d + 1) % 20000 == 0:
            print(f"  bootstrap {d+1}/{BOOT_DRAWS}", flush=True)

    results = {}
    for key, arr in store.items():
        a = np.asarray(arr, dtype=np.float64)
        valid = a[np.isfinite(a)]
        cname, gname, m = key.split("|")
        results.setdefault(cname, {}).setdefault(gname, {})[m] = dict(
            n_draws=int(len(a)),
            n_valid=int(len(valid)),
            point_hint=float(np.nanmean(valid)) if len(valid) else None,
            ci_bonferroni=(
                [float(np.quantile(valid, BOOT_LO)), float(np.quantile(valid, BOOT_HI))]
                if len(valid) >= 100 else None
            ),
            ci_95=(
                [float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))]
                if len(valid) >= 100 else None
            ),
        )
    return dict(
        status="OK",
        n_invalid_change_draws=n_invalid_change,
        family16=results,
        bonferroni_interval=[BOOT_LO, BOOT_HI],
    )


def go_k2_check(per_seed_eval: dict, boot: dict, secondary_ok: dict) -> dict:
    """
    GO_K2 from 02 §6.
    per_seed_eval[seed]['S'] = evaluate_set public rows (overall/change/...)
    per_seed_eval[seed]['native'] = ...
    per_seed_eval[seed]['I'] = ...
    """
    clauses = {}
    ok = True

    # S vs native hard gates on all 3 seeds
    for seed in SEEDS:
        if seed not in per_seed_eval:
            clauses[f"seed{seed}_present"] = False
            ok = False
            continue
        S = per_seed_eval[seed]["S"]
        N = per_seed_eval[seed]["native"]
        chS, chN = S["change"], N["change"]
        # change A/F ≥5% improve
        ade_ok = chS["minADE"] <= chN["minADE"] * 0.95 + 1e-15
        fde_ok = chS["minFDE"] <= chN["minFDE"] * 0.95 + 1e-15
        clauses[f"seed{seed}_change_ade5"] = dict(pass_=ade_ok, S=chS["minADE"], N=chN["minADE"])
        clauses[f"seed{seed}_change_fde5"] = dict(pass_=fde_ok, S=chS["minFDE"], N=chN["minFDE"])
        ok &= ade_ok and fde_ok
        for g in ("overall", "stable", "transition", "change"):
            # protect ≤1% for overall/stable/transition; change uses 5% above
            if g != "change":
                a_ok = S[g]["minADE"] <= N[g]["minADE"] * 1.01 + 1e-15
                f_ok = S[g]["minFDE"] <= N[g]["minFDE"] * 1.01 + 1e-15
                clauses[f"seed{seed}_{g}_protect"] = dict(pass_=a_ok and f_ok)
                ok &= a_ok and f_ok
            lost = S[g]["conditional_lost_frac"]
            lost_ok = True if lost is None else (lost <= 0.01 + 1e-15)
            clauses[f"seed{seed}_{g}_lost"] = dict(pass_=lost_ok, frac=lost)
            ok &= lost_ok
            m_ok = S[g]["motion_gate"]["status"] == "PASS"
            clauses[f"seed{seed}_{g}_motion"] = dict(pass_=m_ok, fails=S[g]["motion_gate"].get("fails"))
            ok &= m_ok

        # S-I change coverage per-seed point diff ≥0
        cov_diff = S["change"]["coverage"] - per_seed_eval[seed]["I"]["change"]["coverage"]
        clauses[f"seed{seed}_S_minus_I_change_cov"] = dict(pass_=cov_diff >= -1e-15, diff=cov_diff)
        ok &= cov_diff >= -1e-15

    # bootstrap CI gates
    fam = boot.get("family16", {})
    def ci(contrast, g, m):
        return fam.get(contrast, {}).get(g, {}).get(m, {})

    sn_cov = ci("S_minus_native", "change", "coverage")
    sn_ci = sn_cov.get("ci_bonferroni")
    cov_ci_ok = bool(sn_ci and sn_ci[0] > 0)
    clauses["boot_S_native_change_cov_lo"] = dict(pass_=cov_ci_ok, ci=sn_ci)
    ok &= cov_ci_ok

    for m in ("minADE", "minFDE", "mean_minJ"):
        sn = ci("S_minus_native", "change", m)
        sn_ci = sn.get("ci_bonferroni")
        # improvement => S-native < 0 for error metrics
        m_ok = bool(sn_ci and sn_ci[1] < 0)
        clauses[f"boot_S_native_change_{m}_hi"] = dict(pass_=m_ok, ci=sn_ci)
        ok &= m_ok

    si = ci("S_minus_I", "change", "mean_minJ")
    si_ci = si.get("ci_bonferroni")
    si_ok = bool(si_ci and si_ci[1] < 0)
    clauses["boot_S_I_change_minJ_hi"] = dict(pass_=si_ok, ci=si_ci)
    ok &= si_ok

    # secondary splits
    for split, info in secondary_ok.items():
        clauses[f"secondary_{split}"] = info
        if info.get("status") in ("BLOCKED", "NOT_DONE"):
            if split == "original_CAL":
                clauses["original_CAL_blocks_K2"] = True
                ok = False  # design: 原CAL不可用则K2暂不触发
            # for inner/FIT_cal must have no motion fail etc.
        elif not info.get("pass_", False):
            ok = False

    return dict(GO_K2=bool(ok), clauses=clauses)
