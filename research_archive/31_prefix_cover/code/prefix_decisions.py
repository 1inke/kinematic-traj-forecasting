"""Compute window-paired, track-clustered prefix-cover decisions."""
import numpy as np

from advisor_study import read, usable
from prefix_train import FOLDS, SEEDS, ROOT
from tracks_analysis import BOOT_SEED, N_BOOT, cluster_draws, interval, ratio

KEYS = tuple(f"riskp_minFDE{k}" for k in (1, 3, 5, 20))


def collect(root, fold, arms):
    reports, values = {}, {}
    reference_ids = None
    fast = None
    for arm in arms:
        run_reports, arrays = [], []
        for seed in SEEDS:
            name = f"{arm}_s{seed}" if fold == "C0" else f"{arm}_{fold}_s{seed}"
            run = root/"runs"/name
            assert usable(run, deep=True), name
            run_reports.append(read(run/"RESULT.json"))
            with np.load(run/"dev_metrics.npz", allow_pickle=False) as d:
                if reference_ids is None:
                    reference_ids = d["ids"]
                    fast = d["fast"]
                assert np.array_equal(reference_ids, d["ids"]), name
                assert np.array_equal(fast, d["fast"]), name
                arrays.append({k: d[k] for k in (*KEYS, "conf_minFDE3", "spread_m", "brier_minFDE")
                               if k in d.files})
        reports[arm] = run_reports
        values[arm] = {k: np.mean([r[k] for r in arrays], axis=0, dtype=np.float64) for k in arrays[0]}
    if fold == "C0":
        tracks = np.array([s.split("|", 1)[0] for s in reference_ids.astype(str)])
        recordings = np.array(["C0"] * len(tracks))
    else:
        with np.load(ROOT/"tracks"/f"{fold}.npz", allow_pickle=True) as z:
            assert np.array_equal(reference_ids, z["ids"]), fold
            tracks = z["tracks"].astype(str)
            recordings = z["recording"].astype(str)
    return tracks, recordings, fast, reports, values


def draws_for(tracks, fold):
    if fold == "C0":
        return cluster_draws(tracks)
    units, codes = np.unique(tracks, return_inverse=True)
    sequences = np.random.SeedSequence(BOOT_SEED).spawn(len(FOLDS))
    rng = np.random.default_rng(sequences[FOLDS.index(fold)])
    chosen = rng.integers(len(units), size=(N_BOOT, len(units)), dtype=np.int32)
    counts = np.zeros_like(chosen)
    np.add.at(counts, (np.arange(N_BOOT)[:, None], chosen), 1)
    return units, codes, counts


def compare(base, arm, cluster, level=.95):
    units, lookup, draws = cluster
    a = np.bincount(lookup, weights=base, minlength=len(units))
    change = np.bincount(lookup, weights=base-arm, minlength=len(units))
    samples = ratio(draws @ change, draws @ a)
    return {"gain": float(change.sum()/a.sum()), "interval": interval(samples, level),
            "level": level, "base_m": float(base.mean()), "arm_m": float(arm.mean())}, samples


def report_fold(root, fold, arms):
    tracks, recordings, fast, reports, data = collect(root, fold, arms)
    cluster = draws_for(tracks, fold)
    units, codes, _ = cluster
    counts = np.bincount(codes)
    cov = data["N-cov"]
    base = data["A"]
    comparisons, samples = {}, {}
    for key in KEYS:
        comparisons[key], samples[key] = compare(base[key], cov[key], cluster)
    level = .983 if fold == "C0" else .975
    specificity, specificity_draws = {}, {}
    for arm in arms:
        if arm not in ("A","N-cov"):
            specificity[arm], specificity_draws[arm] = compare(data[arm][KEYS[1]], cov[KEYS[1]], cluster, level)
    native, _ = compare(data["N-conf"]["conf_minFDE3"], cov[KEYS[1]], cluster, level)
    distinct = {arm: float(np.mean([r["distinct3"] for r in reports[arm]]))
                for arm in arms if arm != "A"}
    mc1 = distinct["N-win"] == 0 and all(v >= .30 for k, v in distinct.items() if k != "N-win")
    reduction = float((base[KEYS[1]] - cov[KEYS[1]]).mean())
    excess = float(((base[KEYS[1]] - base[KEYS[3]]) - (cov[KEYS[1]] - cov[KEYS[3]])).mean())
    spread = {arm: float(d["spread_m"].mean()/base["spread_m"].mean()-1) for arm,d in data.items()}
    splits = {}
    for name in np.unique(recordings):
        keep = recordings == name
        if len(set(recordings)) > 1:
            splits[name] = compare(base[KEYS[1]][keep], cov[KEYS[1]][keep], cluster_draws(tracks[keep]))[0]
    high_speed = (compare(base[KEYS[1]][fast], cov[KEYS[1]][fast], cluster_draws(tracks[fast]))[0]
                  if fold == "eth" and fast.any() else None)
    report = dict(windows=len(tracks), tracks=len(units), top10_window_share=float(np.sort(counts)[-10:].sum()/len(tracks)),
                  comparisons=comparisons, specificity=specificity, vs_native_conf=native,
                  distinct3=distinct, MC1=mc1, E3_reduction_m=excess, FDE3_reduction_m=reduction,
                  E3_fraction=excess/reduction if reduction > 0 else None,
                  MC2=reduction > 0 and excess >= .5*reduction,
                  spread_change=spread, MC3=all(abs(v) <= .15 for v in spread.values()),
                  MC4={a: [r["mc4_top3_set_stable"] for r in rr] for a,rr in reports.items()} if fold == "C0" else {},
                  per_arm={arm: {k: float(v.mean()) for k,v in entry.items()} for arm,entry in data.items()},
                  recording=splits, eth_high_speed=high_speed)
    return report, samples, specificity_draws


def screen(root, arms):
    report, _, _ = report_fold(root, "C0", arms)
    m = report["comparisons"]
    gain3, gain5, gain20 = [m[key]["gain"] for key in KEYS[1:]]
    gates = dict(C1=gain3 >= .015 and m[KEYS[1]]["interval"][0] > 0,
                 C2=gain5 >= .01, C3=gain20 >= -.01,
                 C4=all(v["gain"] > 0 for v in report["specificity"].values()),
                 C5=report["MC1"])
    if all(gates.values()):
        decision = "GO"
    elif gates["C1"] and gates["C2"] and not gates["C3"]:
        decision = "TRADE-OFF"
    elif gates["C1"] and gates["C2"] and gates["C3"] and not gates["C4"]:
        decision = "REDIAGNOSE"
    else:
        decision = "STOP"
    return dict(decision=decision, gates=gates, **report)


def confirm(root, arms):
    folds = {fold: report_fold(root, fold, arms) for fold in FOLDS}
    all_reports = {f:v[0] for f,v in folds.items()}
    summary = {}
    for key in KEYS:
        estimates = [r["comparisons"][key]["gain"] for r in all_reports.values()]
        summary[key] = dict(gain=float(np.mean(estimates)),
                            interval=interval(np.mean([v[1][key] for v in folds.values()], axis=0), .95),
                            leave_one_fold={f:float(np.mean([e for i,e in enumerate(estimates) if i != j]))
                                            for j,f in enumerate(FOLDS)})
    specificity = {a:dict(gain=float(np.mean([v["specificity"][a]["gain"] for v in all_reports.values()])),
                          interval=interval(np.mean([v[2][a] for v in folds.values()], axis=0), .975))
                   for a in ("N-conf","N-win")}
    g1, g3, g5, g20 = (summary[k] for k in KEYS)
    reduction = float(np.mean([r["FDE3_reduction_m"] for r in all_reports.values()]))
    excess = float(np.mean([r["E3_reduction_m"] for r in all_reports.values()]))
    gates = dict(P1=g3["gain"] >= .02 and g5["gain"] >= .015 and min(g3["interval"][0],g5["interval"][0]) > 0,
                 P2=sum(r["comparisons"][KEYS[1]]["gain"] > 0 for r in all_reports.values()) >= 4,
                 P3=all_reports["eth"]["comparisons"][KEYS[1]]["gain"] >= -.01,
                 G1=g20["gain"] >= -.01 and min(r["comparisons"][KEYS[3]]["gain"] for r in all_reports.values()) >= -.02,
                 G2=g1["gain"] >= -.01,
                 S1=specificity["N-conf"]["gain"] >= .01 and specificity["N-conf"]["interval"][0] > 0,
                 S2=specificity["N-win"]["gain"] >= .01 and specificity["N-win"]["interval"][0] > 0,
                 M=all(r["MC1"] for r in all_reports.values()) and reduction > 0 and excess >= .5*reduction)
    if all(gates.values()):
        decision = "F-PASS"
    elif gates["P1"] and not gates["G1"]:
        decision = "TRADE-OFF"
    elif all(gates[k] for k in ("P1","P2","P3","G1","G2")) and not all(gates[k] for k in ("S1","S2","M")):
        decision = "NONSPECIFIC"
    else:
        decision = "STOP"
    return dict(decision=decision, gates=gates, folds=all_reports, aggregate=summary,
                specificity=specificity, MC2=dict(excess_m=excess, reduction_m=reduction))
