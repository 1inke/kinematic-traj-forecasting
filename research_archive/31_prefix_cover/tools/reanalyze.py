"""Verify the archived bytes and reproduce the C0 screening decision with NumPy."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from tracks_analysis import cluster_draws, interval, ratio

ARMS = ("A", "N-cov", "N-conf", "N-rand", "N-win")
SEEDS = (42, 43, 44)
KEYS = tuple(f"riskp_minFDE{k}" for k in (1, 3, 5, 20))
STAT_KEYS = (*KEYS, "conf_minFDE3", "spread_m", "brier_minFDE")


def read(path):
    return json.loads((ROOT / path).read_bytes())


def compare(base, arm, cluster, level=.95):
    units, lookup, draws = cluster
    a = np.bincount(lookup, weights=base, minlength=len(units))
    change = np.bincount(lookup, weights=base - arm, minlength=len(units))
    return dict(gain=float(change.sum() / a.sum()),
                interval=interval(ratio(draws @ change, draws @ a), level),
                level=level, base_m=float(base.mean()), arm_m=float(arm.mean()))


def main():
    for name, wanted in read("FILES.json").items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == wanted, name
    cloud = read("results/C0_RESULT.json")
    release = read("results/CLOUD_RUN_STATUS.json")
    done = read("results/DONE.json")
    assert release["status"] == "COMPLETE" and release["exit_code"] == 0
    assert done["status"] == "PASS" and done["c0"] == cloud["decision"]
    for name, wanted in done["files"].items():
        assert hashlib.sha256((ROOT / "results" / name).read_bytes()).hexdigest() == wanted
    reference_groups = None
    metrics, distinct, stable = {}, {}, {}
    for arm in ARMS:
        by_seed, receipts = [], []
        for seed in SEEDS:
            run = ROOT / "results" / "runs" / f"{arm}_s{seed}"
            receipt = read(run / "RESULT.json")
            contract = receipt["contract"]
            assert (contract["fold"], contract["arm"], contract["seed"]) == ("C0", arm, seed)
            assert receipt["status"] == "PASS" and receipt["updates"] == 21760
            assert receipt["parameters"] == 295474 and contract["epochs"] == 64
            receipts.append(receipt)
            with np.load(run / "window_metrics.npz", allow_pickle=False) as data:
                groups = data["track_group"]
                if reference_groups is None:
                    reference_groups = groups
                np.testing.assert_array_equal(groups, reference_groups)
                np.testing.assert_array_equal(data["riskp_minFDE20"], data["conf_minFDE20"])
                by_seed.append({key: data[key] for key in STAT_KEYS})
        metrics[arm] = {key: np.mean([row[key] for row in by_seed], axis=0, dtype=np.float64)
                        for key in STAT_KEYS}
        if arm != "A":
            distinct[arm] = float(np.mean([row["distinct3"] for row in receipts]))
        stable[arm] = [row["mc4_top3_set_stable"] for row in receipts]
    cluster = cluster_draws(reference_groups)
    base, cov = metrics["A"], metrics["N-cov"]
    comparisons = {key: compare(base[key], cov[key], cluster) for key in KEYS}
    specificity = {arm: compare(metrics[arm][KEYS[1]], cov[KEYS[1]], cluster, .983)
                   for arm in ARMS[2:]}
    native = compare(metrics["N-conf"]["conf_minFDE3"], cov[KEYS[1]], cluster, .983)
    mc1 = distinct["N-win"] == 0 and all(distinct[arm] >= .30 for arm in ARMS[1:-1])
    spread = {arm: float(row["spread_m"].mean() / base["spread_m"].mean() - 1)
              for arm, row in metrics.items()}
    reduction = float((base[KEYS[1]] - cov[KEYS[1]]).mean())
    excess = float(((base[KEYS[1]] - base[KEYS[3]]) - (cov[KEYS[1]] - cov[KEYS[3]])).mean())
    gates = dict(C1=comparisons[KEYS[1]]["gain"] >= .015 and comparisons[KEYS[1]]["interval"][0] > 0,
                 C2=comparisons[KEYS[2]]["gain"] >= .01,
                 C3=comparisons[KEYS[3]]["gain"] >= -.01,
                 C4=all(row["gain"] > 0 for row in specificity.values()), C5=mc1)
    if all(gates.values()):
        decision = "GO"
    elif gates["C1"] and gates["C2"] and not gates["C3"]:
        decision = "TRADE-OFF"
    elif gates["C1"] and gates["C2"] and gates["C3"] and not gates["C4"]:
        decision = "REDIAGNOSE"
    else:
        decision = "STOP"
    group_counts = np.bincount(cluster[1])
    reproduced = dict(decision=decision, gates=gates, windows=len(reference_groups),
                      tracks=len(group_counts),
                      top10_window_share=float(np.sort(group_counts)[-10:].sum() / len(reference_groups)),
                      comparisons=comparisons, specificity=specificity, vs_native_conf=native,
                      distinct3=distinct, MC1=mc1, E3_reduction_m=excess,
                      FDE3_reduction_m=reduction, E3_fraction=excess / reduction if reduction > 0 else None,
                      MC2=reduction > 0 and excess >= .5 * reduction,
                      spread_change=spread, MC3=all(abs(v) <= .15 for v in spread.values()),
                      MC4=stable, per_arm={arm: {k: float(v.mean()) for k, v in row.items()}
                                          for arm, row in metrics.items()},
                      recording={}, eth_high_speed=None)
    check_equal(reproduced, cloud)
    print(json.dumps(dict(status="PASS", decision=decision, gates=gates, groups=len(group_counts),
                          runs=len(ARMS) * len(SEEDS), window_records=len(reference_groups) * len(ARMS) * len(SEEDS))))


def check_equal(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for name in actual:
            check_equal(actual[name], expected[name])
    elif type(actual) is float or isinstance(actual, list):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-14)
    else:
        assert actual == expected, (actual, expected)


if __name__ == "__main__":
    main()
