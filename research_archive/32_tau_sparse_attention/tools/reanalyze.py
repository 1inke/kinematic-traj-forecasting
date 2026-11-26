"""Recompute EXP-TAU C0 from anonymized per-window metrics using NumPy."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from tracks_analysis import cluster_draws, summarize_group

ARMS = ("B0", "T1", "T2", "T3")
SEEDS = (42, 43, 44)


def read(path):
    return json.loads((ROOT / path).read_bytes())


def compare_equal(value, expected):
    if isinstance(value, dict):
        assert value.keys() == expected.keys()
        for key in value:
            compare_equal(value[key], expected[key])
    elif isinstance(value, list):
        assert len(value) == len(expected)
        for left, right in zip(value, expected):
            compare_equal(left, right)
    elif isinstance(value, float):
        np.testing.assert_allclose(value, expected, rtol=1e-9, atol=1e-13)
    else:
        assert value == expected, (value, expected)


def summarize(base_seeds, arm_seeds, biases, groups):
    base = np.mean(base_seeds, axis=0)
    change = np.mean(arm_seeds, axis=0) - base
    result = summarize_group(base, change, groups, np.ones(len(base), dtype=bool))
    _, lookup, draws = cluster_draws(groups)
    numer = draws @ np.bincount(lookup, weights=change)
    denom = draws @ np.bincount(lookup, weights=base)
    result.update(by_seed={str(seed): float(arm.mean() / base.mean() - 1)
                           for seed, arm, base in zip(SEEDS, arm_seeds, base_seeds)},
                  beta_by_seed=biases, upper_one_sided_95=float(np.quantile(numer / denom, .95)))
    return result


def main():
    manifest = read("FILES.json")
    for name, expected in manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    assets = hashlib.sha256((ROOT / "code/tau_assets.json").read_bytes()).hexdigest()
    preflight = read("results/PREFLIGHT.json")
    status = read("results/KAGGLE_STATUS.json")
    assert preflight["status"] == "PASS" and preflight["devices"] == ["Tesla T4", "Tesla T4"]
    assert status["assets"] == preflight["assets"] == assets
    provenance = read("PROVENANCE.json")
    reused = provenance["source_members_reused_from_research_archive"]
    source = read("code/tau_assets.json")["tau_source.zip"]["members"]
    for name, wanted in source.items():
        path = ROOT.parent / reused[name] if name in reused else ROOT / "code" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == wanted, name
    groups = None
    samples, biases = {}, {}
    for arm in ARMS:
        samples[arm], biases[arm] = [], []
        for seed in SEEDS:
            run = Path("results/runs") / f"C0_{arm}_s{seed}"
            result, trained = read(run / "RESULT.json"), read(run / "TRAINED.json")
            contract = result["contract"]
            assert contract == trained["contract"] and result["bias"] == trained["bias"]
            assert (contract["fold"], contract["arm"], contract["seed"]) == ("C0", arm, seed)
            assert (contract["epochs"], contract["batch"], contract["power"]) == (64, 64, 2.0)
            assert contract["assets"] == assets and trained["updates"] == 21760
            metric_file = run / "window_metrics.npz"
            pair = provenance["window_metrics"][metric_file.as_posix()]
            assert pair["kaggle_sha256"] == result["metrics_sha256"]
            assert pair["public_sha256"] == manifest[metric_file.as_posix()]
            with np.load(ROOT / metric_file, allow_pickle=False) as data:
                if groups is None:
                    groups = data["track_group"]
                np.testing.assert_array_equal(data["track_group"], groups)
                assert len(groups) == result["windows"] == 4753
                assert len(np.unique(groups)) == result["tracks"] == 248
                assert set(data.files) == set(result["metrics"]) | {"track_group"}
                compare_equal({key: float(data[key].mean()) for key in result["metrics"]}, result["metrics"])
                samples[arm].append(data["minFDE"])
            biases[arm].append(result["bias"])
    comparisons = {arm: summarize(samples["B0"], samples[arm], biases[arm], groups) for arm in ARMS[1:]}
    decision = "GO" if comparisons["T2"]["relative"] <= .01 else "STOP"
    compare_equal(dict(decision=decision, comparisons=comparisons), read("results/C0_RESULT.json"))
    assert status["status"] == f"COMPLETE_C0_{decision}"
    print(json.dumps(dict(status="PASS", decision=decision, runs=len(ARMS) * len(SEEDS),
                          tracks=248, window_records=len(groups) * len(ARMS) * len(SEEDS),
                          delta={arm: row["relative"] for arm, row in comparisons.items()})))


if __name__ == "__main__":
    main()
