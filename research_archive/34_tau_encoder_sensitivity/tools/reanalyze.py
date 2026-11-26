"""Recalculate E-3/E-5 from anonymized paired window metrics."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from tracks_analysis import cluster_draws, interval, summarize_group

SEEDS = (42, 43, 44)
METRICS = ("minFDE", "minADE", "Top1_FDE", "Top1_ADE")
ENCODERS = ("base", "state", "sparse", "both")
GRID = ("r06_t24", "r06_t30", "r06_t48", "r02_t24", "r02_t30", "r02_t48")
NAMES = [f"encoder_{name}" for name in ENCODERS] + ["tau_100"]
NAMES += [f"sens_{name}" for name in GRID[1:]]


def read(name):
    return json.loads((ROOT / name).read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def compare(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            compare(actual[key], expected[key])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            compare(left, right)
    elif isinstance(actual, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-13)
    else:
        assert actual == expected, (actual, expected)


def recompute():
    status = read("results/STATUS.json")
    plan = read("results/PLAN.json")
    provenance = read("PROVENANCE.json")
    assert status["status"] == "COMPLETE"
    assert status["protocol"] == plan["protocol"] == provenance["protocol"]
    assert plan["gpus"] == ["Tesla T4", "Tesla T4"]
    assert set(plan["tasks"]) == {f"{name}_s{s}" for name in NAMES if name != "tau_100" for s in SEEDS}
    assert len(plan["tasks"]) == 27
    with np.load(ROOT / "results/window_groups.npz", allow_pickle=False) as data:
        assert data.files == ["track_group"]
        groups = data["track_group"]
    assert groups.shape == (4753,)
    np.testing.assert_array_equal(np.unique(groups), np.arange(248))
    rows = {}
    per_seed = {}
    fit_hash = None
    for name in NAMES:
        rows[name] = {metric: [] for metric in METRICS}
        per_seed[name] = {}
        for seed in SEEDS:
            run = f"{name}_s{seed}"
            result = read(f"results/runs/{run}/RESULT.json")
            contract = result["contract"]
            assert (contract["name"], contract["seed"]) == (name, seed)
            assert contract["protocol"] == status["protocol"]
            assert (contract["fit_windows"], contract["epochs"], contract["batch"]) == (21719, 64, 64)
            fit_hash = fit_hash or contract["fit_sha256"]
            assert contract["fit_sha256"] == fit_hash
            assert (result["windows"], result["tracks"]) == (4753, 248)
            if name == "tau_100":
                assert contract["source"] == "EXP_TAU_KAGGLE_v2"
                assert contract["original"]["gpu"] == "Tesla T4"
                trained = read(f"results/runs/{run}/REUSED_TRAINED.json")
                assert trained["contract"] == contract["original"] and trained["updates"] == 21760
            else:
                assert contract["gpu"] == "Tesla T4"
                assert result["updates"] == 21760
            file = ROOT / f"results/runs/{run}/window_metrics.npz"
            hashes = provenance["runs"][run]
            assert digest(file) == result["public_metrics_sha256"] == hashes["public_metrics_sha256"]
            assert result["metrics_sha256"] == hashes["source_metrics_sha256"]
            with np.load(file, allow_pickle=False) as array:
                assert set(array.files) == set(result["metrics"])
                per_seed[name][str(seed)] = {}
                for key in array.files:
                    values = array[key]
                    assert values.shape == groups.shape and np.isfinite(values).all()
                    compare(float(values.mean()), result["metrics"][key])
                    if key in METRICS:
                        rows[name][key].append(values)
                        per_seed[name][str(seed)][key] = float(values.mean())
    vectors = {name: {key: np.mean(v, axis=0) for key, v in row.items()} for name, row in rows.items()}
    means = {name: {key: float(v.mean()) for key, v in row.items()} for name, row in vectors.items()}
    mask = np.ones(len(groups), dtype=bool)

    def pair(base, tested):
        return {key: summarize_group(vectors[base][key], vectors[tested][key] - vectors[base][key], groups, mask)
                for key in METRICS}

    base, state, sparse, both = [f"encoder_{name}" for name in ENCODERS]
    b = vectors[base]["minFDE"]
    delta = vectors[both]["minFDE"] - vectors[state]["minFDE"] - vectors[sparse]["minFDE"] + b
    _, codes, counts = cluster_draws(groups)
    bt = np.bincount(codes, weights=b)
    dt = np.bincount(codes, weights=delta)
    e3 = dict(
        comparisons=dict(state_only=pair(base, state), sparse_only=pair(base, sparse),
                         state_given_sparse=pair(sparse, both), sparse_given_state=pair(state, both)),
        minFDE_interaction=dict(absolute_m=float(delta.mean()), relative=float(dt.sum() / bt.sum()),
                               ci95=interval((counts @ dt) / (counts @ bt), .95)),
        means={name: means[name] for name in (base, state, sparse, both)},
    )
    grid = {name: "tau_100" if name == "r06_t24" else f"sens_{name}" for name in GRID}
    gap = (max(means[name]["minFDE"] for name in grid.values()) -
           min(means[name]["minFDE"] for name in grid.values())) / means["tau_100"]["minFDE"]
    e5 = dict(means={name: means[key] for name, key in grid.items()},
              comparisons={name: pair("tau_100", key) for name, key in grid.items() if key != "tau_100"},
              relative_range=gap, below_1_percent=gap < .01)
    return e3, e5, per_seed


def main():
    for name, checksum in read("FILES.json").items():
        assert digest(ROOT / name) == checksum, name
    for name, metadata in read("PROVENANCE.json")["sealed_sources"].items():
        assert digest(ROOT / name) == metadata["sha256"]
    e3, e5, seeds = recompute()
    compare(e3, read("results/E3_RESULT.json"))
    compare(e5, read("results/E5_RESULT.json"))
    compare(seeds, read("results/BY_SEED.json"))
    print(json.dumps(dict(status="PASS", evaluated_groups=30, newly_trained=27, reused=3,
                          model_window_records=142590, clusters=248, new_training_updates=587520,
                          e5_relative_range=e5["relative_range"], e5_below_1_percent=e5["below_1_percent"])))


if __name__ == "__main__":
    main()
