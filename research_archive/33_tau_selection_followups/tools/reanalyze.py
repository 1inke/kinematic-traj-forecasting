"""Verify the Colab E-1/E-2 archive and recalculate all paired statistics."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("minFDE", "minADE", "Top1_FDE", "Top1_ADE")
SEEDS = (42, 43, 44)
PERCENTS = (25, 50, 100)
CONTROLS = ("distance", "top_p")
CASES = [f"{name}_{pct}" for pct in PERCENTS for name in ("tau", *CONTROLS)]
CASES.append("full_100")


def read(name):
    return json.loads((ROOT / name).read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def compare(value, expected):
    if isinstance(value, dict):
        assert value.keys() == expected.keys()
        for key in value:
            compare(value[key], expected[key])
    elif isinstance(value, list):
        assert len(value) == len(expected)
        for actual, wanted in zip(value, expected, strict=True):
            compare(actual, wanted)
    elif isinstance(value, float):
        np.testing.assert_allclose(value, expected, rtol=1e-9, atol=1e-13)
    else:
        assert value == expected, (value, expected)


def interval(samples):
    return np.quantile(samples, [0.025, 0.975]).tolist()


def count_resamples(size):
    rng = np.random.default_rng(20261124)
    chosen = rng.integers(size, size=(4000, size), dtype=np.int32)
    counts = np.zeros(chosen.shape, dtype=np.int32)
    np.add.at(counts, (np.arange(len(counts))[:, None], chosen), 1)
    return counts


def recompute():
    status = read("STATUS.json")
    runtime = read("results/CLOUD_RUNTIME.json")
    release = read("results/CLOUD_RUN_STATUS.json")
    selection = read("results/STATUS_selection.json")
    plan = read("results/PLAN.json")
    protocol = status["protocol"]
    provenance = read("PROVENANCE.json")
    assert provenance["protocol"] == protocol and provenance["source_files_verified"] == 160
    for member, checksum in provenance["sealed_selection_code_and_design_sha256"].items():
        directory = "design" if member.startswith("source/followup_design/") else "code"
        assert digest(ROOT / directory / Path(member).name) == checksum
    assert runtime["protocol"] == release["protocol"] == selection["protocol"] == plan["protocol"] == protocol
    assert runtime["gpu"] == "NVIDIA L4" and release["exit_code"] == 0
    assert selection["status"] == "COMPLETE"
    with np.load(ROOT / "results/window_groups.npz", allow_pickle=False) as file:
        assert file.files == ["track_group"]
        groups = file["track_group"]
    assert groups.shape == (4753,) and groups.dtype.kind in "iu"
    np.testing.assert_array_equal(np.unique(groups), np.arange(248))
    draws = count_resamples(248)

    runs = {case: {} for case in CASES}
    fit_contracts = {}
    total_updates = 0
    expected_tasks = {f"{case}_s{seed}" for case in CASES for seed in SEEDS}
    assert len(plan["tasks"]) == 30 and set(plan["tasks"]) == expected_tasks
    for case in CASES:
        arm, percent_text = case.rsplit("_", 1)
        pct = int(percent_text)
        for seed in SEEDS:
            path = Path("results/runs") / f"{case}_s{seed}"
            result = read(path / "RESULT.json")
            contract = result["contract"]
            assert contract["seed"] == seed and contract["name"] == case
            assert contract["settings"] == dict(arm=arm, pct=pct, cutoff=2.4, radius=0.6)
            assert (contract["epochs"], contract["batch"], contract["fit_windows"]) == (64, 64, round(21719 * pct / 100))
            assert all(contract[key] == runtime[key] for key in ("protocol", "gpu", "torch", "cuda"))
            fit = (contract["fit_sha256"], contract["fit_indices_sha256"])
            assert fit_contracts.setdefault((seed, pct), fit) == fit
            assert (result["windows"], result["tracks"]) == (4753, 248)
            assert result["updates"] == {25: 5440, 50: 10880, 100: 21760}[pct]
            total_updates += result["updates"]
            metric_path = ROOT / path / "window_metrics.npz"
            hashes = provenance["run_metrics"][path.name]
            assert digest(metric_path) == result["public_metrics_sha256"] == hashes["public_metrics_sha256"]
            assert result["metrics_sha256"] == hashes["source_metrics_sha256"]
            with np.load(metric_path, allow_pickle=False) as array:
                assert set(array.files) == set(result["metrics"])
                runs[case][seed] = {metric: array[metric] for metric in array.files}
            for metric, values in runs[case][seed].items():
                assert values.shape == groups.shape and np.isfinite(values).all()
                compare(float(values.mean()), result["metrics"][metric])
    assert len({fit[0] for fit in fit_contracts.values()}) == 1
    assert total_updates == status["training_updates"] == 408000

    vectors = {
        case: {metric: np.mean([by_seed[seed][metric] for seed in SEEDS], axis=0) for metric in METRICS}
        for case, by_seed in runs.items()
    }
    means = {case: {metric: float(values.mean()) for metric, values in row.items()} for case, row in vectors.items()}

    def totals(values):
        return np.bincount(groups, weights=values, minlength=248)

    def pair(base, tested):
        b, d = totals(base), totals(tested - base)
        largest = np.argsort(-np.abs(d), kind="stable")[:10]
        return dict(
            windows=len(groups), tracks=248, small=False, baseline_m=float(base.mean()),
            mean_delta_m=float((tested - base).mean()), relative=float(d.sum() / b.sum()),
            ci95=interval((draws @ d) / (draws @ b)),
            top10_share_signed=float(d[largest].sum() / d.sum()) if d.sum() else None,
            top10_share_absolute=float(np.abs(d[largest]).sum() / np.abs(d).sum()) if np.abs(d).sum() else None,
            largest_positive_track_ids=np.argsort(-d, kind="stable")[:10].tolist(),
        )

    e1_pairs = {
        control: {metric: pair(vectors[f"{control}_100"][metric], vectors["tau_100"][metric]) for metric in METRICS}
        for control in (*CONTROLS, "full")
    }
    e1 = dict(
        comparisons=e1_pairs,
        upper_ci_at_most_zero={name: row["minFDE"]["ci95"][1] <= 0 for name, row in e1_pairs.items()},
        means={name: row for name, row in means.items() if name.endswith("_100")},
    )

    e2_pairs = {}
    for control in CONTROLS:
        gains, samples = {}, {}
        for pct in PERCENTS:
            base = vectors[f"{control}_{pct}"]["minFDE"]
            tau = vectors[f"tau_{pct}"]["minFDE"]
            b, d = totals(base), totals(base - tau)
            gains[str(pct)] = float(d.sum() / b.sum())
            samples[str(pct)] = (draws @ d) / (draws @ b)
        change = gains["25"] - gains["100"]
        e2_pairs[control] = dict(
            gain=gains, ci95_gain={pct: interval(values) for pct, values in samples.items()},
            change_25_vs_100=change, ci95_change=interval(samples["25"] - samples["100"]),
            meets_1pp=change >= 0.01,
        )
    e2 = dict(comparisons=e2_pairs, means=means)
    by_seed = {
        case: {str(seed): {metric: float(values[metric].mean()) for metric in METRICS} for seed, values in rows.items()}
        for case, rows in runs.items()
    }
    return e1, e2, by_seed


def main():
    manifest = read("FILES.json")
    for name, expected in manifest.items():
        assert digest(ROOT / name) == expected, name
    e1, e2, by_seed = recompute()
    compare(e1, read("results/E1_RESULT.json"))
    compare(e2, read("results/E2_RESULT.json"))
    compare(by_seed, read("results/BY_SEED.json"))
    print(json.dumps(dict(status="PASS", runs=30, model_window_records=142590, clusters=248,
                          training_updates=408000, e1_gates=e1["upper_ci_at_most_zero"],
                          e2_gates={name: row["meets_1pp"] for name, row in e2["comparisons"].items()})))


if __name__ == "__main__":
    main()
