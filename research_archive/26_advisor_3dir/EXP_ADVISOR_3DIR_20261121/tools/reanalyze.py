"""Replay all gates and aggregate metrics from the published window-level data."""
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "code/sealed"))
from advisor_decisions import confirm, screen
from coupling_analysis import TrackBootstrap


def read(path):
    return json.loads(path.read_bytes())


def compare(value, reference, label):
    if isinstance(reference, dict):
        assert value.keys() == reference.keys(), label
        for key, item in reference.items():
            compare(value[key], item, f"{label}/{key}")
    elif isinstance(reference, list):
        assert len(value) == len(reference), label
        for index, (a, b) in enumerate(zip(value, reference)):
            compare(a, b, f"{label}/{index}")
    elif isinstance(reference, float):
        assert math.isclose(value, reference, rel_tol=1e-10, abs_tol=1e-12), (label, value, reference)
    else:
        assert value == reference, (label, value, reference)


def recheck_metrics(run):
    result = read(run / "RESULT.json")
    path = run / "dev_metrics.npz"
    assert sha256(path.read_bytes()).hexdigest() == result["artifacts"][path.name], path
    with np.load(path, allow_pickle=False) as data:
        for name, group in result["dev"].items():
            mask = data[name]
            assert int(mask.sum()) == group["windows"], (run, name)
            for metric, ref in group.items():
                if metric != "windows":
                    compare(float(data[metric][mask].mean()), ref, f"{run.name}/{name}/{metric}")
        return len(data["ids"])


def replay_z2():
    root = ROOT / "Z2"
    receipt = read(root / "DONE.json")
    for name, checksum in receipt["files"].items():
        assert sha256((root / name).read_bytes()).hexdigest() == checksum, name
    expected = read(root / "RESULT.json")
    metric_names = ("minADE", "minFDE", "J")
    base_name = "B0_five_seed_mean"
    nms_names = ("NMS_0.1", "NMS_0.2", "NMS_0.3", "NMS_0.5")
    methods = (base_name, "pool100", *nms_names, "weighted_kmedoids20", "top20")
    counts = []
    curves = {}
    for split in ("fit", "dev"):
        with np.load(root / f"windows_{split}.npz", allow_pickle=False) as data:
            base = data[base_name]
            np.testing.assert_allclose(data["seed_metrics"].mean(axis=1), base, rtol=0, atol=1e-12)
            counts.append(len(base))
            for method in methods[2:]:
                indices = data[f"{method}_indices"]
                assert indices.shape == (len(base), 20) and (indices < 100).all()
                assert (np.diff(np.sort(indices.astype(int), axis=1), axis=1) > 0).all()
            curves[split] = [float(data[name][:, 1].mean()) for name in nms_names]
            overall = np.ones(len(base), bool)
            groups = {"overall": overall, "hard": np.isin(data["maneuver"], ("turn", "turn_and_speed")),
                      **{name: data["observation"] == name for name in ("stationary", "steady", "other")}}
            for scope, include in (("all", overall), ("without_zara02", data["scene"] != "crowds_zara02")):
                for name, group in groups.items():
                    mask = include & group
                    if not mask.any():
                        continue
                    helper = TrackBootstrap(data["tracks"][mask])
                    reference = expected["metrics"][split][scope][name]
                    compare(helper.population, reference["population"], f"Z2/{split}/{scope}/{name}/population")
                    unit = np.ones(mask.sum())
                    pairs = {}
                    for method in methods:
                        for i, metric in enumerate(metric_names):
                            before, after = base[mask, i], data[method][mask, i]
                            pairs[method, metric, "value"] = after, unit
                            pairs[method, metric, "delta"] = after - before, unit
                            if metric != "J":
                                pairs[method, metric, "gain"] = before - after, before
                    for (method, metric, stat), value in helper.summarize(pairs).items():
                        compare(value, reference["metrics"][method][metric][stat],
                                f"Z2/{split}/{scope}/{name}/{method}/{metric}/{stat}")
    compare(curves, expected["NMS_fit_and_dev_curve_m"], "Z2 NMS curve")
    chosen = nms_names[int(np.argmin(curves["fit"]))]
    assert chosen == expected["NMS_chosen_fit_only"]
    metrics = expected["metrics"]["dev"]["all"]["overall"]["metrics"]
    pass_names = [name for name in (chosen, "weighted_kmedoids20")
                  if metrics[name]["minFDE"]["gain"]["estimate"] >= .015]
    assert pass_names == expected["meets_dev_1_5_percent"]
    return sum(counts)


def main():
    receipt = read(RESULTS / "DONE.json")
    for name, checksum in receipt["files"].items():
        assert sha256((RESULTS / name).read_bytes()).hexdigest() == checksum, name
    stage1 = sorted((RESULTS / "stage1/runs").iterdir())
    stage2 = sorted((RESULTS / "stage2/runs").iterdir())
    assert (len(stage1), len(stage2)) == (33, 30)
    windows = sum(recheck_metrics(run) for run in (*stage1, *stage2))
    with tempfile.TemporaryDirectory(prefix="advisor_replay_") as directory:
        scratch = Path(directory)
        for stage in ("stage1", "stage2"):
            shutil.copytree(RESULTS / stage / "runs", scratch / stage / "runs")
        admission = screen(scratch)
        confirm(scratch, admission)
        for name in ("stage1/RESULT.json", "stage1/MECH_R.json", "stage1/G1_compression.json",
                     "stage2/ADMISSION.json", "stage2/RESULT.json"):
            compare(read(scratch / name), read(RESULTS / name), name)
    print(json.dumps(dict(status="PASS", C0_runs=len(stage1), fivefold_runs=len(stage2), windows=windows,
                          Z2_windows=replay_z2(), admission=admission,
                          F_PASS=read(RESULTS / "stage2/RESULT.json")["arms"]["R2"]["F-PASS"])))


if __name__ == "__main__":
    main()
