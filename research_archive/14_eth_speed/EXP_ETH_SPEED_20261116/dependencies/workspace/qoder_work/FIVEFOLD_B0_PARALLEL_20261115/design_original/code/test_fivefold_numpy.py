"""Synthetic checks for the numpy parts. Run: python -I test_fivefold_numpy.py
They check contracts and arithmetic only; they say nothing about model quality."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fivefold_common import FOLDS, FINAL_SEEDS, write_json_once  # noqa: E402
from fivefold_data import load_split, scene_tail_split, to_dict, change_groups  # noqa: E402
from fivefold_metrics import per_window, aggregate, decide_budget, ece  # noqa: E402
import summarize_fivefold as S  # noqa: E402

rng = np.random.default_rng(20261115)


def fake_fold(n, scenes, offset):
    d = np.zeros((n, 9, 20, 6), np.float32)
    v = rng.normal([0.5, 0], [0.1, 0.05], size=(n, 9, 2))
    t = np.arange(20)[None, None, :, None]
    d[..., -2:] = rng.normal(0, 3, (n, 9, 1, 2)) + t * v[:, :, None] + rng.normal(0, .01, (n, 9, 20, 2))
    num = rng.integers(1, 10, n)
    for i in range(n):
        d[i, num[i]:] = 0
    d[:, :, :, 0] = rng.choice(scenes, n)[:, None, None] + offset * 0
    return d, num


def make_data(root):
    for k, f in enumerate(FOLDS):
        train_scenes = [s for s in range(5) if s != k]
        tr, tn = fake_fold(400, train_scenes, 0)
        te, vn = fake_fold(60, [k], 0)
        for name, arr in (("data_train", tr), ("num_train", tn), ("data_test", te), ("num_test", vn)):
            np.save(root / f"{f}_{name}.npy", arr)


def reference_scene_tail(scene_ids, fraction=0.05):
    """Verbatim logic of train_front_variant.py (scene_tail branch)."""
    unique, counts = np.unique(scene_ids, return_counts=True)
    parts = []
    for scene in unique:
        candidates = np.flatnonzero(scene_ids == scene)
        validation_count = max(1, min(len(candidates) - 1, int(round(len(candidates) * fraction))))
        parts.append(candidates[-validation_count:])
    val = np.sort(np.concatenate(parts)); mask = np.zeros(len(scene_ids), bool); mask[val] = True
    return np.flatnonzero(~mask), val


def test_scene_tail_matches_exp156():
    for n in (37, 400, 1001):
        s = rng.integers(0, 4, n)
        a, b = scene_tail_split(s); c, d = reference_scene_tail(s)
        assert np.array_equal(a, c) and np.array_equal(b, d)


def test_to_dict_layout():
    d, num = fake_fold(10, [1], 0)
    x = to_dict(d, num)
    assert x["past"].shape == (10, 9, 8, 2) and x["future"].shape == (10, 12, 2)
    assert np.array_equal(x["past"][:, 0], d[:, 0, :8, -2:]) and np.array_equal(x["future"], d[:, 0, 8:, -2:])
    assert x["num_valid"].max() <= 9 and x["ids"].tolist() == list(range(10))


def test_groups_formula():
    past = np.zeros((3, 8, 2)); past[:, :, 0] = np.arange(-7, 1) * .5
    fut = np.zeros((3, 12, 2))
    fut[0, :, 0] = np.arange(1, 13) * .5                       # straight, same speed -> stable
    fut[1, :, 1] = np.arange(1, 13) * .5                       # 90 deg turn -> turn .5 -> change
    fut[2, :, 0] = np.arange(1, 13) * .35                      # speed ratio .7 -> transition
    g, s = change_groups(past, fut)
    assert g.tolist() == [0, 2, 1], (g, s)


def test_metrics():
    n = 50
    fut = rng.normal(size=(n, 12, 2)).cumsum(1)
    past = rng.normal(size=(n, 8, 2)).cumsum(1)
    paths = fut[:, None] + rng.normal(0, .3, (n, 20, 12, 2))
    paths[:, 7] = fut                                           # slot 7 exact
    logits = rng.normal(size=(n, 20)); logits[:, 7] += 10        # and top-scored
    w = per_window(paths, logits, fut, past)
    assert np.allclose(w["minADE"], 0) and np.allclose(w["top1_ADE"], 0) and w["coverage"].all()
    assert w["top1_is_J"].all() and np.allclose(w["top1_minADE"], w["top1_ADE"])
    assert (w["top5_minADE"] <= w["top3_minADE"] + 1e-12).all()
    agg = aggregate(w, np.zeros(n, int))
    assert agg["overall"]["coverage"] == 1.0 and agg["overall"]["n"] == n
    assert abs(ece(np.array([.9, .9]), np.array([1., 1.])) - .1) < 1e-12


def test_decide_budget():
    rows = {f: {64: dict(minADE=.2, minFDE=.4), 128: dict(minADE=.197, minFDE=.399)} for f in FOLDS}
    assert decide_budget(rows)["E_star"] == 128                 # 1.5% gain, FDE not worse
    rows = {f: {64: dict(minADE=.2, minFDE=.4), 128: dict(minADE=.199, minFDE=.399)} for f in FOLDS}
    assert decide_budget(rows)["E_star"] == 64                  # 0.5% gain
    rows = {f: {64: dict(minADE=.2, minFDE=.4), 128: dict(minADE=.19, minFDE=.41)} for f in FOLDS}
    assert decide_budget(rows)["E_star"] == 64                  # FDE worse


def test_audit_cli_and_summary():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); data = tmp / "data"; data.mkdir(); make_data(data)
        r = subprocess.run([sys.executable, "-I", str(HERE / "audit_folds.py"), "--data-dir", str(data),
                            "--out", str(tmp / "F0/AUDIT.json")], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        audit = json.loads((tmp / "F0/AUDIT.json").read_text())
        assert audit["passed"] and all(f["scene_tail"]["validation"] > 0 for f in audit["folds"])
        tr = load_split(data, "eth", "train"); assert tr["past"].shape[0] == 400
        # fake final metrics -> summary
        out = tmp / "RUNS"
        write_json_once(out / "BUDGET_DECISION.json", dict(E_star=64))
        for f in FOLDS:
            for s in FINAL_SEEDS:
                d = load_split(data, f, "test")
                fut, past = d["future"], d["past"][:, 0]
                paths = fut[:, None] + rng.normal(0, .3, (len(fut), 20, 12, 2))
                agg = aggregate(per_window(paths, rng.normal(size=(len(fut), 20)), fut, past), d["group"])
                agg.setdefault("change", agg["overall"])
                write_json_once(out / "runs" / f"final_{f}_E64_s{s}" / "metrics.json",
                                dict(test=agg, train=dict(parameters=295474)))
        exp156 = tmp / "exp156.json"
        exp156.write_text(json.dumps({"rows": [dict(fold=f, min_ade_20=.2, min_fde_20=.35, parameters=355048) for f in FOLDS]}))
        r = subprocess.run([sys.executable, "-I", str(HERE / "summarize_fivefold.py"), "--out", str(out),
                            "--exp156", str(exp156)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        s = json.loads((out / "SUMMARY.json").read_text())
        manual = np.mean([np.mean([json.loads((out / "runs" / f"final_{f}_E64_s{seed}" / "metrics.json").read_text())
                                   ["test"]["overall"]["minADE"] for f in FOLDS]) for seed in FINAL_SEEDS])
        assert abs(s["equal_weight_average"]["minADE"]["mean"] - manual) < 1e-12
        assert abs(s["comparisons"]["EXP156_M2_single_seed_equal_weight"][0] - .2) < 1e-12


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t(); print("PASS", t.__name__)
    print(f"{len(tests)} numpy checks passed")
