"""Rebuild archived comparisons with NumPy, using coordinate-free artifacts."""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT.parent / "29_refine_pred"


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def json_file(path):
    return json.loads(path.read_bytes())


def same(value, saved, where):
    if isinstance(saved, dict):
        assert value.keys() == saved.keys(), where
        for key, val in saved.items():
            same(value[key], val, f"{where}/{key}")
    elif isinstance(saved, list):
        assert len(value) == len(saved), where
        for i, (a, b) in enumerate(zip(value, saved)):
            same(a, b, f"{where}/{i}")
    elif isinstance(saved, float):
        assert math.isclose(value, saved, rel_tol=1e-9, abs_tol=1e-11), (where, value, saved)
    else:
        assert value == saved, (where, value, saved)


def routines(path, names, constants, env):
    """Load only the data-reduction functions, with no model or launcher imports."""
    tree = ast.parse(path.read_bytes())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in constants:
                    env[target.id] = ast.literal_eval(node.value)
                elif isinstance(target, ast.Tuple):
                    selected = [(i, item.id) for i,item in enumerate(target.elts)
                                if isinstance(item, ast.Name) and item.id in constants]
                    for i,name in selected:
                        env[name] = ast.literal_eval(node.value.elts[i])
    assert set(names) == {node.name for node in nodes}
    assert set(constants) <= env.keys()
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    return env


def layer_pool(stats):
    directory = ROOT / "layer_pool"
    folders = directory / "results/runs"
    records = sorted(folders.glob("*/RESULT.json"))
    assert len(records) == 34
    env = routines(directory / "run.py", ("summarize",), ("GROUPS", "METHODS"), {
        "np": np, "load": load, "source_path":lambda name: folders / name / "METRICS.npz",
        "parent":SimpleNamespace(folders=lambda fold: (PRIOR / "results/tracks" / fold, None)),
        "summarize_group": stats,
    })
    expected = json_file(directory / "results/SUMMARY.json")
    results, rows = {}, 0
    for fold, arms in expected["results"].items():
        results[fold] = {}
        for arm in arms:
            names = [p.parent.name for p in records if p.parent.name.startswith(f"{arm}_{fold}_s")]
            assert len(names) == (5 if fold == "eth" else 3)
            rows += sum(json_file(folders/name/"RESULT.json")["windows"] for name in names)
            results[fold][arm] = env["summarize"](fold, names)
    same(results, expected["results"], "layer_pool")
    gains = {f: a["O-conf+E0"]["all"]["cover4"]["minFDE"]["relative"] for f, a in results.items()}
    passed = sum(x <= -.02 for x in gains.values()) >= 3 and gains["eth"] <= .01
    assert expected["decision"] == ("FIXED_K_OPPORTUNITY" if passed else "NO_FIXED_K_SIGNAL")
    return rows


def mcus_ablation(stats):
    directory = ROOT / "mcus_ablation"
    env = routines(directory / "run.py", ("display_metrics","compare"), ("FOLDS","SEEDS","CONTRASTS"),
                   {"np":np, "summarize_group":stats})
    track = {fold:load(PRIOR/"results/tracks"/fold/"TRACKS.npz") for fold in env["FOLDS"]}
    test = json_file(directory / "SUMMARY.json")
    dev = json_file(directory / "DEV_LOCK.json")
    assert len(test["metrics"]) == len(dev["metrics"]) == 12
    total = 0
    for split, expected in (("dev",dev), ("test",test)):
        data = {name:load(directory/"runs"/name/f"{split}_metrics.npz") for name in expected["metrics"]}
        same({name:env["display_metrics"](d) for name,d in data.items()},expected["metrics"],split)
        same(env["compare"](data,track if split=="test" else None),expected["contrasts"],split+"/contrasts")
        total += sum(len(d["ids"]) for d in data.values())
    return total


def probes():
    rows = 0
    for name, spec in json_file(ROOT/"PROBE_METRIC_SOURCES.json").items():
        d = load(ROOT/name/"window_metrics.npz")
        assert len(d["ids"]) == spec["rows"]
        source = json_file(ROOT/name.split("/")[0]/"RESULT_SUMMARY.json")
        subset = source.get(spec["fold"],source)
        results = subset.get("results",subset)
        for prefix, label in spec["keys"].items():
            for arr, metric, scale in (("minADE","minADE20",1),("minFDE","minFDE20",1),
                                       ("Top1_FDE","Top1_FDE",1),("ES","ES_w",1),
                                       ("brier_minFDE","brier_minFDE",1),("J","J_cov_pct",100)):
                fresh = d[prefix+arr].mean()*scale
                assert math.isclose(fresh,results[label][metric],abs_tol=2e-5,rel_tol=1e-6), (
                    name,label,metric,fresh,results[label][metric])
        rows += len(d["ids"])
    return rows


def main():
    for path,digest in json_file(ROOT/"FILES.json").items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest,path
    stats = routines(PRIOR/"code/analysis.py", ("cluster_draws","interval","ratio","summarize_group"),
                     ("N_BOOT","BOOT_SEED"), {"np":np})["summarize_group"]
    print(json.dumps(dict(status="PASS",layer_pool_windows=layer_pool(stats),
                          mcus_ablation_windows=mcus_ablation(stats),probe_windows=probes())))


if __name__ == "__main__":
    main()
