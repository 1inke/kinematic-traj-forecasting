"""Recompute QRANK's formal results from the published coordinate-free metrics."""

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
from qrank_decisions import FOLDS, confirm, evidence, run, screen, short_compare


def read(path):
    return json.loads(path.read_bytes())


def compare(actual, expected, label):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), label
        for key, value in expected.items():
            compare(actual[key], value, f"{label}/{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), label
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{label}/{i}")
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12) or (math.isnan(actual) and math.isnan(expected)), label
    else:
        assert actual == expected, label


def check_archive():
    receipt = read(ROOT / "ARTIFACT_MANIFEST.json")
    assert len(receipt["files"]) == receipt["count"]
    for name, entry in receipt["files"].items():
        path = ROOT / name
        assert path.stat().st_size == entry["bytes"], name
        assert sha256(path.read_bytes()).hexdigest() == entry["sha256"], name
    done = read(RESULTS / "DONE.json")
    for name, checksum in done["files"].items():
        assert sha256((RESULTS / name).read_bytes()).hexdigest() == checksum, name
    return receipt["count"]


def direct(root, admission):
    expected = read(RESULTS / "QFL_VS_E0.json")
    keys = ("minADE", "minFDE", "Top1_FDE", "brier_minFDE", "brier_penalty")
    actual, speed = {}, {}
    for fold in FOLDS:
        seeds = admission["stage2_seeds"][fold]
        qfl, e0 = (evidence(run(root, "stage2", arm, fold, seed) for seed in seeds)[0]
                   for arm in ("O-conf+QFL", "O-conf+E0"))
        for row in (*qfl, *e0):
            row["brier_penalty"] = row["brier_minFDE"].astype("float64") - row["minFDE"].astype("float64")
            row["nonfast"] = row["overall"] & ~row["fast"]
        actual[fold] = {m: short_compare(qfl, e0, m) for m in keys}
        actual[fold]["stationary_minFDE"] = short_compare(qfl, e0, "minFDE", "stationary")
        if fold == "eth":
            b0 = evidence(run(root, "stage2", "B0", fold, seed) for seed in seeds)[0]
            for row in b0:
                row["nonfast"] = row["overall"] & ~row["fast"]
            for g in ("fast", "nonfast", "stationary"):
                speed[g] = {name: {m: short_compare(rows, b0, m, g) for m in ("minFDE", "Top1_FDE")}
                            for name, rows in (("O-conf+QFL", qfl), ("O-conf+E0", e0))}
    compare(actual, expected["folds"], "direct/QFL versus E0")
    compare(speed, expected["eth_vs_b0"], "direct/ETH groups")
    compare({m: float(np.mean([actual[f][m]["gain"] for f in FOLDS])) for m in keys},
            expected["mean"], "direct/five-fold mean")


def main():
    file_count = check_archive()
    counts, windows = {}, 0
    for stage in ("stage1", "stage2"):
        groups = sorted((RESULTS / stage).glob("*/*/RESULT.json"))
        counts[stage] = len(groups)
        for receipt in groups:
            arrays, details = evidence((receipt.parent,))
            rows, result = arrays[0], details[0]
            windows += len(rows["ids"])
            for group, recorded in result["dev"].items():
                mask = rows[group]
                assert int(mask.sum()) == recorded["windows"], (receipt, group)
                for metric, value in recorded.items():
                    if metric != "windows":
                        compare(float(rows[metric][mask].mean()), value, f"{receipt.parent.name}/{group}/{metric}")
    assert counts == {"stage1": 33, "stage2": 51}
    with tempfile.TemporaryDirectory(prefix="qrank-formal-reanalysis-") as temp:
        root = Path(temp)
        for stage in counts:
            shutil.copytree(RESULTS / stage, root / stage)
        admission = screen(root)
        compare(admission, read(RESULTS / "ADMISSION.json"), "admission")
        compare(read(root / "STAGE1_RESULT.json"), read(RESULTS / "STAGE1_RESULT.json"), "C0")
        formal = confirm(root, admission)
        compare(read(root / "STAGE2_RESULT.json"), read(RESULTS / "STAGE2_RESULT.json"), "five-fold")
        direct(root, admission)
    print(json.dumps(dict(status="PASS", manifest_files=file_count, groups=counts, window_records=windows,
                          F_PASS={name: value["F-PASS"] for name, value in formal.items()})))


if __name__ == "__main__":
    main()
