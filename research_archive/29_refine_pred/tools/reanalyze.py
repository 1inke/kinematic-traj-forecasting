"""Recompute EXP-REFINE-PRED from the published window metrics with NumPy."""

import json
import math
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PHASE27 = ROOT.parent / "27_qrank_mldec/EXP_QRANK_MLDEC_L4_20261122"
PHASE28 = ROOT.parent / "28_qrank_followups_and_ea"
sys.path.insert(0, str(ROOT / "code"))
from analysis import ARMS, CONFIRMATION, EXPLORATORY, LABELS, decide, fold_report


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require(ok, context):
    if not ok:
        raise ValueError(context)


def compare(actual, expected, context):
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), context)
        for key, value in expected.items():
            compare(actual[key], value, f"{context}/{key}")
    elif isinstance(expected, list):
        require(len(actual) == len(expected), context)
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{context}/{i}")
    elif isinstance(expected, float):
        require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), f"{context}: {actual} != {expected}")
    else:
        require(actual == expected, context)


def verify_manifest():
    entries = (ROOT / "FILES.sha256").read_text(encoding="utf-8").splitlines()
    listed = set()
    for row in entries:
        checksum, name = row.split("  ", 1)
        require(digest(ROOT / name) == checksum, name)
        listed.add(name)
    present = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()
               and p.name != "FILES.sha256" and "__pycache__" not in p.parts}
    require(present == listed and len(entries) == len(listed), "The archive file list differs from its manifest")


def load_array(path, keys):
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == keys, f"Unexpected fields in {path}")
        values = {key: data[key] for key in data.files}
    ids = values["ids"]
    require(ids.dtype.kind in "iu" and len(ids) == len(np.unique(ids)), f"Invalid window IDs in {path}")
    require(all(x.shape == ids.shape for x in values.values()), f"Expected one-dimensional window metrics in {path}")
    require(all(np.isfinite(x).all() for x in values.values() if x.dtype.kind == "f"), f"Nonfinite data in {path}")
    return values


def population(fold, summary):
    folder = RESULTS / "tracks" / fold
    info = read(folder / "TRACKS.json")
    compare(info, summary["tracks"][fold], f"track audit {fold}")
    require(info["status"] == "PASS" and info["track_sha256"] == digest(folder / "TRACKS.npz"), fold)
    tracks = load_array(folder / "TRACKS.npz", {"ids", "tracks", "recording"})
    labels = load_array(folder / "LABELS.npz", {"ids", "maneuver"})
    np.testing.assert_array_equal(tracks["ids"], labels["ids"])
    require(set(labels["maneuver"]) <= set(LABELS), f"Unknown maneuver label in {fold}")
    require(len(labels["ids"]) == info["windows"], f"Wrong population size in {fold}")
    np.testing.assert_array_equal(np.unique(tracks["tracks"]), np.arange(info["trajectories"]))
    compare({str(k): int((tracks["recording"] == k).sum()) for k in np.unique(tracks["recording"])},
            info["by_recording"], f"recording counts {fold}")
    return tracks, labels["maneuver"], folder


def run_metrics(name, fold, tracks, folder, summary):
    target = RESULTS / "runs" / name
    receipt = read(target / "RESULT.json")
    binding = receipt["binding"]
    prior = PHASE27 / "results/stage2/runs" / name / "RESULT.json"
    compare(binding["sources"], summary["source_lock"], f"source lock {name}")
    require(binding["receipt"] == digest(prior), f"Formal receipt changed for {name}")
    formal = read(prior)
    require(binding["weights"] == formal["artifacts"]["final_ema.pt"], f"Checkpoint hash changed for {name}")
    require(summary["tracks"][fold]["sources"]["test.npz"] == formal["contract"]["data"]["test.npz"], fold)
    require(binding["track_audit"] == digest(folder / "TRACKS.json") and
            binding["labels"] == digest(folder / "LABELS.npz"), f"Track or label checksum differs for {name}")
    metrics = target / "metrics.npz"
    require(receipt["status"] == "PASS" and receipt["metrics_sha256"] == digest(metrics), f"Invalid receipt for {name}")
    require(receipt["layers_reordered_oracle_exact"] == [0, 3] and receipt["weight_updates"] == 0, name)
    require(receipt["endpoint_movement_median_m"] >= .02, f"Manipulation check failed for {name}")
    require(all(0 <= x <= .0005 for x in receipt["replay_drift"].values()), f"Replay drift exceeded tolerance for {name}")
    layer_metrics = ("minFDE", "minADE", "Top1_FDE", "brier_minFDE", "spread_long", "spread_lat", "entropy",
                     "best_slot", "best_error_long_sq", "best_error_lat_sq")
    fields = {"X1", "X2", "X3", "X4", "valid", "observed_steady", "tracks", "recording", "ids", "movement", "best_slot_switch"}
    fields.update(f"{layer}_{metric}" for layer in ("first", "final") for metric in layer_metrics)
    data = load_array(metrics, fields)
    require(len(data["ids"]) == receipt["windows"], name)
    for key, values in tracks.items():
        np.testing.assert_array_equal(data[key], values)
    if fold in EXPLORATORY:
        old = PHASE28 / "frozen_replay/results/runs" / name
        require(digest(old / "RESULT.json") == binding["archive28"], f"Replay receipt mismatch for {name}")
        with np.load(old / "metrics.npz", allow_pickle=False) as previous:
            np.testing.assert_array_equal(previous["ids"], data["ids"])
            for layer in ("first", "final"):
                np.testing.assert_allclose(data[f"{layer}_minFDE"], previous[f"{layer}_conf/minFDE"], atol=.0005, rtol=0)
    return data


def main():
    verify_manifest()
    current = read(RESULTS / "CURRENT.json")
    summary = read(RESULTS / "SUMMARY.json")
    require(digest(RESULTS / "SUMMARY.json") == current["summary_sha256"], "Summary checksum")
    require(digest(RESULTS / "REPORT.md") == current["report_sha256"], "Report checksum")
    require(digest(RESULTS / "PREFLIGHT.json") == summary["preflight"], "Preflight checksum")
    source = read(RESULTS / "SOURCE_LOCK.json")["files"]
    compare(source, summary["source_lock"], "Source lock")
    for name in ("analysis.py", "measure.py", "run.py", "tracks.py"):
        require(digest(ROOT / "code" / name) == source[f"refine_pred/{name}"], name)
    require(digest(ROOT / "DESIGN.md") == source["refine_pred/DESIGN.md"], "Design checksum")
    require(digest(PHASE28 / "frozen_replay/run.py") == source["frozen_replay/run.py"], "Frozen replay source checksum")
    ea_origin = read(PHASE28 / "e_a_diag/results/LOCAL_PROVENANCE.json")
    require(ea_origin["code_sha256"] == source["e_a_diag/run.py"], "E-A source provenance")
    recomputed, counts, cohorts = {}, {}, 0
    for fold in (*EXPLORATORY, *CONFIRMATION):
        seeds = tuple(range(42, 47 if fold == "eth" else 45))
        tracks, labels, folder = population(fold, summary)
        recomputed[fold] = {}
        for arm in ARMS:
            rows = [run_metrics(f"{arm}_{fold}_s{s}", fold, tracks, folder, summary) for s in seeds]
            cohorts += len(rows)
            result = fold_report(rows, labels, fold, seeds)
            compare(result, summary["folds"][fold][arm], f"{fold}/{arm}")
            recomputed[fold][arm] = result
            if fold == "zara02":
                compare(fold_report(rows[:2], labels, fold, seeds[:2]), summary["zara02_without_local_s44"][arm],
                        f"{fold}/{arm}/without_local_s44")
        counts[fold] = len(tracks["ids"]) * len(seeds) * len(ARMS)
        print(f"PASS {fold}: {counts[fold]} model-window records", flush=True)
    decision = decide(recomputed)
    compare(decision, summary["decision"], "Decision")
    status = read(ROOT / "STATUS.json")
    require(status["status"] == current["status"] == "COMPLETE" and status["outcome"] == decision["outcome"], "Status")
    require(cohorts == status["inference_runs"] == current["completed"], "Replay count")
    require(sum(counts.values()) == status["model_window_records"], "Model-window count")
    print(f"PASS {cohorts} replays, {sum(counts.values()):,} model-window records, decision {decision['outcome']}", flush=True)


if __name__ == "__main__":
    main()
