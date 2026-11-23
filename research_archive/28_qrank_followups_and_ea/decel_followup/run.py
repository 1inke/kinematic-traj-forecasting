"""Replay signed endpoint errors and check an observed-only deceleration cue."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys

import numpy as np

from analysis import BLOCK_SIZE, HORIZONS, LAYERS, SELECTORS, summarize

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUT = HERE / "results"
WORKSPACE = PROJECT / "local/workspace"
OLD_DIAG = PROJECT.parent / "EXP_DECEL_DIAG_EXEC_20261117/code/decel_metrics.py"
sys.path[:0] = [str(PROJECT), str(PROJECT / "local"), str(WORKSPACE), str(OLD_DIAG.parent)]
from cloud_handoff import digest, read, write
from frozen_replay import run as frozen


def receipt(name):
    return OUT / "runs" / name / "RESULT.json"


def binding(name):
    return dict(frozen_result=digest(frozen.OUT / "runs" / name / "RESULT.json"),
                code=digest(Path(__file__)), analysis=digest(HERE / "analysis.py"),
                projection_code=digest(OLD_DIAG))


def completed(name):
    destination = receipt(name)
    metrics = destination.with_name("metrics.npz")
    if not destination.is_file() or not metrics.is_file():
        return False
    done = read(destination)
    return done["status"] == "PASS" and done["binding"] == binding(name) and done["metrics_sha256"] == digest(metrics)


def replay(name):
    import torch
    from advisor_train import core, inputs, load_data
    from decel_metrics import compute
    from qrank_eval import infer
    from qrank_model import attach

    if not frozen.completed(name):
        raise ValueError(f"Missing verified earlier frozen replay for {name}")
    arm, fold, seed = frozen.RUNS[name]
    formal, seal = frozen.bound(name)
    weight = frozen.OUT / "weights" / f"{name}.pt"
    test, size = inputs(fold)[1]
    if digest(weight) != seal["checkpoint"] or digest(test) != formal["contract"]["data"][test.name]:
        raise ValueError(f"Source hash mismatch for {name}")
    data = load_data(test, size)
    torch.set_num_threads(1)
    model, _ = core.make_model("cuda:0", seed)
    attach(model, arm)
    model.decoder.use_kin = formal["contract"]["use_kin"]
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
    model.requires_grad_(False)
    predictions = infer(core, model, data)
    target = receipt(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    prior_path = frozen.OUT / "runs" / name / "metrics.npz"
    drift = {}
    with np.load(prior_path, allow_pickle=False) as prior:
        if not np.array_equal(prior["ids"], data["ids"]):
            raise ValueError(f"Window IDs differ for {name}")
        metrics = {k: prior[k] for k in ("ids", "fast", "fast_deceleration", "fast_steady")}
        for layer, layer_id in zip(LAYERS, (0, 3)):
            computed = compute(data["past"], data["future"], predictions["layer_paths"][:, layer_id],
                               predictions["layer_logits"][:, layer_id])
            for key in ("direction_valid", "history_ratio"):
                if key in metrics and not np.array_equal(metrics[key], computed[key]):
                    raise ValueError(f"Observed history depends on decoder layer for {name}")
                metrics[key] = computed[key]
            for t in HORIZONS:
                suffix = f"_t{t}"
                for selector in SELECTORS:
                    stem = f"{selector}_"
                    longitudinal = computed[f"{stem}prog_error{suffix}"]
                    lateral = computed[f"{stem}lat_error{suffix}"]
                    fde = computed[f"{stem}displacement_error{suffix}"]
                    fields = dict(fde=fde, longitudinal=longitudinal, absolute_longitudinal=np.abs(longitudinal),
                                  absolute_lateral=np.abs(lateral), longitudinal_squared=longitudinal**2,
                                  lateral_squared=lateral**2, overshoot=longitudinal > .4)
                    valid = metrics["direction_valid"]
                    np.testing.assert_allclose((longitudinal**2 + lateral**2)[valid], fde[valid]**2, atol=1e-10)
                    for key, values in fields.items():
                        metrics[f"{layer}/{stem}{key}{suffix}"] = values
                    if t == 12:
                        label = "minFDE" if selector == "best" else "Top1_FDE"
                        drift[f"{layer}/{label}"] = float(np.abs(fde - prior[f"{layer}_conf/{label}"]).mean())
                for key in ("all_overshoot", "longitudinal_reachable", "best_index"):
                    metrics[f"{layer}/{key}{suffix}"] = computed[f"{key}{suffix}"]
        if any(not np.isfinite(x) or x > frozen.TOLERANCE for x in drift.values()):
            raise ValueError(f"Previously replayed endpoint errors disagree for {name}: {drift}")
    temporary = target.with_name("metrics.part.npz")
    np.savez_compressed(temporary, **metrics)
    artifact = target.with_name("metrics.npz")
    temporary.replace(artifact)
    write(target, dict(status="PASS", binding=binding(name), windows=size, metrics_sha256=digest(artifact),
                       checkpoint=seal["checkpoint"], mean_abs_replay_drift=drift))


def aggregate():
    report = {}
    for fold, seeds in frozen.FOLDS.items():
        report[fold] = {}
        for arm in frozen.ARMS:
            rows = []
            for seed in seeds:
                name = f"{arm}_{fold}_s{seed}"
                if not completed(name):
                    raise ValueError(f"Missing verified decomposition: {name}")
                with np.load(receipt(name).with_name("metrics.npz"), allow_pickle=False) as saved:
                    rows.append({key: saved[key] for key in saved.files})
            report[fold][arm] = summarize(rows)
        if report[fold][frozen.ARMS[0]]["observation"] != report[fold][frozen.ARMS[1]]["observation"]:
            raise ValueError(f"Observation cue result differs by model arm in {fold}")
    return report


def coordinate():
    OUT.mkdir(exist_ok=True)
    frozen.verify_sources()
    pending = [name for name in frozen.RUNS if not completed(name)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = {pool.submit(subprocess.run, [sys.executable, "-u", "-B", str(Path(__file__)), "--run", name],
                             check=True): name for name in pending}
        for future in as_completed(tasks):
            future.result()
            print(f"PASS {tasks[future]}", flush=True)
    report = aggregate()
    write(OUT / "SUMMARY.json", dict(
        status="COMPLETE", weight_updates=0, model_runs=len(frozen.RUNS), seeds=frozen.FOLDS,
        design=dict(horizons_frames=HORIZONS, dt_s=.4, observed_fast_m_per_s=1.3,
                    direction="Mean last three observed displacements, norm strictly greater than 0.03 m.",
                    overshoot_m=.4, cue="Mean last 3 observed speeds / mean first 4 (floor 0.1 m/s) <= 0.85.",
                    truth="Frozen replay fast_deceleration: deceleration or stop, using future only to label.",
                    intervals=f"2000 paired bootstrap draws using {BLOCK_SIZE}-row blocks; seeds averaged within window.",
                    scope="Exploratory reanalysis on already examined ETH and ZARA01."),
        report=report,
    ))
    print(f"COMPLETE {len(frozen.RUNS)} held-out replays {digest(OUT / 'SUMMARY.json')}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=tuple(frozen.RUNS))
    args = parser.parse_args()
    if args.run:
        replay(args.run)
    else:
        coordinate()
