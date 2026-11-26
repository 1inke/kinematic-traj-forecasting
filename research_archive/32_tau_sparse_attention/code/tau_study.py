"""Run the fixed C0 screen, then T2 vs B0 five-fold confirmation on a GO."""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from prepare_inputs import digest
from run_b0 import write_json
from tau_run import ARMS, FOLDS, SEEDS
from tracks_analysis import cluster_draws, summarize_group


def read(path):
    return json.loads(path.read_bytes())


def task_dir(output, task):
    fold, arm, seed = task
    return output / "runs" / f"{fold}_{arm}_s{seed}"


def dispatch(args, tasks, command, devices, deadline):
    limit = 2 if command == "train" else 1
    queued = deque(tasks)
    active = {}
    while queued or active:
        for worker, (process, task) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            del active[worker]
            if status:
                for other, _ in active.values():
                    other.terminate()
                for other, _ in active.values():
                    other.wait(timeout=30)
                raise RuntimeError(f"{command} failed for {task}, exit {status}")
        for gpu in devices:
            for slot in range(limit):
                worker = (gpu, slot)
                if not queued or worker in active or time.time() >= deadline:
                    continue
                task = queued.popleft()
                fold, arm, seed = task
                out = task_dir(args.output, task)
                file = out / ("TRAINED.json" if command == "train" else "RESULT.json")
                if file.exists():
                    result = read(file)
                    assert (result["contract"]["fold"], result["contract"]["arm"],
                            result["contract"]["seed"], result["contract"]["assets"]) == (*task, args.assets)
                    filename, key = (("epoch_064.pt", "checkpoint_sha256") if command == "train"
                                     else ("METRICS.npz", "metrics_sha256"))
                    assert digest(out / filename) == result[key]
                    continue
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                           PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                if deadline < float("inf"):
                    extra = ["--deadline", str(deadline)]
                else:
                    extra = []
                cmd = [sys.executable, "-u", str(Path(__file__).with_name("tau_run.py")), command,
                       "--fold", fold, "--arm", arm, "--seed", str(seed), "--out", str(out),
                       "--data", str(args.data), "--assets", args.assets, *extra]
                print(f"START {command} {out.name} physical_gpu={gpu}", flush=True)
                active[worker] = (subprocess.Popen(cmd, env=env), task)
        if time.time() >= deadline and not active:
            break
        if active:
            time.sleep(2)
    receipt = "TRAINED.json" if command == "train" else "RESULT.json"
    return all((task_dir(args.output, task) / receipt).exists() for task in tasks)


def comparison(output, fold, arm):
    samples = {"B0": [], arm: []}
    seed_changes, biases = {}, []
    identifiers = groups = None
    for seed in SEEDS:
        for name in samples:
            run = task_dir(output, (fold, name, seed))
            with np.load(run / "METRICS.npz", allow_pickle=False) as metrics:
                if identifiers is None:
                    identifiers, groups = metrics["ids"], metrics["tracks"]
                np.testing.assert_array_equal(metrics["ids"], identifiers)
                np.testing.assert_array_equal(metrics["tracks"], groups)
                samples[name].append(metrics["minFDE"])
        b0, other = samples["B0"][-1], samples[arm][-1]
        seed_changes[str(seed)] = float(other.mean() / b0.mean() - 1)
        biases.append(read(task_dir(output, (fold, arm, seed)) / "RESULT.json")["bias"])
    base = np.mean(samples["B0"], axis=0)
    change = np.mean(samples[arm], axis=0) - base
    summary = summarize_group(base, change, groups, np.ones(len(base), dtype=bool))
    _, keys, counts = cluster_draws(groups)
    numer = counts @ np.bincount(keys, weights=change)
    denom = counts @ np.bincount(keys, weights=base)
    draws = numer / denom
    summary.update(by_seed=seed_changes, beta_by_seed=biases, upper_one_sided_95=float(np.quantile(draws, .95)))
    return summary, draws


def preflight(args, devices):
    jobs = []
    for gpu in devices:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE="1")
        cmd = [sys.executable, "-u", str(Path(__file__).with_name("tau_run.py")), "preflight",
               "--data", str(args.data), "--out", str(args.output / f"preflight_gpu_{gpu}")]
        jobs.append(subprocess.Popen(cmd, env=env))
    codes = [job.wait() for job in jobs]
    if any(codes):
        raise RuntimeError(f"T4 preflight failed: {codes}")


def run(args):
    devices = list(range(torch.cuda.device_count()))
    names = [torch.cuda.get_device_name(i) for i in devices]
    if len(devices) != 2 or any("T4" not in name for name in names):
        raise RuntimeError(f"Kaggle T4 x2 required, got {names}")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"KAGGLE_T4_CONFIRMED {names}, torch={torch.__version__}, cuda={torch.version.cuda}", flush=True)
    preflight(args, devices)
    write_json(args.output / "PREFLIGHT.json", dict(status="PASS", devices=names, torch=str(torch.__version__),
                                                   cuda=torch.version.cuda, assets=args.assets))
    deadline = time.time() + args.budget_hours * 3600
    c0 = [("C0", arm, seed) for seed in SEEDS for arm in ARMS]
    if not dispatch(args, c0, "train", devices, deadline):
        return "PAUSED_C0"
    dispatch(args, c0, "evaluate", devices, float("inf"))
    result = {arm: comparison(args.output, "C0", arm)[0] for arm in ARMS[1:]}
    go = result["T2"]["relative"] <= .01
    write_json(args.output / "C0_RESULT.json", dict(decision="GO" if go else "STOP", comparisons=result))
    print(f"C0_RESULT decision={'GO' if go else 'STOP'} T2_delta={result['T2']['relative']:.6f}", flush=True)
    if not go:
        return "COMPLETE_C0_STOP"
    if args.c0_only:
        return "COMPLETE_C0_GO"
    # Schedule longer folds first, evaluate only after all final EMA states are locked.
    group = ("eth", "hotel", "zara01", "zara02", "univ")
    formal = [(fold, arm, seed) for fold in group for seed in SEEDS for arm in ("B0", "T2")]
    if not dispatch(args, formal, "train", devices, deadline):
        return "PAUSED_FIVEFOLD"
    dispatch(args, formal, "evaluate", devices, float("inf"))
    paired = {fold: comparison(args.output, fold, "T2") for fold in FOLDS}
    folds = {fold: summary for fold, (summary, _) in paired.items()}
    relative = {fold: row["relative"] for fold, row in folds.items()}
    bootstrap = np.stack([draws for _, draws in paired.values()])
    # Within each fold, draws retain paired trajectories and all three seeds.
    rng = np.random.default_rng(20261125)
    bootstrap = np.stack([rng.permutation(draw) for draw in bootstrap])
    overall_upper = float(np.quantile(bootstrap.mean(axis=0), .95))
    leave_out = {name: float(np.mean([v for f, v in relative.items() if f != name])) for name in FOLDS}
    per_fold = {}
    for fold, summary in folds.items():
        signs = np.sign(np.asarray(summary["beta_by_seed"]))
        per_fold[fold] = ((signs != 0).all(axis=0) & (signs == signs[:1]).all(axis=0)).tolist()
    write_json(args.output / "FIVEFOLD_RESULT.json", dict(
        folds=folds, leave_one_fold_out=leave_out, overall_relative=float(np.mean(list(relative.values()))),
        overall_upper_one_sided_95=overall_upper,
        minFDE_gate=overall_upper <= .01 and folds["eth"]["upper_one_sided_95"] <= .02,
        beta_sign_consistency_by_fold=per_fold,
        beta_sign_gate=all(all(heads) for heads in per_fold.values()),
    ))
    return "COMPLETE_FIVEFOLD"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--budget-hours", type=float, default=9)
    parser.add_argument("--c0-only", action="store_true")
    args = parser.parse_args()
    try:
        status = run(args)
    except Exception as exc:
        write_json(args.output / "STATUS.json", dict(status="FAILED", error=str(exc)))
        raise
    write_json(args.output / "STATUS.json", dict(status=status, assets=args.assets))
    print(f"STUDY_STATUS {status}", flush=True)


if __name__ == "__main__":
    main()
