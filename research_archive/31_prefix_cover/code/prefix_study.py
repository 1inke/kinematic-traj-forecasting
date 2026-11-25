"""Run L4 preflight, the C0 screen, and the conditional five-fold queue."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

import advisor_study as queueing
from prefix_decisions import confirm, screen
from prefix_preflight import execute
from prefix_loss import ARMS
from prefix_train import FOLDS, ROOT, SEEDS
from prepare_inputs import digest
from run_b0 import verify_inputs, write_json

STAGE2 = ("A","N-cov","N-conf","N-win")


def launch(arm, fold, seed, out, bench):
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", "-B", str(ROOT/"prefix_train.py"), "--arm", arm, "--fold", fold,
           "--seed", str(seed), "--out", str(out)]
    if bench:
        cmd.append("--bench")
    env = {**os.environ, "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1"}
    with (out/"worker.log").open("a", encoding="utf-8") as log:
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)


def queue(root, name, jobs, cap, deadline, bench=False):
    receipt = queueing.queue(root, name, jobs, cap, deadline, deadline, bench=bench)
    if not bench and receipt["failed"]:
        raise RuntimeError(f"{name} failed: {receipt['failed']}")
    return receipt


def bench(root, until):
    def trial(cap, sequence):
        jobs = [(a, "C0", 42+i%3, root/"benchmark"/str(cap)/f"{a}_{i}") for i,a in enumerate(sequence)]
        state = queue(root, f"PF-T:{cap}", jobs, cap, until, bench=True)
        runs = [queueing.read(j[-1]/"RESULT.json") for j in jobs if queueing.usable(j[-1])]
        return state, runs

    state, solo = trial(1, ARMS)
    if len(solo) != len(ARMS):
        raise RuntimeError("PF-T solo baseline incomplete")
    reference = {r["contract"]["arm"]: r["seconds"] for r in solo}
    candidates = {}
    for cap in (6,8,10):
        state, runs = trial(cap, [ARMS[i%len(ARMS)] for i in range(cap)])
        slowdown = max((r["seconds"]/reference[r["contract"]["arm"]] for r in runs), default=None)
        valid = len(runs) == cap and not state["failed"] and slowdown <= 2.
        wall = max(r["seconds"] for r in runs) if runs else None
        candidates[str(cap)] = dict(valid=valid, completed=len(runs), failures=state["failed"], slowdown=slowdown,
                                     epochs_per_hour=len(runs)*3600/wall if wall else None)
    eligible = [int(n) for n,r in candidates.items() if r["valid"]]
    if not eligible:
        raise RuntimeError(f"PF-T: no registered concurrency passed: {candidates}")
    chosen = max(eligible, key=lambda n:candidates[str(n)]["epochs_per_hour"])
    return dict(status="PASS", chosen=chosen, solo_seconds=reference, by_lane_count=candidates)


def job(root, stage, arm, fold, seed):
    tag = f"{arm}_s{seed}" if fold == "C0" else f"{arm}_{fold}_s{seed}"
    return arm, fold, seed, root/stage/"runs"/tag


def lock_inputs():
    lock = queueing.read(ROOT/"DESIGN_LOCK.json")
    for name, wanted in {**lock["files"], **lock["data"]}.items():
        assert digest(ROOT/name) == wanted, name
    verify_inputs(ROOT)
    manifest = queueing.read(ROOT/"fivefold_inputs.manifest.json")
    for name, wanted in manifest["members"].items():
        assert digest(ROOT/name) == wanted, name
    return digest(ROOT/"DESIGN_LOCK.json")


def work(root, hours):
    torch.set_num_threads(1)
    until = time.monotonic() + hours*3600
    root.mkdir(parents=True, exist_ok=True)
    design = lock_inputs()
    assert torch.cuda.is_available() and "L4" in torch.cuda.get_device_name(0).split()
    bind = dict(gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__), cuda=torch.version.cuda,
                design=design)
    runtime = root/"RUNTIME.json"
    if runtime.is_file() and queueing.read(runtime) != bind:
        raise RuntimeError("Runtime or design differs from the initial run")
    write_json(runtime, bind)
    queueing.launch = launch
    gate = root/"PREFLIGHT.json"
    if gate.is_file() and queueing.read(gate).get("status") == "PASS":
        report = queueing.read(gate)
    else:
        queueing.status(root, "PREFLIGHT_QUICK", gpu=bind["gpu"])
        quick = root/"GPU_PREFLIGHT.json"
        if not quick.is_file() or queueing.read(quick).get("status") != "PASS":
            execute(quick, "cuda:0")
        throughput = root/"PF_T.json"
        if not throughput.is_file():
            write_json(throughput, bench(root, until))
        report = dict(status="WAITING_PF4", quick=queueing.read(quick), **{"PF-T":queueing.read(throughput)})
        write_json(gate, report)
        baseline = job(root, "stage1", "A", "C0", 42)
        queue(root, "PF-4", [baseline], 1, until)
        observed = queueing.read(baseline[-1]/"RESULT.json")["mean"]["riskp_minFDE20"]
        archived = queueing.read(ROOT/"archive27_b0_s42.json")["dev"]["overall"]["minFDE"]
        delta = observed/archived - 1
        report["PF-4"] = dict(observed=observed, archived=archived, relative_delta=delta, passed=abs(delta)<=.015)
        report["status"] = "PASS" if report["PF-4"]["passed"] else "FAIL"
        write_json(gate, report)
        if report["status"] != "PASS":
            raise RuntimeError(f"PF-4 failed: {report['PF-4']}")
    cap = report["PF-T"]["chosen"]
    if (root/"DONE.json").is_file():
        done = queueing.read(root/"DONE.json")
        assert all(digest(root/n)==h for n,h in done["files"].items())
        return
    c0 = [job(root, "stage1", a, "C0", s) for s in SEEDS for a in ARMS]
    queue(root, "C0", c0, cap, until)
    result = screen(root/"stage1", ARMS)
    write_json(root/"C0_RESULT.json", result)
    names = ["PREFLIGHT.json", "C0_RESULT.json", "RUNTIME.json"]
    if result["decision"] == "GO":
        folds = [job(root, "stage2", a, f, s) for f in FOLDS for s in SEEDS for a in STAGE2]
        queue(root, "FIVEFOLD", folds, cap, until)
        final = confirm(root/"stage2", STAGE2)
        write_json(root/"FIVEFOLD_RESULT.json", final)
        names.append("FIVEFOLD_RESULT.json")
    write_json(root/"DONE.json", dict(status="PASS", c0=result["decision"],
                                      files={n:digest(root/n) for n in names}))
    queueing.status(root, "COMPLETE", c0=result["decision"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hours", type=float, required=True)
    a = p.parse_args()
    try:
        work(a.out, a.hours)
    except queueing.Deadline:
        sys.exit(75)
