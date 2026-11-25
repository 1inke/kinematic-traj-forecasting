"""Paired tests of overlap attention bias and adaptive refinement in MCUS."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "probe_mcus_20261124"))
from run_mcus_probe import (  # noqa: E402
    MCUSProbeTrainer, QROOT, attach_mcus, core, forecast_metrics, infer_mcus, inputs, load_data,
)
from prepare_inputs import digest  # noqa: E402
from run_b0 import write_json  # noqa: E402

sys.path.insert(0, str(QROOT / "refine_pred"))
from analysis import summarize_group  # noqa: E402

ARMS = {
    "fixed_no_overlap": (False, False),
    "fixed_overlap": (True, False),
    "learned_overlap": (True, True),
}
CONTRASTS = (("fixed_no_overlap", "fixed_overlap"), ("fixed_overlap", "learned_overlap"))
FOLDS, SEEDS = ("univ", "eth"), (42, 43)
EPOCHS, WORKERS, DEV_SCENE = 64, 2, 1
JOBS = {f"{fold}_s{seed}_{arm}": (fold, seed, arm)
        for seed in SEEDS for fold in FOLDS for arm in ARMS}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_bytes())


def torch_save(path, state):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def subset(data, mask):
    return {key: values[mask] for key, values in data.items()}


def fit_and_dev(fold):
    train_input, _ = inputs(fold)
    data = load_data(*train_input)
    is_dev = data["scene"] == DEV_SCENE
    if is_dev.sum() != 877 or is_dev.all():
        raise ValueError(f"Unexpected HOTEL holdout in {fold}")
    return subset(data, ~is_dev), subset(data, is_dev)


def trainer_for(data, seed, arm):
    overlap, adaptive = ARMS[arm]
    trainer = MCUSProbeTrainer(data, seed, "cuda", EPOCHS)
    decoder = trainer.model.decoder
    with torch.no_grad():
        decoder.ref_gate[-1].bias.zero_()
        trainer.ema.module.decoder.ref_gate[-1].bias.zero_()
    decoder.layer.overlap_bias.requires_grad_(overlap)
    decoder.ref_gate.requires_grad_(adaptive)
    # The inherited optimizer skips frozen tensors because their gradients stay None.
    return trainer


def preflight():
    fit, dev = fit_and_dev("univ")
    sample = subset(fit, slice(0, 256))
    reference_state, reference_output = None, None
    checks = {}
    for arm, (overlap, adaptive) in ARMS.items():
        trainer = trainer_for(sample, SEEDS[0], arm)
        state = core.cpu_tree(trainer.model.state_dict())
        output = infer_mcus(trainer.model, subset(sample, slice(0, 8)))
        if reference_state is None:
            reference_state, reference_output = state, output
        else:
            assert state.keys() == reference_state.keys()
            assert all(torch.equal(state[key], reference_state[key]) for key in state)
            for actual, expected in zip(output, reference_output):
                np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
        assert np.all(output[4] == 0.5)
        for _ in range(3):
            step = trainer.step()
        decoder = trainer.model.decoder
        bias_grad = decoder.layer.overlap_bias.weight.grad
        gate_grad = decoder.ref_gate[-1].bias.grad
        assert (bias_grad is not None and bias_grad.abs().max().item() > 0) if overlap else bias_grad is None
        assert (gate_grad is not None and gate_grad.abs().max().item() > 0) if adaptive else gate_grad is None
        if not adaptive:
            assert step["gate_mean"] == 0.5 and step["gate_std"] == 0.0
        contract = {"preflight": arm}
        path = ROOT / "PREFLIGHT_RESUME.pt"
        torch_save(path, trainer.snapshot(contract))
        continued = trainer.step()
        resumed = trainer_for(sample, SEEDS[0], arm)
        resumed.load(torch.load(path, map_location="cpu", weights_only=False), contract)
        replayed = resumed.step()
        for key, value in continued.items():
            np.testing.assert_allclose(replayed[key], value, atol=1e-6, rtol=1e-5)
        path.unlink()
        with tempfile.TemporaryDirectory(prefix="mcus_preflight_") as temp:
            path = Path(temp) / "dev_metrics.npz"
            evaluate(resumed.ema.module, subset(dev, slice(0, 16)), path)
            metrics = table(path)
            assert display_metrics(metrics)["all"]["windows"] == 16
        checks[arm] = dict(loss=step["loss"], overlap_gradient=overlap,
                           gate_gradient=adaptive, checkpoint_replay=True, dev_evaluation=True)
    write_json(ROOT / "PREFLIGHT.json", dict(status="PASS", checked_at=now(), checks=checks))
    print("PREFLIGHT PASS", flush=True)


def evaluate(model, data, path):
    paths, logits, aux_paths, _, gate, _ = infer_mcus(model, data)
    target = data["future"].astype(np.float64)
    scores = forecast_metrics(paths, logits, target)
    scores["L0_minFDE"] = np.linalg.norm(aux_paths[:, :, -1] - target[:, None, -1], axis=-1).min(1)
    speed = np.linalg.norm(np.diff(data["past"][:, 0].astype(np.float64), axis=1), axis=-1).mean(1) / 0.4
    scores.update(ids=data["ids"], fast=speed > 1.3, gate=gate.mean(1))
    if not all(np.isfinite(value).all() for key, value in scores.items() if key != "ids"):
        raise FloatingPointError(f"Nonfinite evaluation: {path}")
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **scores)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def job_contract(name):
    manifest = read_json(ROOT / "DESIGN.json")
    return dict(job=name, design_sha256=digest(ROOT / "DESIGN.json"),
                data=manifest["train_sha256"][JOBS[name][0]])


def lock_design():
    code = [Path(__file__), ROOT.parent / "probe_mcus_20261124/run_mcus_probe.py",
            QROOT / "refine_pred/analysis.py"]
    workspace = QROOT / "local/workspace"
    code += [workspace / filename for filename in (
        "advisor_eval.py", "advisor_model.py", "advisor_train.py",
        "qoder_work/NEXT_PHASE_EXEC_20261114/train_core.py",
        "qoder_work/NEXT_PHASE_EXEC_20261114/baseline_factory.py",
    )]
    design = dict(arms={arm: list(flags) for arm, flags in ARMS.items()},
                  jobs={name: list(job) for name, job in JOBS.items()}, epochs=EPOCHS,
                  workers=WORKERS, batch_size=64, dev_scene=DEV_SCENE,
                  held_out_windows=877, fixed_multiplier=0.175, max_multiplier=0.35,
                  train_sha256={fold: digest(inputs(fold)[0][0]) for fold in FOLDS},
                  test_sha256={fold: digest(inputs(fold)[1][0]) for fold in FOLDS},
                  source_sha256={str(path): digest(path) for path in code})
    path = ROOT / "DESIGN.json"
    if path.exists():
        if read_json(path) != design:
            raise ValueError("Frozen design or input changed")
    else:
        write_json(path, design)


def train(name):
    fold, seed, arm = JOBS[name]
    out = ROOT / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    contract = job_contract(name)
    checkpoint = out / "checkpoint.pt"
    fit, dev = fit_and_dev(fold)
    trainer = trainer_for(fit, seed, arm)
    if checkpoint.exists():
        trainer.load(torch.load(checkpoint, map_location="cpu", weights_only=False), contract)
    total = trainer.steps_per_epoch * EPOCHS
    began = time.monotonic()
    print(f"{name} training {len(fit['ids'])} windows from step {trainer.updates}/{total}", flush=True)
    while trainer.updates < total:
        row = trainer.step()
        if trainer.updates == 1 or trainer.cursor == len(fit["ids"]):
            torch_save(checkpoint, trainer.snapshot(contract))
            status = dict(name=name, state="training", epoch=trainer.epoch, updates=trainer.updates,
                          total_updates=total, updated_at=now(), pid=os.getpid(),
                          session_seconds=round(time.monotonic() - began, 1), **row)
            write_json(out / "CURRENT.json", status)
            print(json.dumps(status), flush=True)
    weights = out / "final_ema.pt"
    torch_save(weights, core.cpu_tree(trainer.ema.module.state_dict()))
    metrics = out / "dev_metrics.npz"
    evaluate(trainer.ema.module, dev, metrics)
    write_json(out / "DEV_DONE.json", dict(contract=contract, weights_sha256=digest(weights),
                                         metrics_sha256=digest(metrics), updates=trainer.updates))
    write_json(out / "CURRENT.json", dict(name=name, state="dev_complete", epoch=EPOCHS,
               updates=total, total_updates=total, updated_at=now(), pid=os.getpid()))


def checked_receipt(name):
    out = ROOT / "runs" / name
    path = out / "DEV_DONE.json"
    if not path.exists():
        return None
    receipt = read_json(path)
    if receipt["contract"] != job_contract(name):
        raise ValueError(f"Job contract changed: {name}")
    for file, key in (("final_ema.pt", "weights_sha256"), ("dev_metrics.npz", "metrics_sha256")):
        if digest(out / file) != receipt[key]:
            raise ValueError(f"Artifact failed verification: {name}/{file}")
    return receipt


def table(path):
    with np.load(path, allow_pickle=False) as raw:
        return dict(raw)


def display_metrics(metrics):
    names = ("minADE", "minFDE", "Top1_ADE", "Top1_FDE", "J", "ES", "L0_minFDE", "gate")
    return {group: dict(windows=int(mask.sum()), **{
        key: float(metrics[key][mask].mean()) if mask.any() else None for key in names})
        for group, mask in (("all", np.ones_like(metrics["fast"])), ("fast", metrics["fast"]),
                            ("rest", ~metrics["fast"]))}


def compare(metrics, tracks=None):
    result = {}
    for fold in FOLDS:
        result[fold] = {}
        for base, arm in CONTRASTS:
            paired = [(metrics[f"{fold}_s{seed}_{base}"], metrics[f"{fold}_s{seed}_{arm}"])
                      for seed in SEEDS]
            ids = paired[0][0]["ids"]
            fast = paired[0][0]["fast"]
            for a, b in paired:
                np.testing.assert_array_equal(a["ids"], ids)
                np.testing.assert_array_equal(b["ids"], ids)
                np.testing.assert_array_equal(a["fast"], fast)
                np.testing.assert_array_equal(b["fast"], fast)
            value = dict(seeds={})
            for group, mask in (("all", np.ones_like(fast)), ("fast", fast), ("rest", ~fast)):
                if not mask.any():
                    continue
                baseline = np.stack([a["minFDE"] for a, _ in paired])
                delta = np.stack([b["minFDE"] - a["minFDE"] for a, b in paired])
                value[group] = dict(
                    mean_delta_m=float(delta[:, mask].mean()),
                    baseline_m=float(baseline[:, mask].mean()),
                    relative=float(delta[:, mask].mean() / baseline[:, mask].mean()),
                    rescue_rate=float(np.mean([(~a["J"][mask] & b["J"][mask]).mean() for a, b in paired])),
                    loss_rate=float(np.mean([(a["J"][mask] & ~b["J"][mask]).mean() for a, b in paired])),
                )
                if tracks is not None:
                    np.testing.assert_array_equal(tracks[fold]["ids"], ids)
                    value[group]["track_bootstrap"] = summarize_group(
                        baseline.mean(0), delta.mean(0), tracks[fold]["tracks"], mask)
                for seed, change in zip(SEEDS, delta):
                    value["seeds"].setdefault(str(seed), {})[group] = float(change[mask].mean())
            result[fold][f"{arm}_vs_{base}"] = value
    return result


def report(receipts):
    dev = {name: table(ROOT / "runs" / name / "dev_metrics.npz") for name in JOBS}
    dev_contrasts = compare(dev)
    holdout = dict(contrasts=dev_contrasts, receipts=receipts,
                   metrics={job: display_metrics(data) for job, data in dev.items()},
                   order_by_dev_minFDE=sorted(ARMS, key=lambda arm: np.mean([
                       dev[f"{fold}_s{seed}_{arm}"]["minFDE"].mean()
                       for fold in FOLDS for seed in SEEDS])))
    path = ROOT / "DEV_LOCK.json"
    if path.exists():
        if read_json(path) != holdout:
            raise ValueError("Frozen development results changed")
    else:
        write_json(path, holdout)

    test, tracks = {}, {}
    test_hashes = read_json(ROOT / "DESIGN.json")["test_sha256"]
    for fold in FOLDS:
        _, test_input = inputs(fold)
        if digest(test_input[0]) != test_hashes[fold]:
            raise ValueError(f"Test input changed: {fold}")
        data = load_data(*test_input)
        tracks[fold] = table(QROOT / f"refine_pred/results/tracks/{fold}/TRACKS.npz")
        np.testing.assert_array_equal(tracks[fold]["ids"], data["ids"])
        for name, (job_fold, seed, _) in JOBS.items():
            if job_fold != fold:
                continue
            out = ROOT / "runs" / name
            model, _ = core.make_model("cuda", seed)
            attach_mcus(model)
            model.load_state_dict(torch.load(out / "final_ema.pt", map_location="cpu",
                                             weights_only=True), strict=True)
            path = out / "test_metrics.npz"
            evaluate(model, data, path)
            test[name] = table(path)
            print(f"{name} test evaluation saved", flush=True)
    write_json(ROOT / "SUMMARY.json", dict(
        finished_at=now(), design_sha256=digest(ROOT / "DESIGN.json"),
        dev_lock_sha256=digest(ROOT / "DEV_LOCK.json"), contrasts=compare(test, tracks),
        metrics={job: display_metrics(data) for job, data in test.items()}))


def launch_job(name):
    out = ROOT / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "training.log").open("a", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", "-B", __file__, "--job", name],
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def coordinate():
    path = ROOT / "CURRENT.json"
    started = now()

    def publish(state, **fields):
        write_json(path, dict(state=state, pid=os.getpid(), started_at=started,
                              updated_at=now(), jobs_total=len(JOBS), **fields))

    try:
        if read_json(ROOT / "PREFLIGHT.json")["status"] != "PASS":
            raise ValueError("Preflight must pass")
        lock_design()
        receipts = {name: item for name in JOBS if (item := checked_receipt(name)) is not None}
        publish("training", completed=list(receipts), workers=WORKERS)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            pending = {pool.submit(launch_job, name): name for name in JOBS if name not in receipts}
            for task in as_completed(pending):
                name = pending[task]
                try:
                    task.result()
                    receipt = checked_receipt(name)
                    if receipt is None:
                        raise RuntimeError(f"Missing completion receipt for {name}")
                    receipts[name] = receipt
                except Exception:
                    for future in pending:
                        future.cancel()
                    raise
                publish("training", completed=list(receipts), workers=WORKERS)
        publish("test_evaluation", completed=list(receipts))
        report(receipts)
        publish("complete", completed=list(receipts))
    except Exception as error:
        publish("failed", error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    task = parser.add_mutually_exclusive_group()
    task.add_argument("--preflight", action="store_true")
    task.add_argument("--job", choices=JOBS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if args.preflight:
        preflight()
    elif args.job:
        train(args.job)
    else:
        coordinate()
