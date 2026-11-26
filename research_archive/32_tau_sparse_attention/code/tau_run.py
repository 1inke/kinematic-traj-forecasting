"""Run one paired arm using the exact B0 optimization, loss and output geometry."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from advisor_eval import forecast_metrics, safe_npz
from prepare_inputs import digest
from run_b0 import checkpoint, latest, load_data, write_json
import tau_geometry


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "qoder_work/NEXT_PHASE_EXEC_20261114"
sys.path.insert(0, str(BASE))
import common

common.ASTRA = ROOT / "ASTRA_ACCURACY_Q20"
import train_core as core


ARMS = ("B0", "T1", "T2", "T3")
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
SEEDS = (42, 43, 44)
ORIGINAL_MODEL = core.make_model


def data_paths(data, fold, evaluation):
    if fold == "C0":
        split, count = ("dev", 4753) if evaluation else ("fit", 21719)
        return data / "qoder_work/NEXT_PHASE_EXEC_20261114/C0" / f"{split}.npz", count
    audit = json.loads((data / "fivefold/source_audit.json").read_bytes())
    assert audit["recording_isolation_passed"]
    split = "test" if evaluation else "train"
    return data / f"fivefold/{fold}/{split}.npz", audit["folds"][fold]["splits"][split]["windows"]


def power_for(data, fold):
    law = json.loads((data / "R_TAU2.json").read_bytes())
    return law["method"]["C0_power"] if fold == "C0" else law["folds"][fold]["power"]


def build_trainer(fit, seed, arm, power, device, epochs, batch=64):
    def model_factory(*args, **kwargs):
        model, config = ORIGINAL_MODEL(*args, **kwargs)
        assert model.drse_enabled
        tau_geometry.attach(model, arm, power)
        return model, config

    core.make_model = model_factory
    try:
        # Legacy train_core names T1/T2/T3 for unrelated loss changes.
        trainer = core.Trainer(fit, np.arange(len(fit["ids"])), batch=batch, epochs=epochs,
                               device=device, seed=seed, order_seed=seed, arm="B0")
    finally:
        core.make_model = ORIGINAL_MODEL
    expected = 295478 if arm in ("T2", "T3") else 295474
    assert sum(p.numel() for p in trainer.model.parameters()) == expected
    return trainer


def run(args):
    fit_path, expected = data_paths(args.data, args.fold, False)
    fit = load_data(fit_path, expected)
    power = power_for(args.data, args.fold)
    device = "cuda:0"
    gpu = torch.cuda.get_device_name(0)
    if "T4" not in gpu:
        raise RuntimeError(f"Requested T4, got {gpu}")
    args.out.mkdir(parents=True, exist_ok=True)
    contract = dict(arm=args.arm, fold=args.fold, seed=args.seed, power=power,
                    fit_sha256=digest(fit_path), assets=args.assets, epochs=64, batch=64,
                    torch=str(torch.__version__), cuda=torch.version.cuda, gpu=gpu)
    locked = args.out / "CONTRACT.json"
    if locked.exists():
        assert json.loads(locked.read_bytes()) == contract
    else:
        write_json(locked, contract)
    final = args.out / "TRAINED.json"
    trainer = build_trainer(fit, args.seed, args.arm, power, device, 64)
    start = latest(args.out, core, trainer, contract)
    if final.exists():
        record = json.loads(final.read_bytes())
        assert record["contract"] == contract and start == 64
        assert digest(args.out / "epoch_064.pt") == record["checkpoint_sha256"]
        return
    for epoch in range(start + 1, 65):
        if time.time() >= args.deadline:
            if trainer.epoch and not (args.out / f"epoch_{trainer.epoch:03d}.json").exists():
                checkpoint(args.out, core, trainer, contract)
            print(f"PAUSED {args.out.name} at {trainer.epoch}/64", flush=True)
            return
        now = time.monotonic()
        for _ in range(trainer.steps_per_epoch):
            trainer.step()
        if epoch % 8 == 0 or epoch in (1, 64):
            checkpoint(args.out, core, trainer, contract)
            # Retain the latest two verified recovery points.
            for old in sorted(args.out.glob("epoch_???.json"))[:-2]:
                old.unlink()
                old.with_suffix(".pt").unlink()
        status = dict(epoch=epoch, seconds=time.monotonic() - now, updates=trainer.updates,
                      mean_loss=float(np.mean([x["loss"] for x in trainer.history[-trainer.steps_per_epoch:]])))
        write_json(args.out / "CURRENT.json", status)
        print(f"TRAIN {args.out.name} {json.dumps(status)}", flush=True)
    bias = getattr(trainer.ema.module.agent_aware_trans_encoder, "tau_beta", None)
    write_json(final, dict(contract=contract, updates=trainer.updates,
                          checkpoint_sha256=digest(args.out / "epoch_064.pt"),
                          bias=bias.detach().cpu().tolist() if bias is not None else None))


def evaluate(args):
    trained = json.loads((args.out / "TRAINED.json").read_bytes())
    contract = trained["contract"]
    assert (contract["arm"], contract["fold"], contract["seed"], contract["assets"]) == (
        args.arm, args.fold, args.seed, args.assets)
    weights = args.out / "epoch_064.pt"
    assert digest(weights) == trained["checkpoint_sha256"]
    path, expected = data_paths(args.data, args.fold, True)
    dev = load_data(path, expected)
    if args.fold == "C0":
        tracks = np.array(["|".join(window.split("|")[:2]) for window in dev["ids"]])
    else:
        with np.load(args.data / f"tracks/{args.fold}.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(dev["ids"], archive["ids"])
            tracks = archive["tracks"]
    model, _ = ORIGINAL_MODEL("cuda:0", args.seed)
    tau_geometry.attach(model, args.arm, contract["power"])
    model.load_state_dict(torch.load(weights, weights_only=False, map_location="cpu")["ema"], strict=True)
    paths, logits = core.infer(model, dev)
    if not (np.isfinite(paths).all() and np.isfinite(logits).all()):
        raise ValueError(f"Invalid predictions for {args.out.name}")
    metrics = forecast_metrics(paths, logits, dev["future"])
    safe_npz(args.out / "METRICS.npz", ids=dev["ids"], tracks=tracks, **metrics)
    report = dict(metrics={k: float(v.mean()) for k, v in metrics.items()}, windows=expected,
                  tracks=len(Counter(tracks)), bias=trained["bias"], contract=contract,
                  metrics_sha256=digest(args.out / "METRICS.npz"))
    write_json(args.out / "RESULT.json", report)
    print(f"EVAL {args.out.name} minFDE={report['metrics']['minFDE']:.6f}", flush=True)


def preflight(args):
    device = "cpu" if args.cpu else "cuda:0"
    if not args.cpu:
        assert "T4" in torch.cuda.get_device_name(0)
    fit = load_data(*data_paths(args.data, "C0", False))
    idx = np.random.default_rng(42).choice(len(fit["ids"]), 64, replace=False)
    fit = {k: v[idx] for k, v in fit.items()}
    p, n = (torch.as_tensor(fit[k], device=device) for k in ("past", "num_valid"))
    positions = torch.tensor([[2., 0.], [2., 0.], [.5, 0.], [2., 1.]], device=device)
    speeds = torch.tensor([[-1., 0.], [1., 0.], [0., 0.], [-1., 0.]], device=device)
    found = tau_geometry.collision_time(positions, speeds, .6)
    torch.testing.assert_close(found, found.new_tensor([1.4, torch.inf, 0., torch.inf]))
    initial = {}
    for arm in ARMS:
        t = build_trainer(fit, 42, arm, 2.0, device, 2, 32)
        t.model.eval()
        with torch.no_grad():
            output = core.prediction(t.model, p, n)
            encoder = t.model.agent_aware_trans_encoder
            initial[arm] = (output[0].detach().clone(), encoder.last_allowed.clone())
        t.step()
        if arm in ("T2", "T3"):
            # The frozen H3 zero-initialized output branch can block encoder
            # gradients on the first update before its output head moves.
            t.step()
            assert any(name.endswith("tau_beta") for name in t.connected)
        if arm != "T2":
            continue
        args.out.mkdir(parents=True, exist_ok=True)
        contract = {"preflight": True}
        checkpoint(args.out, core, t, contract)
        t.step()
        saved = t.snapshot(contract)
        restored = build_trainer(fit, 42, arm, 2.0, device, 2, 32)
        assert latest(args.out, core, restored, contract) == 1
        restored.step()
        core.exact(saved, restored.snapshot(contract), "TAU_RESUME")
    for left, right in (("B0", "T3"), ("T1", "T2")):
        for a, b in zip(initial[left], initial[right]):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    changed = torch.count_nonzero(initial["T1"][1] ^ initial["B0"][1]).item()
    assert changed > 0
    print(f"PREFLIGHT_PASS hardware={device} altered_graph_edges={changed} exact_resume=true", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate", "preflight"))
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fold", choices=("C0", *FOLDS), default="C0")
    parser.add_argument("--arm", choices=ARMS, default="B0")
    parser.add_argument("--seed", choices=SEEDS, type=int, default=42)
    parser.add_argument("--assets", default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--deadline", type=float, default=float("inf"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "train":
        run(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        preflight(args)


if __name__ == "__main__":
    main()
