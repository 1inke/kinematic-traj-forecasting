"""Run the original B0 training core with an auditable, resumable Colab output directory."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
from prepare_inputs import digest


DRIVE_MOUNT = Path("/content/drive")
DRIVE_ATTEMPTS = 6


def on_drive(path):
    return DRIVE_MOUNT in path.parents


def replace_verified(path, stage, verify):
    remote = on_drive(path)
    attempts = DRIVE_ATTEMPTS if remote else 1
    for attempt in range(attempts):
        part = None
        try:
            if remote and path.is_file() and verify(path):
                return
            fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
            os.close(fd)
            part = Path(name)
            stage(part)
            if not verify(part):
                raise OSError(f"Staged data failed verification: {path}")
            part.replace(path)
            if remote and not verify(path):
                raise OSError(f"Published data failed verification: {path}")
            return
        except OSError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.25 * 2**attempt)
        finally:
            if part is not None:
                try:
                    part.unlink(missing_ok=True)
                except OSError:
                    # Drive may already have completed a reported failed rename.
                    pass


def write_json(path, data):
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    if path.name == "CURRENT.json" and on_drive(path):
        # Display-only status; recovery uses the verified checkpoint and result receipts.
        for attempt in range(3):
            try:
                path.write_bytes(payload)
                return
            except OSError as error:
                if attempt == 2:
                    print(f"Could not update progress at {path}: {error}", file=sys.stderr, flush=True)
                else:
                    time.sleep(0.25 * (attempt + 1))
        return
    replace_verified(path, lambda part: part.write_bytes(payload),
                     lambda candidate: candidate.read_bytes() == payload)


def verify_inputs(workspace):
    info = {}
    for bundle in ("c0_inputs", "baseline_sources"):
        path = workspace / f"{bundle}.manifest.json"
        manifest = json.loads(path.read_bytes())
        for name, wanted in manifest["members"].items():
            if digest(workspace / name) != wanted:
                raise ValueError(f"Cannot run with changed input: {name}")
        info[bundle] = digest(path)
    return info


def load_data(path, expected):
    with np.load(path, allow_pickle=False) as raw:
        data = {key: raw[key] for key in raw.files}
    assert len(np.unique(data["ids"])) == expected
    assert data["past"].shape == (expected, 9, 8, 2)
    assert data["future"].shape == (expected, 12, 2)
    assert np.isfinite(data["past"]).all() and np.isfinite(data["future"]).all()
    assert ((1 <= data["num_valid"]) & (data["num_valid"] <= 9)).all()
    return data


def publish(local, target):
    # Stage and verify bytes before creating an output that a new session may consume.
    wanted = digest(local)
    replace_verified(target, lambda part: shutil.copyfile(local, part),
                     lambda candidate: digest(candidate) == wanted)
    return wanted


def checkpoint(out, core, trainer, contract):
    path = out / f"epoch_{trainer.epoch:03d}.pt"
    with tempfile.TemporaryDirectory(prefix="b0_checkpoint_") as temp:
        local = Path(temp) / path.name
        core.save_checkpoint(local, trainer.snapshot(contract))
        h = publish(local, path)
    write_json(path.with_suffix(".json"), dict(sha256=h, epoch=trainer.epoch,
                                                updates=trainer.updates))


def latest(out, core, trainer, contract):
    for manifest in sorted(out.glob("epoch_???.json"), reverse=True):
        path = manifest.with_suffix(".pt")
        try:
            data = json.loads(manifest.read_bytes())
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Skipped unreadable checkpoint receipt {manifest.name}", flush=True)
            continue
        if path.is_file() and digest(path) == data["sha256"]:
            with tempfile.TemporaryDirectory(prefix="b0_restore_") as temp:
                local = Path(temp) / path.name
                shutil.copyfile(path, local)
                if digest(local) != data["sha256"]:
                    raise IOError(f"Checkpoint changed during restore: {path}")
                state = core.torch.load(local, map_location="cpu", weights_only=False)
            trainer.load(state, contract)
            assert trainer.epoch == data["epoch"] and trainer.updates == data["updates"]
            assert trainer.cursor == len(trainer.indices)
            return trainer.epoch
        print(f"Skipped incomplete checkpoint {path.name}", flush=True)
    return 0


def evaluate(core, trainer, dev, out, contract):
    paths, logits = core.infer(trainer.ema.module, dev, batch=64)
    error = np.linalg.norm(paths - dev["future"][:, None], axis=-1)
    ade, fde = error.mean(2), error[:, :, -1]
    best, top = fde.argmin(1), logits.argmax(1)
    row = np.arange(len(ade))
    prob = np.exp(logits.astype(np.float64) - logits.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    miss_probability = 1 - prob[row, best]
    metrics = dict(windows=len(ade), minADE=float(ade.min(1).mean()),
                   minFDE=float(fde.min(1).mean()), Top1_ADE=float(ade[row, top].mean()),
                   Top1_FDE=float(fde[row, top].mean()),
                   J=float(((ade<=.2) & (fde<=.4)).any(1).mean()),
                   brier_minFDE=float((fde[row, best] + miss_probability**2).mean()))
    with tempfile.TemporaryDirectory(prefix="b0_predictions_") as temp:
        path = Path(temp) / "dev_predictions.npz"
        np.savez_compressed(path, ids=dev["ids"], paths=paths, logits=logits,
                            future=dev["future"], past=dev["past"][:, 0], groups=dev["group"])
        h = publish(path, out / path.name)
    row = dict(status="PASS", contract=contract, dev=metrics, epochs=trainer.epoch,
               updates=trainer.updates, predictions_sha256=h)
    if contract["mode"] == "formal":
        ref = contract["archive10_minFDE"]
        delta = (metrics["minFDE"] - ref) / ref
        row["archive10_relative_minFDE_change"] = delta
        if contract["seed"] == 42:
            row["PF2_within_1_5_percent"] = abs(delta) <= .015
    write_json(out/"RESULT.json", row)
    print(json.dumps(row,ensure_ascii=False), flush=True)


def run(workspace, out, seed, smoke, stop_after):
    verified = verify_inputs(workspace)
    base = workspace / "qoder_work/NEXT_PHASE_EXEC_20261114"
    sys.path.insert(0, str(base))
    import common
    common.ASTRA = workspace / "ASTRA_ACCURACY_Q20"
    import train_core as core

    torch = core.torch
    torch.set_num_threads(1)
    if smoke:
        device, epochs, interval = "cpu", 2, 1
        name = "cpu"
    else:
        if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name(0).split():
            raise RuntimeError("The formal B0 run requires an allocated NVIDIA L4.")
        device, epochs, interval = "cuda:0", 64, 8
        name = torch.cuda.get_device_name(0)
    limit = stop_after if stop_after is not None else epochs
    if not 1 <= limit <= epochs:
        raise ValueError(f"stop-after-epoch must be in [1, {epochs}]")
    fit = load_data(base/"C0/fit.npz", 21719)
    dev = load_data(base/"C0/dev.npz", 4753)
    if smoke:
        fit, dev = ({k: v[:8] for k,v in data.items()} for data in (fit,dev))
    reference = json.loads((workspace/f"reference/B0_s{seed}_dev_metrics.json").read_bytes())
    contract = dict(mode="smoke" if smoke else "formal", inputs=verified, seed=seed, order_seed=seed,
                    epochs=epochs, checkpoint_every=interval, batch=64, gradient_clip=5,
                    final_ema_decay=.995, Torch=str(torch.__version__), cuda=torch.version.cuda,
                    device=name, archive10_minFDE=reference["overall"]["minFDE"],
                    runner_sha256=digest(Path(__file__)))
    out.mkdir(parents=True, exist_ok=True)
    spec = out/"CONTRACT.json"
    if spec.exists() and json.loads(spec.read_bytes()) != contract:
        raise ValueError(f"Existing run directory has a different contract: {out}")
    if not spec.exists():
        write_json(spec, contract)
    completed = out/"RESULT.json"
    if completed.exists():
        result = json.loads(completed.read_bytes())
        assert result["contract"] == contract
        assert digest(out/"dev_predictions.npz") == result["predictions_sha256"]
        print(f"Verified completed run: {completed}", flush=True)
        return
    trainer = core.Trainer(fit, np.arange(len(fit["ids"])), batch=64, epochs=epochs, device=device,
                           seed=seed, order_seed=seed, arm="B0")
    assert sum(p.numel() for p in trainer.model.parameters()) == 295474
    assert trainer.steps_per_epoch == math.ceil(len(fit["ids"])/64)
    start = latest(out, core, trainer, contract)
    for epoch in range(start+1, limit+1):
        t = time.monotonic()
        for _ in range(trainer.steps_per_epoch):
            trainer.step()
        row = dict(epoch=epoch, seconds=time.monotonic()-t, updates=trainer.updates,
                   mean_loss=float(np.mean([x["loss"] for x in trainer.history[-trainer.steps_per_epoch:]])))
        write_json(out/f"timing_{epoch:03d}.json", row)
        if epoch % interval == 0 or epoch == limit:
            checkpoint(out, core, trainer, contract)
        print(json.dumps(row), flush=True)
    if trainer.epoch == epochs:
        evaluate(core,trainer,dev,out,contract)
    else:
        print(f"Stopped at saved epoch {trainer.epoch}/{epochs}", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--seed", type=int, choices=(42,43,44), default=42)
    cli.add_argument("--stop-after-epoch", type=int)
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--formal", action="store_true")
    args = cli.parse_args()
    run(args.workspace.resolve(),args.out.resolve(),args.seed,args.smoke,args.stop_after_epoch)

