"""Run one fixed-architecture prefix-cover arm with durable checkpoints."""
import argparse
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from advisor_eval import forecast_metrics, safe_npz
from advisor_train import core, inputs, load_data
from order_numpy import risk_order, softmax
from prefix_eval import prefix_metrics
from prefix_loss import ARMS, KS, make_generator, prefix_loss
from prepare_inputs import digest
from run_b0 import checkpoint, latest, publish, write_json

ROOT = Path(__file__).resolve().parent
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
SEEDS = (42, 43, 44)


class PrefixTrainer(core.Trainer):
    def __init__(self, data, arm, seed, device, epochs):
        super().__init__(data, np.arange(len(data["ids"])), batch=64, epochs=epochs,
                         device=device, seed=seed, order_seed=seed, arm="B0")
        self.prefix_arm = arm
        self.prefix_rng = make_generator(device, seed) if arm == "N-rand" else None

    def objective(self, paths, target, logits):
        base, detail = core.loss_value(paths, target, logits, self.loss_fn)
        if self.prefix_arm == "A":
            return base, detail
        extra, distinct = prefix_loss(self.prefix_arm, paths, target, logits,
                                      self.loss_fn, self.prefix_rng)
        return base + extra, {**detail, **distinct}

    def step(self):
        if self.order is None or self.cursor == len(self.order):
            if self.epoch >= self.epochs:
                raise StopIteration
            self.order = self.indices[torch.randperm(len(self.indices), generator=self.generator).numpy()]
            self.cursor, self.epoch = 0, self.epoch + 1
        index = self.order[self.cursor:self.cursor + self.batch]
        self.model.train()
        past, future, tokens = (torch.as_tensor(self.data[k][index], device=self.device)
                                for k in ("past", "future", "num_valid"))
        paths, logits, target, _ = core.prediction(self.model, past, tokens, future, augment=True)
        loss, detail = self.objective(paths, target, logits)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite prefix training loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5., error_if_nonfinite=True)
        self.optimizer.step()
        self.ema.update(self.model)
        self.scheduler.step()
        self.updates += 1
        self.cursor += len(index)
        self.batch_orders.append(self.data["ids"][index].tolist())
        row = dict(loss=float(loss.detach()), grad_norm=float(grad),
                   regression=float(detail["regression"]), classification=float(detail["classification"]),
                   windows=len(index))
        row.update({key: value for key, value in detail.items() if key.startswith("distinct")})
        self.history.append(row)
        return row

    def snapshot(self, contract):
        state = super().snapshot(contract)
        if self.prefix_rng is not None:
            state["prefix_rng"] = self.prefix_rng.get_state().cpu()
        return state

    def load(self, state, contract):
        super().load(state, contract)
        if self.prefix_rng is not None:
            self.prefix_rng.set_state(state["prefix_rng"].cpu())


def subset(data, n):
    return {k: v[:n] for k, v in data.items()}


def evaluate(trainer, dev, out, fold):
    paths, logits = core.infer(trainer.ema.module, dev)
    metrics = forecast_metrics(paths, logits, dev["future"])
    pairs = []
    for begin in range(0, len(paths), 512):
        s = slice(begin, begin + 512)
        pairs.append(prefix_metrics(paths[s], logits[s], dev["future"][s]))
    pref = {k: np.concatenate([p[k] for p in pairs]) for k in pairs[0]}
    assert np.array_equal(pref["conf_minFDE20"], pref["riskp_minFDE20"])
    past = dev["past"][:, 0].astype("float64")
    fast = np.linalg.norm(np.diff(past, axis=1), axis=-1).mean(1) / .4 > 1.3
    files = {"dev_metrics.npz": safe_npz(out/"dev_metrics.npz", ids=dev["ids"], fast=fast,
                                       **{**metrics, **pref})}
    if fold == "C0":
        path = out/"prefix56.npz"
        s = subset(dev, 512)
        top3 = np.sort(risk_order(paths[:len(s["ids"]), :, -1],
                                  softmax(logits[:len(s["ids"])]))[:, :3], axis=1)
        if path.is_file():
            with np.load(path, allow_pickle=False) as saved:
                assert np.array_equal(saved["ids"], s["ids"])
                mc4 = float((saved["top3"] == top3).all(1).mean())
            files[path.name] = digest(path)
        else:
            mc4 = None
    else:
        mc4 = None
    with tempfile.TemporaryDirectory(prefix="prefix_weights_") as temp:
        path = Path(temp)/"final_ema.pt"
        torch.save(core.cpu_tree(trainer.ema.module.state_dict()), path)
        files[path.name] = publish(path, out/path.name)
    return {"mean": {k: float(v.mean()) for k, v in {**metrics, **pref}.items()},
            "mc4_top3_set_stable": mc4, "artifacts": files}


def run(args):
    torch.set_num_threads(1)
    device = "cpu" if args.smoke else "cuda:0"
    if not args.smoke and (not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name(0).split()):
        raise RuntimeError("NVIDIA L4 is required")
    sources = inputs(args.fold)
    fit, dev = (load_data(p, n) for p, n in sources)
    if args.smoke:
        fit, dev = subset(fit, 8), subset(dev, 8)
    epochs = 2 if args.smoke else 1 if args.bench else 64
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    contract = dict(arm=args.arm, fold=args.fold, seed=args.seed, epochs=epochs, batch=64,
                    torch=str(torch.__version__), cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if not args.smoke else None,
                    design=digest(ROOT/"DESIGN_LOCK.json"), inputs={p.name: digest(p) for p, _ in sources})
    spec = out/"CONTRACT.json"
    if spec.exists():
        from advisor_study import read
        if read(spec) != contract:
            raise ValueError(f"Resume contract differs: {spec}")
    else:
        write_json(spec, contract)
    t = PrefixTrainer(fit, args.arm, args.seed, device, epochs)
    assert sum(p.numel() for p in t.model.parameters()) == 295474
    start = 0 if args.bench else latest(out, core, t, contract)
    if not args.smoke:
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for epoch in range(start + 1, epochs + 1):
        epoch_start = time.monotonic()
        t.step()
        if epoch == 1:
            write_json(out/"CURRENT.json", dict(status="TRAINING", pid=os.getpid(),
                                                epoch=epoch, updates=t.updates, first_batch=True))
        for _ in range(t.steps_per_epoch - 1):
            t.step()
        elapsed = time.monotonic() - epoch_start
        if not args.bench and args.fold == "C0" and epoch == 56:
            data = subset(dev, 512)
            paths, logits = core.infer(t.ema.module, data)
            top3 = np.sort(risk_order(paths[:, :, -1], softmax(logits))[:, :3], axis=1)
            safe_npz(out/"prefix56.npz", ids=data["ids"], top3=top3)
        if not args.bench and (args.smoke or epoch % 8 == 0 or epoch == epochs):
            checkpoint(out, core, t, contract)
        current = dict(status="TRAINING", pid=os.getpid(), epoch=epoch, updates=t.updates,
                       seconds=elapsed, mean_loss=float(np.mean([r["loss"] for r in t.history[-t.steps_per_epoch:]])))
        write_json(out/"CURRENT.json", current)
        print(args.fold, args.arm, args.seed, current, flush=True)
    result = dict(status="PASS", contract=contract, parameters=295474, updates=t.updates,
                  seconds=time.monotonic() - started,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved() if not args.smoke else 0)
    if args.arm != "A":
        weights = np.array([row["windows"] for row in t.history])
        for k in KS:
            key = f"distinct{k}"
            result[key] = float(np.average([row[key] for row in t.history], weights=weights))
    if not args.bench:
        result["classification_loss_by_epoch"] = [
            float(np.mean([r["classification"] for r in t.history[i:i+t.steps_per_epoch]]))
            for i in range(0, len(t.history), t.steps_per_epoch)
        ]
        result.update(evaluate(t, dev, out, args.fold))
    write_json(out/"RESULT.json", result)
    write_json(out/"CURRENT.json", dict(status="COMPLETE", epoch=epochs, updates=t.updates))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--fold", choices=("C0", *FOLDS), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--bench", action="store_true")
    run(parser.parse_args())
