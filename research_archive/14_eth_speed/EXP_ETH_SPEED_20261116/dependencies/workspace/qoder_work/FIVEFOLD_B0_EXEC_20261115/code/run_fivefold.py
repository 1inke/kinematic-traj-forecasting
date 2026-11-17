"""Five-fold ETH/UCY run orchestration for the current E2E B0 (torch; NOT executed in authoring env).

Re-uses the NEXT_PHASE execution code unchanged (train_core.Trainer, train_core.infer,
baseline_factory.make_model / prediction) by putting --exec-dir on sys.path.
Model, loss (ModeQueryLoss + 0.20 soft coverage), AdamW 2.5e-4 / wd 5e-4, 1-epoch warmup +
cosine to the budget, EMA 0.995 fixed final, batch 64, reflection + 0.9-1.1 speed augmentation:
all exactly as NEXT_PHASE B0.  No checkpoint selection on any split.

Phases (each refuses to run if its prerequisites are missing):
  budget  fold x E in {64,128}, seed 42, train on scene_tail-train (95%), evaluate scene_tail-val
  decide  E* from the 10 budget runs (rule in fivefold_common / fivefold_metrics.decide_budget)
  final   fold x seed in {42,43,44}, E*, train on 100% fold train, evaluate the held-out test ONCE
  all     budget -> decide -> final, skipping runs that already have DONE.json

The test arrays are opened only inside `final`, after BUDGET_DECISION.json exists.

usage:
  python run_fivefold.py budget --smoke --folds hotel --out FIVEFOLD_SMOKE ...   (first, minutes)
  python run_fivefold.py all --exec-dir <qoder_work/NEXT_PHASE_EXEC_20261114> \
      --data-dir <ASTRA_ACCURACY_Q20/datasets/eth_ucy/processed_data_diverse> \
      --audit F0/AUDIT.json --out FIVEFOLD_RUNS --device cuda:0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fivefold_common import (BUDGETS, BUDGET_SEED, FINAL_SEEDS, FOLDS, now, read_json,  # noqa: E402
                             sha256, write_json_once)
from fivefold_data import load_split, scene_tail_split, subset  # noqa: E402
from fivefold_metrics import aggregate, decide_budget, per_window  # noqa: E402

STUDY = "FIVEFOLD_B0_20261115"
KEEP_CHECKPOINTS = 2
B0_PARAMETERS = 295474      # NEXT_PHASE E1C B0 total_parameters; any other count = wrong model


def import_exec(exec_dir):
    exec_dir = Path(exec_dir).resolve()
    for name in ("train_core.py", "baseline_factory.py", "common.py"):
        if not (exec_dir / name).is_file():
            raise FileNotFoundError(exec_dir / name)
    sys.path.insert(0, str(exec_dir))
    import torch  # noqa: F401
    import train_core
    sources = {n: sha256(exec_dir / n) for n in ("train_core.py", "baseline_factory.py", "common.py")}
    sources.update({p.name: sha256(p) for p in sorted(HERE.glob("*.py"))})
    return train_core, sources


def run_name(phase, fold, epochs, seed):
    return f"{phase}_{fold}_E{epochs}_s{seed}"


def index_sha(index):
    return hashlib.sha256(np.asarray(index, np.int64).tobytes()).hexdigest()


def train_one(tc, data, index, epochs, seed, device, run_dir, contract):
    """Train with per-epoch checkpoints and exact resume. Returns (trainer, info)."""
    import torch
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer = tc.Trainer(data, index, batch=64, epochs=epochs, device=device, seed=seed, arm="B0")
    order_hash = hashlib.sha256(b"FIVEFOLD_ORDER").hexdigest()
    ckpts = sorted(run_dir.glob("ckpt_epoch*.pt"))
    if ckpts:
        state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        trainer.load(state, contract)
        order_hash = state["fivefold_order_hash"]
        print(f"[resume] {run_dir.name} from {ckpts[-1].name}", flush=True)
    t0 = time.time()
    while True:
        try:
            trainer.step()
        except StopIteration:
            break
        # Keep a hash chain of batch order instead of the full id list (memory/disk).
        idx = np.asarray(trainer.batch_orders.pop(), np.int64)
        order_hash = hashlib.sha256(bytes.fromhex(order_hash) + idx.tobytes()).hexdigest()
        if trainer.cursor == len(trainer.order):
            state = trainer.snapshot(contract); state["fivefold_order_hash"] = order_hash
            tc.save_checkpoint(run_dir / f"ckpt_epoch{trainer.epoch:03d}.pt", state)
            for old in sorted(run_dir.glob("ckpt_epoch*.pt"))[:-KEEP_CHECKPOINTS]:
                old.unlink()
            last = trainer.history[-1]
            print(f"[{run_dir.name}] epoch {trainer.epoch}/{epochs} loss {last['loss']:.4f} "
                  f"lr {last['lr']:.2e} {time.time() - t0:.0f}s", flush=True)
    if trainer.updates != epochs * trainer.steps_per_epoch:
        raise ValueError("UPDATE_COUNT_MISMATCH")
    n_params = int(sum(p.numel() for p in trainer.model.parameters()))
    if n_params != B0_PARAMETERS:
        raise ValueError(f"B0_PARAMETER_COUNT {n_params} != {B0_PARAMETERS}")
    info = dict(updates=trainer.updates, steps_per_epoch=trainer.steps_per_epoch,
                order_hash=order_hash, train_seconds_this_session=time.time() - t0,
                connected_parameters=len(trainer.connected),
                parameters=int(sum(p.numel() for p in trainer.model.parameters())),
                final_train_loss=trainer.history[-1]["loss"])
    return trainer, info


def evaluate(tc, model, data, run_dir, split_name):
    paths, logits = tc.infer(model, data)
    w = per_window(paths, logits, data["future"], data["past"][:, 0])
    np.savez_compressed(run_dir / f"{split_name}_predictions.npz", paths=paths.astype(np.float32),
                        logits=logits.astype(np.float32), ids=data["ids"], scene=data["scene"], group=data["group"])
    np.savez_compressed(run_dir / f"{split_name}_window_metrics.npz", ids=data["ids"], group=data["group"],
                        **{k: v for k, v in w.items()})
    return aggregate(w, data["group"])


def base_contract(args, sources, audit, phase, fold, epochs, seed, index, extra=None):
    c = dict(study=STUDY, phase=phase, fold=fold, epochs=int(epochs), seed=int(seed), arm="B0",
             n_train=int(len(index)), train_index_sha=index_sha(index),
             data_sha=audit["by_fold"][fold]["sha256"], sources=sources, device=str(args.device))
    if extra:
        c.update(extra)
    return c


def load_audit(path):
    a = read_json(path)
    if not a.get("passed"):
        raise RuntimeError("F0 audit did not pass")
    a["by_fold"] = {r["fold"]: r for r in a["folds"]}
    return a


def phase_budget(args, tc, sources, audit, folds):
    out = Path(args.out)
    for fold in folds:
        data = load_split(args.data_dir, fold, "train")
        tr, va = scene_tail_split(data["scene"])
        budgets, tag = BUDGETS, "budget"
        if args.smoke:     # wiring check only: 1 epoch on 512 train / 256 val windows
            tr, va, budgets, tag = tr[:512], va[:256], (1,), "smoke"
        val = subset(data, va)
        for E in budgets:
            rd = out / "runs" / run_name(tag, fold, E, BUDGET_SEED)
            if (rd / "DONE.json").exists():
                continue
            contract = base_contract(args, sources, audit, tag, fold, E, BUDGET_SEED, tr,
                                     dict(val_index_sha=index_sha(va), n_val=int(len(va))))
            trainer, info = train_one(tc, data, tr, E, BUDGET_SEED, args.device, rd, contract)
            metrics = evaluate(tc, trainer.ema.module, val, rd, "val")
            write_json_once(rd / "metrics.json", dict(contract=contract, train=info, val=metrics, at=now()))
            write_json_once(rd / "DONE.json", dict(at=now()))


def phase_decide(args):
    out = Path(args.out)
    target = out / "BUDGET_DECISION.json"
    if target.exists():
        return read_json(target)
    rows = {}
    for fold in FOLDS:
        rows[fold] = {}
        for E in BUDGETS:
            m = read_json(out / "runs" / run_name("budget", fold, E, BUDGET_SEED) / "metrics.json")
            rows[fold][E] = dict(minADE=m["val"]["overall"]["minADE"], minFDE=m["val"]["overall"]["minFDE"])
    decision = decide_budget(rows)
    decision.update(at=now(), budget_rows={f: {str(E): v for E, v in r.items()} for f, r in rows.items()},
                    test_accessed=False)
    write_json_once(target, decision)
    print("[decide] E* =", decision["E_star"], json.dumps({k: decision[k] for k in
          ("mean_relative_minADE_gain_64_to_128", "mean_minFDE_delta_128_minus_64",
           "possibly_not_saturated_at_128")}))
    return decision


def phase_final(args, tc, sources, audit, folds, seeds):
    out = Path(args.out)
    decision_path = out / "BUDGET_DECISION.json"
    if not decision_path.exists():
        raise RuntimeError("BUDGET_DECISION_MISSING: run `decide` first; test stays closed")
    E = int(read_json(decision_path)["E_star"])
    for fold in folds:
        train = load_split(args.data_dir, fold, "train")
        index = np.arange(len(train["ids"]))
        for seed in seeds:
            rd = out / "runs" / run_name("final", fold, E, seed)
            if (rd / "DONE.json").exists():
                continue
            contract = base_contract(args, sources, audit, "final", fold, E, seed, index,
                                     dict(budget_decision_sha=sha256(decision_path)))
            trainer, info = train_one(tc, train, index, E, seed, args.device, rd, contract)
            test = load_split(args.data_dir, fold, "test")      # opened here, once per run
            metrics = evaluate(tc, trainer.ema.module, test, rd, "test")
            write_json_once(rd / "metrics.json", dict(contract=contract, train=info, test=metrics, at=now()))
            write_json_once(rd / "DONE.json", dict(at=now()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("budget", "decide", "final", "all"))
    ap.add_argument("--exec-dir", required=False)
    ap.add_argument("--data-dir", required=False)
    ap.add_argument("--audit", required=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--folds", nargs="+", choices=FOLDS, default=list(FOLDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(FINAL_SEEDS))
    ap.add_argument("--smoke", action="store_true", help="budget phase only, 1 epoch, tiny subset, --out must contain SMOKE")
    args = ap.parse_args()
    if args.smoke and (args.phase != "budget" or "SMOKE" not in str(args.out)):
        ap.error("--smoke needs phase=budget and an --out path containing SMOKE (kept apart from real runs)")
    if args.phase == "decide":
        phase_decide(args); return
    for name in ("exec_dir", "data_dir", "audit"):
        if getattr(args, name) is None:
            ap.error(f"--{name.replace('_', '-')} is required for {args.phase}")
    if set(args.seeds) - set(FINAL_SEEDS):
        ap.error("seeds are fixed to 42 43 44 by the design")
    audit = load_audit(args.audit)
    tc, sources = import_exec(args.exec_dir)
    env = Path(args.out) / "ENVIRONMENT.json"
    if not env.exists():
        import torch
        write_json_once(env, dict(at=now(), python=sys.version, platform=platform.platform(),
                                  torch=torch.__version__, cuda=torch.version.cuda,
                                  device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                  numpy=np.__version__, sources=sources))
    if args.phase in ("budget", "all"):
        phase_budget(args, tc, sources, audit, args.folds)
    if args.phase == "all":
        phase_decide(args)
    if args.phase in ("final", "all"):
        phase_final(args, tc, sources, audit, args.folds, args.seeds)
    print("[done]", args.phase)


if __name__ == "__main__":
    main()
