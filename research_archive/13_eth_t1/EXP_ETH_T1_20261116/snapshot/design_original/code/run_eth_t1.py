"""【待执行】实验一 EXP-ETH-T1: does T1 improve the ETH fold?  (torch; NOT executed in authoring env)

One new training run: T1, ETH fold, 95% scene_tail-train (identical index to the existing budget
run), 64 epochs, seed 42, fixed final EMA.  The paired B0 is the EXISTING five-fold budget model
budget_eth_E64_s42 (not retrained).  Both are evaluated once on the ETH test set (364 windows).

Reuses, unchanged:
  explore_common.load_training()  -> real train_core / run_fivefold / resource guard
  run_explore.training_loop()     -> same Trainer loop, per-epoch checkpoints, exact resume
  run_fivefold.load_split / scene_tail_split / index_sha, fivefold_metrics.per_window / aggregate
Phases:
  preflight : data, B0 checkpoint identity (epoch 64, fold, train-index SHA), T1 builds with 295,474 params
  train     : T1 (~30 min alone); rerun the same command to resume
  evaluate  : B0-E64 checkpoint and T1 on ETH test -> test_predictions / test_window_metrics / result.json
usage:
  python run_eth_t1.py preflight --explore-dir <qoder_work/EXPLORE_FAST_20261116> --out <qoder_work/EXP_ETH_T1_20261116>
  python run_eth_t1.py train     --explore-dir ... --out ...
  python run_eth_t1.py evaluate  --explore-dir ... --out ...
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

FOLD, EPOCHS, SEED, ARM, PARAMETERS = "eth", 64, 42, "T1", 295474
B0_RUN = f"budget_{FOLD}_E{EPOCHS}_s{SEED}"


def setup(explore_dir):
    code = Path(explore_dir).resolve() / "code"
    sys.path.insert(0, str(code))
    import explore_common as ec
    import run_explore as rx
    cfg = ec.read(ec.PARALLEL / "PARALLEL_CONFIG.json")       # data_dir, output (five-fold runs), exp156
    return ec, rx, cfg


def b0_checkpoint(ec, cfg, rf, train_index):
    run = Path(cfg["output"]) / "runs" / B0_RUN
    if not (run / "DONE.json").exists():
        raise FileNotFoundError(f"{run} not finished")
    ckpts = sorted(run.glob("ckpt_epoch*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint left in {run} (the final-epoch weights are required)")
    side = ckpts[-1].with_suffix(".json")
    if side.exists() and ec.read(side)["sha256"] != ec.sha(ckpts[-1]):
        raise ValueError("B0 checkpoint SHA does not match its receipt")
    import torch
    state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    c = state["contract"]
    if state["epoch"] != EPOCHS or c["fold"] != FOLD or c["seed"] != SEED or c["epochs"] != EPOCHS:
        raise ValueError("B0 checkpoint is not ETH / 64 epochs / seed 42")
    if c["train_index_sha"] != rf.index_sha(train_index):
        raise ValueError("B0 was trained on a different index than the T1 run would use")
    return run, ckpts[-1], state


def contract(ec, rf, cfg, train_index):
    files = {n: ec.sha(Path(ec.EXEC) / n) for n in ("train_core.py", "baseline_factory.py", "training_ext.py")}
    files[Path(__file__).name] = ec.sha(Path(__file__))
    return dict(study="EXP_ETH_T1_20261116", phase="explore", fold=FOLD, arm=ARM, epochs=EPOCHS, seed=SEED,
                train_set="eth_scene_tail_train_95pct", n_train=int(len(train_index)),
                train_index_sha=rf.index_sha(train_index), final_weights="EMA_0.995", sources=files,
                paired_b0=B0_RUN)


def preflight(a):
    ec, rx, cfg = setup(a.explore_dir)
    tc, rf, guard = ec.load_training()
    train = rf.load_split(cfg["data_dir"], FOLD, "train")
    idx, _ = rf.scene_tail_split(train["scene"])
    run, ck, state = b0_checkpoint(ec, cfg, rf, idx)
    t = tc.Trainer(train, idx[:256], batch=64, epochs=1, device="cuda:0", seed=SEED, arm=ARM)
    n = sum(p.numel() for p in t.model.parameters())
    row = t.step()
    ok = n == PARAMETERS and np.isfinite(row["loss"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = dict(at=ec.now(), passed=bool(ok), b0_checkpoint=str(ck), b0_epoch=state["epoch"], t1_parameters=n,
               t1_first_loss=row["loss"], n_train=int(len(idx)), steps_per_epoch=int(np.ceil(len(idx) / 64)))
    ec.replace_json(out / "PREFLIGHT.json", rep)
    print(json.dumps(rep, ensure_ascii=False, indent=1)); print("PREFLIGHT", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


def train(a):
    out = Path(a.out)
    if not (out / "PREFLIGHT.json").exists() or not json.loads((out / "PREFLIGHT.json").read_text(encoding="utf-8"))["passed"]:
        raise RuntimeError("run `preflight` first")
    ec, rx, cfg = setup(a.explore_dir)
    d = out / "runs" / f"T1_{FOLD}_E{EPOCHS}_s{SEED}"
    if (d / "TRAIN_DONE.json").exists():
        print("already trained:", d); return
    tc, rf, guard = ec.load_training()
    train_data = rf.load_split(cfg["data_dir"], FOLD, "train")
    idx, _ = rf.scene_tail_split(train_data["scene"])
    c = contract(ec, rf, cfg, idx)
    guard.check(force=True)
    t0 = time.perf_counter()
    trainer, info = rx.training_loop(tc, rf, guard, train_data, idx, ARM, EPOCHS, SEED, d, c, progress=out / "PROGRESS.json")
    if info["parameters"] != PARAMETERS:
        raise ValueError("T1 parameter count changed")
    ec.replace_json(d / "TRAIN_DONE.json", dict(at=ec.now(), contract=c, train=info, wall_seconds=time.perf_counter() - t0))
    print("trained", d, f"{(time.perf_counter() - t0) / 60:.1f} min")


def evaluate_model(ec, tc, guard, model, test, directory):
    from fivefold_metrics import per_window, aggregate
    paths, logits = tc.infer(model, test, guard=guard)
    paths, logits = paths.astype(np.float32), logits.astype(np.float32)
    w = per_window(paths, logits, test["future"], test["past"][:, 0])
    m = aggregate(w, test["group"])
    past = test["past"][:, 0].astype(np.float64)
    v1 = (paths[:, :, 0].astype(np.float64) - past[:, None, -1]) / .4
    m["overall"]["junction_accel_allK20"] = float((np.linalg.norm(v1 - ((past[:, -1] - past[:, -2]) / .4)[:, None], axis=-1) / .4).mean())
    ec.save_npz(directory / "test_predictions.npz", paths=paths, logits=logits, ids=test["ids"], group=test["group"])
    ec.save_npz(directory / "test_window_metrics.npz", ids=test["ids"], group=test["group"], **w)
    return m


def evaluate(a):
    out = Path(a.out)
    ec, rx, cfg = setup(a.explore_dir)
    tc, rf, guard = ec.load_training()
    import torch
    train = rf.load_split(cfg["data_dir"], FOLD, "train")
    idx, _ = rf.scene_tail_split(train["scene"])
    test = rf.load_split(cfg["data_dir"], FOLD, "test")
    # B0: existing budget checkpoint (read-only)
    bdir = out / "runs" / f"B0_{FOLD}_E{EPOCHS}_s{SEED}"
    if not (bdir / "result.json").exists():
        run, ck, state = b0_checkpoint(ec, cfg, rf, idx)
        t = tc.Trainer(train, idx, batch=64, epochs=EPOCHS, device="cuda:0", seed=SEED, arm="B0")
        t.load(state, state["contract"])
        bdir.mkdir(parents=True, exist_ok=True)
        m = evaluate_model(ec, tc, guard, t.ema.module, test, bdir)
        ec.replace_json(bdir / "result.json", dict(at=ec.now(), arm="B0", source=str(ck), checkpoint_sha256=ec.sha(ck),
                                                   fold=FOLD, epochs=EPOCHS, seed=SEED, test=m))
        del t, state; gc.collect(); torch.cuda.empty_cache()
    # T1: last checkpoint of this study's run
    tdir = out / "runs" / f"T1_{FOLD}_E{EPOCHS}_s{SEED}"
    if not (tdir / "TRAIN_DONE.json").exists():
        raise RuntimeError("T1 not trained yet")
    if not (tdir / "result.json").exists():
        ck = sorted(tdir.glob("ckpt_epoch*.pt"))[-1]
        state = torch.load(ck, map_location="cpu", weights_only=False)
        if state["epoch"] != EPOCHS:
            raise ValueError("T1 checkpoint is not the final epoch")
        t = tc.Trainer(train, idx, batch=64, epochs=EPOCHS, device="cuda:0", seed=SEED, arm=ARM)
        t.load(state, state["contract"])
        m = evaluate_model(ec, tc, guard, t.ema.module, test, tdir)
        ec.replace_json(tdir / "result.json", dict(at=ec.now(), arm=ARM, source=str(ck), checkpoint_sha256=ec.sha(ck),
                                                   fold=FOLD, epochs=EPOCHS, seed=SEED, test=m))
    print("evaluated -> run analyze_eth_t1.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("preflight", "train", "evaluate"))
    ap.add_argument("--explore-dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"preflight": preflight, "train": train, "evaluate": evaluate}[a.phase](a)


if __name__ == "__main__":
    main()
