"""Exploration runner: short, single-seed-first runs on C0 fit -> dev, light metrics, one ledger.
(torch; uses the NEXT_PHASE execution code unchanged; NOT executed in authoring env)

What it deliberately does NOT do (exploration mode): per-epoch checkpoints/resume, clustering,
bootstrap, motion panels, audits, locks, multi-page reports.  A crashed run is simply rerun.

What it keeps (what makes a method-selection conclusion trustworthy):
  same budget and seed pairing vs B0; fixed final EMA; primary metric chosen before running;
  an optional interim futility stop on CALIB (never on dev), compared with the B0 curve of the
  same seed and budget; one CSV ledger row per run.

Arms are whatever train_core.Trainer accepts: B0, T1, T2, T3, E2_<arm>, E3_<arm>.
A new method only needs a new arm name in train_core.Trainer.

usage:
  python fast_run.py --exec-dir <NEXT_PHASE_EXEC_20261114> --out EXPLORE \
      --arms B0 T1 E3_self --epochs 16 32 --seeds 42 [--calib-every 4 --stop-if-behind coverage]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fivefold_metrics import aggregate, per_window  # noqa: E402

LEDGER = ["at", "arm", "epochs", "seed", "status", "stopped_at_epoch", "minADE", "minFDE", "coverage",
          "change_coverage", "top1_coverage", "junction_accel_allK20", "wall_seconds", "note"]
DT = 0.4


def light_metrics(paths, logits, data):
    w = per_window(paths, logits, data["future"], data["past"][:, 0])
    agg = aggregate(w, data["group"])
    past = data["past"][:, 0].astype(np.float64)
    v1 = (np.asarray(paths, np.float64)[:, :, 0] - past[:, None, -1]) / DT
    vo = (past[:, -1] - past[:, -2]) / DT
    agg["overall"]["junction_accel_allK20"] = float((np.linalg.norm(v1 - vo[:, None], axis=-1) / DT).mean())
    return agg


def brief(agg):
    o, c = agg["overall"], agg.get("change", {})
    return dict(minADE=o["minADE"], minFDE=o["minFDE"], coverage=100 * o["coverage"],
                change_coverage=100 * c["coverage"] if c else None, top1_coverage=100 * o["top1_good"],
                junction_accel_allK20=o.get("junction_accel_allK20"))


def run_dir(out, arm, E, seed, tag=""):
    return Path(out) / "runs" / f"{arm}_E{E}_s{seed}{tag}"


def behind(curve, ref_curve, key, E):
    """Futility: from 50% of the budget on, mean of the last 3 paired CALIB differences <= 0."""
    ref = {r["epoch"]: r for r in ref_curve}
    pairs = [(r[key] - ref[r["epoch"]][key]) for r in curve if r["epoch"] in ref and r[key] is not None]
    if not curve or curve[-1]["epoch"] < E / 2 or len(pairs) < 3:
        return False, None
    m = float(np.mean(pairs[-3:]))
    return m <= 0, m


def run_one(tc, fit, train_index, dev, calib, arm, E, seed, args):
    tag = "_fit85" if calib is not None else ""
    rd = run_dir(args.out, arm, E, seed, tag)
    if (rd / "result.json").exists():
        return json.loads((rd / "result.json").read_text(encoding="utf-8"))
    rd.mkdir(parents=True, exist_ok=True)
    ref_curve = None
    if args.stop_if_behind and arm != "B0":
        ref_path = run_dir(args.out, "B0", E, seed, tag) / "result.json"
        if ref_path.exists():
            ref_curve = json.loads(ref_path.read_text(encoding="utf-8")).get("calib_curve") or None
    t0 = time.time()
    trainer = tc.Trainer(fit, train_index, batch=64, epochs=E, device=args.device, seed=seed, arm=arm)
    curve, status, stopped, futility = [], "DONE", None, None
    while True:
        try:
            trainer.step()
        except StopIteration:
            break
        trainer.batch_orders.clear()
        if trainer.cursor == len(trainer.order) and calib is not None and trainer.epoch % args.calib_every == 0 \
                and trainer.epoch < E:
            p, l = tc.infer(trainer.ema.module, calib)
            curve.append(dict(epoch=trainer.epoch, **brief(light_metrics(p, l, calib))))
            print(f"  [{arm} E{E} s{seed}] calib ep{trainer.epoch} cov {curve[-1]['coverage']:.1f} "
                  f"chg {curve[-1]['change_coverage']:.1f} minADE {curve[-1]['minADE']:.4f}", flush=True)
            if ref_curve:
                stop, futility = behind(curve, ref_curve, args.stop_if_behind, E)
                if stop:
                    status, stopped = "STOPPED_FUTILE", trainer.epoch
                    break
    result = dict(arm=arm, epochs=E, seed=seed, train_set="fit85" if tag else "fit", status=status, stopped_at_epoch=stopped,
                  futility_mean_diff=futility, calib_curve=curve, wall_seconds=time.time() - t0,
                  parameters=int(sum(p.numel() for p in trainer.model.parameters())))
    if status == "DONE":
        p, l = tc.infer(trainer.ema.module, dev)
        result["dev"] = light_metrics(p, l, dev)
        if args.save_predictions:
            np.savez_compressed(rd / "dev_predictions.npz", paths=np.asarray(p, np.float32),
                                logits=np.asarray(l, np.float32), ids=dev["ids"])
    (rd / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    row = dict(at=time.strftime("%Y-%m-%d %H:%M"), arm=arm, epochs=E, seed=seed, status=status,
               stopped_at_epoch=stopped, wall_seconds=round(result["wall_seconds"]),
               note=(args.note + " fit85" if tag else args.note).strip(),
               **(brief(result["dev"]) if status == "DONE" else {}))
    ledger = Path(args.out) / "LEDGER.csv"
    new = not ledger.exists()
    with ledger.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER)
        if new:
            w.writeheader()
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"[{status}] {arm} E{E} s{seed} {round(result['wall_seconds'])}s", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--epochs", nargs="+", type=int, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--calib-every", type=int, default=0, help="0 = no interim CALIB evaluation")
    ap.add_argument("--stop-if-behind", choices=("coverage", "change_coverage", None), default=None)
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    if args.stop_if_behind and not args.calib_every:
        ap.error("--stop-if-behind needs --calib-every")
    ex = Path(args.exec_dir).resolve()
    sys.path.insert(0, str(ex))
    import train_core as tc
    fit = dict(np.load(ex / "C0" / "fit.npz", allow_pickle=False))
    dev = dict(np.load(ex / "C0" / "dev.npz", allow_pickle=False))
    calib, train_index = None, np.arange(len(fit["ids"]))
    if args.calib_every:
        # CALIB is held out from training: such runs train on FIT85 (as E1B did) and are kept
        # apart (run dir suffix _fit85); never mix them with full-fit runs in one comparison.
        split = np.load(ex / "E1" / "calib_split.npz")
        train_index = split["fit85_index"]
        calib = {k: v[split["calib_index"]] for k, v in fit.items()}
    # B0 first so its CALIB curve exists for the futility comparison of the other arms
    arms = sorted(args.arms, key=lambda a: a != "B0")
    for E in args.epochs:
        for seed in args.seeds:
            for arm in arms:
                run_one(tc, fit, train_index, dev, calib, arm, E, seed, args)


if __name__ == "__main__":
    main()
