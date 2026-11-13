from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from kdec.data import load_bundle
from kdec.train import train_arm
from kdec.metrics import motion_stats

KIN_CEILING = 0.5646


def decide(outer):
    cov = outer["overall"]["coverage"]
    native = outer["overall"]["native_coverage"]
    gate = outer["_motion"]["gate"]["status"]
    return dict(
        coverage=cov, native_coverage=native, gap_pp=(cov-native)*100,
        minADE=outer["overall"]["minADE"], motion=gate,
        beat_native=bool(gate=="PASS" and cov>=native),
        beat_kin_ceiling=bool(gate=="PASS" and cov>=KIN_CEILING),
        weak_positive=bool(gate=="PASS" and cov>=KIN_CEILING),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["smoke","short"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    data = load_bundle()
    fit_m = motion_stats(data["native"][data["split"]["FIT"]])
    v_max = fit_m["speed_p95"]*0.995; a_max = fit_m["accel_p95"]*0.90
    motion_refs = dict(speed_p95=fit_m["speed_p95"], accel_mean=fit_m["accel_mean"])
    out = HERE / "results" / args.phase; out.mkdir(parents=True, exist_ok=True)
    if args.phase == "smoke":
        cfg = dict(epochs=args.epochs or 2, batch=128, subset=2048, seed=args.seed)
    else:
        cfg = dict(epochs=args.epochs or 8, batch=256, subset=None, seed=args.seed)
    print(f"==== MDK {args.phase} seed={args.seed} ====", flush=True)
    res, _ = train_arm(data, v_max=v_max, a_max=a_max, motion_refs=motion_refs, out_dir=out, **cfg)
    decision = decide(res["metrics"]["outer"])
    compare = dict(phase=args.phase, seed=args.seed, cfg=cfg, decision=decision,
                   outer=res["metrics"]["outer"], FIT_overall=res["metrics"]["FIT"]["overall"],
                   params=res["params"])
    name = f"COMPARE_r1_seed{args.seed}.json"
    (out / name).write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
