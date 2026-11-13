"""smoke / short 驱动。"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from kdec.data import load_bundle
from kdec.train import train_arm
from kdec import models as M
from kdec.metrics import motion_stats


def decide(short_results: dict) -> dict:
    base = short_results["BASE"]["metrics"]["outer"]
    kin = short_results["KIN"]["metrics"]["outer"]
    b_gate = base["_motion"]["gate"]["status"]
    k_gate = kin["_motion"]["gate"]["status"]
    b_cov = base["overall"]["coverage"]
    k_cov = kin["overall"]["coverage"]
    b_ade = base["overall"]["minADE"]
    k_ade = kin["overall"]["minADE"]
    b_ratio = base["_motion"]["gate"]["ratios"]["accel_mean"]
    k_ratio = kin["_motion"]["gate"]["ratios"]["accel_mean"]
    motion_better = (k_gate == "PASS" and b_gate == "FAIL") or (
        k_gate == "PASS" and b_gate == "PASS" and k_ratio <= b_ratio * 0.95
    )
    cov_ok = k_cov >= b_cov - 0.01
    ade_ok = k_ade <= b_ade * 1.10
    native_cov = base["overall"]["native_coverage"]
    return dict(
        weak_positive=bool(motion_better and cov_ok and ade_ok),
        stronger=bool(k_gate == "PASS" and k_cov >= native_cov and k_cov >= b_cov and ade_ok),
        motion_better=motion_better,
        cov_ok=cov_ok,
        ade_ok=ade_ok,
        BASE_motion=b_gate,
        KIN_motion=k_gate,
        BASE_coverage=b_cov,
        KIN_coverage=k_cov,
        BASE_minADE=b_ade,
        KIN_minADE=k_ade,
        BASE_accel_ratio=b_ratio,
        KIN_accel_ratio=k_ratio,
        KIN_speed_ratio=kin["_motion"]["gate"]["ratios"]["speed_p95"],
        native_coverage=native_cov,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["smoke", "short"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--v-max-scale", type=float, default=0.995)
    ap.add_argument("--a-max-scale", type=float, default=0.90)
    ap.add_argument("--motion-w", type=float, default=0.05)
    ap.add_argument("--round-tag", type=str, default="r1")
    args = ap.parse_args()

    data = load_bundle()
    fit_m = motion_stats(data["native"][data["split"]["FIT"]])
    v_max = fit_m["speed_p95"] * args.v_max_scale
    a_max = fit_m["accel_p95"] * args.a_max_scale
    motion_refs = dict(speed_p95=fit_m["speed_p95"], accel_mean=fit_m["accel_mean"])

    out = HERE / "results" / args.phase
    out.mkdir(parents=True, exist_ok=True)

    if args.phase == "smoke":
        cfg = dict(epochs=args.epochs or 2, batch=128, subset=2048, seed=args.seed)
        compare_name = f"COMPARE_{args.round_tag}_seed{args.seed}.json"
    else:
        cfg = dict(epochs=args.epochs or 8, batch=256, subset=None, seed=args.seed)
        compare_name = f"COMPARE_{args.round_tag}_seed{args.seed}.json"

    common = dict(
        v_max=v_max, a_max=a_max, motion_w=args.motion_w, motion_refs=motion_refs,
        out_dir=out, **cfg,
    )
    results = {}
    for arm in ("BASE", "KIN"):
        print(f"==== {args.phase} {args.round_tag} seed={args.seed} {arm} v_max={v_max:.4f} a_max={a_max:.4f} ====", flush=True)
        res, _ = train_arm(arm, data, **common)
        results[arm] = res

    compare = dict(
        phase=args.phase, round=args.round_tag, seed=args.seed, cfg=cfg,
        clamp=dict(v_max=v_max, a_max=a_max, v_max_scale=args.v_max_scale, a_max_scale=args.a_max_scale),
        motion_w=args.motion_w, fit_native_motion=fit_m,
        BASE_params=results["BASE"]["params"], KIN_params=results["KIN"]["params"],
        BASE_outer=results["BASE"]["metrics"]["outer"],
        KIN_outer=results["KIN"]["metrics"]["outer"],
        BASE_FIT_overall=results["BASE"]["metrics"]["FIT"]["overall"],
        KIN_FIT_overall=results["KIN"]["metrics"]["FIT"]["overall"],
        decision=decide(results),
    )
    (out / compare_name).write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"decision": compare["decision"], "file": compare_name}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
