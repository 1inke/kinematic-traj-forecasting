"""smoke / short：RES vs ID；可选 KIN 再生参考。"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from kdec.data import load_bundle
from kdec.train import train_arm
from kdec.metrics import motion_stats

KIN_CEILING = 0.5646  # SPEED_CLAMP 3-seed mean
NATIVE_REF = 0.6039


def decide(res_outer, id_outer=None):
    cov = res_outer["overall"]["coverage"]
    native_cov = res_outer["overall"]["native_coverage"]
    ade = res_outer["overall"]["minADE"]
    gate = res_outer["_motion"]["gate"]["status"]
    rescue = res_outer["overall"]["rescue"]
    lost = res_outer["overall"]["lost"]
    beat_native = bool(gate == "PASS" and cov + 1e-12 >= native_cov)
    beat_kin_ceiling = bool(gate == "PASS" and cov >= KIN_CEILING)
    retain_ok = bool(lost <= max(2, int(0.02 * res_outer["overall"]["N"])))
    return dict(
        beat_native=beat_native,
        beat_kin_ceiling=beat_kin_ceiling,
        motion=gate,
        coverage=cov,
        native_coverage=native_cov,
        gap_vs_native_pp=(cov - native_cov) * 100,
        minADE=ade,
        rescue=rescue,
        lost=lost,
        retain_ok=retain_ok,
        weak_positive=bool(gate == "PASS" and cov >= native_cov - 0.01 and rescue >= lost),
        KIN_CEILING=KIN_CEILING,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["smoke", "short"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--a-res-max", type=float, default=0.25)
    ap.add_argument("--res-w", type=float, default=0.1)
    ap.add_argument("--motion-w", type=float, default=0.05)
    ap.add_argument("--with-kin-ref", action="store_true")
    ap.add_argument("--round-tag", type=str, default="r1")
    args = ap.parse_args()

    data = load_bundle()
    fit_m = motion_stats(data["native"][data["split"]["FIT"]])
    v_max = fit_m["speed_p95"] * 0.995
    a_max = fit_m["accel_p95"] * 0.90
    motion_refs = dict(speed_p95=fit_m["speed_p95"], accel_mean=fit_m["accel_mean"])

    out = HERE / "results" / args.phase
    out.mkdir(parents=True, exist_ok=True)

    if args.phase == "smoke":
        cfg = dict(epochs=args.epochs or 2, batch=128, subset=2048, seed=args.seed)
    else:
        cfg = dict(epochs=args.epochs or 8, batch=256, subset=None, seed=args.seed)

    common = dict(
        v_max=v_max, a_max=a_max, a_res_max=args.a_res_max,
        motion_w=args.motion_w, res_w=args.res_w, motion_refs=motion_refs,
        out_dir=out, **cfg,
    )
    results = {}
    # identity baseline (no train)
    print(f"==== {args.phase} {args.round_tag} seed={args.seed} ID ====", flush=True)
    results["ID"], _ = train_arm("ID", data, **common)

    print(f"==== {args.phase} {args.round_tag} seed={args.seed} RES a_res_max={args.a_res_max} ====", flush=True)
    results["RES"], _ = train_arm("RES", data, **common)

    if args.with_kin_ref:
        print(f"==== {args.phase} KIN_REF ====", flush=True)
        results["KIN"], _ = train_arm("KIN", data, **common)

    decision = decide(results["RES"]["metrics"]["outer"])
    compare = dict(
        phase=args.phase, round=args.round_tag, seed=args.seed, cfg=cfg,
        a_res_max=args.a_res_max, res_w=args.res_w, motion_w=args.motion_w,
        clamp=dict(v_max=v_max, a_max=a_max), fit_native_motion=fit_m,
        ID_outer=results["ID"]["metrics"]["outer"],
        RES_outer=results["RES"]["metrics"]["outer"],
        RES_FIT_overall=results["RES"]["metrics"]["FIT"]["overall"],
        RES_params=results["RES"]["params"],
        decision=decision,
    )
    if "KIN" in results:
        compare["KIN_outer"] = results["KIN"]["metrics"]["outer"]
    name = f"COMPARE_{args.round_tag}_seed{args.seed}.json"
    (out / name).write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"decision": decision, "file": name}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
