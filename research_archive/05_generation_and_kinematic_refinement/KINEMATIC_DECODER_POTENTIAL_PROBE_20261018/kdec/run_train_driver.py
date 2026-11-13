"""潜力探测驱动：LONG / LONG_RELAX，主训 KIN；可选 BASE 参考。"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from kdec.data import load_bundle
from kdec.train import train_arm
from kdec.metrics import motion_stats

VARIANTS = {
    "LONG": dict(epochs=32, v_max_scale=0.995, a_max_scale=0.90, motion_w=0.05),
    "LONG_RELAX": dict(epochs=32, v_max_scale=1.00, a_max_scale=0.95, motion_w=0.05),
}


def decide_vs_native(kin_outer: dict, base_outer: dict | None = None) -> dict:
    k_gate = kin_outer["_motion"]["gate"]["status"]
    k_cov = kin_outer["overall"]["coverage"]
    k_ade = kin_outer["overall"]["minADE"]
    native_cov = kin_outer["overall"]["native_coverage"]
    out = dict(
        motion_PASS=k_gate == "PASS",
        cov_ge_native=bool(k_cov >= native_cov),
        native_coverage=native_cov,
        KIN_coverage=k_cov,
        KIN_minADE=k_ade,
        KIN_motion=k_gate,
        KIN_accel_ratio=kin_outer["_motion"]["gate"]["ratios"]["accel_mean"],
        KIN_speed_ratio=kin_outer["_motion"]["gate"]["ratios"]["speed_p95"],
        gap_vs_native_pp=(k_cov - native_cov) * 100,
        seed_gate_A=bool(k_gate == "PASS" and k_cov >= native_cov),
    )
    if base_outer is not None:
        b_cov = base_outer["overall"]["coverage"]
        b_ade = base_outer["overall"]["minADE"]
        b_gate = base_outer["_motion"]["gate"]["status"]
        out.update(
            BASE_coverage=b_cov,
            BASE_minADE=b_ade,
            BASE_motion=b_gate,
            ade_better_than_BASE=bool(k_ade <= b_ade * 1.10),
            cov_better_than_BASE=bool(k_cov >= b_cov - 0.01),
        )
    return out


def summarize_splits(metrics: dict) -> dict:
    rows = {}
    for sp, m in metrics.items():
        rows[sp] = dict(
            coverage=m["overall"]["coverage"],
            minADE=m["overall"]["minADE"],
            minFDE=m["overall"]["minFDE"],
            native_coverage=m["overall"]["native_coverage"],
            motion=m["_motion"]["gate"]["status"],
            fails=m["_motion"]["gate"]["fails"],
            accel_ratio=m["_motion"]["gate"]["ratios"]["accel_mean"],
            speed_ratio=m["_motion"]["gate"]["ratios"]["speed_p95"],
            rescue=m["overall"].get("rescue"),
            lost=m["overall"].get("lost"),
        )
    if "FIT" in rows and "outer" in rows:
        rows["migration"] = dict(
            FIT_minus_outer_cov_pp=(rows["FIT"]["coverage"] - rows["outer"]["coverage"]) * 100,
            abs_gap_le_3pp=abs(rows["FIT"]["coverage"] - rows["outer"]["coverage"]) <= 0.03,
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS) + ["BASE_REF"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--arms", type=str, default="KIN", help="comma list: KIN or BASE,KIN")
    ap.add_argument("--eval-every", type=int, default=8)
    args = ap.parse_args()

    data = load_bundle()
    fit_m = motion_stats(data["native"][data["split"]["FIT"]])

    out = HERE / "results" / "long"
    out.mkdir(parents=True, exist_ok=True)

    if args.variant == "BASE_REF":
        cfg = dict(epochs=args.epochs or 32, v_max_scale=0.995, a_max_scale=0.90, motion_w=0.0)
        arms = ["BASE"]
        tag = "BASE_REF"
    else:
        cfg = dict(VARIANTS[args.variant])
        if args.epochs is not None:
            cfg["epochs"] = args.epochs
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        tag = args.variant

    v_max = fit_m["speed_p95"] * cfg["v_max_scale"]
    a_max = fit_m["accel_p95"] * cfg["a_max_scale"]
    motion_refs = dict(speed_p95=fit_m["speed_p95"], accel_mean=fit_m["accel_mean"])

    results = {}
    for arm in arms:
        print(
            f"==== long {tag} seed={args.seed} {arm} epochs={cfg['epochs']} "
            f"v_max={v_max:.4f} a_max={a_max:.4f} ====",
            flush=True,
        )
        res, _ = train_arm(
            arm,
            data,
            seed=args.seed,
            epochs=cfg["epochs"],
            batch=256,
            subset=None,
            motion_w=cfg["motion_w"] if arm == "KIN" else 0.0,
            motion_refs=motion_refs,
            v_max=v_max,
            a_max=a_max,
            out_dir=out,
            tag=tag,
            eval_every=args.eval_every,
        )
        results[arm] = res

    kin = results.get("KIN")
    base = results.get("BASE")
    compare = dict(
        phase="long",
        variant=tag,
        seed=args.seed,
        cfg=cfg,
        clamp=dict(v_max=v_max, a_max=a_max, v_max_scale=cfg["v_max_scale"], a_max_scale=cfg["a_max_scale"]),
        motion_w=cfg["motion_w"],
        fit_native_motion=fit_m,
        splits={},
        decision={},
    )
    if kin is not None:
        compare["splits"]["KIN"] = summarize_splits(kin["metrics"])
        compare["decision"] = decide_vs_native(
            kin["metrics"]["outer"],
            base["metrics"]["outer"] if base is not None else None,
        )
        compare["KIN_params"] = kin["params"]
        compare["KIN_elapsed_sec"] = kin["elapsed_sec"]
        compare["KIN_mid_evals"] = kin.get("mid_evals", [])
    if base is not None:
        compare["splits"]["BASE"] = summarize_splits(base["metrics"])
        compare["BASE_params"] = base["params"]
        compare["BASE_elapsed_sec"] = base["elapsed_sec"]

    name = f"COMPARE_{tag}_seed{args.seed}.json"
    (out / name).write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"decision": compare.get("decision"), "file": name, "splits_KIN": compare["splits"].get("KIN")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
