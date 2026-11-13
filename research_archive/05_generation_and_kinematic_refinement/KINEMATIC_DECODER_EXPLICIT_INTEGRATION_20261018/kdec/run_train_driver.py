"""smoke / short 训练驱动。"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from kdec.data import load_bundle
from kdec.train import train_arm
from kdec import models as M


def decide(short_results: dict) -> dict:
    """按 DESIGN 弱阳性门（单 seed）。"""
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
    weak_positive = bool(motion_better and cov_ok and ade_ok)
    return dict(
        weak_positive=weak_positive,
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
        note="弱阳性只支持结构假设；不自动升级覆盖/迁移结论。阴性则停止，不扫参。",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["smoke", "short"], required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    data = load_bundle()
    out = HERE / "results" / args.phase
    out.mkdir(parents=True, exist_ok=True)

    if args.phase == "smoke":
        seed = 42 if args.seed is None else args.seed
        cfg = dict(epochs=2, batch=128, subset=2048, seed=seed)
        compare_name = "COMPARE.json"
    else:
        seed = 42 if args.seed is None else args.seed
        cfg = dict(epochs=8, batch=256, subset=None, seed=seed)
        compare_name = f"COMPARE_seed{seed}.json"

    results = {}
    for arm in ("BASE", "KIN"):
        print(f"==== {args.phase} seed={seed} {arm} params={M.count_params(M.build(arm))} ====", flush=True)
        res, _ = train_arm(arm, data, out_dir=out, **cfg)
        results[arm] = res

    pb = results["BASE"]["params"]
    pk = results["KIN"]["params"]
    compare = dict(
        phase=args.phase,
        cfg=cfg,
        seed=seed,
        BASE_params=pb,
        KIN_params=pk,
        param_ratio_KIN_over_BASE=pk / pb,
        BASE_outer=results["BASE"]["metrics"]["outer"],
        KIN_outer=results["KIN"]["metrics"]["outer"],
        BASE_FIT_overall=results["BASE"]["metrics"]["FIT"]["overall"],
        KIN_FIT_overall=results["KIN"]["metrics"]["FIT"]["overall"],
    )
    if args.phase == "short":
        compare["decision"] = decide(results)
    (out / compare_name).write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")
    # keep legacy COMPARE.json as alias of seed42 for compatibility
    if args.phase == "short" and seed == 42:
        (out / "COMPARE.json").write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(compare, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
