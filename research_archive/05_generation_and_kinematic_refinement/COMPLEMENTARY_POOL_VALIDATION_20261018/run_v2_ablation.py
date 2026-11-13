"""V2：固定池下条件特征 vs geometry-only 一项消融。"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kdec.io_data import load_bundle
from kdec.features import build_pool_features, geometry_only_features
from kdec.hgb_locked import train_hgb, score_hgb
from kdec.select_locked import select_cond_boost_11_9
from kdec.metrics_full import errors, evaluate_portfolio
from run_v1 import make_fit_cal_split, kin_for_indices, load_kin, soft_labels

OUT = HERE / "results" / "v2"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 43, 44)


def main():
    data = load_bundle()
    fit_train_ix, fit_cal_ix, _ = make_fit_cal_split(data)
    rows = []
    for seed in SEEDS:
        kin_fit, _, _ = load_kin("FIT", seed, data)
        past_tr = data["past"][fit_train_ix]
        nat_tr = data["native"][fit_train_ix]
        kin_tr = kin_for_indices(fit_train_ix, seed, data, kin_fit)
        gt_tr = data["gt"][fit_train_ix]
        pool = np.concatenate([nat_tr, kin_tr], 1)
        soft, _ = soft_labels(pool, gt_tr)

        arms = {
            "cond_features": build_pool_features(past_tr, nat_tr, kin_tr),
            "geometry_only": geometry_only_features(nat_tr, kin_tr),
        }
        models = {}
        for arm, (fn, fk) in arms.items():
            feat = np.concatenate([fn, fk], 1)
            models[arm] = train_hgb(feat.reshape(-1, feat.shape[-1]), soft.reshape(-1), seed=seed)

        for split_name, ix in (("FIT_cal", fit_cal_ix), ("outer", data["split"]["outer"])):
            past = data["past"][ix]
            native = data["native"][ix]
            gt = data["gt"][ix]
            groups = data["groups"][ix]
            kin = kin_for_indices(ix, seed, data, kin_fit) if split_name.startswith("FIT") else load_kin(split_name, seed, data)[0]
            for arm, builder in (
                ("cond_features", lambda: build_pool_features(past, native, kin)),
                ("geometry_only", lambda: geometry_only_features(native, kin)),
            ):
                fn, fk = builder()
                F = fn.shape[-1]
                B = len(ix)
                sn = score_hgb(fn.reshape(-1, F), models[arm]).reshape(B, -1)
                sk = score_hgb(fk.reshape(-1, F), models[arm]).reshape(B, -1)
                paths, *_ = select_cond_boost_11_9(native, kin, sn, sk)
                ev = evaluate_portfolio(paths, gt, native, groups, do_cluster=False)
                rows.append(dict(
                    seed=seed, split=split_name, arm=arm,
                    coverage=ev["overall"]["coverage"],
                    minADE=ev["overall"]["minADE"],
                    minFDE=ev["overall"]["minFDE"],
                    lost=ev["overall"]["lost"],
                    rescue=ev["overall"]["rescue"],
                    change_coverage=ev["change"]["coverage"],
                    change_motion=ev["change"]["motion_gate"]["status"],
                ))
                print(rows[-1], flush=True)

    summary = {"rows": rows, "note": "单项消融：删 past 派生通道；容量随输入变，解释有局限。"}
    for split in ("FIT_cal", "outer"):
        summary[split] = {}
        for arm in ("cond_features", "geometry_only"):
            rs = [r for r in rows if r["split"] == split and r["arm"] == arm]
            summary[split][arm] = dict(
                coverage_mean=float(np.mean([r["coverage"] for r in rs])),
                minADE_mean=float(np.mean([r["minADE"] for r in rs])),
                lost_mean=float(np.mean([r["lost"] for r in rs])),
                change_coverage_mean=float(np.mean([r["change_coverage"] for r in rs])),
            )
    (OUT / "V2_ABLATION.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("FIT_cal", "outer")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
