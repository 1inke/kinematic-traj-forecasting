"""自改进续：固定互补池下小配额消融（更多原生槽）。FIT_cal 选；outer 仅确认。不改 HGB/不训 generator。"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kdec.io_data import load_bundle
from kdec.features import build_pool_features
from kdec.hgb_locked import load_hgb, score_hgb
from kdec.select_locked import select_cond_boost_11_9, select_cond_boost_kinsafe, BOOST
from kdec.metrics_full import evaluate_portfolio
from run_v1 import make_fit_cal_split, kin_for_indices, load_kin

OUT = HERE / "results" / "improve_quota"
ART = HERE / "artifacts" / "v1"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 43, 44)


def score_split(past, native, kin, model):
    fn, fk = build_pool_features(past, native, kin)
    F = fn.shape[-1]
    B = native.shape[0]
    sn = score_hgb(fn.reshape(-1, F), model).reshape(B, -1)
    sk = score_hgb(fk.reshape(-1, F), model).reshape(B, -1)
    return sn, sk


def make_selector(n_nat, n_kin, kinsafer=False):
    def fn(native, kin, sn, sk):
        if kinsafer:
            paths, pool, *_ = select_cond_boost_kinsafe(
                native, kin, sn, sk, boost=BOOST, n_nat=n_nat, n_kin=n_kin
            )
            return paths, pool
        paths, pool, *_ = select_cond_boost_11_9(
            native, kin, sn, sk, boost=BOOST, n_nat=n_nat, n_kin=n_kin
        )
        return paths, pool
    return fn


METHODS = {
    "boost_11_9": make_selector(11, 9, False),
    "boost_12_8": make_selector(12, 8, False),
    "boost_13_7": make_selector(13, 7, False),
    "boost_14_6": make_selector(14, 6, False),
    "kinsafe_11_9": make_selector(11, 9, True),
    "kinsafe_12_8": make_selector(12, 8, True),
    "kinsafe_13_7": make_selector(13, 7, True),
}


def eval_method(paths, gt, native, groups):
    ev = evaluate_portfolio(paths, gt, native, groups, do_cluster=False)
    o, c = ev["overall"], ev.get("change")
    return dict(
        coverage=o["coverage"], minADE=o["minADE"], minFDE=o["minFDE"],
        rescue=o["rescue"], lost=o["lost"],
        motion_overall=o["motion_gate"]["status"],
        change_coverage=c["coverage"] if c else None,
        change_minADE=c["minADE"] if c else None,
        change_lost=c["lost"] if c else None,
        change_motion=c["motion_gate"]["status"] if c else None,
    )


def main():
    data = load_bundle()
    _, fit_cal_ix, cal_rec = make_fit_cal_split(data)
    rows = []
    for seed in SEEDS:
        print(f"quota seed {seed}", flush=True)
        model = load_hgb(ART / f"hgb_seed{seed}.joblib")
        kin_fit, _, _ = load_kin("FIT", seed, data)
        for split_name, ix in (("FIT_cal", fit_cal_ix), ("outer", data["split"]["outer"])):
            past = data["past"][ix]
            native = data["native"][ix]
            gt = data["gt"][ix]
            groups = data["groups"][ix]
            if split_name.startswith("FIT"):
                kin = kin_for_indices(ix, seed, data, kin_fit)
            else:
                kin, _, _ = load_kin(split_name, seed, data)
            sn, sk = score_split(past, native, kin, model)
            for mname, fn in METHODS.items():
                paths, pool_idx = fn(native, kin, sn, sk)
                if split_name == "outer":
                    np.savez_compressed(
                        OUT / f"outer_seed{seed}_{mname}.npz",
                        selected_pool_indices=pool_idx.astype(np.int16),
                        selected_paths=paths.astype(np.float32),
                    )
                met = eval_method(paths, gt, native, groups)
                rows.append(dict(seed=seed, split=split_name, method=mname, **met))
                print(seed, split_name, mname, met["coverage"], met["minADE"], met["lost"], met["change_motion"], flush=True)

    cal = [r for r in rows if r["split"] == "FIT_cal"]
    methods = sorted({r["method"] for r in cal})
    ranking = []
    for m in methods:
        rs = [r for r in cal if r["method"] == m]
        cov = float(np.mean([r["coverage"] for r in rs]))
        ade = float(np.mean([r["minADE"] for r in rs]))
        fde = float(np.mean([r["minFDE"] for r in rs]))
        lost = float(np.mean([r["lost"] for r in rs]))
        ch_pass = sum(1 for r in rs if r["change_motion"] == "PASS")
        # Prefer lower lost & ADE harm, higher cov; bonus change motion
        score = cov * 100 - lost * 0.05 - (ade - 0.17) * 50 - (fde - 0.30) * 20 + ch_pass
        ranking.append(dict(
            method=m, FIT_cal_cov=cov, FIT_cal_ade=ade, FIT_cal_fde=fde,
            FIT_cal_lost=lost, FIT_cal_change_motion_pass=ch_pass, select_score=score,
        ))
    ranking.sort(key=lambda d: -d["select_score"])
    chosen = ranking[0]["method"]

    outer = [r for r in rows if r["split"] == "outer"]
    confirm = {}
    for m in methods:
        rs = [r for r in outer if r["method"] == m]
        confirm[m] = dict(
            coverage_mean=float(np.mean([r["coverage"] for r in rs])),
            minADE_mean=float(np.mean([r["minADE"] for r in rs])),
            minFDE_mean=float(np.mean([r["minFDE"] for r in rs])),
            lost_mean=float(np.mean([r["lost"] for r in rs])),
            rescue_mean=float(np.mean([r["rescue"] for r in rs])),
            change_coverage_mean=float(np.mean([r["change_coverage"] for r in rs])),
            change_motion_pass=sum(1 for r in rs if r["change_motion"] == "PASS"),
            overall_motion_pass=sum(1 for r in rs if r["motion_overall"] == "PASS"),
            per_seed=[{k: v for k, v in r.items() if k != "per_seed"} for r in rs],
        )

    out = dict(
        selection_split="FIT_cal",
        selection_rule="cov*100 - 0.05*lost - 50*(ade-0.17) - 20*(fde-0.30) + change_motion_pass",
        ranking_FIT_cal=ranking,
        chosen_on_FIT_cal=chosen,
        outer_confirmation=confirm,
        fit_cal_rec=cal_rec,
        note="配额消融：更多原生槽以保留支撑、降 lost/ADE；FIT_cal 选，outer 仅确认。主方法仍锁定 11/9 作对照。",
        all_rows=rows,
    )
    (OUT / "QUOTA_IMPROVE_SUMMARY.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"chosen": chosen, "ranking": ranking[:4], "outer_chosen": confirm[chosen],
                      "outer_locked": confirm["boost_11_9"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
