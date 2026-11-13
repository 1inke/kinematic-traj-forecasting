"""自改进：在锁定 11/9 下做保留原生独特性 / KIN 速度安全 小消融。
用 FIT_cal 选改进臂；outer 只作一次确认报告，不回头改方法。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kdec.io_data import load_bundle, SEL
from kdec.features import build_pool_features
from kdec.hgb_locked import load_hgb, score_hgb
from kdec.select_locked import (
    select_cond_boost_11_9, select_cond_boost_natunique, select_cond_boost_kinsafe,
    select_cond_boost_natunique as _nu,
)
from kdec.metrics_full import evaluate_portfolio, errors
from run_v1 import make_fit_cal_split, kin_for_indices, load_kin, jsonable_eval

OUT = HERE / "results" / "improve"
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


def select_combo(native, kin, sn, sk):
    """natunique + kinsafer：先独特 native，再对 KIN 做速度过滤。"""
    # get natunique native indices then kinsafer kin
    paths_u, pool_u, idx_n, idx_k, _ = select_cond_boost_natunique(native, kin, sn, sk)
    # re-do kin with safety given those natives
    from kdec.select_locked import gather, stable_topk, BOOST, N_KIN
    B = native.shape[0]
    def mean_speed(p):
        v = np.diff(p.astype(np.float64), axis=2, prepend=np.zeros_like(p[:, :, :1])) / 0.4
        return np.linalg.norm(v, axis=-1).mean(-1)
    sn_spd = mean_speed(native)
    sk_spd = mean_speed(kin)
    ep_n = native[:, :, -1, :].astype(np.float64)
    ep_k = kin[:, :, -1, :].astype(np.float64)
    idx_k2 = np.zeros((B, N_KIN), dtype=np.int64)
    for b in range(B):
        thr = float(np.quantile(sn_spd[b], 0.95)) * 1.01
        sk_b = sk[b].astype(np.float64).copy()
        sk_b[sk_spd[b] > thr + 1e-12] = -1e9
        dmin = np.linalg.norm(ep_k[b][:, None, :] - ep_n[b, idx_n[b]][None, :, :], axis=-1).min(1)
        sk_b = sk_b * (BOOST + dmin)
        if (sk_b > -1e8).sum() < N_KIN:
            sk_b = sk[b] * (BOOST + dmin)
        idx_k2[b] = stable_topk(sk_b, N_KIN)
    paths = np.concatenate([gather(native, idx_n), gather(kin, idx_k2)], 1)
    pool_idx = np.concatenate([idx_n, idx_k2 + 20], 1)
    return paths, pool_idx


METHODS = {
    "locked_boost_11_9": lambda n, k, sn, sk: select_cond_boost_11_9(n, k, sn, sk)[0:2],
    "natunique_11_9": lambda n, k, sn, sk: select_cond_boost_natunique(n, k, sn, sk)[0:2],
    "kinsafe_11_9": lambda n, k, sn, sk: select_cond_boost_kinsafe(n, k, sn, sk)[0:2],
    "natunique_kinsafe_11_9": select_combo,
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
        change_motion_fails=c["motion_gate"]["fails"] if c else None,
    )


def main():
    data = load_bundle()
    fit_train_ix, fit_cal_ix, cal_rec = make_fit_cal_split(data)
    rows = []
    for seed in SEEDS:
        print(f"improve seed {seed}", flush=True)
        model = load_hgb(ART / f"hgb_seed{seed}.joblib")
        kin_fit, _, _ = load_kin("FIT", seed, data)
        # FIT_cal for selection among improvements
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
                # save outer selected for audit
                if split_name == "outer":
                    np.savez_compressed(
                        OUT / f"outer_seed{seed}_{mname}.npz",
                        selected_pool_indices=pool_idx.astype(np.int16),
                        selected_paths=paths.astype(np.float32),
                    )
                met = eval_method(paths, gt, native, groups)
                rows.append(dict(seed=seed, split=split_name, method=mname, **met))
                print(seed, split_name, mname, met["coverage"], met["minADE"], met["lost"], met["change_motion"], flush=True)

    # choose on FIT_cal: maximize coverage - 0.5*lost_rate - 10*ade_penalty; require change motion pass if possible
    cal = [r for r in rows if r["split"] == "FIT_cal"]
    # aggregate per method
    methods = sorted({r["method"] for r in cal})
    ranking = []
    for m in methods:
        rs = [r for r in cal if r["method"] == m]
        cov = np.mean([r["coverage"] for r in rs])
        ade = np.mean([r["minADE"] for r in rs])
        lost = np.mean([r["lost"] for r in rs])
        ch_pass = sum(1 for r in rs if r["change_motion"] == "PASS")
        # score: prefer lower lost & ade, higher cov, change motion
        score = cov * 100 - lost * 0.05 - (ade - 0.17) * 50 + ch_pass
        ranking.append(dict(method=m, FIT_cal_cov=cov, FIT_cal_ade=ade, FIT_cal_lost=lost,
                            FIT_cal_change_motion_pass=ch_pass, select_score=score))
    ranking.sort(key=lambda d: -d["select_score"])
    chosen = ranking[0]["method"]

    # outer confirmation for chosen + locked baseline
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
            per_seed=rs,
        )

    out = dict(
        selection_split="FIT_cal",
        selection_rule="maximize cov*100 - 0.05*lost - 50*(ade-0.17) + change_motion_pass",
        ranking_FIT_cal=ranking,
        chosen_on_FIT_cal=chosen,
        outer_confirmation=confirm,
        note="改进臂在 FIT_cal 选择；outer 仅确认。不回头用 outer 改方法。",
        all_rows=rows,
    )
    (OUT / "IMPROVE_SUMMARY.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("chosen_on_FIT_cal", "ranking_FIT_cal")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
