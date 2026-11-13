"""自改进：按分数质量自适应原生配额（10–14），无 GT；FIT_cal 选，outer 确认。"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kdec.io_data import load_bundle
from kdec.features import build_pool_features
from kdec.hgb_locked import load_hgb, score_hgb
from kdec.select_locked import select_cond_boost_11_9, select_cond_boost_kinsafe, gather, stable_topk, BOOST
from kdec.metrics_full import evaluate_portfolio
from run_v1 import make_fit_cal_split, kin_for_indices, load_kin

OUT = HERE / "results" / "improve_adaptive"
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


def select_adaptive(native, kin, sn, sk, kinsafer=False, n_min=10, n_max=14):
    """n_nat = clip(round(10 + 4 * mass_n/(mass_n+mass_k)), n_min, n_max); n_kin=20-n_nat."""
    B = native.shape[0]
    ep_n = native[:, :, -1, :].astype(np.float64)
    ep_k = kin[:, :, -1, :].astype(np.float64)
    idx_n = np.zeros((B, 20), dtype=np.int64)  # fill later per-row varying — pad unused
    # variable length: build paths by stacking per batch with fixed 20
    paths = np.zeros((B, 20) + native.shape[2:], dtype=native.dtype)
    pool_idx = np.zeros((B, 20), dtype=np.int64)
    quotas = np.zeros(B, dtype=np.int64)

    def mean_speed(p):
        v = np.diff(p.astype(np.float64), axis=2, prepend=np.zeros_like(p[:, :, :1])) / 0.4
        return np.linalg.norm(v, axis=-1).mean(-1)

    sn_spd = mean_speed(native) if kinsafer else None
    sk_spd = mean_speed(kin) if kinsafer else None

    for b in range(B):
        mass_n = float(np.sort(sn[b])[-11:].sum())
        mass_k = float(np.sort(sk[b])[-9:].sum()) + 1e-9
        frac = mass_n / (mass_n + mass_k)
        n_nat = int(np.clip(round(n_min + (n_max - n_min) * frac), n_min, n_max))
        n_kin = 20 - n_nat
        quotas[b] = n_nat
        in_ = stable_topk(sn[b], n_nat)
        sk_b = sk[b].astype(np.float64).copy()
        if kinsafer:
            thr = float(np.quantile(sn_spd[b], 0.95)) * 1.01
            sk_b[sk_spd[b] > thr + 1e-12] = -1e9
        dmin = np.linalg.norm(ep_k[b][:, None, :] - ep_n[b, in_][None, :, :], axis=-1).min(1)
        sk_b = sk_b * (BOOST + dmin)
        if kinsafer and (sk_b > -1e8).sum() < n_kin:
            sk_b = sk[b] * (BOOST + dmin)
        ik = stable_topk(sk_b, n_kin)
        paths[b] = np.concatenate([native[b, in_], kin[b, ik]], 0)
        pool_idx[b] = np.concatenate([in_, ik + 20], 0)
    return paths, pool_idx, quotas


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


METHODS = {
    "locked_11_9": lambda n, k, sn, sk: select_cond_boost_11_9(n, k, sn, sk)[0:2] + (None,),
    "kinsafe_11_9": lambda n, k, sn, sk: select_cond_boost_kinsafe(n, k, sn, sk)[0:2] + (None,),
    "adaptive": lambda n, k, sn, sk: select_adaptive(n, k, sn, sk, False),
    "adaptive_kinsafe": lambda n, k, sn, sk: select_adaptive(n, k, sn, sk, True),
}


def main():
    data = load_bundle()
    _, fit_cal_ix, _ = make_fit_cal_split(data)
    rows = []
    quota_stats = {}
    for seed in SEEDS:
        print(f"adaptive seed {seed}", flush=True)
        model = load_hgb(ART / f"hgb_seed{seed}.joblib")
        kin_fit, _, _ = load_kin("FIT", seed, data)
        for split_name, ix in (("FIT_cal", fit_cal_ix), ("outer", data["split"]["outer"])):
            past = data["past"][ix]
            native = data["native"][ix]
            gt = data["gt"][ix]
            groups = data["groups"][ix]
            kin = kin_for_indices(ix, seed, data, kin_fit) if split_name.startswith("FIT") else load_kin(split_name, seed, data)[0]
            sn, sk = score_split(past, native, kin, model)
            for mname, fn in METHODS.items():
                out = fn(native, kin, sn, sk)
                paths, pool_idx = out[0], out[1]
                quotas = out[2] if len(out) > 2 else None
                if quotas is not None and split_name == "outer":
                    quota_stats[f"{mname}_seed{seed}"] = dict(
                        mean=float(quotas.mean()), min=int(quotas.min()), max=int(quotas.max()),
                        hist={str(i): int((quotas == i).sum()) for i in range(10, 15)},
                    )
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
        score = cov * 100 - lost * 0.05 - (ade - 0.17) * 50 - (fde - 0.30) * 20 + ch_pass
        ranking.append(dict(method=m, FIT_cal_cov=cov, FIT_cal_ade=ade, FIT_cal_fde=fde,
                            FIT_cal_lost=lost, FIT_cal_change_motion_pass=ch_pass, select_score=score))
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
        )

    out = dict(
        selection_split="FIT_cal",
        chosen_on_FIT_cal=chosen,
        ranking_FIT_cal=ranking,
        outer_confirmation=confirm,
        quota_stats=quota_stats,
        note="自适应原生配额；主方法仍为锁定 11/9。本臂为探索性自改进。",
        all_rows=rows,
    )
    (OUT / "ADAPTIVE_SUMMARY.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"chosen": chosen, "ranking": ranking, "outer": confirm}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
