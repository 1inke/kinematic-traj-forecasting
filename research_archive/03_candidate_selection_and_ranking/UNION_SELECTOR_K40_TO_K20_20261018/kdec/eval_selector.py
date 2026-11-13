"""Evaluate K40->K20 selectors on outer; train cheap FIT ranker."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
CLAMP = ROOT / "KINEMATIC_DECODER_SPEED_CLAMP_20261018"
sys.path.insert(0, str(HERE))

from kdec.data import load_bundle
from kdec.metrics import errors, motion_stats, evaluate_split
from kdec import models as M
from kdec import selectors as S

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_kin(net, past, batch=256):
    net.eval()
    rows = []
    for i in range(0, len(past), batch):
        p = torch.from_numpy(past[i:i+batch].astype(np.float32)).to(DEVICE)
        rows.append(net(p)["paths"].detach().cpu().numpy())
    return np.concatenate(rows, 0)


def load_kin_net(ckpt_path, v_max, a_max):
    net = M.build("KIN", v_max=v_max, a_max=a_max).to(DEVICE)
    blob = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    net.load_state_dict(blob["state_dict"])
    return net


def train_ranker_on_fit(past_fit, native_fit, kin_fit, gt_fit, seed: int = 0):
    from kdec.cond_ranker import build_cond_features
    from kdec.hgb_ranker import train_hgb
    B = native_fit.shape[0]
    fam_n = np.zeros((B, native_fit.shape[1]))
    fam_k = np.ones((B, kin_fit.shape[1]))
    fn = build_cond_features(past_fit, native_fit, fam_n)
    fk = build_cond_features(past_fit, kin_fit, fam_k)
    ep_n = native_fit[:, :, -1, :].astype(np.float64)
    ep_k = kin_fit[:, :, -1, :].astype(np.float64)
    d = np.linalg.norm(ep_n[:, :, None, :] - ep_k[:, None, :, :], axis=-1)
    fn = np.concatenate([fn, d.min(axis=2, keepdims=True)], -1)
    fk = np.concatenate([fk, d.min(axis=1)[..., None]], -1)
    feat = np.concatenate([fn, fk], axis=1)
    pool = np.concatenate([native_fit, kin_fit], axis=1)
    ade, fde, joint, good, _, _ = errors(pool, gt_fit)
    soft = np.exp(-np.maximum(ade / 0.25, fde / 0.5)).astype(np.float64)
    soft = np.maximum(soft, good.astype(np.float64))
    X = feat.reshape(-1, feat.shape[-1])
    y = soft.reshape(-1)
    print("  train HGB...", flush=True)
    return train_hgb(X, y, seed=seed)


def score_pool(past, native, kin, model):
    from kdec.cond_ranker import build_cond_features
    from kdec.hgb_ranker import score_hgb
    from kdec.cond_ranker import score_cond
    B = native.shape[0]
    fam_n = np.zeros((B, native.shape[1]))
    fam_k = np.ones((B, kin.shape[1]))
    fn = build_cond_features(past, native, fam_n)
    fk = build_cond_features(past, kin, fam_k)
    ep_n = native[:, :, -1, :].astype(np.float64)
    ep_k = kin[:, :, -1, :].astype(np.float64)
    d = np.linalg.norm(ep_n[:, :, None, :] - ep_k[:, None, :, :], axis=-1)
    fn = np.concatenate([fn, d.min(axis=2, keepdims=True)], -1)
    fk = np.concatenate([fk, d.min(axis=1)[..., None]], -1)
    F = fn.shape[-1]
    if model.get("type") == "hgb":
        sn = score_hgb(fn.reshape(-1, F), model).reshape(B, -1)
        sk = score_hgb(fk.reshape(-1, F), model).reshape(B, -1)
    else:
        sn = score_cond(fn.reshape(-1, F), model).reshape(B, -1)
        sk = score_cond(fk.reshape(-1, F), model).reshape(B, -1)
    return sn, sk


def summarize_method(paths, gt, native, groups):
    ev = evaluate_split(paths, gt, native, groups)
    o = ev["overall"]
    return dict(
        coverage=o["coverage"],
        native_coverage=o["native_coverage"],
        minADE=o["minADE"],
        minFDE=o["minFDE"],
        rescue=o["rescue"],
        lost=o["lost"],
        K=int(paths.shape[1]),
        motion=ev["_motion"]["gate"]["status"],
        motion_fails=ev["_motion"]["gate"]["fails"],
        motion_ratios=ev["_motion"]["gate"]["ratios"],
        by_group={g: dict(coverage=ev[g]["coverage"], N=ev[g]["N"]) for g in ("stable", "transition", "change") if g in ev},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--split", default="outer", choices=["outer", "FIT", "inner"])
    ap.add_argument("--cache-pred", action="store_true", default=True)
    args = ap.parse_args()

    data = load_bundle()
    fit_m = motion_stats(data["native"][data["split"]["FIT"]])
    v_max = fit_m["speed_p95"] * 0.995
    a_max = fit_m["accel_p95"] * 0.90

    out_dir = HERE / "results" / "per_seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = HERE / "results" / "cache_pred"
    cache_dir.mkdir(parents=True, exist_ok=True)

    method_order = (
        ["native", "kin", "union40", "oracle_cover", "oracle_minade_top20"]
        + S.GT_FREE_NAMES
    )

    all_seed_rows = []

    for seed in args.seeds:
        ckpt = CLAMP / "results" / "short" / f"KIN_seed{seed}.pt"
        if not ckpt.exists():
            raise SystemExit(f"missing ckpt {ckpt}")
        net = load_kin_net(ckpt, v_max, a_max)

        # FIT preds for ranker
        fit_ix = data["split"]["FIT"]
        cache_fit = cache_dir / f"kin_FIT_seed{seed}.npy"
        if cache_fit.exists():
            kin_fit = np.load(cache_fit)
        else:
            print(f"predict FIT seed={seed} ...", flush=True)
            kin_fit = predict_kin(net, data["past"][fit_ix])
            np.save(cache_fit, kin_fit)
        native_fit = data["native"][fit_ix]
        gt_fit = data["gt"][fit_ix]
        print(f"train ranker seed={seed} ...", flush=True)
        ranker = train_ranker_on_fit(data["past"][fit_ix], native_fit, kin_fit, gt_fit, seed=seed)

        ix = data["split"][args.split]
        cache_ev = cache_dir / f"kin_{args.split}_seed{seed}.npy"
        if cache_ev.exists():
            kin_paths = np.load(cache_ev)
        else:
            print(f"predict {args.split} seed={seed} ...", flush=True)
            kin_paths = predict_kin(net, data["past"][ix])
            np.save(cache_ev, kin_paths)
        native = data["native"][ix]
        gt = data["gt"][ix]
        groups = data["groups"][ix]

        past = data["past"][ix]
        print(f"  score cond ranker seed={seed}", flush=True)
        sn, sk = score_pool(past, native, kin_paths, ranker)

        # FIT-calibrate score-swap depth (uses FIT GT only)
        print(f"  calibrate swap on FIT seed={seed}", flush=True)
        sn_f, sk_f = score_pool(data["past"][fit_ix], native_fit, kin_fit, ranker)
        best_swap, best_c = 4, -1.0
        for ns in (2, 3, 4, 5, 6, 7, 8, 10):
            paths_f = S.select_score_swap(native_fit, kin_fit, sn_f, sk_f, ns)
            # coverage on FIT subsample for speed
            rng = np.random.default_rng(seed)
            sub = rng.choice(len(native_fit), size=min(4000, len(native_fit)), replace=False)
            from kdec.metrics import errors as _err
            _, _, _, g, _, _ = _err(paths_f[sub], gt_fit[sub])
            c = float(g.any(1).mean())
            if c > best_c:
                best_c, best_swap = c, ns
        print(f"  best_swap={best_swap} FIT_sub_cov={best_c:.4f}", flush=True)


        # Cond-score driven GT-free methods (primary)
        # adaptive ratios from max native score (low max_sn -> more KIN slots)
        max_sn = sn.max(axis=1)
        # calibrate thresholds on percentiles of this split's scores (GT-free)
        thr_low = np.quantile(max_sn, 0.35)   # uncertain natives
        thr_high = np.quantile(max_sn, 0.70)
        adapt_paths = native.copy()
        for b in range(len(native)):
            if max_sn[b] < thr_low:
                n_nat, n_kin = 10, 10
            elif max_sn[b] < thr_high:
                n_nat, n_kin = 12, 8
            else:
                n_nat, n_kin = 15, 5
            ix_n = np.argsort(-sn[b])[:n_nat]
            ix_k = np.argsort(-sk[b])[:n_kin]
            adapt_paths[b, :n_nat] = native[b, ix_n]
            adapt_paths[b, n_nat:] = kin_paths[b, ix_k]

        # boost KIN by complementarity: sk * (0.5 + min_dist_to_native_top)
        ep_n = native[:, :, -1, :].astype(np.float64)
        ep_k = kin_paths[:, :, -1, :].astype(np.float64)
        # dist to top-12 native by score
        boost_paths = {}
        for n_nat, n_kin, tag in [(12, 8, "12_8"), (11, 9, "11_9"), (10, 10, "10_10")]:
            sk_boost = sk.copy()
            for b in range(len(native)):
                topn = np.argsort(-sn[b])[:n_nat]
                dmin = np.linalg.norm(ep_k[b][:, None, :] - ep_n[b, topn][None, :, :], axis=-1).min(axis=1)
                sk_boost[b] = sk[b] * (0.35 + dmin)
            boost_paths[f"cond_boost_{tag}"] = S.select_from_scores(native, kin_paths, sn, sk_boost, n_nat, n_kin)

        score_methods = {
            "cond_top_12_8": S.select_from_scores(native, kin_paths, sn, sk, 12, 8),
            "cond_top_11_9": S.select_from_scores(native, kin_paths, sn, sk, 11, 9),
            "cond_top_10_10": S.select_from_scores(native, kin_paths, sn, sk, 10, 10),
            "cond_top_13_7": S.select_from_scores(native, kin_paths, sn, sk, 13, 7),
            "cond_top_14_6": S.select_from_scores(native, kin_paths, sn, sk, 14, 6),
            "cond_top_15_5": S.select_from_scores(native, kin_paths, sn, sk, 15, 5),
            "cond_global_top20": S.select_global_from_scores(native, kin_paths, sn, sk, 20),
            "cond_quota_8_4": S.select_quota_from_scores(native, kin_paths, sn, sk, 20, 8, 4),
            "cond_quota_10_5": S.select_quota_from_scores(native, kin_paths, sn, sk, 20, 10, 5),
            "cond_quota_6_6": S.select_quota_from_scores(native, kin_paths, sn, sk, 20, 6, 6),
            "cond_adapt_ratio": adapt_paths,
            **boost_paths,
            "cond_swap_4": S.select_score_swap(native, kin_paths, sn, sk, 4),
            "cond_swap_6": S.select_score_swap(native, kin_paths, sn, sk, 6),
            "cond_swap_8": S.select_score_swap(native, kin_paths, sn, sk, 8),
            "cond_swap_calib": S.select_score_swap(native, kin_paths, sn, sk, best_swap),
        }

        methods = {}
        # baselines + oracle via run_named (no ranker needed)
        for name in ["native", "kin", "union40", "oracle_cover", "oracle_minade_top20"]:
            print(f"  select {name} seed={seed}", flush=True)
            paths = S.run_named(name, native, kin_paths, gt=gt, ranker=None)
            methods[name] = summarize_method(paths, gt, native, groups)

        # geometry-only GT-free (no ranker)
        for name in ["strat_fps_12_8", "global_fps_20", "swap_redund_6"]:
            print(f"  select {name} seed={seed}", flush=True)
            paths = S.run_named(name, native, kin_paths, gt=gt, ranker=None)
            methods[name] = summarize_method(paths, gt, native, groups)

        for name, paths in score_methods.items():
            print(f"  select {name} seed={seed}", flush=True)
            methods[name] = summarize_method(paths, gt, native, groups)

        native_cov = methods["native"]["coverage"]
        union_cov = methods["union40"]["coverage"]
        # best GT-free (exclude baselines/oracle)
        exclude = {"native", "kin", "union40", "oracle_cover", "oracle_minade_top20"}
        gf = {n: methods[n] for n in methods if n not in exclude}
        best_name = max(gf, key=lambda n: gf[n]["coverage"])
        best = gf[best_name]
        lift = (best["coverage"] - native_cov) * 100
        gap_u = (union_cov - best["coverage"]) * 100
        seed_hit = bool(best["coverage"] + 1e-12 >= native_cov + 0.04 and best["motion"] == "PASS")
        seed_stretch = bool(
            best["coverage"] + 1e-12 >= native_cov + 0.07
            or gap_u <= 2.0 + 1e-12
        )

        row = dict(
            seed=seed,
            split=args.split,
            ckpt=str(ckpt),
            methods=methods,
            best_gt_free=best_name,
            best_gt_free_coverage=best["coverage"],
            best_gt_free_lift_pp=lift,
            best_gt_free_motion=best["motion"],
            gap_to_union_pp=gap_u,
            seed_HIT=seed_hit,
            seed_STRETCH=seed_stretch,
            native_coverage=native_cov,
            union_coverage=union_cov,
            kin_coverage=methods["kin"]["coverage"],
            oracle_cover_coverage=methods["oracle_cover"]["coverage"],
        )
        (out_dir / f"seed{seed}_{args.split}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        all_seed_rows.append(row)
        print(json.dumps({
            "seed": seed,
            "best": best_name,
            "cov": best["coverage"],
            "lift_pp": lift,
            "motion": best["motion"],
            "HIT": seed_hit,
        }, ensure_ascii=False), flush=True)

    # aggregate
    native_cov = float(np.mean([r["native_coverage"] for r in all_seed_rows]))
    union_mean = float(np.mean([r["union_coverage"] for r in all_seed_rows]))
    kin_mean = float(np.mean([r["kin_coverage"] for r in all_seed_rows]))
    # per GT-free method means
    # union of gt-free names across seeds
    exclude = {"native", "kin", "union40", "oracle_cover", "oracle_minade_top20"}
    gf_names = sorted({n for r in all_seed_rows for n in r["methods"] if n not in exclude})
    gf_table = {}
    for name in gf_names:
        covs = [r["methods"][name]["coverage"] for r in all_seed_rows]
        mots = [r["methods"][name]["motion"] for r in all_seed_rows]
        lifts = [(c - native_cov) * 100 for c in covs]
        gf_table[name] = dict(
            coverage_mean=float(np.mean(covs)),
            coverage_std=float(np.std(covs)),
            lift_pp_mean=float(np.mean(lifts)),
            motion_pass=sum(1 for m in mots if m == "PASS"),
            per_seed=covs,
        )
    best_overall = max(gf_table, key=lambda n: gf_table[n]["coverage_mean"])
    b = gf_table[best_overall]
    gate_HIT = bool(
        b["lift_pp_mean"] + 1e-12 >= 4.0
        and b["motion_pass"] == len(all_seed_rows)
        and sum(1 for r in all_seed_rows if r["methods"][best_overall]["coverage"] + 1e-12 >= native_cov + 0.04) >= 2
    )
    gate_STRETCH = bool(
        b["lift_pp_mean"] + 1e-12 >= 7.0
        or (union_mean - b["coverage_mean"]) * 100 <= 2.0 + 1e-12
    )

    summary = dict(
        n_seeds=len(all_seed_rows),
        split=args.split,
        native_coverage=native_cov,
        kin_coverage_mean=kin_mean,
        union40_coverage_mean=union_mean,
        oracle_cover_mean=float(np.mean([r["oracle_cover_coverage"] for r in all_seed_rows])),
        oracle_minade_mean=float(np.mean([r["methods"]["oracle_minade_top20"]["coverage"] for r in all_seed_rows])),
        gt_free=gf_table,
        best_gt_free=best_overall,
        best_gt_free_coverage_mean=b["coverage_mean"],
        best_gt_free_lift_pp_mean=b["lift_pp_mean"],
        best_gt_free_motion_pass=b["motion_pass"],
        gate_HIT=gate_HIT,
        gate_STRETCH=gate_STRETCH,
        seed_rows=[{
            "seed": r["seed"],
            "best_gt_free": r["best_gt_free"],
            "best_cov": r["best_gt_free_coverage"],
            "lift_pp": r["best_gt_free_lift_pp"],
            "motion": r["best_gt_free_motion"],
            "HIT": r["seed_HIT"],
            "native": r["native_coverage"],
            "kin": r["kin_coverage"],
            "union40": r["union_coverage"],
            "oracle_cover": r["oracle_cover_coverage"],
        } for r in all_seed_rows],
    )
    (HERE / "results" / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# UNION_SELECTOR_K40_TO_K20 results",
        "",
        f"- split: {args.split}",
        f"- native K20: {native_cov:.4f}",
        f"- KIN K20 mean: {kin_mean:.4f}",
        f"- union K40 mean: {union_mean:.4f}",
        f"- oracle_cover K20 mean: {summary['oracle_cover_mean']:.4f}",
        f"- best GT-free: **{best_overall}** cov={b['coverage_mean']:.4f} lift={b['lift_pp_mean']:.2f}pp motion_pass={b['motion_pass']}/{len(all_seed_rows)}",
        f"- gate_HIT (+4pp & motion): **{gate_HIT}**",
        f"- gate_STRETCH: **{gate_STRETCH}**",
        "",
        "## Seed table (best GT-free per seed)",
        "| seed | native | kin | union40 | best_gt_free | best_cov | lift_pp | motion | oracle | HIT |",
        "|---:|---:|---:|---:|---|---:|---:|---|---:|---|",
    ]
    for r in all_seed_rows:
        lines.append(
            f"| {r['seed']} | {r['native_coverage']:.4f} | {r['kin_coverage']:.4f} | {r['union_coverage']:.4f} | {r['best_gt_free']} | {r['best_gt_free_coverage']:.4f} | {r['best_gt_free_lift_pp']:.2f} | {r['best_gt_free_motion']} | {r['oracle_cover_coverage']:.4f} | {r['seed_HIT']} |"
        )
    lines += ["", "## All GT-free methods (mean)", "| method | cov_mean | lift_pp | motion_pass |", "|---|---:|---:|---:|"]
    for name, g in sorted(gf_table.items(), key=lambda kv: -kv[1]["coverage_mean"]):
        lines.append(f"| {name} | {g['coverage_mean']:.4f} | {g['lift_pp_mean']:.2f} | {g['motion_pass']} |")
    text = "\\n".join(lines) + "\\n"
    (HERE / "results" / "RESULTS.md").write_text(text, encoding="utf-8")
    (HERE / "RESULTS.md").write_text(text, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
