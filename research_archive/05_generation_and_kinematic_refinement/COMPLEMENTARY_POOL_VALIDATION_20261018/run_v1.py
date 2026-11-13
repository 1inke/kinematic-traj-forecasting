"""V1：冻结重建 cond_boost_11_9；保存完整产物；分组运动学与尾部审核。不改 generator。"""
from __future__ import annotations
import json, sys, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kdec.io_data import load_bundle, array_sha, file_sha, CLAMP, SEL, ROOT
from kdec.features import build_pool_features, FEATURE_NAMES
from kdec.hgb_locked import train_hgb, score_hgb, save_hgb
from kdec.select_locked import (
    select_cond_boost_11_9, select_cond_top_11_9, N_NAT, N_KIN, BOOST,
)
from kdec.metrics_full import errors, evaluate_portfolio, summarize_deltas, MOTION_KEYS
from kdec.ci import hierarchical_ci

OUT = HERE / "results" / "v1"
ART = HERE / "artifacts" / "v1"
OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)

SEEDS = (42, 43, 44)
SPLITS_EVAL = ("outer", "inner", "FIT_cal")  # FIT_cal = held-out tracks from FIT
# original_CAL optional later


def soft_labels(pool, gt):
    e = errors(pool, gt)
    soft = np.exp(-np.maximum(e["ade"] / 0.25, e["fde"] / 0.5)).astype(np.float64)
    soft = np.maximum(soft, e["good"].astype(np.float64))
    return soft, e["good"]


def make_fit_cal_split(data, tag="v1_track_cal_20261018", frac=0.15, seed=20261018):
    """从 FIT 按完整 track 划出校准集；一次登记。"""
    fit_ix = data["split"]["FIT"]
    tracks = data["track_key"][fit_ix]
    uniq = np.unique(tracks)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_cal = max(1, int(round(len(uniq) * frac)))
    cal_tracks = set(uniq[:n_cal].tolist())
    is_cal = np.array([t in cal_tracks for t in tracks], dtype=bool)
    train_local = np.where(~is_cal)[0]
    cal_local = np.where(is_cal)[0]
    rec = dict(
        tag=tag, frac=frac, seed=seed,
        n_tracks_total=int(len(uniq)),
        n_tracks_cal=int(n_cal),
        n_windows_train=int(len(train_local)),
        n_windows_cal=int(len(cal_local)),
        cal_track_list_sha=array_sha(np.array(sorted(cal_tracks), dtype=object).astype(str)),
    )
    return fit_ix[train_local], fit_ix[cal_local], rec


def load_kin(split_name, seed, data, force_cpu=True):
    """优先读已有缓存；缺失则 frozen forward。默认 CPU，规避本机 torch CUDA DLL。"""
    cache = SEL / "results" / "cache_pred" / f"kin_{split_name}_seed{seed}.npy"
    # FIT_cal / inner may need custom
    if split_name in ("FIT", "outer") and cache.exists():
        return np.load(cache), str(cache), True
    # try our artifacts (incl. pre-generated inner)
    ours = ART / f"kin_{split_name}_seed{seed}.npy"
    if ours.exists():
        return np.load(ours), str(ours), True
    # also accept legacy name kin_inner_seedN.npy
    if split_name == "inner":
        legacy = ART / f"kin_inner_seed{seed}.npy"
        if legacy.exists():
            return np.load(legacy), str(legacy), True
    # frozen forward
    try:
        import torch
    except Exception as e:
        raise RuntimeError(
            f"torch import failed while needing forward for {split_name} seed{seed}: {e}. "
            "Pre-generate cache with CPU torch or copy kin_*.npy into artifacts/v1/."
        ) from e
    sys.path.insert(0, str(SEL))
    from kdec import models as M  # noqa type: ignore
    # Prefer CPU to avoid Windows CUDA DLL crashes unless explicitly disabled
    import os
    use_cuda = (not force_cpu) and os.environ.get("KIN_FORCE_CPU", "1") != "1" and torch.cuda.is_available()
    DEVICE = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"  load_kin forward {split_name} seed{seed} on {DEVICE}", flush=True)
    from kdec.metrics import motion_stats as ms  # type: ignore
    fit_m = ms(data["native"][data["split"]["FIT"]])
    v_max = fit_m["speed_p95"] * 0.995
    a_max = fit_m["accel_p95"] * 0.90
    ckpt = CLAMP / "results" / "short" / f"KIN_seed{seed}.pt"
    net = M.build("KIN", v_max=v_max, a_max=a_max).to(DEVICE)
    blob = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    net.load_state_dict(blob["state_dict"])
    net.eval()
    if split_name == "FIT_cal":
        raise RuntimeError("FIT_cal should use subset of FIT cache")
    ix = data["split"][split_name]
    past = data["past"][ix]
    rows = []
    bs = 64 if DEVICE.type == "cpu" else 256
    with torch.no_grad():
        for i in range(0, len(past), bs):
            p = torch.from_numpy(past[i:i+bs].astype(np.float32)).to(DEVICE)
            rows.append(net(p)["paths"].detach().cpu().numpy())
            if i % (bs * 8) == 0:
                print(f"    forward {i}/{len(past)}", flush=True)
    kin = np.concatenate(rows, 0)
    np.save(ours, kin)
    return kin, str(ours), False


def kin_for_indices(global_ix, seed, data, kin_fit_full):
    """Map global indices into FIT-ordered kin cache via position in FIT."""
    fit_ix = data["split"]["FIT"]
    # build map fit_global -> row
    pos = {int(g): i for i, g in enumerate(fit_ix)}
    rows = [pos[int(g)] for g in global_ix]
    return kin_fit_full[rows]


def score_split(past, native, kin, model):
    fn, fk = build_pool_features(past, native, kin)
    F = fn.shape[-1]
    B = native.shape[0]
    sn = score_hgb(fn.reshape(-1, F), model).reshape(B, -1)
    sk = score_hgb(fk.reshape(-1, F), model).reshape(B, -1)
    return sn, sk, fn, fk


def jsonable_eval(ev):
    """Strip large per-window arrays for JSON summary; keep class index lists."""
    out = {}
    for k, v in ev.items():
        if k == "_per_window":
            continue
        if k == "_classes":
            out[k] = {ck: cv.tolist() for ck, cv in v.items()}
            continue
        out[k] = v
    return out


def lost_attribution(native, gt, pool_idx, sn, n_nat=20):
    """对丢失窗口：原唯一 good slot、其 HGB 分位、是否仍在 40 池（总是）、为何被删。"""
    ne = errors(native, gt)
    good = ne["good"]
    n_good_count = good.sum(1)
    # selected native slots are first 11 of pool_idx
    B = native.shape[0]
    rows = []
    for b in range(B):
        goods = np.where(good[b])[0]
        if len(goods) == 0:
            continue
        # check if any good native kept
        kept_n = set(pool_idx[b, :11].tolist())
        kept_any = any(int(g) in kept_n for g in goods)
        if kept_any:
            continue
        # lost: all good natives dropped
        # unique good?
        uniq = len(goods) == 1
        g0 = int(goods[np.argmin(ne["joint"][b, goods])])
        rank = int((-sn[b]).argsort().tolist().index(g0))  # 0 = best score
        rows.append(dict(
            window=int(b),
            unique_good=bool(uniq),
            best_good_slot=g0,
            score=float(sn[b, g0]),
            score_rank0=rank,
            n_native_good=int(len(goods)),
            reason="dropped_by_top11_score",
            still_in_pool40=True,
        ))
    return rows


def main():
    data = load_bundle()
    fit_train_ix, fit_cal_ix, cal_rec = make_fit_cal_split(data)
    (OUT / "FIT_CAL_SPLIT.json").write_text(json.dumps(cal_rec, ensure_ascii=False, indent=2), encoding="utf-8")

    # primary deltas for CI across seeds on outer
    outer_deltas = {m: [] for m in (
        "coverage_ind", "minADE", "minFDE", "minJ",
        "change_coverage_ind", "change_minADE", "change_minFDE", "change_minJ",
    )}
    # coverage_ind is 0/1 per window difference method-native
    seed_summaries = []

    for seed in SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        kin_fit, kin_fit_src, cached = load_kin("FIT", seed, data)
        # train HGB on FIT_train only
        past_tr = data["past"][fit_train_ix]
        nat_tr = data["native"][fit_train_ix]
        kin_tr = kin_for_indices(fit_train_ix, seed, data, kin_fit)
        gt_tr = data["gt"][fit_train_ix]
        fn, fk = build_pool_features(past_tr, nat_tr, kin_tr)
        feat = np.concatenate([fn, fk], axis=1)
        pool = np.concatenate([nat_tr, kin_tr], axis=1)
        soft, good = soft_labels(pool, gt_tr)
        print(f"  train HGB on {len(fit_train_ix)} windows...", flush=True)
        model = train_hgb(feat.reshape(-1, feat.shape[-1]), soft.reshape(-1), seed=seed)
        hgb_path = ART / f"hgb_seed{seed}.joblib"
        save_hgb(model, hgb_path)

        seed_row = dict(seed=seed, hgb=str(hgb_path), kin_fit_source=kin_fit_src, kin_fit_cached=cached,
                        feature_names=FEATURE_NAMES, n_nat=N_NAT, n_kin=N_KIN, boost=BOOST)
        split_results = {}

        # prepare split specs
        split_specs = {
            "outer": data["split"]["outer"],
            "inner": data["split"]["inner"],
            "FIT_cal": fit_cal_ix,
            "FIT_train": fit_train_ix,
        }

        for split_name, ix in split_specs.items():
            print(f"  eval {split_name} N={len(ix)}", flush=True)
            past = data["past"][ix]
            native = data["native"][ix]
            gt = data["gt"][ix]
            groups = data["groups"][ix]
            ids = data["sample_id"][ix]
            tracks = data["track_key"][ix]
            # kin
            if split_name.startswith("FIT"):
                kin = kin_for_indices(ix, seed, data, kin_fit)
                kin_src = kin_fit_src
            else:
                kin, kin_src, _ = load_kin(split_name, seed, data)
                assert kin.shape[0] == len(ix)

            sn, sk, fn_e, fk_e = score_split(past, native, kin, model)
            scores40 = np.concatenate([sn, sk], axis=1)

            # locked primary
            paths_b, pool_idx_b, idx_n_b, idx_k_b, sk_boost = select_cond_boost_11_9(native, kin, sn, sk)
            paths_t, pool_idx_t, _, _ = select_cond_top_11_9(native, kin, sn, sk)
            union = np.concatenate([native, kin], axis=1)

            # save artifacts
            stem = ART / f"{split_name}_seed{seed}"
            np.savez_compressed(
                str(stem) + "_bundle.npz",
                sample_id=ids,
                track_key=tracks.astype(str),
                groups=groups,
                scores40=scores40.astype(np.float32),
                sn=sn.astype(np.float32),
                sk=sk.astype(np.float32),
                sk_boost=sk_boost.astype(np.float32),
                selected_pool_indices=pool_idx_b.astype(np.int16),
                selected_native_indices=idx_n_b.astype(np.int16),
                selected_kin_indices=idx_k_b.astype(np.int16),
                selected_paths=paths_b.astype(np.float32),
                pool_sha_native=np.array(array_sha(native)),
                pool_sha_kin=np.array(array_sha(kin)),
            )

            # evaluate portfolios
            ports = {
                "native20": native,
                "kin20": kin,
                "union40": union,
                "cond_top_11_9": paths_t,
                "cond_boost_11_9": paths_b,
            }
            port_eval = {}
            for pname, pth in ports.items():
                # cluster only for primary + native on outer to save time? do all but skip cluster on FIT_train large
                # 聚簇较慢：仅 outer 的 native/primary 做；其余跳过
                do_cl = (split_name == "outer" and pname in ("native20", "cond_boost_11_9"))
                ev = evaluate_portfolio(pth, gt, native, groups, do_cluster=do_cl)
                # union-good-but-selected-fail for primary
                if pname == "cond_boost_11_9":
                    ue = errors(union, gt)
                    u_cov = ue["good"].any(1)
                    p_cov = ev["_per_window"]["covered"]
                    ug_fail = np.where(u_cov & (~p_cov))[0].astype(np.int64)
                    ev["_classes"]["union_good_but_selected_fail"] = ug_fail
                    # lost attribution
                    lost_attr = lost_attribution(native, gt, pool_idx_b, sn)
                    ev["lost_attribution_n"] = len(lost_attr)
                    ev["lost_attribution_sample"] = lost_attr[:50]
                    # delta summaries overall + by group
                    pw = ev["_per_window"]
                    ev["delta_summaries"] = {
                        "overall": {
                            "minADE": summarize_deltas(pw["delta_minADE"]),
                            "minFDE": summarize_deltas(pw["delta_minFDE"]),
                            "J": summarize_deltas(pw["delta_J"]),
                        }
                    }
                    for gname, gmask in (("stable", groups == 0), ("transition", groups == 1), ("change", groups == 2)):
                        if gmask.any():
                            ev["delta_summaries"][gname] = {
                                "minADE": summarize_deltas(pw["delta_minADE"], gmask),
                                "minFDE": summarize_deltas(pw["delta_minFDE"], gmask),
                                "J": summarize_deltas(pw["delta_J"], gmask),
                            }
                port_eval[pname] = jsonable_eval(ev)

            meta = dict(
                split=split_name, N=int(len(ix)), kin_source=kin_src,
                pool_sha_native=array_sha(native), pool_sha_kin=array_sha(kin),
                scores_sha=array_sha(scores40),
                selected_indices_sha=array_sha(pool_idx_b),
                selected_paths_sha=array_sha(paths_b),
                portfolios=port_eval,
            )
            split_results[split_name] = meta
            (OUT / f"seed{seed}_{split_name}.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # collect outer deltas for primary CI
            if split_name == "outer":
                pw = evaluate_portfolio(paths_b, gt, native, groups, do_cluster=False)["_per_window"]
                # indicator coverage delta
                cov_d = pw["covered"].astype(np.float64) - pw["native_covered"].astype(np.float64)
                outer_deltas["coverage_ind"].append(cov_d)
                outer_deltas["minADE"].append(pw["delta_minADE"])
                outer_deltas["minFDE"].append(pw["delta_minFDE"])
                outer_deltas["minJ"].append(pw["delta_J"])
                ch = groups == 2
                # for change metrics store full-length with nan outside change? design: only change windows
                # hierarchical_ci expects same N — use zeros for non-change and mask via track? 
                # Better: store change-only arrays separately with change track ids
                seed_row["outer_change_ix"] = np.where(ch)[0]
                seed_row["_cov_d"] = cov_d
                seed_row["_dade"] = pw["delta_minADE"]
                seed_row["_dfde"] = pw["delta_minFDE"]
                seed_row["_dj"] = pw["delta_J"]
                seed_row["_groups"] = groups
                seed_row["_tracks"] = tracks

        # drop bulky arrays from seed json
        slim = {k: v for k, v in seed_row.items() if not k.startswith("_") and k != "outer_change_ix"}
        slim["splits"] = {k: dict(N=split_results[k]["N"],
                                  primary=split_results[k]["portfolios"]["cond_boost_11_9"].get("overall"),
                                  primary_change=split_results[k]["portfolios"]["cond_boost_11_9"].get("change"),
                                  motion_by_group={
                                      g: split_results[k]["portfolios"]["cond_boost_11_9"].get(g, {}).get("motion_gate")
                                      for g in ("overall", "stable", "transition", "change")
                                  }) for k in split_results}
        seed_summaries.append(dict(slim=slim, _raw=seed_row, split_results=split_results))
        (OUT / f"seed{seed}_SUMMARY.json").write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"seed": seed, "outer_primary": slim["splits"]["outer"]["primary"]}, ensure_ascii=False), flush=True)

    # CI on outer
    tracks0 = seed_summaries[0]["_raw"]["_tracks"]
    ci_out = {}
    mapping = [
        ("overall_coverage", "coverage_ind", None),
        ("overall_minADE", "minADE", None),
        ("overall_minFDE", "minFDE", None),
        ("overall_minJ", "minJ", None),
    ]
    # build list of arrays
    for key, field, _ in mapping:
        arrs = [s["_raw"][f"_{ 'cov_d' if field=='coverage_ind' else 'dade' if field=='minADE' else 'dfde' if field=='minFDE' else 'dj'}"] for s in seed_summaries]
        # fix field map
    field_map = {
        "overall_coverage": "_cov_d",
        "overall_minADE": "_dade",
        "overall_minFDE": "_dfde",
        "overall_minJ": "_dj",
    }
    for name, attr in field_map.items():
        arrs = [s["_raw"][attr] for s in seed_summaries]
        ci_out[name] = hierarchical_ci(arrs, tracks0.astype(str), n_draws=100000, seed=20261018)

    # change-only CI: restrict to change windows (same indices across seeds)
    ch_mask = seed_summaries[0]["_raw"]["_groups"] == 2
    ch_tracks = tracks0[ch_mask].astype(str)
    for name, attr in (
        ("change_coverage", "_cov_d"),
        ("change_minADE", "_dade"),
        ("change_minFDE", "_dfde"),
        ("change_minJ", "_dj"),
    ):
        arrs = [s["_raw"][attr][ch_mask] for s in seed_summaries]
        if ch_mask.sum() == 0:
            ci_out[name] = dict(valid=0, ci=None, note="no change windows")
        else:
            ci_out[name] = hierarchical_ci(arrs, ch_tracks, n_draws=100000, seed=20261018)

    # aggregate table
    def mean_metric(split, port, group, key):
        vals = []
        for s in seed_summaries:
            vals.append(s["split_results"][split]["portfolios"][port][group][key])
        return float(np.mean(vals)), [float(v) for v in vals]

    agg = {"seeds": list(SEEDS), "fit_cal": cal_rec, "ci_exploratory": ci_out, "tables": {}}
    for split in ("outer", "inner", "FIT_cal"):
        rows = {}
        for port in ("native20", "kin20", "union40", "cond_top_11_9", "cond_boost_11_9"):
            rows[port] = {}
            for group in ("overall", "stable", "transition", "change"):
                try:
                    cov, covs = mean_metric(split, port, group, "coverage")
                    ade, ades = mean_metric(split, port, group, "minADE")
                    fde, fdes = mean_metric(split, port, group, "minFDE")
                    gates = [s["split_results"][split]["portfolios"][port][group]["motion_gate"]["status"] for s in seed_summaries]
                    resc = [s["split_results"][split]["portfolios"][port][group]["rescue"] for s in seed_summaries]
                    lost = [s["split_results"][split]["portfolios"][port][group]["lost"] for s in seed_summaries]
                    rows[port][group] = dict(
                        coverage_mean=cov, coverage_per_seed=covs,
                        minADE_mean=ade, minADE_per_seed=ades,
                        minFDE_mean=fde, minFDE_per_seed=fdes,
                        motion_pass=sum(1 for g in gates if g == "PASS"),
                        motion_status=gates,
                        rescue=resc, lost=lost,
                    )
                except Exception as e:
                    rows[port][group] = dict(error=str(e))
        agg["tables"][split] = rows

    # gates
    prim = agg["tables"]["outer"]["cond_boost_11_9"]["overall"]
    nat = agg["tables"]["outer"]["native20"]["overall"]
    lift_pp = (prim["coverage_mean"] - nat["coverage_mean"]) * 100
    ade_worse = prim["minADE_mean"] > nat["minADE_mean"]
    fde_worse = prim["minFDE_mean"] > nat["minFDE_mean"]
    # all seed x group motion
    motion_all_pass = True
    motion_fails = []
    for s in seed_summaries:
        for g in ("overall", "stable", "transition", "change"):
            st = s["split_results"]["outer"]["portfolios"]["cond_boost_11_9"][g]["motion_gate"]
            if st["status"] != "PASS":
                motion_all_pass = False
                motion_fails.append(dict(seed=s["slim"]["seed"], group=g, fails=st["fails"], ratios=st["ratios"]))

    decision = dict(
        coverage_lift_pp=lift_pp,
        ade_worsened=ade_worse,
        fde_worsened=fde_worse,
        label=("覆盖—误差折中" if (lift_pp > 0 and (ade_worse or fde_worse))
               else ("全面优于native(本轮探索)" if lift_pp > 0 and not ade_worse and not fde_worse
                     else "未优于native")),
        motion_all_seed_group_pass=motion_all_pass,
        motion_fails=motion_fails,
        major_deploy_gate=False,
        note="outer 曾参与历史方法选择；本 V1 锁定方法重现，区间为探索性。",
    )
    agg["decision"] = decision
    (OUT / "SUMMARY.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
