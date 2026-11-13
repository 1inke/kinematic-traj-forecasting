#!/usr/bin/env python3
"""K1 orchestrator: I vs S set-marginal supervision. Generator frozen. No K0 teacher."""
from __future__ import annotations
import os
for _k, _v in (("OMP_NUM_THREADS", "2"), ("MKL_NUM_THREADS", "2"), ("OPENBLAS_NUM_THREADS", "2"), ("NUMEXPR_NUM_THREADS", "2"), ("VECLIB_MAXIMUM_THREADS", "2")):
    os.environ.setdefault(_k, _v)
import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
import hashlib
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from k1.config import (
    ROOT, EXEC, K1_DIR, DESIGN, SEEDS, GROUPS, N_NAT,
)
from k1.build_dataset import build_training_matrix
from k1.train_hgb import fit_arm, load_arm
from k1.features113 import build_pool_features_22, pairwise_D_matrix
from k1.select_set import select_batch
from k1.bootstrap import per_window_metrics, family16_bootstrap, point_estimates, go_k2_check
from k0.data_io import (
    load_bundle, make_fit_cal_indices, load_kin, load_v1_selected_indices, file_sha, array_sha,
)
from k0.metrics import evaluate_set, check_W, gather
from k0.serialize import dump, eval_public, jsonable


def now_hkt():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S+09:00")


def lock_design(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    h_all = hashlib.sha256()
    paths = []
    if DESIGN.exists():
        paths = sorted(DESIGN.glob("*.md"))
    else:
        # fallback: bundled copies next to exec
        for name in ("00_contract.md", "01_K0.md", "02_K1.md", "04.md"):
            p = HERE / name
            if p.exists():
                paths.append(p)
    for p in paths:
        sha = file_sha(p)
        files[p.name] = dict(path=str(p), sha256=sha, size=p.stat().st_size)
        h_all.update(sha.encode())
    rec = dict(locked_at=now_hkt(), files=files, design_bundle_sha256=h_all.hexdigest())
    dump(out / "DESIGN_LOCK.json", rec)
    return rec


def slice_split(data, ix):
    return dict(
        sample_id=data["sample_id"][ix],
        past=data["past"][ix],
        gt=data["gt"][ix],
        native=data["native"][ix],
        groups=data["groups"][ix],
        track_key=data["track_key"][ix],
        ix=ix,
    )


def prepare_pool(native, kin):
    return np.concatenate([native, kin], axis=1)


def eval_indices(pool40, indices, gt, native, groups, do_cluster=False):
    paths = gather(pool40, indices)
    ev = evaluate_set(paths, gt, native, groups, do_cluster=do_cluster)
    w = check_W(ev, groups)
    return paths, ev, w


def run_seed(seed: int, data, fit_train, fit_cal, k1_root: Path, smoke: int | None):
    seed_dir = k1_root / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    models_dir = seed_dir / "models"
    pred_dir = seed_dir / "predictions"
    trace_dir = seed_dir / "action_traces"
    data_dir = seed_dir / "train_data"
    for d in (models_dir, pred_dir, trace_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    done_marker = seed_dir / "SEED_DONE.json"
    if done_marker.exists() and not smoke:
        print(f"[seed{seed}] already DONE, skip", flush=True)
        return json.loads(done_marker.read_text(encoding="utf-8"))

    # Load KIN for FIT (full FIT then slice train/cal)
    kin_fit, kin_src = load_kin("FIT", seed, data)
    fit_ix = data["split"]["FIT"]
    pos = {int(g): i for i, g in enumerate(fit_ix)}
    tr_rows = [pos[int(g)] for g in fit_train]
    cal_rows = [pos[int(g)] for g in fit_cal]
    kin_tr = kin_fit[tr_rows]
    kin_cal = kin_fit[cal_rows]

    past_tr = data["past"][fit_train]
    gt_tr = data["gt"][fit_train]
    native_tr = data["native"][fit_train]
    sid_tr = data["sample_id"][fit_train]

    print(f"[seed{seed}] build training matrix", flush=True)
    t0 = time.time()
    X, YI, YS, diag = build_training_matrix(
        data["sample_id"], past_tr, gt_tr, native_tr, kin_tr, sid_tr,
        data_dir, seed, max_windows=smoke,
    )
    dump(data_dir / "diag.json", {**diag, "kin_src": kin_src, "build_s": time.time() - t0})

    from k1.config import OUTPUT_NAMES as _OUT
    def _arm_ready(arm: str) -> bool:
        return all((models_dir / f"{arm}_{n}_seed{seed}.joblib").exists() for n in _OUT)
    if _arm_ready("I"):
        print(f"[seed{seed}] fit I SKIP (models present)", flush=True)
    else:
        print(f"[seed{seed}] fit I", flush=True)
        fit_arm(X, YI, seed, "I", models_dir)
    if _arm_ready("S"):
        print(f"[seed{seed}] fit S SKIP (models present)", flush=True)
    else:
        print(f"[seed{seed}] fit S", flush=True)
        fit_arm(X, YS, seed, "S", models_dir)
    models_I = load_arm(models_dir, "I", seed)
    models_S = load_arm(models_dir, "S", seed)

    split_results = {}
    # splits to evaluate
    split_specs = []
    # outer
    kin_o, src_o = load_kin("outer", seed, data)
    split_specs.append(("outer", slice_split(data, data["split"]["outer"]), kin_o, src_o, True))
    # inner
    kin_i, src_i = load_kin("inner", seed, data)
    split_specs.append(("inner", slice_split(data, data["split"]["inner"]), kin_i, src_i, False))
    # FIT_cal
    cal_pack = slice_split(data, fit_cal)
    split_specs.append(("FIT_cal", cal_pack, kin_cal, kin_src, False))
    # original_CAL
    kin_c, src_c = load_kin("original_CAL", seed, data)
    if kin_c is None:
        split_results["original_CAL"] = dict(status="BLOCKED_NO_KIN_CACHE", src=src_c)
    else:
        split_specs.append(("original_CAL", slice_split(data, data["split"]["original_CAL"]), kin_c, src_c, False))

    for split_name, pack, kin, src, do_cl in split_specs:
        print(f"[seed{seed}] deploy {split_name} N={len(pack['sample_id'])}", flush=True)
        out_s = pred_dir / split_name
        out_s.mkdir(parents=True, exist_ok=True)
        marker = out_s / "done.json"
        if marker.exists() and not smoke:
            split_results[split_name] = json.loads(marker.read_text(encoding="utf-8"))
            continue

        native = pack["native"]
        pool40 = prepare_pool(native, kin.astype(np.float64))
        feat40 = build_pool_features_22(pack["past"], native, kin.astype(np.float64))
        # D cache can be heavy; compute on the fly inside select for memory
        # For outer 1035 * 40*40 is fine to cache
        D_cache = None
        if len(native) <= 2000:
            print(f"  caching D matrices...", flush=True)
            D_cache = [pairwise_D_matrix(pool40[b]) for b in range(len(native))]

        ports = {}
        # native
        _, ev_n, w_n = eval_indices(pool40, np.tile(np.arange(N_NAT), (len(native), 1)), pack["gt"], native, pack["groups"], do_cl)
        ports["native20"] = dict(eval=eval_public(ev_n), W=jsonable(w_n), W_pass=bool(w_n["pass_"]))

        # V1 historical reference only
        v1_idx, v1_src = load_v1_selected_indices(split_name, seed)
        if v1_idx is not None and len(v1_idx) == len(native):
            # verify ids if possible
            _, ev_v, w_v = eval_indices(pool40, v1_idx, pack["gt"], native, pack["groups"])
            ports["v1_cond_boost_11_9"] = dict(
                eval=eval_public(ev_v), W=jsonable(w_v), W_pass=bool(w_v["pass_"]), src=v1_src,
                note="historical reference only; not single-factor arm",
            )

        arm_out = {}
        for arm, models in (("I", models_I), ("S", models_S)):
            print(f"  select arm={arm}", flush=True)
            t1 = time.time()
            indices, traces, rejects = select_batch(
                pool40, native, feat40, models, D_cache=D_cache,
                progress_every=100 if split_name == "outer" else 200,
            )
            paths, ev, w = eval_indices(pool40, indices, pack["gt"], native, pack["groups"], do_cl)
            np.save(out_s / f"{arm}_indices.npy", indices)
            dump(trace_dir / f"{split_name}_{arm}_rejects.json", rejects)
            # save compact traces (counts + first/last few)
            dump(trace_dir / f"{split_name}_{arm}_trace_summary.json", dict(
                n_windows=len(traces),
                n_swaps_mean=float(np.mean([len(t) for t in traces])),
                n_swaps_hist={str(k): int(v) for k, v in zip(*np.unique([len(t) for t in traces], return_counts=True))},
                rejects=rejects,
                wall_s=time.time() - t1,
            ))
            # full traces as npz-ish json only for outer small? save joblib-style npz of lengths
            np.save(trace_dir / f"{split_name}_{arm}_n_swaps.npy", np.array([len(t) for t in traces], dtype=np.int16))
            arm_out[arm] = dict(
                eval=eval_public(ev),
                W=jsonable(w),
                W_pass=bool(w["pass_"]),
                indices_sha=array_sha(indices),
                rejects=rejects,
                n_swaps_mean=float(np.mean([len(t) for t in traces])),
                wall_s=time.time() - t1,
            )
            ports[arm] = arm_out[arm]
            # keep per-window for bootstrap (outer only)
            if split_name == "outer":
                pw = per_window_metrics(paths, pack["gt"])
                np.savez_compressed(out_s / f"{arm}_per_window.npz", **pw)

        # native per-window for outer
        if split_name == "outer":
            pw_n = per_window_metrics(native, pack["gt"])
            np.savez_compressed(out_s / "native_per_window.npz", **pw_n)
            np.save(out_s / "groups.npy", pack["groups"])
            np.save(out_s / "track_key.npy", pack["track_key"].astype(str))
            np.save(out_s / "sample_id.npy", pack["sample_id"])

        rec = dict(
            seed=seed, split=split_name, N=int(len(native)), kin_src=src,
            ports=ports, finished_at=now_hkt(),
        )
        dump(marker, rec)
        split_results[split_name] = rec
        # free D
        D_cache = None

    summary = dict(seed=seed, splits=split_results, finished_at=now_hkt())
    dump(done_marker, summary)
    return summary


def secondary_pass_check(seed_summaries) -> dict:
    """Check inner/FIT_cal/original_CAL for GO_K2 secondary conditions."""
    out = {}
    for split in ("inner", "FIT_cal", "original_CAL"):
        infos = []
        blocked = False
        not_done = False
        fail = False
        for seed in SEEDS:
            if seed not in seed_summaries:
                not_done = True
                continue
            sp = seed_summaries[seed]["splits"].get(split)
            if sp is None:
                not_done = True
                continue
            if sp.get("status", "").startswith("BLOCKED") or sp.get("status") == "NOT_DONE":
                blocked = True
                infos.append(sp)
                continue
            S = sp["ports"]["S"]["eval"]
            N = sp["ports"]["native20"]["eval"]
            seed_fail = False
            reasons = []
            for g in GROUPS:
                if S[g]["motion_gate"]["status"] != "PASS":
                    seed_fail = True
                    reasons.append(f"{g}_motion")
                if g != "change":
                    if S[g]["minADE"] > N[g]["minADE"] * 1.01 + 1e-15:
                        seed_fail = True
                        reasons.append(f"{g}_ade")
                    if S[g]["minFDE"] > N[g]["minFDE"] * 1.01 + 1e-15:
                        seed_fail = True
                        reasons.append(f"{g}_fde")
                lost = S[g]["conditional_lost_frac"]
                if lost is not None and lost > 0.01 + 1e-15:
                    seed_fail = True
                    reasons.append(f"{g}_lost")
            if seed_fail:
                fail = True
            infos.append(dict(seed=seed, pass_=not seed_fail, reasons=reasons))
        if blocked:
            out[split] = dict(status="BLOCKED", pass_=False, detail=infos)
        elif not_done:
            out[split] = dict(status="NOT_DONE", pass_=False, detail=infos)
        else:
            out[split] = dict(status="DONE", pass_=not fail, detail=infos)
    return out


def write_analysis(k1_root: Path, seed_summaries, boot, points, go):
    lines = []
    lines.append("# K1 RESULT_ANALYSIS")
    lines.append("")
    lines.append(f"- 生成时间：{now_hkt()}（Asia/Tokyo）")
    lines.append(f"- 执行目录：`{k1_root}`")
    lines.append("- 唯一因素：I=individual-surrogate vs S=set-marginal；生成器冻结；未读 K0 oracle teacher。")
    lines.append("")
    lines.append("## Outer 三 seed 关键数字")
    lines.append("")
    lines.append("| seed | arm | change_good | change_cov | change_minADE | change_minFDE | change_minJ | W_pass | n_swaps |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---|---:|")
    for seed in SEEDS:
        if seed not in seed_summaries:
            continue
        outer = seed_summaries[seed]["splits"]["outer"]["ports"]
        for arm in ("native20", "I", "S", "v1_cond_boost_11_9"):
            if arm not in outer:
                continue
            ev = outer[arm]["eval"]
            ch = ev["change"]
            lines.append(
                f"| {seed} | {arm} | {ch['good_count']} | {ch['coverage']:.4f} | "
                f"{ch['minADE']:.5f} | {ch['minFDE']:.5f} | {ch['mean_minJ']:.5f} | "
                f"{outer[arm].get('W_pass')} | {outer[arm].get('n_swaps_mean', '—')} |"
            )
    lines.append("")
    lines.append("## S − I / S − native 点估计（outer，三 seed 均值）")
    lines.append("")
    if points:
        for contrast, gd in points.items():
            lines.append(f"### {contrast}")
            for g, md in gd.items():
                for metric, rec in md.items():
                    lines.append(f"- {g}.{metric}: mean={rec.get('mean')} per_seed={rec.get('per_seed')}")
            lines.append("")
    lines.append("## Bootstrap family16（Bonferroni）")
    lines.append("")
    fam = boot.get("family16", {})
    for contrast in ("S_minus_I", "S_minus_native"):
        lines.append(f"### {contrast}")
        for g in ("overall", "change"):
            for m in ("coverage", "minADE", "minFDE", "mean_minJ"):
                rec = fam.get(contrast, {}).get(g, {}).get(m, {})
                lines.append(f"- {g}.{m}: ci_bonf={rec.get('ci_bonferroni')} ci95={rec.get('ci_95')} n_valid={rec.get('n_valid')}")
        lines.append("")
    lines.append("## GO_K2")
    lines.append("")
    lines.append(f"**GO_K2 = {go.get('GO_K2')}**")
    lines.append("")
    lines.append("关键失败条款见 `GO_K2_REGISTER.json`。")
    lines.append("")
    lines.append("## 未做 / 缺口")
    lines.append("")
    lines.append("- original_CAL：若 BLOCKED_NO_KIN_CACHE，按合同不触发 K2。")
    lines.append("- 未重训 generator；未用 K0 oracle 作 teacher；未扫配额/阈值。")
    lines.append("- V1 仅历史参照，非单因素对照臂。")
    text = "\n".join(lines) + "\n"
    (k1_root / "RESULT_ANALYSIS.md").write_text(text, encoding="utf-8")
    dump(k1_root / "RESULT_ANALYSIS.json", dict(
        points=points, bootstrap=boot, go=go, seeds=jsonable(seed_summaries),
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=None, help="truncate train windows for dry run")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--skip-boot", action="store_true")
    args = ap.parse_args()

    k1_root = K1_DIR
    k1_root.mkdir(parents=True, exist_ok=True)
    print(f"K1_DIR={k1_root}", flush=True)
    design_lock = lock_design(k1_root / "freeze")
    print(f"design_bundle_sha={design_lock['design_bundle_sha256']}", flush=True)

    print("loading bundle...", flush=True)
    data = load_bundle()
    fit_train, fit_cal, rec = make_fit_cal_indices(data)
    dump(k1_root / "FIT_CAL_REPLAY.json", dict(rec=rec, n_train=len(fit_train), n_cal=len(fit_cal)))

    seed_summaries = {}
    for seed in args.seeds:
        try:
            seed_summaries[seed] = run_seed(seed, data, fit_train, fit_cal, k1_root, args.smoke)
        except Exception as e:
            dump(k1_root / f"seed{seed}_ERROR.json", dict(error=str(e), tb=traceback.format_exc()))
            print(f"SEED {seed} FAILED: {e}", flush=True)
            raise

    # Bootstrap on outer
    per_seed = {}
    track_keys = {}
    per_seed_eval = {}
    for seed in args.seeds:
        outer_dir = k1_root / f"seed{seed}" / "predictions" / "outer"
        groups = np.load(outer_dir / "groups.npy")
        tracks = np.load(outer_dir / "track_key.npy", allow_pickle=True)
        rec = {"groups": groups}
        for arm, fname in (("I", "I_per_window.npz"), ("S", "S_per_window.npz"), ("native", "native_per_window.npz")):
            z = np.load(outer_dir / fname)
            rec[arm] = {k: z[k] for k in z.files}
        per_seed[seed] = rec
        track_keys[seed] = tracks
        ports = seed_summaries[seed]["splits"]["outer"]["ports"]
        per_seed_eval[seed] = dict(
            S=ports["S"]["eval"],
            I=ports["I"]["eval"],
            native=ports["native20"]["eval"],
        )

    points = point_estimates(per_seed)
    if args.skip_boot or args.smoke:
        boot = dict(status="SKIPPED")
    else:
        print("bootstrap family16...", flush=True)
        boot = family16_bootstrap(per_seed, track_keys)
        dump(k1_root / "bootstrap.json", boot)

    secondary = secondary_pass_check(seed_summaries)
    if boot.get("status") != "OK":
        go = dict(GO_K2=False, clauses={"bootstrap": dict(pass_=False, status=boot.get("status"))})
    else:
        go = go_k2_check(per_seed_eval, boot, secondary)
    dump(k1_root / "GO_K2_REGISTER.json", dict(
        document="GO_K2_REGISTER",
        created_at=now_hkt(),
        timezone="Asia/Tokyo",
        exec_dir=str(k1_root),
        GO_K2=go["GO_K2"],
        clauses=go["clauses"],
        secondary=secondary,
        design_lock=design_lock,
        note="未过门不启动 K2",
        author="tosaka_executor",
    ))
    write_analysis(k1_root, seed_summaries, boot, points, go)

    # ARITHMETIC_AUDIT light
    dump(k1_root / "ARITHMETIC_AUDIT.json", dict(
        quality=(0.2, 0.4),
        motion_factor=1.01,
        feature_dim=113,
        hgb=dict(max_iter=200, max_depth=6, lr=0.08),
        n_train_windows_design=4096,
        actions_per_state=32,
        states=["S0", "S4", "S12"],
        pred_L_max=0.01,
        checked_at=now_hkt(),
    ))
    dump(k1_root / "runtime_state.json", dict(status="COMPLETE", finished_at=now_hkt(), GO_K2=go["GO_K2"]))
    print(f"DONE GO_K2={go['GO_K2']}", flush=True)


if __name__ == "__main__":
    main()
