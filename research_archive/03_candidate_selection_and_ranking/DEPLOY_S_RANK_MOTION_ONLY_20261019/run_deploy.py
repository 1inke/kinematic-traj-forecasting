#!/usr/bin/env python3
"""DEPLOY_S_RANK_MOTION_ONLY: A=locked K1-S; B=S rank motion-only; C=I rank motion-only.
No retrain. Outer seeds 42/43/44 only. Quiet THREADS=2.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "2"

import json
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXEC = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018")
K1_EXEC = EXEC / "k1_exec"
OUT = HERE
SEEDS = (42, 43, 44)
N_NAT = 20

sys.path.insert(0, str(EXEC))
from k0.data_io import load_bundle, load_kin, array_sha
from k0.metrics import evaluate_set, check_W, gather
from k0.serialize import dump, eval_public, jsonable
from k1.features113 import build_pool_features_22, pairwise_D_matrix, action_feature_113
from k1.sampling import enumerate_all_legal_actions
from k1.train_hgb import load_arm, predict_arm
from k0.metrics import motion_stats, motion_gate

from select_motion_only import select_batch


def now_hkt():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S+09:00")


def prepare_pool(native, kin):
    return np.concatenate([native, kin], axis=1)


def eval_indices(pool40, indices, gt, native, groups):
    paths = gather(pool40, indices)
    ev = evaluate_set(paths, gt, native, groups, do_cluster=False)
    w = check_W(ev, groups)
    return paths, ev, w


def summarize_port(ev, w, extra=None):
    ch = ev["change"]
    ov = ev["overall"]
    rec = dict(
        W_pass=bool(w["pass_"]),
        W=jsonable(w),
        eval=eval_public(ev),
        change_good=int(ch["good_count"]),
        change_cov=float(ch["coverage"]),
        change_minADE=float(ch["minADE"]),
        change_minFDE=float(ch["minFDE"]),
        native_change_good=int(ch["native_good_count"]),
        native_change_minADE=float(ch["native_minADE"]),
        native_change_minFDE=float(ch["native_minFDE"]),
        overall_cov=float(ov["coverage"]),
        overall_minADE=float(ov["minADE"]),
        overall_minFDE=float(ov["minFDE"]),
        change_ade_improve_pct=float(
            (ch["native_minADE"] - ch["minADE"]) / ch["native_minADE"] * 100.0
        ),
        change_fde_improve_pct=float(
            (ch["native_minFDE"] - ch["minFDE"]) / ch["native_minFDE"] * 100.0
        ),
    )
    if extra:
        rec.update(extra)
    return rec


def load_k0c_summary(seed, pool40, gt, native, groups):
    idx_path = EXEC / "results" / f"seed{seed}" / "outer" / "k0c_indices.npy"
    w_path = EXEC / "results" / f"seed{seed}" / "outer" / "k0c_W.json"
    out = dict(src=str(idx_path))
    if w_path.exists():
        out["W_file"] = json.loads(w_path.read_text(encoding="utf-8"))
    if idx_path.exists():
        idx = np.load(idx_path)
        _, ev, w = eval_indices(pool40, idx, gt, native, groups)
        out["recomputed"] = summarize_port(ev, w, dict(indices_sha=array_sha(idx)))
    return out


def load_arm_A_from_k1(seed, pool40, gt, native, groups):
    """Locked baseline: existing K1-S deploy; do not re-run selector."""
    pred = K1_EXEC / f"seed{seed}" / "predictions" / "outer"
    idx_path = pred / "S_indices.npy"
    done_path = pred / "done.json"
    assert idx_path.exists(), f"missing {idx_path}"
    idx = np.load(idx_path)
    _, ev, w = eval_indices(pool40, idx, gt, native, groups)
    extra = dict(
        arm="A_K1S_locked",
        indices_sha=array_sha(idx),
        indices_path=str(idx_path),
        n_swaps_mean=None,
    )
    # try n_swaps from traces
    nsw = K1_EXEC / f"seed{seed}" / "action_traces" / "outer_S_n_swaps.npy"
    if nsw.exists():
        arr = np.load(nsw)
        extra["n_swaps_mean"] = float(arr.mean())
    if done_path.exists():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        ports = done.get("ports", {})
        if "S" in ports:
            extra["k1_done_W_pass"] = ports["S"].get("W_pass")
            extra["k1_done_n_swaps_mean"] = ports["S"].get("n_swaps_mean")
    return idx, summarize_port(ev, w, extra)


def run_seed(seed, data):
    seed_out = OUT / "results" / f"seed{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)
    marker = seed_out / "SEED_DONE.json"
    if marker.exists():
        print(f"[seed{seed}] already DONE", flush=True)
        return json.loads(marker.read_text(encoding="utf-8"))

    ix = data["split"]["outer"]
    native = data["native"][ix]
    gt = data["gt"][ix]
    groups = data["groups"][ix]
    past = data["past"][ix]
    kin, kin_src = load_kin("outer", seed, data)
    assert kin is not None, f"no KIN outer seed{seed}"
    kin = kin.astype(np.float64)
    pool40 = prepare_pool(native, kin)
    feat40 = build_pool_features_22(past, native, kin)
    print(f"[seed{seed}] cache D N={len(native)}", flush=True)
    D_cache = [pairwise_D_matrix(pool40[b]) for b in range(len(native))]

    ports = {}

    # native
    nat_idx = np.tile(np.arange(N_NAT), (len(native), 1))
    _, ev_n, w_n = eval_indices(pool40, nat_idx, gt, native, groups)
    ports["native20"] = summarize_port(ev_n, w_n, dict(arm="native20"))

    # A locked
    print(f"[seed{seed}] arm A locked K1-S", flush=True)
    idx_A, rec_A = load_arm_A_from_k1(seed, pool40, gt, native, groups)
    np.save(OUT / "indices" / f"seed{seed}_A_S_indices.npy", idx_A)
    ports["A_K1S"] = rec_A

    models_dir = K1_EXEC / f"seed{seed}" / "models"

    # B: S motion-only
    print(f"[seed{seed}] arm B S motion-only (no pred_gate)", flush=True)
    t0 = time.time()
    models_S = load_arm(models_dir, "S", seed)
    idx_B, traces_B, rej_B = select_batch(
        pool40, native, feat40, models_S, D_cache,
        action_feature_113, enumerate_all_legal_actions, predict_arm,
        pairwise_D_matrix, motion_stats, motion_gate,
        progress_every=100, tag="B-S",
    )
    _, ev_B, w_B = eval_indices(pool40, idx_B, gt, native, groups)
    np.save(OUT / "indices" / f"seed{seed}_B_S_motion_indices.npy", idx_B)
    np.save(seed_out / "B_n_swaps.npy", np.array([len(t) for t in traces_B], dtype=np.int16))
    dump(seed_out / "B_rejects.json", rej_B)
    ports["B_S_motion"] = summarize_port(ev_B, w_B, dict(
        arm="B_S_motion",
        indices_sha=array_sha(idx_B),
        n_swaps_mean=float(np.mean([len(t) for t in traces_B])),
        rejects=rej_B,
        wall_s=time.time() - t0,
    ))

    # C: I motion-only (optional cheap)
    print(f"[seed{seed}] arm C I motion-only (no pred_gate)", flush=True)
    t1 = time.time()
    models_I = load_arm(models_dir, "I", seed)
    idx_C, traces_C, rej_C = select_batch(
        pool40, native, feat40, models_I, D_cache,
        action_feature_113, enumerate_all_legal_actions, predict_arm,
        pairwise_D_matrix, motion_stats, motion_gate,
        progress_every=100, tag="C-I",
    )
    _, ev_C, w_C = eval_indices(pool40, idx_C, gt, native, groups)
    np.save(OUT / "indices" / f"seed{seed}_C_I_motion_indices.npy", idx_C)
    np.save(seed_out / "C_n_swaps.npy", np.array([len(t) for t in traces_C], dtype=np.int16))
    dump(seed_out / "C_rejects.json", rej_C)
    ports["C_I_motion"] = summarize_port(ev_C, w_C, dict(
        arm="C_I_motion",
        indices_sha=array_sha(idx_C),
        n_swaps_mean=float(np.mean([len(t) for t in traces_C])),
        rejects=rej_C,
        wall_s=time.time() - t1,
    ))

    # K0-C reference
    ports["K0C_ref"] = load_k0c_summary(seed, pool40, gt, native, groups)

    rec = dict(
        seed=seed, N=int(len(native)), kin_src=kin_src,
        ports=ports, finished_at=now_hkt(),
    )
    dump(marker, rec)
    dump(seed_out / "ports.json", ports)
    print(f"[seed{seed}] DONE W: A={rec_A['W_pass']} B={ports['B_S_motion']['W_pass']} "
          f"C={ports['C_I_motion']['W_pass']}", flush=True)
    return rec


def breakthrough_check(seed_recs):
    """Hard door: all 3 outer seeds B pass W (treatment). Also report A/C."""
    arms = {}
    for arm_key in ("A_K1S", "B_S_motion", "C_I_motion"):
        passes = []
        rows = []
        for s, rec in seed_recs.items():
            p = rec["ports"][arm_key]
            passes.append(bool(p["W_pass"]))
            rows.append(dict(
                seed=s,
                W_pass=bool(p["W_pass"]),
                change_good=p["change_good"],
                change_cov=p["change_cov"],
                change_minADE=p["change_minADE"],
                change_minFDE=p["change_minFDE"],
                change_ade_improve_pct=p["change_ade_improve_pct"],
                change_fde_improve_pct=p["change_fde_improve_pct"],
                n_swaps_mean=p.get("n_swaps_mean"),
            ))
        arms[arm_key] = dict(all_W=all(passes), per_seed=rows)
    # Breakthrough only for treatment B
    bt = bool(arms["B_S_motion"]["all_W"])
    return dict(
        BREAKTHROUGH=bt,
        rule="outer 3 seeds B_S_motion all pass check_W (includes change ADE/FDE >=5% improve)",
        arms=arms,
        decided_at=now_hkt(),
    )


def write_analysis(seed_recs, bt):
    lines = []
    lines.append("# DEPLOY_S_RANK_MOTION_ONLY — RESULT_ANALYSIS")
    lines.append("")
    lines.append(f"- time: {now_hkt()} (Asia/Tokyo)")
    lines.append(f"- dir: `{OUT}`")
    lines.append("- factor: remove pred_gate; keep true kinematics ≤1.01 native; S (B) / I (C) rank")
    lines.append(f"- **BREAKTHROUGH={'true' if bt['BREAKTHROUGH'] else 'false'}**")
    lines.append("")
    lines.append("## Outer summary")
    lines.append("")
    lines.append("| seed | arm | W | change_good | change_cov | change_ADE | change_FDE | ADE↑% | FDE↑% | swaps |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in SEEDS:
        ports = seed_recs[s]["ports"]
        for key, label in (
            ("native20", "native"),
            ("A_K1S", "A_K1S"),
            ("B_S_motion", "B_S_mot"),
            ("C_I_motion", "C_I_mot"),
        ):
            p = ports[key]
            if key == "native20":
                ade_pct = 0.0
                fde_pct = 0.0
            else:
                ade_pct = p["change_ade_improve_pct"]
                fde_pct = p["change_fde_improve_pct"]
            lines.append(
                f"| {s} | {label} | {p['W_pass']} | {p['change_good']} | {p['change_cov']:.4f} | "
                f"{p['change_minADE']:.5f} | {p['change_minFDE']:.5f} | {ade_pct:.2f} | {fde_pct:.2f} | "
                f"{p.get('n_swaps_mean', '—')} |"
            )
        k0 = ports.get("K0C_ref", {}).get("recomputed")
        if k0:
            lines.append(
                f"| {s} | K0C | {k0['W_pass']} | {k0['change_good']} | {k0['change_cov']:.4f} | "
                f"{k0['change_minADE']:.5f} | {k0['change_minFDE']:.5f} | "
                f"{k0['change_ade_improve_pct']:.2f} | {k0['change_fde_improve_pct']:.2f} | — |"
            )
    lines.append("")
    lines.append("## B vs A (treatment vs locked K1-S)")
    lines.append("")
    for s in SEEDS:
        a = seed_recs[s]["ports"]["A_K1S"]
        b = seed_recs[s]["ports"]["B_S_motion"]
        lines.append(
            f"- seed{s}: Δchange_good={b['change_good']-a['change_good']}, "
            f"Δchange_ADE={b['change_minADE']-a['change_minADE']:+.5f}, "
            f"Δchange_FDE={b['change_minFDE']-a['change_minFDE']:+.5f}, "
            f"W A={a['W_pass']}→B={b['W_pass']}, "
            f"swaps {a.get('n_swaps_mean')}→{b.get('n_swaps_mean')}"
        )
    lines.append("")
    lines.append("## Judgment")
    lines.append("")
    if bt["BREAKTHROUGH"]:
        lines.append("B passes W on all outer seeds. pred_gate was the binding block under frozen S.")
        lines.append("Next: productionize motion-only + calibrated stop / margin rank (not more weak ablations).")
    else:
        lines.append(
            "Removing pred_gate alone does **not** clear W on all outer seeds "
            "(or fails protect/error clauses). S rank without the gate is insufficient."
        )
        lines.append("")
        lines.append("### Next scientific question (standby — no further weak variants)")
        lines.append("")
        lines.append(
            "Given motion-feasible actions exist and K0-C accepts many of them, but neither "
            "pred_gate-on (A) nor pred_gate-off+S-rank (B) reaches W: **what stop / accept rule "
            "is learnable without GT?** Candidate design: *calibrated stop* — predict whether "
            "one more motion-PASS swap improves set margin with a holdout-calibrated threshold "
            "(or pairwise preference over motion-PASS actions), not another ON/OFF of the same gate."
        )
        lines.append("Do **not** open another threshold/rank-key tweak as a third/fourth weak variant.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- `BREAKTHROUGH.json`, `RESULT_ANALYSIS.json`, `results/seed{{42,43,44}}/`")
    lines.append(f"- indices under `indices/`")
    text = "\n".join(lines) + "\n"
    (OUT / "RESULT_ANALYSIS.md").write_text(text, encoding="utf-8")
    dump(OUT / "RESULT_ANALYSIS.json", dict(breakthrough=bt, seeds={str(k): v for k, v in seed_recs.items()}))
    dump(OUT / "BREAKTHROUGH.json", bt)


def main():
    print(f"START {now_hkt()}", flush=True)
    data = load_bundle()
    seed_recs = {}
    for seed in SEEDS:
        seed_recs[seed] = run_seed(seed, data)
    bt = breakthrough_check(seed_recs)
    write_analysis(seed_recs, bt)
    # overnight log append
    log_path = EXEC / "OVERNIGHT_LOG.md"
    line = (
        f"\n- {now_hkt()}: DEPLOY_S_RANK_MOTION_ONLY_20261019 done; "
        f"BREAKTHROUGH={'true' if bt['BREAKTHROUGH'] else 'false'}; "
        f"B_all_W={bt['arms']['B_S_motion']['all_W']}; "
        f"see {OUT}\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)
    dump(OUT / "STATUS.json", dict(
        status="DONE", BREAKTHROUGH=bt["BREAKTHROUGH"], finished_at=now_hkt(),
        standby=(not bt["BREAKTHROUGH"]),
    ))
    print(f"DONE BREAKTHROUGH={bt['BREAKTHROUGH']}", flush=True)


if __name__ == "__main__":
    main()
