#!/usr/bin/env python3
"""One-factor GT-free rank vs native / V1 / K1-S / K0-C. Outer 3 seeds. Quiet CPU."""
from __future__ import annotations
import os
for _k, _v in (
    ("OMP_NUM_THREADS", "2"),
    ("MKL_NUM_THREADS", "2"),
    ("OPENBLAS_NUM_THREADS", "2"),
    ("NUMEXPR_NUM_THREADS", "2"),
    ("VECLIB_MAXIMUM_THREADS", "2"),
):
    os.environ.setdefault(_k, _v)

import json
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXEC = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXEC))

from gtf_select import select_batch
from k0.data_io import load_bundle, load_kin, load_v1_selected_indices, array_sha
from k0.metrics import evaluate_set, check_W, gather, motion_stats, motion_gate
from k0.serialize import dump, eval_public, jsonable

OUT = HERE
SEEDS = (42, 43, 44)
K1_EXEC = EXEC / "k1_exec"
K0_RESULTS = EXEC / "results"


def now_hk():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def slim_row(ev_name, ports_blk):
    e = ports_blk["eval"]
    ch = e.get("change", {})
    ov = e.get("overall", {})
    return dict(
        change_good=ch.get("good_count"),
        change_cov=ch.get("coverage"),
        change_minADE=ch.get("minADE"),
        change_minFDE=ch.get("minFDE"),
        change_minJ=ch.get("mean_minJ"),
        change_motion=ch.get("motion_gate", {}).get("status"),
        overall_good=ov.get("good_count"),
        overall_minADE=ov.get("minADE"),
        overall_minFDE=ov.get("minFDE"),
        overall_motion=ov.get("motion_gate", {}).get("status"),
        W_pass=ports_blk["W_pass"],
        n_swaps_mean=ports_blk.get("n_swaps_mean"),
    )


def portfolio(name, paths, gt, native, groups, indices=None, extra=None):
    ev = evaluate_set(paths, gt, native, groups, do_cluster=False)
    w = check_W(ev, groups)
    blk = dict(name=name, eval=eval_public(ev), W=jsonable(w), W_pass=bool(w["pass_"]))
    if indices is not None:
        blk["indices_sha"] = array_sha(indices.astype(np.int64))
    if extra:
        blk.update(extra)
    return blk


def load_k1s(seed, split="outer"):
    p = K1_EXEC / f"seed{seed}" / "predictions" / split / "S_indices.npy"
    if not p.exists():
        return None, str(p)
    return np.load(p).astype(np.int64), str(p)


def load_k0c(seed, split="outer"):
    p = K0_RESULTS / f"seed{seed}" / split / "k0c_indices.npy"
    if not p.exists():
        return None, str(p)
    return np.load(p).astype(np.int64), str(p)


def run_seed(seed, data):
    out_dir = OUT / "results" / f"seed{seed}" / "outer"
    out_dir.mkdir(parents=True, exist_ok=True)
    ix = data["split"]["outer"]
    native = data["native"][ix]
    gt = data["gt"][ix]
    groups = data["groups"][ix]
    ids = data["sample_id"][ix]
    kin, kin_src = load_kin("outer", seed, data)
    assert kin is not None
    pool40 = np.concatenate([native, kin], axis=1)
    print(f"[seed{seed}] outer N={len(ix)} kin={kin_src}", flush=True)

    ports = {}
    ports["native20"] = portfolio("native20", native, gt, native, groups)

    v1_idx, v1_src = load_v1_selected_indices("outer", seed)
    if v1_idx is not None:
        ports["v1_cond_boost_11_9"] = portfolio(
            "v1_cond_boost_11_9", gather(pool40, v1_idx), gt, native, groups, indices=v1_idx,
            extra=dict(src=v1_src),
        )

    k1s, k1s_src = load_k1s(seed)
    if k1s is not None:
        ports["K1_S"] = portfolio(
            "K1_S", gather(pool40, k1s), gt, native, groups, indices=k1s,
            extra=dict(src=k1s_src, n_swaps_mean=float(np.mean([20 - np.sum(row < 20) for row in k1s]))),
        )

    k0c, k0c_src = load_k0c(seed)
    if k0c is not None:
        ports["K0_C"] = portfolio(
            "K0_C", gather(pool40, k0c), gt, native, groups, indices=k0c,
            extra=dict(src=k0c_src, tag="ORACLE_DIAGNOSTIC_ONLY",
                       n_swaps_mean=float(np.mean([20 - np.sum(row < 20) for row in k0c]))),
        )

    t0 = time.time()
    print(f"[seed{seed}] GTF select...", flush=True)
    gtf_idx, traces, rejects = select_batch(
        pool40, native, motion_stats, motion_gate, progress_every=100,
    )
    wall = time.time() - t0
    np.save(out_dir / "gtf_indices.npy", gtf_idx)
    with (out_dir / "gtf_action_traces.jsonl").open("w", encoding="utf-8") as f:
        for b, tr in enumerate(traces):
            f.write(json.dumps(dict(window=b, sample_id=str(ids[b]), actions=tr), ensure_ascii=False) + "\n")
    ports["GTF"] = portfolio(
        "GTF", gather(pool40, gtf_idx), gt, native, groups, indices=gtf_idx,
        extra=dict(
            n_swaps_mean=float(np.mean([len(t) for t in traces])),
            rejects=rejects,
            wall_s=wall,
            score="min_D_then_neg_ep_to_mean_then_slots",
        ),
    )
    dump(out_dir / "portfolios.json", ports)
    dump(out_dir / "rejects.json", rejects)
    summary = {k: slim_row(k, v) for k, v in ports.items()}
    dump(out_dir / "summary.json", dict(seed=seed, split="outer", summary=summary, wall_s=wall))
    print(f"[seed{seed}] done wall={wall:.1f}s GTF W={ports['GTF']['W_pass']} "
          f"cg={summary['GTF']['change_good']} swaps={summary['GTF']['n_swaps_mean']:.2f}", flush=True)
    return dict(seed=seed, summary=summary, wall_s=wall, rejects=rejects)


def breakthrough(all_summaries):
    """True iff all 3 outer seeds: GTF W_pass and change ADE/FDE >=5% vs native."""
    if len(all_summaries) < 3:
        return dict(BREAKTHROUGH=False, reason="incomplete")
    clauses = {}
    ok = True
    for s in all_summaries:
        seed = s["seed"]
        g = s["summary"]["GTF"]
        n = s["summary"]["native20"]
        w = bool(g["W_pass"])
        ade_ok = g["change_minADE"] <= n["change_minADE"] * 0.95 + 1e-15
        fde_ok = g["change_minFDE"] <= n["change_minFDE"] * 0.95 + 1e-15
        clauses[str(seed)] = dict(
            W_pass=w,
            change_ade5=ade_ok,
            change_fde5=fde_ok,
            change_good=g["change_good"],
            change_minADE=g["change_minADE"],
            change_minFDE=g["change_minFDE"],
            native_ade=n["change_minADE"],
            native_fde=n["change_minFDE"],
        )
        ok &= w and ade_ok and fde_ok
    return dict(BREAKTHROUGH=bool(ok), clauses=clauses)


def write_analysis(all_summaries, bt):
    lines = []
    lines.append("# RESULT_ANALYSIS ? STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019")
    lines.append("")
    lines.append(f"- time: {now_hk()} (Asia/Tokyo)")
    lines.append(f"- dir: `{OUT}`")
    lines.append("- factor: GT-free min_D hole-fill rank + real motion gate; **no** HGB / GT / K0 teacher")
    lines.append("")
    lines.append("## Outer 3-seed table")
    lines.append("")
    lines.append("| seed | arm | change_good | change_minADE | change_minFDE | W_pass | n_swaps |")
    lines.append("|---:|---|---:|---:|---:|---|---:|")
    for s in all_summaries:
        seed = s["seed"]
        for arm, row in s["summary"].items():
            lines.append(
                f"| {seed} | {arm} | {row.get('change_good')} | "
                f"{row.get('change_minADE')} | {row.get('change_minFDE')} | "
                f"{row.get('W_pass')} | {row.get('n_swaps_mean')} |"
            )
    lines.append("")
    lines.append("## vs native (GTF)")
    lines.append("")
    for s in all_summaries:
        g = s["summary"]["GTF"]
        n = s["summary"]["native20"]
        d_cg = g["change_good"] - n["change_good"]
        d_ade = (g["change_minADE"] / n["change_minADE"] - 1.0) * 100.0
        d_fde = (g["change_minFDE"] / n["change_minFDE"] - 1.0) * 100.0
        lines.append(
            f"- seed{s['seed']}: ?change_good={d_cg}, "
            f"ADE%={d_ade:.2f}, FDE%={d_fde:.2f}, W={g['W_pass']}, swaps={g['n_swaps_mean']:.3f}"
        )
    lines.append("")
    lines.append("## BREAKTHROUGH")
    lines.append("")
    lines.append(f"**BREAKTHROUGH = {bt['BREAKTHROUGH']}**")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(bt, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- K0-C is oracle diagnostic only (uses GT in selection).")
    lines.append("- K1-S uses learned set-marginal pred-gate; GTF removes that gate.")
    lines.append("- No generator retrain; no HGB hyperparam scan.")
    (OUT / "RESULT_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    dump(OUT / "RESULT_ANALYSIS.json", dict(summaries=all_summaries, breakthrough=bt, time=now_hk()))
    dump(OUT / "BREAKTHROUGH.json", bt)


def main():
    print(f"GTF start {now_hk()}", flush=True)
    data = load_bundle()
    all_s = []
    for seed in SEEDS:
        all_s.append(run_seed(seed, data))
    bt = breakthrough(all_s)
    write_analysis(all_s, bt)
    dump(OUT / "DONE.json", dict(time=now_hk(), BREAKTHROUGH=bt["BREAKTHROUGH"], n_seeds=len(all_s)))
    print(f"GTF DONE BREAKTHROUGH={bt['BREAKTHROUGH']}", flush=True)


if __name__ == "__main__":
    main()
