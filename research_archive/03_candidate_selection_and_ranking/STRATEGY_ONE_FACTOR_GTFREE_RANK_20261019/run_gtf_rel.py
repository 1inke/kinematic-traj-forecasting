#!/usr/bin/env python3
from __future__ import annotations
import os
for _k,_v in (("OMP_NUM_THREADS","2"),("MKL_NUM_THREADS","2"),("OPENBLAS_NUM_THREADS","2"),("NUMEXPR_NUM_THREADS","2")):
    os.environ.setdefault(_k,_v)
import json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXEC = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXEC))

from gtf_select_rel import select_batch_rel
from k0.data_io import load_bundle, load_kin, load_v1_selected_indices, array_sha
from k0.metrics import evaluate_set, check_W, gather, motion_stats, motion_gate
from k0.serialize import dump, eval_public, jsonable

OUT = HERE / "followup_rel"
OUT.mkdir(exist_ok=True)
SEEDS = (42, 43, 44)
K1_EXEC = EXEC / "k1_exec"
K0_RESULTS = EXEC / "results"

def now_hk():
    return datetime.now().astimezone().isoformat(timespec="seconds")

def portfolio(name, paths, gt, native, groups, indices=None, extra=None):
    ev = evaluate_set(paths, gt, native, groups, do_cluster=False)
    w = check_W(ev, groups)
    blk = dict(name=name, eval=eval_public(ev), W=jsonable(w), W_pass=bool(w["pass_"]))
    if indices is not None:
        blk["indices_sha"] = array_sha(indices.astype(np.int64))
    if extra:
        blk.update(extra)
    return blk

def slim(blk):
    e = blk["eval"]; ch=e.get("change",{}); ov=e.get("overall",{})
    return dict(
        change_good=ch.get("good_count"), change_cov=ch.get("coverage"),
        change_minADE=ch.get("minADE"), change_minFDE=ch.get("minFDE"),
        change_minJ=ch.get("mean_minJ"), change_motion=ch.get("motion_gate",{}).get("status"),
        overall_good=ov.get("good_count"), overall_minADE=ov.get("minADE"),
        overall_minFDE=ov.get("minFDE"), overall_motion=ov.get("motion_gate",{}).get("status"),
        W_pass=blk["W_pass"], n_swaps_mean=blk.get("n_swaps_mean"),
    )

def run_seed(seed, data):
    out_dir = OUT / f"seed{seed}" / "outer"
    out_dir.mkdir(parents=True, exist_ok=True)
    ix = data["split"]["outer"]
    native = data["native"][ix]; gt=data["gt"][ix]; groups=data["groups"][ix]
    ids = data["sample_id"][ix]
    kin,_ = load_kin("outer", seed, data)
    pool40 = np.concatenate([native, kin], axis=1)
    print(f"[REL seed{seed}] select", flush=True)
    t0=time.time()
    idx, traces, rejects = select_batch_rel(pool40, native, motion_stats, motion_gate, 100)
    wall=time.time()-t0
    np.save(out_dir/"gtf_rel_indices.npy", idx)
    ports = {
        "native20": portfolio("native20", native, gt, native, groups),
        "GTF_REL": portfolio("GTF_REL", gather(pool40, idx), gt, native, groups, indices=idx,
            extra=dict(n_swaps_mean=float(np.mean([len(t) for t in traces])), rejects=rejects, wall_s=wall)),
    }
    # attach prior GTF / K1S / K0C from parent results if present
    parent = HERE / "results" / f"seed{seed}" / "outer" / "portfolios.json"
    if parent.exists():
        prev = json.loads(parent.read_text(encoding="utf-8"))
        for k in ("GTF","K1_S","K0_C","v1_cond_boost_11_9"):
            if k in prev:
                ports[k] = prev[k]
    dump(out_dir/"portfolios.json", ports)
    summary = {k: slim(v) for k,v in ports.items()}
    dump(out_dir/"summary.json", dict(seed=seed, summary=summary, wall_s=wall, rejects=rejects))
    print(f"[REL seed{seed}] done W={ports['GTF_REL']['W_pass']} cg={summary['GTF_REL']['change_good']} swaps={summary['GTF_REL']['n_swaps_mean']:.2f}", flush=True)
    return dict(seed=seed, summary=summary, wall_s=wall, rejects=rejects)

def breakthrough(all_s):
    clauses={}; ok=True
    for s in all_s:
        g=s["summary"]["GTF_REL"]; n=s["summary"]["native20"]
        w=bool(g["W_pass"])
        ade=g["change_minADE"]<=n["change_minADE"]*0.95+1e-15
        fde=g["change_minFDE"]<=n["change_minFDE"]*0.95+1e-15
        clauses[str(s["seed"])]=dict(W_pass=w, change_ade5=ade, change_fde5=fde,
            change_good=g["change_good"], change_minADE=g["change_minADE"], change_minFDE=g["change_minFDE"])
        ok &= w and ade and fde
    return dict(BREAKTHROUGH=bool(ok), clauses=clauses, arm="GTF_REL")

def write_analysis(all_s, bt):
    lines=["# RESULT_ANALYSIS ? FOLLOWUP GTF_REL (relative set-NN)", "", f"- time: {now_hk()} (Asia/Tokyo)",
           f"- dir: `{OUT}`", "- factor: rel=min_D(l)-min_D(r)>0 + motion; no HGB/GT/teacher", "",
           "| seed | arm | change_good | change_minADE | change_minFDE | W_pass | n_swaps |",
           "|---:|---|---:|---:|---:|---|---:|"]
    for s in all_s:
        for arm,row in s["summary"].items():
            lines.append(f"| {s['seed']} | {arm} | {row.get('change_good')} | {row.get('change_minADE')} | {row.get('change_minFDE')} | {row.get('W_pass')} | {row.get('n_swaps_mean')} |")
    lines += ["", f"**BREAKTHROUGH = {bt['BREAKTHROUGH']}**", "", "```json", json.dumps(bt,indent=2), "```"]
    (OUT/"RESULT_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    dump(OUT/"RESULT_ANALYSIS.json", dict(summaries=all_s, breakthrough=bt, time=now_hk()))
    dump(OUT/"BREAKTHROUGH.json", bt)
    # also update parent RESULT with followup pointer
    parent_md = HERE/"RESULT_ANALYSIS.md"
    extra = f"\n\n## Follow-up GTF_REL\n\nSee `{OUT}/RESULT_ANALYSIS.md`. BREAKTHROUGH={bt['BREAKTHROUGH']}.\n"
    if parent_md.exists():
        txt = parent_md.read_text(encoding="utf-8")
        if "Follow-up GTF_REL" not in txt:
            parent_md.write_text(txt + extra, encoding="utf-8")

def main():
    print(f"GTF_REL start {now_hk()}", flush=True)
    data=load_bundle(); all_s=[]
    for seed in SEEDS:
        all_s.append(run_seed(seed, data))
    bt=breakthrough(all_s)
    write_analysis(all_s, bt)
    dump(OUT/"DONE.json", dict(time=now_hk(), BREAKTHROUGH=bt["BREAKTHROUGH"]))
    print(f"GTF_REL DONE BREAKTHROUGH={bt['BREAKTHROUGH']}", flush=True)

if __name__=="__main__":
    main()
