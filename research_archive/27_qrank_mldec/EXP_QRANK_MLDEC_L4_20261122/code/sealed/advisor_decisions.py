"""Seed-paired C0 screening and fixed metric five-fold confirmation."""
import json
from pathlib import Path

import numpy as np

from advisor_model import ARMS
from coupling_analysis import TrackBootstrap
from prepare_inputs import digest
from run_b0 import write_json

SEEDS = (42,43,44)
FOLDS = ("eth","zara01","hotel","zara02","univ")


def run_path(out,arm,fold,seed):
    stage = "stage1" if fold=="C0" else "stage2"
    key = f"{arm}_s{seed}" if fold=="C0" else f"{arm}_{fold}_s{seed}"
    return out/stage/"runs"/key


def results(out,arm,fold):
    return [json.loads((run_path(out,arm,fold,s)/"RESULT.json").read_bytes()) for s in SEEDS]


def effects(lead,base,metric,group="overall"):
    a = np.array([r["dev"][group][metric] for r in lead])
    b = np.array([r["dev"][group][metric] for r in base])
    return (b-a)/np.maximum(b,1e-12)


def favorable(x, threshold):
    return bool(x.mean()>=threshold and (x>0).all())


def paired(out,baseline,arm):
    ref, treatment = [],[]
    for s in SEEDS:
        for name,store in ((baseline,ref),(arm,treatment)):
            p = run_path(out,name,"C0",s)/"dev_metrics.npz"
            receipt = json.loads((p.parent/"RESULT.json").read_bytes())
            assert digest(p)==receipt["artifacts"][p.name]
            with np.load(p,allow_pickle=False) as z:
                store.append({k:z[k] for k in z.files})
    ids = ref[0]["ids"]
    assert all(np.array_equal(ids,r["ids"]) for r in (*ref,*treatment))
    cols = ("minADE","minFDE","J","Top1_ADE","Top1_FDE","rank","ES","brier_minFDE")
    tracks = np.asarray([x.rsplit("|",1)[0] for x in ids])
    output = {}
    for name in ("overall","hard","stationary","steady","other","without_zara02","hard_without_zara02"):
        mask = ref[0]["hard"]&ref[0]["without_zara02"] if name=="hard_without_zara02" else ref[0][name]
        helper = TrackBootstrap(tracks[mask])
        unit = np.ones(mask.sum())
        deltas = {k:np.mean([t[k][mask].astype(float)-b[k][mask].astype(float) for b,t in zip(ref,treatment)],axis=0)
                  for k in cols}
        output[name] = dict(population=helper.population,delta=helper.summarize({k:(v,unit) for k,v in deltas.items()}))
    return output


def screen(out):
    all_runs = {a:results(out,a,"C0") for a in ARMS}
    base = all_runs["B0"]

    def compare(a,k="minFDE",reference="B0",group="overall"):
        return effects(all_runs[a],all_runs[reference],k,group)

    compare_r = compare("R2",reference="R3")
    rpass = (favorable(compare("R2"),.015) and favorable(compare_r,.010) and
             compare("R2","minADE").mean()>=0 and compare("R2",group="hard").mean()>=-.03)
    first = {k:np.array([v["mechanism"]["first_pass_minFDE"] for v in all_runs[k]]) for k in ("R1","R2")}
    bv = np.array([r["dev"]["overall"]["minFDE"] for r in base])
    s_pass,audit_s = [],{}
    for arm in ("S2","S4"):
        for metric,other,cut in (("Top1_FDE","brier_minFDE",.03),("brier_minFDE","Top1_FDE",.02)):
            gain = compare(arm,metric)
            value = (favorable(gain,cut) and favorable(compare(arm,metric,"S3"),.015) and
                     favorable(compare(arm,metric,"S1"),.015) and compare(arm,other).mean()>=0 and
                     compare(arm,"minFDE").mean()>=-.01)
            audit_s[f"{arm}/{metric}"] = bool(value)
            if value:
                s_pass.append((float(gain.mean()),arm,metric))
    gpass = (favorable(compare("G2"),.015) and favorable(compare("G2",reference="G1"),.01) and
             favorable(compare("G2",reference="S3"),.01) and compare("G2",group="hard").mean()>=-.03)
    saved = {a:[r["dev"] for r in rs] for a,rs in all_runs.items()}
    current = {a:{g:{k:compare(a,k,group=g).tolist() for k in ("minADE","minFDE","Top1_FDE","brier_minFDE")}
                  for g in ("overall","hard","without_zara02")} for a in ARMS}
    initial = {a:(v/bv-1).tolist() for a,v in first.items()}
    markers = {"R-PASS":bool(rpass),"R-INFO":favorable(compare("R3"),.015),
               "R-DS":bool(all(v<=.03 for v in initial["R2"]) and all(v>.10 for v in initial["R1"])),
               "S-PASS":bool(s_pass),"S-GEOM":any(favorable(compare(a),.015) for a in ("S2","S4")),
               "S-FDE":favorable(compare("S1"),.015) or favorable(compare("S1","Top1_FDE"),.03),
               "G-POOL":bool(np.mean([1-r["pool60"]["overall"]["minFDE"]/bv[i]
                                      for i,r in enumerate(all_runs["G1"])])>=.10),
               "G-NONLEARN":favorable(compare("G1"),.015),"G-PASS":bool(gpass),
               "G-INTER":favorable(compare("G2",reference="G2-intra"),.01)}
    summaries = {a:paired(out,"B0",a) for a in ARMS if a!="B0"}
    controls = {"R2/R3":paired(out,"R3","R2"),"S2/S3":paired(out,"S3","S2"),
                "S4/S3":paired(out,"S3","S4"),"S4/S2":paired(out,"S2","S4"),
                "G2/G1":paired(out,"G1","G2"),"G2/G2-intra":paired(out,"G2-intra","G2")}
    selected = {}
    if rpass:
        selected["R2"]="minFDE"
    if s_pass:
        _,a,k = sorted(s_pass,key=lambda row:(-row[0],row[1],row[2]))[0]
        selected[a]=k
    if gpass:
        selected["G2"]="minFDE"
    write_json(out/"stage1/RESULT.json",dict(status="COMPLETE",markers=markers,summary=saved,
        relative_gain_by_seed=current,paired_bootstrap=summaries,controls=controls,sorting_cases=audit_s,
        pool60={k:[r["pool60"] for r in all_runs[k]] for k in ("G1","G2","G2-intra")}))
    write_json(out/"stage1/MECH_R.json",dict(first_pass_relative_harm=initial,mechanism={
        a:[r["mechanism"] for r in all_runs[a]] for a in ("R1","R2","R3")}))
    write_json(out/"stage1/G1_compression.json",[r["compression"] for r in all_runs["G1"]])
    (out/"stage2").mkdir(exist_ok=True)
    write_json(out/"stage2/ADMISSION.json",dict(status="LOCKED",always=["B0"],admitted=selected,
        criteria=markers,for_each_direction_limit=1,fold_order=FOLDS))
    return selected


def confirm(out,admitted):
    base = {f:results(out,"B0",f) for f in FOLDS}
    table = {}
    for arm,metric in admitted.items():
        rows = {}
        for fold in FOLDS:
            treatment = results(out,arm,fold)
            rows[fold] = {k:effects(treatment,base[fold],k).tolist() for k in sorted({"minADE","minFDE",metric})}
        means = {k:float(np.mean([np.mean(row[k]) for row in rows.values()])) for k in rows["eth"]}
        wins = sum(float(np.mean(row[metric]))>0 for row in rows.values())
        eth = float(np.mean(rows["eth"][metric]))
        qualifies = means[metric]>=.01 and wins>=4 and eth>0 and min(means["minADE"],means["minFDE"])>=-.01
        table[arm] = {"primary_metric":metric,"mean":means,"folds_improving":wins,"F-ETH":eth,
                      "F-PASS":bool(qualifies),"foldwise_relative_gain":rows}
    write_json(out/"stage2/RESULT.json",dict(status="COMPLETE",arms=table,
        B0={f:[r["dev"] for r in rows] for f,rows in base.items()}))
    return table

