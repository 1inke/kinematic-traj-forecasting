"""Recompute the diagnostic and intervention decisions from saved derived arrays."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIAG, PROBE = (ROOT/"snapshot"/name for name in ("DIAG","PROBE"))
sys.path.insert(0,str(DIAG))
from coupling_analysis import BOOT, HARD, TrackBootstrap, grouping

ARMS = ("R-couple","R-detach","R-nogeom")
SEEDS = ("0","1","2")


def read(p):
    return json.loads(p.read_text("utf-8-sig"))


def load(p):
    with np.load(p,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def close(a,b,atol=1e-12):
    np.testing.assert_allclose(a,b,rtol=0,atol=atol)


def diagnostic():
    final=read(DIAG/"RESULT.json")
    audit=read(DIAG/"PREFLIGHT.json")
    groups=read(DIAG/"A_summary.json")
    misses=read(DIAG/"B_miss_geometry.json")
    passes=[]
    for seed, row in final["per_seed"].items():
        d=load(DIAG/f"A_forces_s{seed}.npz")
        np.testing.assert_array_equal(d["fit_index"],audit["fit_indices"])
        assert len(d["ids"])==15913==len(np.unique(d["ids"]))
        cls,geom=[list(d["terms"]).index(k) for k in ("cls","geom")]
        r=(d["quality_rank"]>=2)&(d["quality_rank"]<=6)
        hard=np.isin(d["maneuver"],HARD)
        steady=d["maneuver"]=="steady"

        def ratio(mask):
            vals=d["norm"][mask[:,None]&r].astype(np.float64)
            return float(vals[:,cls].sum()/vals[:,geom].sum())

        hard_d,steady_d=ratio(hard),ratio(steady)
        close(hard_d,row["K1"]["D"],atol=1e-8)
        close(steady_d,row["K4"]["D"],atol=1e-8)
        valid=d["P_valid"][hard]&r[hard]
        p=d["P"][hard]
        boot=TrackBootstrap(d["track_id"][hard])
        proj=boot.summarize({k:(np.where(valid,p[:,:,j],0.).astype(np.float64).sum(1),valid.sum(1))
                              for k,j in (("cls",cls),("geom",geom))})
        close(proj["cls"]["estimate"],row["K2"]["cls_P"],atol=1e-9)
        close(proj["geom"]["estimate"],row["K2"]["geom_P"],atol=1e-9)
        close(proj["cls"]["ci95"],row["K2"]["cls_P_ci95"],atol=1e-9)
        ix=np.arange(len(d["ids"]))
        k=d["fde_winner"]
        valid=d["P_valid"][ix,k]
        push=d["P"][ix,k,cls]
        blocked=(push<0)&(-push>=.5*d["P"][ix,k,geom])
        cover=d["covered"]
        uncov_rate=blocked[hard&valid&~cover].mean()
        cov_rate=blocked[hard&valid&cover].mean()
        close([uncov_rate,cov_rate],[row["K3"]["uncovered"],row["K3"]["covered"]])
        passes.append([hard_d>=1,proj["cls"]["estimate"]<0 and -proj["cls"]["estimate"]>=.5*proj["geom"]["estimate"]
                      and proj["cls"]["ci95"][1]<0,uncov_rate>=.5 and uncov_rate-cov_rate>=.15,steady_d<.5])
        for split in ("fit","dev"):
            for item in misses[seed][split].values():
                total=item["uncovered_population"]["windows"]
                assert sum(x["population"]["windows"] for x in item["categories"].values())==total
        assert groups[seed]["A1_A3"]["hard"]["population"]["windows"]==int(hard.sum())
    for i,n in enumerate(np.sum(passes,axis=0),1):
        m=final["flags"][f"K{i}"]
        assert m["seeds_passing"]==n and m["active"]==bool(n>=4)
    assert final["phase_two_triggered"] is final["flags"]["K1"]["active"]
    return len(passes)*15913


def ablation():
    pop=load(ROOT/"analysis_inputs/dev_population.npz")
    masks=grouping(pop["maneuver"],pop["c0_group"])
    masks.update({f"obs_{g}":pop["observation"]==g for g in ("stationary","steady","other")})
    output=read(PROBE/"RESULT.json")
    comparisons=output["decision"]
    results={}
    history={}
    cover={}
    raw_fields=("minADE","minFDE","top1_ADE","top1_FDE","winner_NLL","J_coverage","HPD90_coverage")
    for arm in ARMS:
        results[arm]={}
        for seed in SEEDS:
            folder=PROBE/f"runs/{arm}_s{seed}"
            value=read(folder/"RESULT.json")
            assert value==output["runs"][arm][seed]
            w=load(folder/"dev_metrics.npz")
            np.testing.assert_array_equal(w["ids"],pop["ids"])
            assert len(w["ids"])==4753 and np.array_equal(w["HPD90_coverage"],w["hpd_level"]<=.90)
            rows={}
            for g,m in masks.items():
                row={k:float(w[k][m].mean()) for k in raw_fields}
                row["median_spread"]=float(np.median(w["spread"][m]))
                row["ECE_J"]=float(np.abs(w["ece_bins"][m].sum(0)).sum()/(20*m.sum()))
                for k,x in row.items():
                    close(x,value["groups"][g][k]["estimate"])
                rows[g]=row
            results[arm][seed]=rows
            cover[arm,seed]=w["J_coverage"][masks["hard"]]
            audit=value.get("A1_A2_audit")
            if arm=="R-detach":
                assert audit["cls_geometric_norm_max"]==0
            elif arm=="R-couple":
                assert audit["cls_geometric_norm_max"]>0
            hist=read(folder/"TRAINING.json")["history"]
            assert len(hist)==64 and all(r["steps"]==i*340 and r["epoch"]==i for i,r in enumerate(hist,1))
            order=[r["order_sha256"] for r in hist]
            assert history.setdefault(seed,order)==order

    def series(arm,g,k):
        return np.asarray([results[arm][s][g][k] for s in SEEDS])

    changes={}
    for arm,ref in (("R-detach","R-couple"),("R-nogeom","R-detach")):
        out={}
        for g,metrics in comparisons["paired"][arm].items():
            out[g]={}
            for k,stored in metrics.items():
                delta=series(arm,g,k)-series(ref,g,k)
                mean=float(delta.mean())
                band=float(2*series("R-couple",g,k).std(ddof=1)/np.sqrt(3))
                signal=bool(abs(mean)>band and (np.all(delta>0) or np.all(delta<0)))
                close(delta,stored["deltas"])
                close([mean,band],[stored["mean"],stored["threshold"]])
                assert signal==stored["signal"]
                out[g][k]=(mean,signal)
        changes[arm]=out
    b0=series("R-couple","overall","minFDE")/output["B0_seed42_minFDE"]-1
    close(b0,comparisons["B0_comparison"])
    dd=changes["R-detach"]
    ng=changes["R-nogeom"]
    h=dd["hard"]
    overall=dd["overall"]
    stat=dd["obs_stationary"]["median_spread"]
    flags=dict(S0=bool(np.all(abs(b0)<=.05)),
               S1=h["J_coverage"][1] and h["J_coverage"][0]>0 and h["minFDE"][0]<=0 and
                   all(overall[k][0]<=.01*series("R-couple","overall",k).mean() for k in ("minADE","minFDE")),
               S2=any(overall[k][0]>0 and overall[k][1] for k in ("top1_FDE","winner_NLL")),
               S3=any(ng["overall"][k][0]>0 and ng["overall"][k][1] for k in ("top1_FDE","winner_NLL")),
               S4=stat[1] and stat[0]<0 and bool(np.all(series("R-detach","obs_stationary","median_spread")<
                                                          series("R-detach","obs_steady","median_spread"))))
    assert flags==comparisons["flags"]
    paired=np.mean([cover["R-detach",s].astype(float)-cover["R-couple",s] for s in SEEDS],axis=0)
    boot=TrackBootstrap(pop["track_id"][masks["hard"]])
    extra=boot.summarize({"hard_J_detach_minus_couple":(paired,np.ones(len(paired)))})
    return len(masks),extra


def main():
    manifest=read(ROOT/"ARTIFACT_MANIFEST.json")["files"]
    assert set(manifest)=={p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
                           if p.is_file() and p.name!="ARTIFACT_MANIFEST.json"}
    for p, h in manifest.items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    a=diagnostic()
    b,extra=ablation()
    print(json.dumps(dict(status="PASS",gradient_window_records=a,dev_window_records=9*4753,
                          bootstrap_draws=BOOT,dev_groups=b,epochs=576,updates=195840,
                          diagnostic_markers=4,probe_markers=5,exploratory_paired_trajectory_interval=extra)))


if __name__=="__main__":
    main()
