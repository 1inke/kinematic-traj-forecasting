"""Coordinate the fixed-budget C0 screen and conditional five-fold training."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

import advisor_study as queueing
from advisor_train import inputs
from prepare_inputs import digest
from qrank_decisions import FOLDS, SEEDS, confirm, evidence, matches, read, screen, short_compare
from qrank_model import FORMAL
from qrank_preflight import gates, latency
from run_b0 import publish, verify_inputs, write_json


ROOT = Path(__file__).resolve().parent
OLD = "ADVISOR_FIVEFOLD_20261122_031129"
BENCH = ("B0","D3-full","O-conf","O-risk","O-risk+QFL")
OPTIONAL = ("O-risk+E1","O-risk-full","O-qual-q","D2-full","O-risk-L2","D4-full","O-risk-L4")


def runner(use_kin):
    def launch(arm,fold,seed,out,bench):
        out.mkdir(parents=True,exist_ok=True)
        args = [sys.executable,"-u","-B",str(ROOT/"qrank_train.py"),
                "--arm",arm,"--fold",fold,"--seed",str(seed),"--out",str(out)]
        if bench: args.append("--bench")
        if not use_kin: args.append("--drop-kin")
        env = {**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
        with (out/"worker.log").open("a",encoding="utf-8") as log:
            return subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT)
    return launch


def dispatch(root,label,jobs,cap,until,bench=False,admission=None):
    end = queueing.queue(root,label,jobs,cap,until if admission is None else admission,until,bench=bench)
    if end["failed"] and not bench:
        raise RuntimeError(f"{label} failed: {end['failed']}")
    return end


def job(root,phase,arm,fold,seed):
    name = f"{arm}_s{seed}" if fold=="C0" else f"{arm}_{fold}_s{seed}"
    return arm,fold,seed,root/phase/"runs"/name


def trial(root,label,arms,cap,until):
    jobs = [(a,"C0",42+i%3,root/"benchmark"/label/f"{a}_{i}") for i,a in enumerate(arms)]
    state = dispatch(root,label,jobs,cap,until,bench=True)
    state["runs"] = [read(j[-1]/"RESULT.json") for j in jobs if queueing.usable(j[-1])]
    return state


def bench(root,until):
    solo = trial(root,"PF_T_SOLO",BENCH,1,until)
    if solo["failed"] or len(solo["runs"])!=len(BENCH):
        raise RuntimeError("PF-T single-lane benchmark failed")
    baseline = {x["contract"]["arm"]:x["total_training_seconds"] for x in solo["runs"]}
    choices,perf = [],{}
    for cap in (6,8,10):
        stage = trial(root,f"PF_T_{cap}",[BENCH[k%len(BENCH)] for k in range(cap)],cap,until)
        measurements = stage["runs"]
        # Benchmark receipts survive a runtime interruption. Epoch sums preserve an estimate on resume.
        wall = stage["seconds"] if stage["completed"]==cap else max((r["seconds"] for r in measurements),default=0.)
        slowdown = max((r["total_training_seconds"]/baseline[r["contract"]["arm"]] for r in measurements),
                       default=None)
        valid = len(measurements)==cap and not stage["failed"] and slowdown<=2.
        perf[str(cap)] = dict(wall=wall,epochs_per_hour=3600*cap/max(wall,.1),max_slowdown=slowdown,
                              valid=valid,receipt=stage)
        if valid:
            choices.append(cap)
    if not choices:
        # Retain the registered experiment when multi-lane benchmarking crosses the slowdown guard.
        chosen = 1
    elif 6 in choices:
        eligible = [n for n in choices if n==6 or perf[str(n)]["epochs_per_hour"]>=1.3*perf["6"]["epochs_per_hour"]]
        chosen = max(eligible,key=lambda n:perf[str(n)]["epochs_per_hour"])
    else:
        chosen = max(choices,key=lambda n:perf[str(n)]["epochs_per_hour"])
    times = baseline.copy()
    for r in perf.get(str(chosen),{}).get("receipt",{}).get("runs",[]):
        a = r["contract"]["arm"]
        times[a] = max(times[a],r["total_training_seconds"])
    return dict(chosen=chosen,measurements=perf,epoch_seconds=times,solo=solo)


def select_baseline(root,until):
    lock = root/"B0_REFERENCE.json"
    if lock.is_file():
        return read(lock)
    check = job(root,"stage2","B0","univ",42)
    dispatch(root,"PF_B0_UNIV",[check],1,until)
    ref = ROOT/"data/B0_REUSE"
    done = read(check[-1]/"RESULT.json")
    compare = read(ref/"B0_univ_s42/RESULT.json")
    measured = done["dev"]["overall"]["minFDE"]
    old = compare["dev"]["overall"]["minFDE"]
    all_refs = [p for p in sorted(ref.iterdir()) if p.name!="B0_univ_s42"]
    env = lambda r: (r["contract"]["torch"],r["contract"]["cuda"])
    reuse = abs(measured/old-1)<=.003 and env(done)==env(compare) and all(
        env(read(p/"RESULT.json"))==env(done) and not read(p/"RESULT.json")["contract"].get("gpu_name") for p in all_refs)
    selected = []
    if reuse:
        for src in all_refs:
            dst = root/"stage2/reused"/src.name
            dst.mkdir(parents=True,exist_ok=True)
            r = read(src/"RESULT.json")
            assert digest(src/"dev_metrics.npz")==r["artifacts"]["dev_metrics.npz"]
            for name in ("dev_metrics.npz","RESULT.json"):
                publish(src/name,dst/name)
            selected.append(src.name)
    result = dict(status="LOCKED",reference=old,observed=measured,delta=measured/old-1,
                  compatible=bool(reuse),reused=selected)
    write_json(lock,result)
    return result


def variants(jobs, pft):
    rates = pft["epoch_seconds"]
    # Median benchmark scale from C0 to each fold is proportional to training batches.
    sizes = {f:math.ceil(inputs(f)[0][1]/64) for f in {j[1] for j in jobs}}
    return [rates.get(a,rates["O-risk+QFL"] if a.startswith("O-risk") else rates["D3-full"])
            *64*sizes[f]/340 for a,f,s,out in jobs]


def optional_summary(root):
    extra = [root/"optional/runs"/f"O-risk+E1_s{s}" for s in SEEDS]
    if not all(queueing.usable(p,deep=True) for p in extra):
        return
    a,ar = evidence(extra)
    b,br = evidence(root/"stage1/runs"/f"O-risk+E0_s{s}" for s in SEEDS)
    fast = short_compare(a,b,"minFDE","fast")
    all_ = short_compare(a,b,"minFDE")
    am,bm = matches(ar),matches(br)
    check = am["fast"]["rate"]-bm["fast"]["rate"]>=.10 and abs(am["slow"]["rate"]-bm["slow"]["rate"])<=.01
    write_json(root/"E1_ABLATION.json",dict(eligible=check,passed=check and fast["gain"]>=.03 and
                                            fast["ci"][0]>0 and all_["gain"]>=-.005,
                                            fast=fast,overall=all_,E1_matches=am,E0_matches=bm))


def work(root,old,hours):
    torch.set_num_threads(1)
    stop = time.monotonic()+hours*3600
    root.mkdir(parents=True,exist_ok=True)
    binding = verify_inputs(ROOT)
    f = ROOT/"fivefold_inputs.manifest.json"
    for name,checksum in read(f)["members"].items():
        assert digest(ROOT/name)==checksum,name
    assert torch.cuda.is_available() and "L4" in torch.cuda.get_device_name(0).split()
    runtime = dict(gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),cuda=torch.version.cuda,
                   data=binding,fivefold=digest(f),
                   code={p.name:digest(p) for p in sorted(ROOT.glob("*.py"))})
    write_json(root/"RUNTIME.json",runtime)
    local = read(ROOT/"data/Z0_LOCAL.json")
    kshare = float(np.mean([local[f"B0_s{s}"]["kinematic"]["share_candidates_positive"] for s in SEEDS]))
    use_kin = kshare>=.05
    queueing.launch = runner(use_kin)
    z0 = root/"Z0.json"
    process = log = None
    if not z0.is_file():
        log = (root/"Z0.log").open("a",encoding="utf-8")
        process = subprocess.Popen([sys.executable,"-u","-B",str(ROOT/"qrank_z0.py"),"--old",str(old),
                                    "--out",str(z0)],stdout=log,stderr=subprocess.STDOUT,
                                   env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"})
    try:
        pre = root/"PREFLIGHT.json"
        if pre.is_file():
            checks = read(pre)
            assert checks["runtime"]==runtime and checks["status"]=="PASS"
        else:
            queueing.status(root,"STRUCTURAL_PREFLIGHT")
            smoke = gates(ROOT,"cuda:0",use_kin)
            smoke["PF-LAT"] = latency(use_kin)
            if process is not None and process.wait(timeout=max(1,min(1800,stop-time.monotonic()))):
                raise RuntimeError("Z0 comparison failed; inspect Z0.log")
            assert read(z0)["status"]=="PASS" and read(z0)["MC5"]==use_kin
            pft = bench(root,stop)
            first = job(root,"stage1","B0","C0",42)
            dispatch(root,"PF_B0_C0",[first],1,stop)
            b0 = read(first[-1]/"RESULT.json")["dev"]["overall"]["minFDE"]
            previous = read(ROOT/"data/C0_B0_S42.json")["dev"]["overall"]["minFDE"]
            relative = b0/previous-1
            checks = dict(status="PASS" if abs(relative)<=.003 else "PF1_FAILED",
                          runtime=runtime,structural=smoke,PF_T=pft,PF_1=dict(observed=b0,reference=previous,delta=relative),
                          MC5=dict(share=kshare,use_kinematic_term=use_kin))
            write_json(pre,checks)
            assert checks["status"]=="PASS",f"PF-1 B0 deviation exceeds 0.3%: {relative:+.3%}"
        pft,cap = checks["PF_T"],checks["PF_T"]["chosen"]
        queueing.status(root,"STARTING_C0",max_parallel=cap)
        first = [job(root,"stage1",a,"C0",s) for part in (FORMAL[:6],FORMAL[6:10],FORMAL[10:])
                 for s in SEEDS for a in part]
        dispatch(root,"STAGE1_C0",first,cap,stop)
        admission = read(root/"ADMISSION.json") if (root/"ADMISSION.json").is_file() else screen(root)
        baseline = select_baseline(root,stop)
        second = [job(root,"stage2",a,f,s) for f in FOLDS for s in admission["stage2_seeds"][f]
                  for a in ("B0",*admission["arms"]) if f"B0_{f}_s{s}" not in baseline["reused"] or a!="B0"]
        dispatch(root,"STAGE2_FIVEFOLD",second,cap,stop)
        result = confirm(root,admission)
        optional = {}
        for arm in OPTIONAL:
            extra = [job(root,"optional",arm,"C0",s) for s in SEEDS]
            missing = [j for j in extra if not queueing.usable(j[-1])]
            estimate = max(variants(missing,pft),default=0)*math.ceil(len(missing)/cap)+120
            if missing and time.monotonic()+estimate>=stop:
                optional[arm] = "INSUFFICIENT_TIME"
                break
            try:
                dispatch(root,"OPTIONAL_"+arm,extra,min(cap,len(SEEDS)),stop,admission=stop-estimate)
            except queueing.Deadline:
                optional[arm] = "TIME_LIMIT_RESUMABLE"
                break
            except RuntimeError as error:
                optional[arm] = str(error)
                break
            optional[arm] = "COMPLETE"
        write_json(root/"OPTIONAL.json",optional)
        optional_summary(root)
        lines = ["# EXP-QRANK-ML","",f"完成 33 组 C0 和 51 组五折结果，五折模式为 {admission['mode']}。",
                 f"五折复用 {len(baseline['reused'])} 组历史 L4 B0。",
                 "质量置信度使用固定阈值与训练期的连续误差标签。E1 仅作为可选消融。",""]
        for arm,row in result.items():
            lines.append(f"- {arm} minFDE 等权改善 {row['mean']['minFDE']:+.2%}，F-PASS {row['F-PASS']}。")
        lines.extend(("", "ETH 与 C0 dev 无相同窗口。其他折曾用于开发，五折区间按连续 15 行分块。", ""))
        (root/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
        names = ("PREFLIGHT.json","Z0.json","STAGE1_RESULT.json","ADMISSION.json","B0_REFERENCE.json","STAGE2_RESULT.json","REPORT.md")
        write_json(root/"DONE.json",dict(status="PASS",stage1=33,stage2=51,at_utc=datetime.now(timezone.utc).isoformat(),
                                        files={name:digest(root/name) for name in names}))
        queueing.status(root,"DONE",stage1=33,stage2=51)
    finally:
        if process is not None and process.poll() is None:
            queueing.stop_processes([process])
        if log is not None:
            log.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--old",type=Path,required=True)
    p.add_argument("--hours",type=float,default=23.)
    a=p.parse_args()
    try:
        work(a.out,a.old,a.hours)
    except queueing.Deadline:
        queueing.status(a.out,"TIME_LIMIT_RESUMABLE")
        raise SystemExit(75)
