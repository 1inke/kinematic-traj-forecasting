"""Replay the archived B0 models and evaluate the stage-one gate."""

import argparse
from collections import defaultdict
from datetime import datetime
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "NEXT_PHASE_EXEC_20261114"
PREC = ROOT.parent / "EXP_PREC_DIAG_EXEC_20261119"
WIDTH = ROOT.parent / "EXP_WIDTH_CAUSE_DIAG_EXEC_20261120"
sys.path[:0] = [str(WIDTH), str(OLD), str(PREC)]

import numpy as np
import torch

from run_prec_diag import arrays, maneuver_labels, read, sha, write
from run_width import as_numpy, check, check_replay, load_baseline, save_npz
from width_model import TERMS, WEIGHTS, checked_terms, model_for, replay
from coupling_analysis import RNG, BOOT, HARD, SECONDARY, a_stat, b_stat, decision, misses

SEEDS = (42,43,44,45,46)
BATCH = 64
NAMES = (*TERMS,"geom")
OUT = ("PREFLIGHT.json","A_summary.json","B_miss_geometry.json","RESULT.json","REPORT.md")


def status(phase, **kw):
    write(ROOT/"CURRENT.json",dict(at=datetime.now().astimezone().isoformat(), pid=os.getpid(), phase=phase, **kw))


def setup():
    inputs = {}
    pins = read(WIDTH/"PREFLIGHT.json")["files"]
    for p in (WIDTH/"width_model.py", OLD/"C0/fit.npz", OLD/"C0/dev.npz"):
        check(p,pins[str(p)],inputs)
    for p in (*ROOT.glob("*.py"), ROOT/"PROTOCOL.md", ROOT/"DESIGN.md", *ROOT.glob("standalone/*.ps1")):
        inputs[str(p)] = sha(p)
    data = {}
    for split, n in (("fit",21719),("dev",4753)):
        d = arrays(OLD/f"C0/{split}.npz")
        assert len(d["ids"]) == n == len(np.unique(d["ids"]))
        d["maneuver"] = maneuver_labels(d["past"],d["future"])
        d["track"] = np.asarray([x.rsplit("|",1)[0] for x in d["ids"]])
        data[split] = d
    labels = data["fit"]["maneuver"]
    chosen = np.isin(labels,(*HARD,*SECONDARY))
    rng = np.random.default_rng(RNG)
    for key,n in (("steady",3000), ("stationary",2000)):
        chosen[rng.choice(np.flatnonzero(labels==key), size=n, replace=False)] = True
    return data,chosen,inputs


def unit(v):
    norm = torch.linalg.vector_norm(v,dim=-1,keepdim=True)
    valid = norm>=1e-6
    return torch.where(valid,v/norm.clamp_min(1e-6),0),valid.squeeze(-1).any(-1)


def forces(p,y,terms,winner,aux,selected,audit):
    rel,last = aux[4],aux[5]
    rel0 = rel[:,0].detach()
    # Decoder outputs and target are centered within the history. Force
    # projection uses displacements from the actual last observed point.
    yt = y[:,0] - last[:,0,0]
    toward,p_valid = unit(yt[:,None,[3,7,11]] - rel0)
    outward,e_valid = unit(rel0 - rel0.mean(1,keepdim=True))
    weighted = terms * terms.new_tensor(WEIGHTS)
    gradients = []
    for i in range(len(TERMS)):
        grad, = torch.autograd.grad(weighted[selected,i].sum(),rel,retain_graph=True)
        gradients.append(grad[selected,0])
    geom = sum(g for i,g in enumerate(gradients) if TERMS[i]!="cls")
    combined, = torch.autograd.grad(weighted[selected].sum(),rel)
    error = float((combined[selected,0]-geom-gradients[TERMS.index("cls")]).abs().max())
    assert error <= 2e-6, error
    audit["gradient_sum_max_abs"] = max(error,audit.get("gradient_sum_max_abs",0.))
    g = torch.stack([*gradients,geom],dim=2)
    pp = (-g*toward[selected,:,None]).sum((3,4))
    ee = (-g*outward[selected,:,None]).sum((3,4))
    gn = torch.linalg.vector_norm(g.flatten(3),dim=3)
    distances = torch.linalg.vector_norm(p.detach() - y.unsqueeze(-3),dim=-1)[:,0]
    ade,fde = distances.mean(-1), distances[:,:,-1]
    rank_order = (ade+.25*fde).argsort(dim=1,stable=True)
    assert torch.equal(rank_order[:,0],winner)
    ranks = rank_order.argsort(dim=1)+1
    return dict(P=pp,E=ee,norm=gn,P_valid=p_valid[selected],E_valid=e_valid[selected],quality_rank=ranks[selected],
                fde_winner=fde.argmin(1)[selected],covered=((ade<=.2)&(fde<=.4)).any(1)[selected])


def run_seed(seed,data,chosen,device,inputs,sample=False):
    ckpt,originals = load_baseline(seed,data,inputs)
    model,criterion = model_for(ckpt,device,seed)
    audit = {}
    a = defaultdict(list)
    b = {}
    for split,d in data.items():
        cache = defaultdict(list)
        is_fit = split=="fit"
        n = len(d["ids"])
        limit = min(n,512 if is_fit else 128) if sample else n
        for st in range(0,limit,BATCH):
            end = min(limit,st+BATCH)
            sl = slice(st,end)
            sel = chosen[sl] if is_fit else np.zeros(end-st,bool)
            tracking = bool(sel.any())
            if not sample and st%2048==0:
                status("FIT_GRADIENTS" if is_fit else "DEV_GEOMETRY", seed=seed, window=st,total=n)
            p,z,y,tf,capture = replay(model,d,sl,device,track=tracking,check_original=st==0)
            audit["decoder_max_abs"] = max(audit.get("decoder_max_abs",0.),*capture.get("original_forward_max_abs",[0.]))
            check_replay(p,z,tf,originals[split],sl,audit)
            aux = capture["aux"]
            if tracking:
                terms,winner,_,_ = checked_terms(p,y,z,criterion,audit)
                for k,v in forces(p,y,terms,winner,aux,torch.as_tensor(sel,device=device),audit).items():
                    a[k].append(as_numpy(v))
            offset = as_numpy(aux[5][:,0,0]).astype(np.float64)
            pr = as_numpy(p[:,0]).astype(np.float64)-offset[:,None]
            gt = as_numpy(y[:,0]).astype(np.float64)-offset
            for k,v in misses(pr,gt,as_numpy(z[:,0]).astype(np.float64)).items():
                cache[k].append(v)
        if not sample:
            status("TRACK_BOOTSTRAP_GEOMETRY",seed=seed,split=split,replicates=BOOT)
            b[split] = b_stat({k:np.concatenate(v) for k,v in cache.items()},d["maneuver"],d["group"],d["track"])
    af = {k:np.concatenate(v) for k,v in a.items()}
    if sample:
        subset = np.isin(data["fit"]["maneuver"][:512][chosen[:512]],HARD)
        rank = (af["quality_rank"]>=2)&(af["quality_rank"]<=6)&subset[:,None]
        sample_ratio = af["norm"][:,:,NAMES.index("cls")][rank].sum()/af["norm"][:,:,NAMES.index("geom")][rank].sum()
        return audit, float(sample_ratio)
    d = data["fit"]
    af.update(ids=d["ids"][chosen],track_id=d["track"][chosen],maneuver=d["maneuver"][chosen],
              c0_group=d["group"][chosen],fit_index=np.flatnonzero(chosen),terms=np.asarray(NAMES))
    status("TRACK_BOOTSTRAP_FORCES",seed=seed,replicates=BOOT)
    aa = a_stat(af,NAMES)
    save_npz(ROOT/f"A_forces_s{seed}.npz",**af)
    return aa,b,audit


def report(result,b):
    marker = result["flags"]
    lines=["# EXP-SCORE-COUPLING-DIAG","",
           "达到预设判定的标记为 "+("、".join(k for k,v in marker.items() if v["active"]) or "无")+"。",
           ("K1 达到，阶段二进入触发条件。" if result["phase_two_triggered"] else "K1 未达到，阶段二未触发。"),
           "", "| seed | K1 难组 D | K2 分类投影均值 | K2 轨迹区间 | K3 未覆盖减已覆盖 | K4 匀速 D |",
           "|---:|---:|---:|---:|---:|---:|"]
    for s,v in result["per_seed"].items():
        lo,hi=v["K2"]["cls_P_ci95"]
        lines.append(f"| {s} | {v['K1']['D']:.3f} | {v['K2']['cls_P']:.4g} | [{lo:.4g}, {hi:.4g}] | "
                     f"{v['K3']['gap_pp']:+.1f} pp | {v['K4']['D']:.3f} |")
    lines += ["","梯度、轨迹区间和窗口占比见 [A_summary.json](A_summary.json)。",
              "", "| seed | dev 难组漏覆盖数 | 形状 | 范围 | 方向 | 近距 | 其余扇区内候选 |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for s,v in b.items():
        x=v["dev"]["hard"]
        shares=[f"{x['categories'][k]['fraction']['estimate']:.1%}" for k in
                ("shape","range","direction","near","sector_radius_compatible")]
        lines.append(f"| {s} | {x['uncovered_population']['windows']} | "+" | ".join(shares)+" |")
    lines += ["","fit/dev 分组的轨迹区间、纵横误差、概率和 30° 方向直方图见 [B_miss_geometry.json](B_miss_geometry.json)。",
              "", "梯度来自冻结 EMA 在 eval 模式下的快照。轨迹区间按录制序列和行人编号聚类重抽样。"
              "同一 C0 dev 已用于多轮探索。阶段二的正式训练对照才能检验切断梯度的后果。"]
    return "\n".join(lines)+"\n"


def main(device,sample):
    start=time.monotonic()
    if not sample and (ROOT/"DONE.json").exists():
        assert all(sha(ROOT/p)==h for p,h in read(ROOT/"DONE.json")["files"].items())
        return
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data,selected,inputs=setup()
    if sample:
        audit, ratio = run_seed(42,data,selected,device,inputs,True)
        print(dict(preflight=audit, selected=int(selected.sum()), technical_only_D=ratio),flush=True)
        return
    a,b,checks={}, {}, {}
    for i,seed in enumerate(SEEDS):
        aa,bb,audit=run_seed(seed,data,selected,device,inputs)
        a[str(seed)],b[str(seed)],checks[str(seed)]=aa,bb,audit
        status("SEED_COMPLETE",seed=seed,completed=i+1,total=len(SEEDS))
        print(f"seed {seed} {i+1}/{len(SEEDS)} done",flush=True)
    outcome=decision(a)
    labels=data["fit"]["maneuver"][selected]
    pre=dict(seed=RNG,bootstrap=BOOT,fit_indices=np.flatnonzero(selected).tolist(),
             fit_counts={name:int((labels==name).sum()) for name in sorted(set(labels))},
             checks=checks,files=inputs,device=device)
    for name,value in zip(OUT,(pre,a,b,outcome)):
        write(ROOT/name,value)
    (ROOT/"REPORT.md").write_text(report(outcome,b),encoding="utf-8")
    files=[*OUT,*(f"A_forces_s{s}.npz" for s in SEEDS)]
    write(ROOT/"DONE.json",dict(status="SUCCESS",elapsed_s=time.monotonic()-start,
                               files={p:sha(ROOT/p) for p in files},phase_two_triggered=outcome["phase_two_triggered"]))
    status("DONE",completed=len(SEEDS),total=len(SEEDS),phase_two_triggered=outcome["phase_two_triggered"])


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device",choices=("cpu","cuda:0"),default="cuda:0")
    ap.add_argument("--sample",action="store_true")
    args=ap.parse_args()
    try:
        main(args.device,args.sample)
    except Exception:
        if not args.sample:
            status("FAILED",traceback=traceback.format_exc())
        raise
