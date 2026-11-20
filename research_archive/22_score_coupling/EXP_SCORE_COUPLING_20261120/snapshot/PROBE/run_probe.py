"""Conditionally execute the frozen-encoder h3 score-path intervention."""

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
DIAG = ROOT.parent/"EXP_SCORE_COUPLING_DIAG_EXEC_20261120"
DEC = ROOT.parent/"EXP_DEC_INFO_PROBE_EXEC_20261117"
CAL = ROOT.parent/"EXP_SET_CALIB_DIAG_EXEC_20261119"
sys.path[:0]=[str(DIAG),str(CAL)]

import numpy as np
import torch

from run_coupling import NAMES, forces, setup, check, check_replay, load_baseline, as_numpy, read, save_npz, sha, write
from baseline_factory import ModelEMA, ModeQueryLoss, loss_value, make_model, optimizer_scheduler, prepare, prediction, world_paths
from width_model import checked_terms, h3_forward_diag, model_for, tensors, WEIGHTS, TERMS
from run_history_diag import observation_masks
from coupling_analysis import HARD, a_stat
from calib_data import split_halves
from calib_metrics import SEED as MC_SEED, DRAW_COUNT
from probe_metrics import ARMS, SEEDS, aggregate, calculate_decisions, metrics

BATCH, EPOCHS = 64, 64
DEVICE = "cuda:0"


def status(phase,**kw):
    write(ROOT/"CURRENT.json",dict(at=datetime.now().astimezone().isoformat(),pid=os.getpid(),phase=phase,**kw))


def save_pt(path,obj):
    tmp=path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        torch.save(obj,f)
    tmp.replace(path)


def verify_done(folder):
    stamp=read(folder/"DONE.json")
    assert stamp["status"]=="SUCCESS"
    for p,h in stamp["files"].items():
        assert sha(folder/p)==h,p
    return stamp


def gate():
    verify_done(DIAG)
    outcome=read(DIAG/"RESULT.json")
    return outcome["phase_two_triggered"]


@torch.no_grad()
def cache_fp32(data, originals, ckpt, inputs):
    folder=ROOT/"cache"
    folder.mkdir(exist_ok=True)
    if (folder/"DONE.json").exists():
        verify_done(folder)
        return {s:torch.load(folder/f"{s}.pt",map_location=DEVICE,weights_only=True) for s in ("fit","dev")},read(folder/"VALIDATION.json")
    model,_=model_for(ckpt,DEVICE,42)
    stamp=read(DEC/"cache/DONE.json")
    audit={}
    cache={}
    for split,d in data.items():
        status("CACHE_FP32",split=split)
        oldfile=DEC/f"cache/{split}.pt"
        check(oldfile,stamp["files"][str(oldfile)],inputs)
        legacy=torch.load(oldfile,map_location="cpu",weights_only=False)
        np.testing.assert_array_equal(legacy["meta"]["ids"],d["ids"])
        old_tokens=legacy["tensors"]["tokens_self"]
        assert old_tokens.shape==(len(d["ids"]),8,96) and old_tokens.dtype==torch.float16
        capture={}

        def record(module,args):
            capture["token"]=args[0][:,:1].detach().contiguous()
            capture["past"]=args[1][:,:1].detach().contiguous()

        h=model.decoder.register_forward_pre_hook(record)
        blocks=defaultdict(list)
        table={}
        fp16_path=fp16_logits=fp32_path=fp32_logits=0.
        try:
            for st in range(0,len(d["ids"]),BATCH):
                i=slice(st,st+BATCH)
                p,z,y,tf=prediction(model,*tensors(d,i,DEVICE))
                check_replay(p,z,tf,originals[split],i,table)
                token,past=capture["token"],capture["past"]
                assert token.dtype==torch.float32
                assert torch.equal(token[:,0].half().cpu(),old_tokens[i]),(split,st,"FP16 token lineage")
                base_token,base_past=token,past
                # Remove the capture hook during the two small decoder-only forwards.
                h.remove()
                pf,zf=model.decoder(base_token,base_past)
                qf,lf=model.decoder(old_tokens[i,None].to(DEVICE,torch.float32),base_past)
                h=model.decoder.register_forward_pre_hook(record)
                fp32_path=max(fp32_path,float(np.max(abs(as_numpy(world_paths(pf,tf))-originals[split][0][i]))))
                fp32_logits=max(fp32_logits,float(np.max(abs(as_numpy(zf[:,0])-originals[split][1][i]))))
                fp16_path=max(fp16_path,float(np.max(abs(as_numpy(world_paths(qf,tf))-originals[split][0][i]))))
                fp16_logits=max(fp16_logits,float(np.max(abs(as_numpy(lf[:,0])-originals[split][1][i]))))
                for k,x in (("tokens",base_token),("past",base_past),("target",y)):
                    blocks[k].append(x.cpu())
        finally:
            h.remove()
        assert max(fp32_path,fp32_logits)<=1e-4,(split,fp32_path,fp32_logits)
        out={k:torch.cat(x) for k,x in blocks.items()}
        save_pt(folder/f"{split}.pt",out)
        cache[split]={k:v.to(DEVICE) for k,v in out.items()}
        audit[split]=dict(windows=len(d["ids"]),full_encoder=table,archive18_roundtrip_fp16_equal=True,
                          fp16_path_max_abs=fp16_path,fp16_logit_max_abs=fp16_logits,
                          fp32_path_max_abs=fp32_path,fp32_logit_max_abs=fp32_logits)
        del legacy, out, blocks
    write(folder/"VALIDATION.json",audit)
    write(folder/"DONE.json",dict(status="SUCCESS",files={n:sha(folder/n) for n in ("fit.pt","dev.pt","VALIDATION.json")}))
    return cache,audit


def arm_model(arm,seed):
    model,cfg=make_model(DEVICE,seed)
    decoder=model.decoder
    del model
    criterion=ModeQueryLoss(cfg).to(DEVICE)
    if arm!="R-couple":
        def apply(module,args):
            base,geom=args[0].split((64,6),dim=-1)
            return (torch.cat((base,geom.detach() if arm=="R-detach" else torch.zeros_like(geom)),dim=-1),)
        decoder.score_head.register_forward_pre_hook(apply)
    return decoder,criterion


def batch(cache,index):
    return (cache[k][index] for k in ("tokens","past","target"))


def preflight(cache):
    a,b,c=(arm_model(arm,0) for arm in ARMS)
    x,p,y=batch(cache["fit"],slice(0,64))
    states=[t[0].state_dict() for t in (a,b,c)]
    for name in states[0]:
        assert torch.equal(states[0][name],states[1][name]) and torch.equal(states[0][name],states[2][name])
    # The true initialization has a zero output head. A nonzero test-only
    # fixture makes the intervention test discriminate between the three arms.
    with torch.no_grad():
        for m,_ in (a,b,c):
            m.score_head[-1].weight.copy_(torch.linspace(-.1,.1,64,device=DEVICE)[None])
    result={}
    base=None
    for arm,(m,criterion) in zip(ARMS,(a,b,c)):
        pp,z,aux=h3_forward_diag(m,x,p,track=True)
        terms,_,_,_=checked_terms(pp,y,z,criterion,{})
        g,=torch.autograd.grad(terms[:,TERMS.index("cls")].sum(),aux[4])
        if arm=="R-couple":
            assert g.abs().max()>0
            base=(pp.detach(),z.detach())
        else:
            assert not bool(torch.count_nonzero(g))
            assert torch.equal(pp,base[0])
            if arm=="R-detach":
                assert torch.equal(z,base[1])
        result[arm]=dict(params=sum(v.numel() for v in m.parameters()),fixture_direct_cls_geom_grad_max=float(g.abs().max()))
    assert len({v["params"] for v in result.values()})==1
    return result


@torch.no_grad()
def predict_cached(model,c):
    paths,scores=[],[]
    for i in range(0,len(c["target"]),BATCH):
        x,p,_=batch(c,slice(i,i+BATCH))
        a,z=model(x,p)
        paths.append(as_numpy(a[:,0]))
        scores.append(as_numpy(z[:,0]))
    return np.concatenate(paths),np.concatenate(scores)


def force_audit(model,criterion,cache,data,selected):
    pieces=defaultdict(list)
    model.eval().requires_grad_(False)
    for i in range(0,len(data["ids"]),BATCH):
        index=slice(i,i+BATCH)
        if not selected[index].any():
            continue
        p,loc,target=batch(cache,index)
        paths,z,aux=h3_forward_diag(model,p,loc,track=True)
        terms,winner,_,_=checked_terms(paths,target,z,criterion,{})
        for k,v in forces(paths,target,terms,winner,aux,torch.as_tensor(selected[index],device=DEVICE),{}).items():
            pieces[k].append(as_numpy(v))
    result={k:np.concatenate(v) for k,v in pieces.items()}
    result.update(maneuver=data["maneuver"][selected],track_id=data["track"][selected],c0_group=data["group"][selected])
    cls_max=float(result["norm"][:,:,NAMES.index("cls")].max())
    return dict(cls_geometric_norm_max=cls_max, summaries=a_stat(result,NAMES))


def train_one(arm,seed,data,cache,selected,bandwidth,draws,contract,completed):
    out=ROOT/f"runs/{arm}_s{seed}"
    out.mkdir(parents=True,exist_ok=True)
    if (out/"DONE.json").exists():
        verify_done(out)
        return read(out/"RESULT.json")
    model,criterion=arm_model(arm,seed)
    optim,scheduler=optimizer_scheduler(model,math.ceil(len(data["fit"]["ids"])/BATCH),EPOCHS)
    ema=ModelEMA(model,.995)
    rng=torch.Generator().manual_seed(seed)
    history=[]
    resume=out/"resume.pt"
    cached=cache["fit"]
    steps=0
    if resume.exists():
        state=torch.load(resume,map_location=DEVICE,weights_only=True)
        assert state["contract"]==contract
        model.load_state_dict(state["model"]); ema.module.load_state_dict(state["ema"])
        optim.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        rng.set_state(state["order_rng"].cpu())
        history,steps=state["history"],state["steps"]
    for epoch in range(len(history),EPOCHS):
        status("TRAIN",arm=arm,seed=seed,epoch=epoch+1,epochs=EPOCHS,completed=completed,total=9)
        order=torch.randperm(len(data["fit"]["ids"]),generator=rng)
        total=0.
        start=time.monotonic()
        model.train()
        for index in order.split(BATCH):
            x,p,y=batch(cached,index.to(DEVICE))
            path,z=model(x,p)
            loss,_=loss_value(path,y,z,criterion)
            assert bool(torch.isfinite(loss))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optim.step()
            scheduler.step()
            ema.update(model)
            steps+=1
            total+=float(loss.detach())*len(index)
        history.append(dict(epoch=epoch+1,loss=total/len(order),seconds=time.monotonic()-start,steps=steps,
                            order_sha256=hashlib.sha256(order.numpy().astype("<i8").tobytes()).hexdigest()))
        save_pt(resume,dict(model=model.state_dict(),ema=ema.module.state_dict(),optimizer=optim.state_dict(),
                             scheduler=scheduler.state_dict(),order_rng=rng.get_state(),history=history,steps=steps,
                             contract=contract))
        write(out/"TRAINING.json",dict(history=history,epochs=epoch+1,steps=steps))
        print(f"{arm} seed={seed} epoch={epoch+1}/{EPOCHS} loss={history[-1]['loss']:.8f}",flush=True)
        if (ROOT/"PAUSE.request").exists():
            status("PAUSED",arm=arm,seed=seed,epoch=epoch+1,completed=completed,total=9)
            return None
    save_pt(out/"final_ema.pt",ema.module.state_dict())
    status("EVALUATE",arm=arm,seed=seed,completed=completed,total=9)
    for split in ("fit","dev"):
        paths,z=predict_cached(ema.module,cache[split])
        save_npz(out/f"{split}_predictions.npz",paths=paths,logits=z,ids=data[split]["ids"])
    y=as_numpy(cache["dev"]["target"][:,0]).astype(np.float64)
    w=metrics(paths.astype(np.float64),y,z.astype(np.float64),bandwidth,draws)
    save_npz(out/"dev_metrics.npz",**w,ids=data["dev"]["ids"])
    result=dict(groups=aggregate(w,data["dev"]))
    if arm in ("R-couple","R-detach"):
        fa=force_audit(ema.module,criterion,cached,data["fit"],selected)
        if arm=="R-detach":
            assert fa["cls_geometric_norm_max"]==0.
        result["A1_A2_audit"]=fa
    write(out/"RESULT.json",result)
    names=("TRAINING.json","final_ema.pt","fit_predictions.npz","dev_predictions.npz","dev_metrics.npz","RESULT.json")
    write(out/"DONE.json",dict(status="SUCCESS",files={x:sha(out/x) for x in names}))
    return result


def main(pre_only):
    if not gate():
        status("NOT_TRIGGERED")
        return
    if (ROOT/"DONE.json").exists() and not pre_only:
        verify_done(ROOT)
        return
    start=time.monotonic()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data,selected,inputs=setup()
    ckpt,originals=load_baseline(42,data,inputs)
    obs = observation_masks(data["dev"]["past"][:,0].astype(np.float64))
    data["dev"]["observation"]=np.select([obs["stationary"],obs["steady"]],["stationary","steady"],default="other")
    cache,audit=cache_fp32(data,originals,ckpt,inputs)
    checks=preflight(cache)
    gatehash=sha(DIAG/"DONE.json")
    calibration_path=CAL/"RESULT.json"
    check(calibration_path,read(CAL/"DONE.json")["files"]["RESULT.json"],inputs)
    calibration=read(calibration_path)["runs"]["B0"]["42"]["t12"]["hpd"]["equal"]["fitted_on_other_half"]
    half,_=split_halves(data["dev"]["ids"])
    widths=[calibration[name]["global"]["chosen"] for name in ("A","B")]
    bandwidth=np.asarray(widths)[half]
    contract=dict(stage1=gatehash,cache=sha(ROOT/"cache/DONE.json"),fit=inputs[str(DIAG.parent/"NEXT_PHASE_EXEC_20261114/C0/fit.npz")],
                  source={p.name:sha(p) for p in (*ROOT.glob("*.py"),ROOT/"PROTOCOL.md")},epochs=EPOCHS,batch=BATCH)
    write(ROOT/"PREFLIGHT.json",dict(status="PASS",files=inputs,token_replay=audit,interventions=checks,
                                    bandwidth=widths,contract=contract))
    status("PREFLIGHT_PASS",completed=0,total=9)
    if pre_only:
        return
    random=np.random.default_rng(MC_SEED)
    shape=(len(half),DRAW_COUNT)
    draws=(random.random(shape),random.standard_normal((*shape,2)))
    runs={arm:{} for arm in ARMS}
    done=0
    for arm in ARMS:
        for seed in SEEDS:
            record=train_one(arm,int(seed),data,cache,selected,bandwidth,draws,contract,done)
            if record is None:
                return
            runs[arm][seed]=record
            done+=1
    b0=originals["dev"][0]
    b0_fde=float(np.linalg.norm(b0[:,:,-1]-data["dev"]["future"][:,None,-1],axis=2).min(1).mean())
    verdict=calculate_decisions(runs,b0_fde)
    result=dict(decision=verdict,B0_seed42_minFDE=b0_fde,runs=runs)
    write(ROOT/"RESULT.json",result)
    flags=verdict["flags"]
    rows=["# EXP-SCORE-COUPLING-PROBE","","达到预设门槛的为 "+("、".join(k for k,v in flags.items() if v) or "无")+"。","",
          "| 训练臂 | seed | minADE | minFDE | 转向组 J 覆盖 | Top1 FDE |",
          "|---|---:|---:|---:|---:|---:|"]
    for arm,rr in runs.items():
        for seed,r in rr.items():
            g=r["groups"]
            rows.append(f"| {arm} | {seed} | {g['overall']['minADE']['estimate']:.4f} | {g['overall']['minFDE']['estimate']:.4f} | "
                        f"{g['hard']['J_coverage']['estimate']:.1%} | {g['overall']['top1_FDE']['estimate']:.4f} |")
    rows += ["","详细分组、轨迹区间、两组配对差和梯度干预核验见 RESULT.json。",
             "三个臂使用同一个冻结的 seed42 编码器和原始 C0 fit/dev。训练未做数据增强，阶段一的关联性只用于触发本次干预。"]
    (ROOT/"REPORT.md").write_text("\n".join(rows)+"\n",encoding="utf-8")
    files=["PREFLIGHT.json","RESULT.json","REPORT.md"]+[f"runs/{a}_s{s}/DONE.json" for a in ARMS for s in SEEDS]
    write(ROOT/"DONE.json",dict(status="SUCCESS",seconds=time.monotonic()-start,files={p:sha(ROOT/p) for p in files}))
    status("DONE",completed=9,total=9)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only",action="store_true")
    args=parser.parse_args()
    try:
        main(args.preflight_only)
    except Exception:
        status("FAILED",traceback=traceback.format_exc())
        raise
