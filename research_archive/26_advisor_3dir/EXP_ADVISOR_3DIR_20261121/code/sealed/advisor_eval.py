"""Fixed-K scientific metrics and G1 nonlearned compression."""
from pathlib import Path
import tempfile
import math

import numpy as np

from eth_speed_math import maneuver_labels
from pool_select import NMS_THRESHOLDS, medoids, nms
from run_b0 import publish

COMPRESSION = tuple(f"NMS_{v:.1f}" for v in NMS_THRESHOLDS) + ("kmedoids","top20")


def probs(logits):
    z = logits.astype(np.float64)
    out = np.exp(z-z.max(1,keepdims=True))
    return out/out.sum(1,keepdims=True)


def forecast_metrics(paths, logits, future):
    w = probs(logits)
    err = np.linalg.norm(paths-future[:,None],axis=-1)
    ade,fde = err.mean(2),err[:,:,-1]
    indices = np.arange(len(paths))
    best,top = fde.argmin(1),logits.argmax(1)
    rank = np.argsort(np.argsort(-logits,axis=1,kind="stable"),axis=1)
    scatter = np.empty(len(paths))
    i,j = np.triu_indices(paths.shape[1],1)
    for lo in range(0,len(paths),64):
        sl = slice(lo,lo+64)
        d = paths[sl,i]-paths[sl,j]
        scatter[sl] = (np.sqrt(np.square(d).sum((2,3)))*w[sl,i]*w[sl,j]).sum(1)
    score = (np.linalg.norm(err,axis=2)*w).sum(1)-scatter
    return dict(minADE=ade.min(1),minFDE=fde.min(1),Top1_ADE=ade[indices,top],Top1_FDE=fde[indices,top],
                J=((ade<=.2)&(fde<=.4)).any(1),rank=rank[indices,best]+1,ES=score,
                brier_minFDE=fde[indices,best]+(1-w[indices,best])**2)


def groups(data):
    p = data["past"][:,0]
    delta = np.diff(p,axis=1)
    length = np.hypot(delta[...,0],delta[...,1])
    speed = length/.4
    avg = speed.mean(1)
    stationary = speed.max(1)<=.1
    net = delta.sum(1)
    size = np.hypot(net[:,0],net[:,1])
    eff = size/np.maximum(length.sum(1),1e-12)
    direction = (delta[:,-1]*net).sum(1)/np.maximum(length[:,-1]*size,1e-12)
    steady = (~stationary)&(avg>=.1)&(np.max(abs(speed-avg[:,None]),axis=1)<=.25*avg)&(eff>=.98)&(
        direction>=math.cos(math.radians(15)))
    hard = np.isin(maneuver_labels(data["past"],data["future"]),("turn","turn_and_speed"))
    ids = data["ids"]
    scene = np.array([s.split("|",1)[0] for s in ids]) if ids.dtype.kind in "US" else data["scene"]
    return dict(overall=np.ones(len(ids),bool),hard=hard,stationary=stationary,steady=steady,
                other=~(stationary|steady),without_zara02=(scene!="crowds_zara02") if ids.dtype.kind in "US" else scene!=4)


def metric_summary(arrays,masks):
    return {g:dict(windows=int(mask.sum()),**{k:float(v[mask].mean()) for k,v in arrays.items()})
            for g,mask in masks.items() if mask.any()}


def compression(paths,logits):
    indices = {name:np.empty((len(paths),20),int) for name in COMPRESSION}
    for lo in range(0,len(paths),128):
        sl = slice(lo,lo+128)
        p = paths[sl,:,-1]
        w = probs(logits[sl])
        d = np.linalg.norm(p[:,:,None]-p[:,None],axis=-1)
        for name,threshold in zip(COMPRESSION,NMS_THRESHOLDS):
            indices[name][sl] = nms(d,w,threshold)
        indices["kmedoids"][sl] = medoids(d,w)[0]
        indices["top20"][sl] = np.argsort(-w,axis=1,kind="stable")[:,:20]
    return indices


def subset(paths, logits, idx):
    return paths[np.arange(len(paths))[:,None],idx],logits[np.arange(len(paths))[:,None],idx]


def safe_npz(target, **fields):
    with tempfile.TemporaryDirectory(prefix="advisor_npz_") as tmp:
        path = Path(tmp)/target.name
        np.savez_compressed(path,**fields)
        return publish(path,target)


def mechanism(paths,aux,dev,hard):
    if aux is None or aux.shape!=paths.shape:
        return None
    truth = dev["future"]
    past = dev["past"][:,0].astype(np.float64)
    direction = past[:,-1]-past[:,-4]
    norm = np.linalg.norm(direction,axis=1,keepdims=True)
    unit = np.where(norm>.03,direction/np.maximum(norm,1e-12),[1.,0.])
    normal = unit[:,::-1]*[-1.,1.]
    first = np.linalg.norm(aux[:,:,-1]-truth[:,None,-1],axis=2)
    index = first.argmin(1)
    row = np.arange(len(paths))
    first_lat = np.einsum("nti,ni->nt",aux[row,index]-past[:,-1,None],normal)
    last_lat = np.einsum("nti,ni->nt",paths[row,index]-past[:,-1,None],normal)
    move = np.linalg.norm(paths-aux,axis=-1)
    return dict(first_pass_minFDE=float(first.min(1).mean()), median_shift_m=float(np.median(move)),
                hard_first_winner_peak_lateral_before_m=float(abs(first_lat[hard]).max(1).mean()) if hard.any() else None,
                hard_first_winner_peak_lateral_after_m=float(abs(last_lat[hard]).max(1).mean()) if hard.any() else None)


def save_evaluation(out, arm, infer, train, dev):
    p,z,aux,aux_logits = infer(dev)
    artifacts = {}
    masks = groups(dev)
    table = forecast_metrics(p,z,dev["future"])
    extra = {}
    if arm=="G1":
        ft,fl,_,_ = infer(train)
        fit_choices = compression(ft,fl)
        test_choices = compression(p,z)
        curves = {"fit":{},"dev":{}}
        for k in COMPRESSION:
            fx,fz = subset(ft,fl,fit_choices[k])
            curves["fit"][k] = metric_summary(forecast_metrics(fx,fz,train["future"]),groups(train))["overall"]
            tx,tz = subset(p,z,test_choices[k])
            extra[k] = forecast_metrics(tx,tz,dev["future"])
            curves["dev"][k] = metric_summary(extra[k],masks)["overall"]
        best_nms = min(COMPRESSION[:len(NMS_THRESHOLDS)],key=lambda k:curves["fit"][k]["minFDE"])
        best = min((best_nms,"kmedoids","top20"),key=lambda k:curves["fit"][k]["minFDE"])
        curves["selected_NMS_on_fit"] = best_nms
        curves["selected_compressor_on_fit"] = best
        extra["pool60"] = table
        extra["NMS_fit"] = extra[best_nms]
        table = extra[best]
        artifacts["dev_pool60.npz"] = safe_npz(out/"dev_pool60.npz",paths=p,logits=z,ids=dev["ids"])
        p,z = subset(p,z,test_choices[best])
    else:
        curves = None
    comparison = mechanism(p,aux,dev,masks["hard"])
    if aux is not None and aux.shape[1]==60:
        extra["pool60"] = forecast_metrics(aux,aux_logits,dev["future"])
    fields = {**table, **{f"{variant}__{k}":v for variant,metric in extra.items() for k,v in metric.items()}}
    for name,payload in (("dev_predictions.npz",dict(paths=p,logits=z,ids=dev["ids"])),
                         ("dev_metrics.npz",dict(ids=dev["ids"],**masks,**fields))):
        artifacts[name] = safe_npz(out/name,**payload)
    return dict(dev=metric_summary(table,masks),compression=curves,mechanism=comparison,artifacts=artifacts,
                pool60=metric_summary(extra["pool60"],masks) if "pool60" in extra else None)

