"""Verify the interventions and run short real-window forward/backward checks."""
import json
import time

import numpy as np
import torch

from advisor_eval import compression
from advisor_model import ARMS, CrossLayer, attach, emta_indices
from advisor_train import BASE, Trainer, core, load_data


def structural(device, windows=4, latency=True):
    torch.set_num_threads(1)
    p = load_data(BASE/"C0/fit.npz",21719)
    d = {k:v[:windows] for k,v in p.items()}
    past,count = (torch.from_numpy(d[k]).to(device) for k in ("past","num_valid"))
    y = torch.from_numpy(d["future"]).to(device)
    report = {}
    base,_ = core.make_model(device,42)
    base.eval()
    with torch.no_grad():
        pred,z,_,_ = core.prediction(base,past,count)
    for arm in (*ARMS,"R2-half","S4-wide"):
        trainer = Trainer(d,arm,42,device,2)
        model = trainer.model
        model.eval()
        with torch.no_grad():
            q,a,_,_ = core.prediction(model,past,count,y)
            q2,a2,_,_ = core.prediction(model,past,count,torch.rand_like(y))
            assert torch.equal(q,q2) and torch.equal(a,a2), arm
            if arm in ("B0","S1","R1","R2","R2-half","S2","S3","S4","S4-wide"):
                torch.testing.assert_close(q,pred,rtol=0,atol=1e-6)
                torch.testing.assert_close(a,z,rtol=0,atol=1e-6)
            if arm in ("G2","G2-intra"):
                w = model.decoder.group_weights.sum(-1)
                torch.testing.assert_close(w,torch.ones_like(w),rtol=0,atol=1e-6)
            assert q.shape[-3]==(60 if arm=="G1" else 20)
            timing = None
            if latency:
                times = []
                core.prediction(model,past[:1],count[:1])
                for _ in range(24):
                    start=time.perf_counter()
                    core.prediction(model,past[:1],count[:1])
                    times.append(time.perf_counter()-start)
                timing = float(np.median(times)*1000)
        batch=trainer.step()
        assert np.isfinite(batch["loss"]) and np.isfinite(batch["grad_norm"]),arm
        report[arm]=dict(parameters=sum(p.numel() for p in model.parameters()), windows=len(past),
                         CPU_single_sample_ms=timing,loss=batch["loss"])
        print(f"{device} preflight {arm} PASS",flush=True)
    return report


def deterministic():
    layer = CrossLayer("causal").eval()
    q,m = torch.randn(4,20,64),torch.randn(4,17,64)
    mask = torch.zeros(4,17,dtype=torch.bool)
    with torch.no_grad():
        full = layer(q,m,mask)
        for j in (1,5,11,19):
            altered = q.clone()
            altered[:,j] += torch.randn_like(altered[:,j])
            torch.testing.assert_close(layer(altered,m,mask)[:,:j],full[:,:j],rtol=0,atol=1e-6)
    dist = torch.full((5,4,12),2.)
    dist[0,0] = 0.
    dist[0,2] = .01
    dist[1,3] = .5
    dist[2,3] = .01
    dist[3,2] = .01
    dist[3,0] = .01
    dist[4,1] = .499
    order = torch.tensor([[2,0,1,3],[0,1,2,3],[2,1,0,3],[0,2,1,3],[3,2,0,1]])
    idx,matched = emta_indices(dist,order)
    assert idx.tolist()==[2,3,3,0,1] and matched.tolist()==[True,False,True,True,False]
    rng=np.random.default_rng(42)
    traj=rng.normal(size=(4,60,12,2))
    traj[:,3]=traj[:,0]
    z=rng.normal(size=(4,60))
    a,b = compression(traj,z),compression(traj,z)
    assert all(np.array_equal(a[k],b[k]) for k in a)
    return dict(PF4_causality=True,PF5_EMTA=True,PF9_deterministic_compression=True)


if __name__=="__main__":
    result=dict(deterministic(),arms=structural("cpu"))
    print(json.dumps(result,indent=2))

