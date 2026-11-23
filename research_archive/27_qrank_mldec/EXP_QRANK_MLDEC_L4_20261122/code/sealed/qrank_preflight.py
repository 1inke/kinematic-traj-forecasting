"""QRANK structural and loss checks on sealed observations and predictions."""
from pathlib import Path
import time

import numpy as np
import torch

from advisor_model import CrossLayer
from advisor_train import core, inputs, load_data
from order_numpy import risk_order as numpy_risk
from qrank_model import (FORMAL, confidence_loss, emta_indices, emta_threshold, kinematic_excess,
                         kinematic_reference, observed_speed, quality_loss, risk_order)
from qrank_train import Trainer


def first(data,n):
    return {k:v[:n] for k,v in data.items()}


def ordered_check(root,device):
    rng = np.random.default_rng(20261122)
    end = rng.normal(size=(2000,20,2))
    w = rng.uniform(.1,1.,size=(2000,20))
    with np.load(root/"data/PF_R1_REFERENCE.npz",allow_pickle=False) as saved:
        other = saved["endpoints"]
        z = saved["logits"].astype("float64")
        other_w = np.exp(z-z.max(1,keepdims=True))
    for endpoints,weights in ((end,w),(other,other_w)):
        weights /= weights.sum(1,keepdims=True)
        for start in range(0,len(endpoints),64):
            sl = slice(start,start+64)
            actual = risk_order(torch.as_tensor(endpoints[sl],device=device),
                                torch.as_tensor(weights[sl],device=device)).cpu().numpy()
            np.testing.assert_array_equal(actual,numpy_risk(endpoints[sl],weights[sl]))
    return 2000+len(other)


def gates(root,device,use_kin=True):
    torch.set_num_threads(1)
    (path,n),(dev_path,dev_n) = inputs("C0")
    fit = load_data(path,n)
    dev = load_data(dev_path,dev_n)
    data = first(fit,16)
    p,f,c = (torch.as_tensor(data[k],device=device) for k in ("past","future","num_valid"))
    checks = {"PF-R1":ordered_check(root,device)}
    memory = torch.randn(3,17,64,device=device)
    x = torch.randn(3,20,64,device=device)
    padding = torch.zeros(3,17,device=device,dtype=torch.bool)
    causal = CrossLayer("causal").to(device).eval()
    with torch.no_grad():
        result = causal(x,memory,padding)
        for k in (5,10):
            altered = x.clone()
            altered[:, k] += 2.
            torch.testing.assert_close(causal(altered,memory,padding)[:, :k],result[:, :k],atol=1e-6,rtol=0)
    checks["PF-R2"] = True

    baseline = Trainer(data,"B0",42,device,2)
    baseline.model.eval()
    with torch.no_grad():
        bpaths,bz,_,_ = core.prediction(baseline.model,p,c)
    reference = None
    for arm in FORMAL[1:]:
        t = Trainer(data,arm,42,device,2,use_kin)
        info = core.cpu_tree(t.model.state_dict())
        if reference is None: reference=info
        else: core.exact(info,reference)
        m = t.model.eval()
        d = m.decoder
        with torch.no_grad():
            d.eval_generator = torch.Generator(device=device).manual_seed(12345)
            a,z,_,_ = core.prediction(m,p,c,f)
            initial = core.cpu_tree([{k:o[k] for k in ("paths","logits","order","qraw")} for o in d.outs])
            torch.testing.assert_close(a,bpaths,atol=1e-6,rtol=0)
            torch.testing.assert_close(z,bz,atol=1e-6,rtol=0)
            d.eval_generator = torch.Generator(device=device).manual_seed(12345)
            a2,z2,_,_ = core.prediction(m,p,c,f+100.)
            core.exact(a,a2)
            core.exact(z,z2)
            core.exact(initial,core.cpu_tree([{k:o[k] for k in ("paths","logits","order","qraw")} for o in d.outs]))
        print(f"Structural checks passed: {arm}",flush=True)
    checks.update({"PF-2":True,"PF-3":True,"PF-L1":True})

    t = Trainer(data,"O-risk+QFL",42,device,2,use_kin)
    _,_,target,_ = core.prediction(t.model,p,c,f)
    (.5*sum(quality_loss(o,target) for o in t.model.decoder.outs)).backward()
    for name,param in t.model.named_parameters():
        if not name.startswith("decoder.quality."):
            assert param.grad is None or not torch.count_nonzero(param.grad),name
    checks["PF-Q1"] = True
    small = first(fit,8)
    traces = {}
    for arm in ("O-conf","O-rand"):
        t = Trainer(small,arm,42,device,20,use_kin,batch=4)
        rows = []
        h = t.model.register_forward_pre_hook(lambda _,args: rows.append(args[0].detach().cpu().clone()))
        for _ in range(10): t.step()
        h.remove()
        traces[arm] = rows
    core.exact(traces["O-conf"],traces["O-rand"])
    checks["PF-R3"] = True
    main_parameters = []
    for train_head in (True,False):
        t = Trainer(small,"D3-full",42,device,25,use_kin,batch=4)
        t.supervise_quality = train_head
        for _ in range(50): t.step()
        main_parameters.append({k:v.detach().cpu() for k,v in t.model.state_dict().items()
                                if not k.startswith("decoder.quality.")})
    core.exact(*main_parameters)
    checks["PF-Q2"] = True

    dist = torch.full((2,3,12),2.,device=device)
    dist[0,2] = 1.
    dist[1,0] = dist[1,1] = 0.
    order = torch.tensor([[0,1,2],[1,0,2]],device=device)
    winner,_ = emta_indices(dist,order,torch.full((2,),.5,device=device))
    assert winner.tolist()==[2,1]
    speed = torch.tensor([0.,2.,4.],device=device)
    paths = .4 * torch.arange(8,device=device)[None,:,None] * speed[:,None,None]
    torch.testing.assert_close(emta_threshold("E1",paths),paths.new_tensor([.5,.96,1.5]))
    core.exact(emta_threshold("QFL",paths),emta_threshold("E0",paths))
    z = torch.tensor([[0.,1.]],device=device,requires_grad=True)
    q = torch.tensor([[.5,1.]],device=device,requires_grad=True)
    actual = confidence_loss(z,torch.tensor([0],device=device),q,True)
    y = z.new_tensor([[np.exp(-1),0.]])
    expect = ((z.sigmoid()-y).square()*torch.nn.functional.binary_cross_entropy_with_logits(z,y,reduction="none")).mean()
    torch.testing.assert_close(actual,expect)
    actual.backward()
    assert q.grad is None and torch.isfinite(z.grad).all()
    checks["PF-E1"] = checks["PF-QFL"] = True
    observed = torch.as_tensor(dev["past"][:1000],device=device,dtype=torch.float64)
    local,*_ = core.prepare(observed)
    torch.testing.assert_close(observed_speed(observed[:,0]),observed_speed(local[:,0]),atol=1e-6,rtol=0)
    past = .4*torch.arange(8,device=device,dtype=torch.float64)[None,:,None]
    future = .4*torch.arange(8,20,device=device,dtype=torch.float64)[None,None,:,None]
    ref = kinematic_reference(past)
    assert kinematic_excess(future,past,ref).item()==0
    future[:,:,0] += 3.
    assert kinematic_excess(future,past,ref).item()>0
    checks["PF-K1"] = True
    return checks


def latency(use_kin):
    (p,n),_ = inputs("C0")
    data = first(load_data(p,n),1)
    p,n = (torch.as_tensor(data[k]) for k in ("past","num_valid"))
    report = {}
    with torch.no_grad():
        for arm in ("B0","D3-full","O-conf","O-risk"):
            t = Trainer(data,arm,42,"cpu",2,use_kin)
            m = t.model.eval()
            core.prediction(m,p,n)
            samples = []
            for _ in range(24):
                t0=time.perf_counter()
                core.prediction(m,p,n)
                samples.append((time.perf_counter()-t0)*1000)
            report[arm] = float(np.median(samples))
    return dict(ms=report,risk_over_16_3_ms=report["O-risk"]>16.3)


if __name__=="__main__":
    import argparse, json
    cli=argparse.ArgumentParser()
    cli.add_argument("--device",default="cpu")
    args=cli.parse_args()
    print(json.dumps(gates(Path(__file__).resolve().parent,args.device),indent=2),flush=True)
