import common as c
import numpy as np,torch,random,os,hashlib
from torch import nn
from features import history_torch
torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
DEVICE=torch.device('cuda:0');ARMS=('S','H');SEEDS=(42,43,44)
c.extract(c.PARENT/'model.py',('seed','optimizer','rng_state','restore_rng','checkpoint','restore','exact'),globals())
class Readout(nn.Module):
    def __init__(self,st):
        super().__init__()
        for k in ('mean','std'):self.register_buffer(k,torch.from_numpy(st[k].copy()))
        self.register_buffer('native_identity_flag',torch.tensor(True))
        self.head=nn.Sequential(nn.Linear(146,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,24))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,p2,native,past,arm):
        h=history_torch(past,arm);x=torch.cat((p2[:,None].expand(-1,20,-1),native.reshape(-1,20,24),h[:,None].expand(-1,20,-1)),2)
        raw=self.head(((x.double()-self.mean)/self.std).to(p2.dtype)).reshape(-1,20,12,2)
        return native.double()+raw.double(),raw
def inputs(data,ix):return tuple(torch.from_numpy(data[k][ix].astype(np.float32)).to(DEVICE) for k in ('p2','native','past'))
def loss(p,gt):return (p-gt.double()[:,None]).square().sum(-1).mean()
def state_SHA(net):return c.digest({k:hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest() for k,x in net.state_dict().items()})
def step(net,opt,data,ix,arm):
    p=net(*inputs(data,ix),arm)[0];gt=torch.from_numpy(data['gt'][ix]).to(DEVICE);l=loss(p,gt);opt.zero_grad(set_to_none=True);l.backward()
    assert torch.isfinite(l) and all(x.grad is not None and torch.isfinite(x.grad).all() for x in net.parameters())
    grad=float(torch.linalg.vector_norm(torch.cat([x.grad.reshape(-1) for x in net.parameters()])));opt.step();return dict(n=len(ix),total=float(l),gradnorm=grad)
def epoch(net,opt,data,indices,g,e,arm):
    net.train();order=indices[torch.randperm(len(indices),generator=g).numpy()];rows=[]
    for start in range(0,len(order),256):c.guard();rows.append(step(net,opt,data,order[start:start+256],arm))
    n=sum(x['n'] for x in rows)
    return dict(epoch=e,n=n,optimization_steps=len(rows),total=sum(x['n']*x['total'] for x in rows)/n,mean_gradnorm=float(np.mean([x['gradnorm'] for x in rows])),preupdate_window_weighted=True,objective='DIRECT_READOUT_MSE')
def generated(net,data,ix,arm):
    net.eval();p=[];raw=[]
    with torch.inference_mode():
        for start in range(0,len(ix),128):
            c.guard();a,b=net(*inputs(data,ix[start:start+128]),arm);p.append(a.cpu().numpy());raw.append(b.cpu().numpy())
    return np.concatenate(p),np.concatenate(raw)
