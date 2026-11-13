import common as c
import numpy as np,torch,random,os
from torch import nn
from features import candidate_features
import objectives as o
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
DEVICE=torch.device('cuda:0');CELLS=('A','B','C','D');SEEDS=(42,43,44)
c.extract(c.PARENT/'model.py',('kernels','seed','inputs','optimizer','rng_state','restore_rng','checkpoint','restore','exact','reconstruct'),globals())
ARMS=('free_position','integrated_acceleration')

class Refiner(nn.Module):
    def __init__(self,st,representation='position'):
        super().__init__();assert representation in ('position','velocity');self.representation=representation
        for key in ('mean','std'):self.register_buffer(key,torch.from_numpy(st[key].copy()))
        self.register_buffer('kernelbank',torch.from_numpy(kernels()));self.register_buffer('native_identity_flag',torch.tensor(True))
        self.head=nn.Sequential(nn.Linear(120,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,24))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,p2,native):
        raw=torch.cat((p2[:,None].expand(-1,20,-1),candidate_features(native,self.representation)),dim=-1)
        z=self.head(((raw.double()-self.mean)/self.std).to(p2.dtype)).reshape(-1,20,12,2)
        u=torch.tanh(z).double();u=u/torch.linalg.vector_norm(u,dim=-1,keepdim=True).clamp_min(1.)
        r=torch.einsum('tj,bkjd->bktd',self.kernelbank[1],u)
        return native.double()+r,z,u,r

def coordinate_diagnostic(net,data,ix,cell):
    native=inputs(data,ix)[1];p=net(*inputs(data,ix))[0].detach().requires_grad_();gt=torch.from_numpy(data['gt'][ix]).to(DEVICE)
    _,d=o.loss(p,native,gt,cell);vec=[torch.autograd.grad(d[k],p,retain_graph=True)[0].flatten() for k in ('task','anchor','diversity')]
    norms=[float(x.norm()) for x in vec];cos={}
    for i in range(3):
        for j in range(i+1,3):cos[str(i)+':'+str(j)]=float((vec[i]*vec[j]).sum()/(norms[i]*norms[j])) if norms[i] and norms[j] else None
    return dict(n=len(ix),components=['task','anchor','diversity'],coordinate_norms=norms,cos=cos,not_parameter_gradient=True)
def snapshot(net,data,folder,index,cell):
    ix=data['split']['FIT'][:256];net.eval()
    with torch.no_grad():
        p,raw,u,r=net(*inputs(data,ix));_,d=o.loss(p,inputs(data,ix)[1],torch.from_numpy(data['gt'][ix]).to(DEVICE),cell)
    np.savez_compressed(folder/f'diagnostic_epoch{index:02d}.npz',sample_id=data['ids'][ix],paths=p.cpu().numpy(),raw=raw.cpu().numpy(),controls=u.cpu().numpy(),Q=d['Q'].cpu().numpy(),Q0=d['Q0'].cpu().numpy(),owner=d['owner'].cpu().numpy(),dynamic=d['dynamic'].cpu().numpy(),fixed=d['fixed'].cpu().numpy())
    return coordinate_diagnostic(net,data,ix,cell)
def step(net,opt,data,ix,cell):
    native=inputs(data,ix)[1];out=net(*inputs(data,ix))[0];gt=torch.from_numpy(data['gt'][ix]).to(DEVICE)
    total,d=o.loss(out,native,gt,cell);opt.zero_grad(set_to_none=True);total.backward()
    assert torch.isfinite(total) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    norm=float(torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in net.parameters()])));opt.step()
    with torch.no_grad():
        ties=(d['Q']==d['Q'].min(1).values[:,None]).sum(1)>1;ties0=(d['Q0']==d['Q0'].min(1).values[:,None]).sum(1)>1
        return dict(n=len(ix),total=float(total),task=float(d['task']),diversity=float(d['diversity']),anchor=float(d['anchor']),gradnorm=norm,disagree=int((d['dynamic']!=d['fixed']).sum()),ties=int(ties.sum()),native_ties=int(ties0.sum()),owner_counts=np.bincount(d['owner'].cpu().numpy(),minlength=20),dynamic_counts=np.bincount(d['dynamic'].cpu().numpy(),minlength=20),fixed_counts=np.bincount(d['fixed'].cpu().numpy(),minlength=20))
def epoch(net,opt,data,indices,g,index,cell,batch=256,diagnostics=True):
    net.train();order=indices[torch.randperm(len(indices),generator=g).numpy()];rows=[]
    for start in range(0,len(order),batch):c.guard();rows.append(step(net,opt,data,order[start:start+batch],cell))
    n=sum(r['n'] for r in rows);out=dict(epoch=index,n=n,optimization_steps=len(rows),preupdate_batch_weighted=True,batchorder_SHA=c.digest(order.tolist()))
    out.update({k:sum(r[k]*r['n'] for r in rows)/n for k in ('total','task','diversity','anchor')});out['mean_gradnorm']=float(np.mean([r['gradnorm'] for r in rows]))
    for k in ('disagree','ties','native_ties','owner_counts','dynamic_counts','fixed_counts'):out[k]=np.sum([r[k] for r in rows],axis=0)
    out['steps']=rows
    return out
def generated(net,data,ix):
    net.eval();rows=[[],[],[],[]]
    with torch.inference_mode():
        for first in range(0,len(ix),128):
            c.guard();out=net(*inputs(data,ix[first:first+128]))
            for r,z in zip(rows,out):r.append(z.cpu().numpy())
    arrays=[np.concatenate(r) for r in rows];assert all(np.isfinite(x).all() for x in arrays);return arrays

