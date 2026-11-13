import common as c
import numpy as np,torch,random,os
from torch import nn
from features import candidate_features
import objectives as o
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
DEVICE=torch.device('cuda:0');CELLS=('C',);SEEDS=(42,43,44)
c.extract(c.PARENT/'model.py',('kernels','seed','inputs','optimizer','rng_state','restore_rng','checkpoint','restore','exact','reconstruct'),globals())
ARMS=('free_position','integrated_acceleration')

class Refiner(nn.Module):
    def __init__(self,st,representation='velocity'):
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

def coordinate_diagnostic(net,data,ix):
    p=net(*inputs(data,ix))[0].detach().requires_grad_();gt=torch.from_numpy(data['gt'][ix]).to(DEVICE);base=inputs(data,ix)[1]
    total,d=o.loss(p,base,gt,'B');a=torch.autograd.grad(d['base'],p,retain_graph=True)[0];b=torch.autograd.grad(d['keep'],p)[0];an=float(torch.linalg.vector_norm(a));bn=float(torch.linalg.vector_norm(b))
    return dict(n=len(ix),base_coordinate_norm=an,keep_coordinate_norm=bn,cos=float((a*b).sum()/(an*bn)) if an and bn else None,not_parameter_gradient=True)

def step(net,opt,data,ix,cell):
    native=inputs(data,ix)[1];out=net(*inputs(data,ix))[0];gt=torch.from_numpy(data['gt'][ix]).to(DEVICE);total,d=o.loss(out,native,gt,cell);opt.zero_grad(set_to_none=True);total.backward()
    assert torch.isfinite(total) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    norm=float(torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in net.parameters()])));opt.step()
    with torch.no_grad():
        diff=torch.linalg.vector_norm(out-gt.double()[:,None],dim=-1);a=diff.mean(-1)/.2;f=diff[...,-1]/.4;valid=d['valid'];sl=d['slot'];table=np.zeros((20,20),np.int64)
        np.add.at(table,(d['winner'][valid].cpu().numpy(),sl[valid].cpu().numpy()),1)
        return dict(n=len(ix),total=float(total),base=float(d['base']),quality=float(d['quality']),diversity=float(d['diversity']),keep=float(d['keep']),gradnorm=norm,success=int(valid.sum()),keep_active=int(d['active'].sum()),g0_counts=np.bincount(d['good0'].sum(1).cpu().numpy(),minlength=21),winner_counts=np.bincount(d['winner'].cpu().numpy(),minlength=20),keep_slot_counts=np.bincount(sl[valid].cpu().numpy(),minlength=20),winner_keep_cross=table,component_ADE=int((a>=f).sum()),component_FDE=int((f>a).sum()),component_tie=int((a==f).sum()))

def epoch(net,opt,data,indices,g,index,cell,batch=256,diagnostics=True):
    net.train();order=indices[torch.randperm(len(indices),generator=g).numpy()];rows=[]
    for start in range(0,len(order),batch):c.guard();rows.append(step(net,opt,data,order[start:start+batch],cell))
    n=sum(r['n'] for r in rows);out=dict(epoch=index,n=n,optimization_steps=len(rows),preupdate_batch_weighted=True)
    out.update({k:sum(r[k]*r['n'] for r in rows)/n for k in ('total','base','quality','diversity','keep')});out['mean_gradnorm']=float(np.mean([r['gradnorm'] for r in rows]))
    for key in ('success','keep_active','g0_counts','winner_counts','keep_slot_counts','winner_keep_cross','component_ADE','component_FDE','component_tie'):out[key]=np.sum([r[key] for r in rows],axis=0)
    if diagnostics and index in (1,13,25):out['fixed_FIT_coordinate_gradient']=coordinate_diagnostic(net,data,data['split']['FIT'][:256])
    return out

def generated(net,data,ix):
    net.eval();rows=[[],[],[],[]]
    with torch.inference_mode():
        for first in range(0,len(ix),128):
            c.guard();out=net(*inputs(data,ix[first:first+128]))
            for r,z in zip(rows,out):r.append(z.cpu().numpy())
    arrays=[np.concatenate(r) for r in rows];assert all(np.isfinite(x).all() for x in arrays);return arrays
