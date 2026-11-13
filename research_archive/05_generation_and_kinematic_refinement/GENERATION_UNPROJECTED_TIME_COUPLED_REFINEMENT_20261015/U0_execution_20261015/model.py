import common as c
import numpy as np,torch,random,os
from torch import nn
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
DEVICE=torch.device('cuda:0');ARMS=('free_position','integrated_acceleration');SEEDS=(42,43,44)

def kernels():
    bank=np.zeros((2,12,12),np.float64)
    for t in range(1,13):
        bank[0,t-1,t-1]=.20/144*t*t
        for j in range(1,t+1):bank[1,t-1,j-1]=.20/144*(2*(t-j)+1)
    return bank

class Refiner(nn.Module):
    def __init__(self,st):
        super().__init__()
        for key in ('mean','std'):self.register_buffer(key,torch.from_numpy(st[key].copy()))
        self.register_buffer('kernelbank',torch.from_numpy(kernels()));self.register_buffer('native_identity_flag',torch.tensor(True))
        self.head=nn.Sequential(nn.Linear(120,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,24))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,p2,native,arm):
        assert arm in ARMS
        raw=torch.cat((p2[:,None].expand(-1,20,-1),native.reshape(-1,20,24).to(p2.dtype)),dim=-1)
        z=self.head(((raw.double()-self.mean)/self.std).to(p2.dtype)).reshape(-1,20,12,2)
        u=torch.tanh(z).double();u=u/torch.linalg.vector_norm(u,dim=-1,keepdim=True).clamp_min(1.)
        r=torch.einsum('tj,bkjd->bktd',self.kernelbank[ARMS.index(arm)],u)
        return native.double()+r,z,u,r

def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def inputs(data,ix):return tuple(torch.from_numpy(data[k][ix].astype(np.float32)).to(DEVICE) for k in ('p2','native'))
def optimizer(net):return torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8,foreach=False)
def objective(p,gt):
    e=torch.linalg.vector_norm(p-gt.double()[:,None],dim=-1);q=e.mean(-1)+.5*e[:,:,-1];v,w=q.min(1)
    pair=torch.triu_indices(20,20,1,device=p.device);dist=torch.linalg.vector_norm(p[:,pair[0]]-p[:,pair[1]],dim=-1).mean(-1);div=torch.relu(.3-dist).square().mean()
    return v.mean()+.05*div,v.mean(),div,w
def objective_numpy(p,gt):
    d=p-gt[:,None];e=np.sqrt((d*d).sum(-1));q=e.mean(-1)+.5*e[:,:,-1];div=[]
    for j in range(20):
        for k in range(j+1,20):div.append(np.maximum(0,.3-np.sqrt(((p[:,j]-p[:,k])**2).sum(-1)).mean(-1))**2)
    return float(q.min(1).mean()+.05*np.mean(div)),q.argmin(1)
def objective_independent_torch(p,gt):
    e=torch.linalg.vector_norm(p-gt.double()[:,None],dim=-1);q=e.mean(-1)+.5*e[:,:,-1];winner=q.argmin(1);v=q.gather(1,winner[:,None]).mean();pieces=[]
    for j in range(20):
        for k in range(j+1,20):pieces.append(torch.relu(.3-torch.linalg.vector_norm(p[:,j]-p[:,k],dim=-1).mean(-1)).square())
    return v+.05*torch.stack(pieces).mean()
def rng_state(g):return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),loader=g.get_state(),numpy=np.random.get_state(),python=random.getstate())
def restore_rng(r,g):torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda']);g.set_state(r['loader']);np.random.set_state(r['numpy']);random.setstate(r['python'])
def checkpoint(path,net,opt,g,history,manifest):
    path=c.H/path;path.parent.mkdir(parents=True,exist_ok=True);tmp=str(path)+'.tmp';torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),rng=rng_state(g),history=history,manifest=manifest),tmp);os.replace(tmp,path)
def restore(path,net,opt,g,manifest):
    v=torch.load(c.H/path,map_location='cpu',weights_only=False);assert v['manifest']==manifest,'BAD_RESUME_SOURCE';net.load_state_dict(v['model']);opt.load_state_dict(v['optimizer']);restore_rng(v['rng'],g);return v['history']
def exact(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b,(a,b)
def step(net,opt,data,ix,arm):
    out=net(*inputs(data,ix),arm)[0];gt=torch.from_numpy(data['gt'][ix]).to(DEVICE);loss,q,d,w=objective(out,gt);opt.zero_grad(set_to_none=True);loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    norm=float(torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in net.parameters()])).item());opt.step()
    return dict(n=len(ix),total=loss.item(),quality=q.item(),diversity=d.item(),gradnorm=norm,winner_counts=np.bincount(w.detach().cpu().numpy(),minlength=20).tolist())
def epoch(net,opt,data,indices,g,index,arm,batch=256):
    net.train();order=indices[torch.randperm(len(indices),generator=g).numpy()];rows=[]
    for start in range(0,len(order),batch):c.guard();rows.append(step(net,opt,data,order[start:start+batch],arm))
    n=sum(x['n'] for x in rows)
    return dict(epoch=index,n=n,optimization_steps=len(rows),**{k:sum(r[k]*r['n'] for r in rows)/n for k in ('total','quality','diversity')},mean_gradnorm=float(np.mean([r['gradnorm'] for r in rows])),winner_counts=np.sum([r['winner_counts'] for r in rows],0).tolist(),preupdate_batch_weighted=True)
def generated(net,data,ix,arm):
    net.eval();rows=[[],[],[],[]]
    with torch.inference_mode():
        for first in range(0,len(ix),128):
            c.guard();out=net(*inputs(data,ix[first:first+128]),arm)
            for r,z in zip(rows,out):r.append(z.cpu().numpy())
    arrays=[np.concatenate(r) for r in rows];assert np.isfinite(arrays[0]).all();return arrays

def reconstruct(native,raw,arm,controls=None):
    # Independent recurrence, not einsum or production kernel multiplication.
    u=np.tanh(raw).astype(np.float64);u/=np.maximum(1,np.sqrt((u*u).sum(-1,keepdims=True)))
    if controls is not None:
        assert np.max(abs(u-controls))<=2e-7,'TANH_BACKEND_ARITHMETIC';u=controls
    if arm==ARMS[0]:r=.20*(np.arange(1,13)/12.)[None,None,:,None]**2*u
    else:
        velocity=np.zeros_like(u[...,0,:]);position=np.zeros_like(velocity);r=np.empty_like(u);unit=.20/144
        for t in range(12):position=position+velocity+unit*u[...,t,:];velocity=velocity+2*unit*u[...,t,:];r[...,t,:]=position
    return native.astype(float)+r,u,r
