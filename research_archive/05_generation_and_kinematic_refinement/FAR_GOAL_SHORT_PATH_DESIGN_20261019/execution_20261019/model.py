import common as c
import torch
from torch import nn
import numpy as np

def setup(seed):
    import random
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False

class GoalConditionedShortPath(nn.Module):
    def __init__(self, stats):
        super().__init__()
        for key in ('p2_mean','p2_std','v_mean','v_std','z_mean','z_std','centers12'):
            self.register_buffer(key,torch.tensor(stats[key],dtype=torch.float32))
        self.register_buffer('s',torch.arange(1,13,dtype=torch.float64)/12)
        self.context=nn.Sequential(nn.Linear(96,128),nn.GELU(),nn.Linear(128,64),nn.GELU())
        self.query=nn.Parameter(torch.empty(20,16));nn.init.normal_(self.query,0,.02)
        self.goal_head=nn.Sequential(nn.Linear(80,64),nn.GELU(),nn.Linear(64,2))
        self.process_head=nn.Sequential(nn.Linear(84,64),nn.GELU(),nn.Linear(64,6))
        for layer in (self.goal_head[-1],self.process_head[-1]):nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
    def encode(self,p2):
        assert p2.ndim==2 and p2.shape[1]==96
        h=self.context((p2-self.p2_mean)/self.p2_std)
        return torch.cat((h[:,None,:].expand(-1,20,-1),self.query[None].expand(len(p2),-1,-1)),-1)
    def decode(self,hq,v0,z):
        assert hq.shape[1:]==(20,80) and v0.shape==(len(hq),2) and z.shape==(len(hq),20,2)
        x=torch.cat((hq,(z-self.z_mean)/self.z_std,((v0-self.v_mean)/self.v_std)[:,None,:].expand(-1,20,-1)),-1)
        abc=self.process_head(x).reshape(-1,20,3,2);a,b,cc=abc.double().unbind(2)
        s=self.s[None,None,:,None];v=v0.double()[:,None,None,:];zz=z.double()[:,:,None,:]
        path=4.8*s*v+s.square()*(zz-4.8*v+a[:,:,None,:])+s.square()*(1-s)*b[:,:,None,:]+s.square()*(1-s)*(2*s-1)*cc[:,:,None,:]
        return dict(path=path,z=z,process_coeff=abc)
    def forward(self,p2,v0):
        hq=self.encode(p2);z=4.8*v0[:,None,:]+self.centers12[None]+self.goal_head(hq)
        return self.decode(hq,v0,z)
    def decode_with_z(self,p2,v0,z_override,*,diagnostic=False):
        assert diagnostic is True,'GT/override not allowed in normal prediction API'
        return self.decode(self.encode(p2),v0,z_override)

def objective(out,y12,y20,m20,horizon):
    assert horizon in (12,20) and m20.dtype==torch.bool
    p,z=out['path'],out['z'];d=torch.linalg.vector_norm(p-y12[:,None],dim=-1)
    ade=d.mean(-1);fde=d[:,:,-1];q=ade+.5*fde;j=q.argmin(1);row=torch.arange(len(p),device=p.device)
    path=q[row,j].mean()
    if bool(m20.any()):
        target=y12[m20,-1] if horizon==12 else y20[m20]
        goal=torch.linalg.vector_norm(z[row,j][m20].double()-(12/horizon)*target,dim=-1).sum()/len(p)
    else:goal=z.sum()*0
    a,b=torch.triu_indices(20,20,1,device=p.device)
    pd=torch.linalg.vector_norm(p[:,a]-p[:,b],dim=-1).mean(-1)
    div=torch.relu(.3-pd).square().mean()
    return path+.5*goal+.05*div,dict(path=path,goal=goal,diversity=div,winner=j,ties=(q==q.min(1,keepdim=True).values).sum(1))

def optimizer(model):return torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8,foreach=False)

def reconstruct(z,abc,v0):
    z=z.astype(float);a,b,cc=np.moveaxis(abc.astype(float),2,0)
    s=np.arange(1,13,dtype=float)[None,None,:,None]/12;v=v0.astype(float)[:,None,None,:]
    return 4.8*s*v+s*s*(z[:,:,None]-4.8*v+a[:,:,None])+s*s*(1-s)*b[:,:,None]+s*s*(1-s)*(2*s-1)*cc[:,:,None]

def numpy_loss(out,y12,y20,m20,horizon):
    p=out['path'];z=out['z'];d=np.sqrt(np.sum((p-y12[:,None])**2,-1));q=d.mean(-1)+.5*d[:,:,-1];j=q.argmin(1);path=q[np.arange(len(p)),j].mean()
    target=y12[m20,-1] if horizon==12 else y20[m20]
    goal=np.sqrt(np.sum((z[np.arange(len(p)),j][m20].astype(float)-(12/horizon)*target)**2,-1)).sum()/len(p)
    pair=[]
    for a in range(20):
        for b in range(a+1,20):pair.append(np.maximum(.3-np.sqrt(np.sum((p[:,a]-p[:,b])**2,-1)).mean(-1),0)**2)
    div=np.array(pair).mean()
    return float(path+.5*goal+.05*div),dict(path=float(path),goal=float(goal),diversity=float(div),winner=j)
