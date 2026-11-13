"""Exact decoder mechanisms. Forward accepts features only (no GT/groups/IDs)."""
import numpy as np
import torch
from torch import nn

class ContextK20(nn.Module):
    def __init__(self,mean,std,arm,projection=None):
        super().__init__()
        allowed=('A_cv_free','A_cv_basis','C1_hard','C1_awta','C2_setmix','C2_isomlp')
        if arm not in allowed: raise ValueError(arm)
        self.arm=arm
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32).clone())
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32).clone())
        self.trunk=nn.Sequential(nn.Linear(210,128),nn.GELU(),nn.Linear(128,128),nn.GELU())
        self.queries=nn.Parameter(torch.randn(20,16)*.02)
        self.fuse=nn.Sequential(nn.Linear(144,128),nn.GELU(),nn.Linear(128,128),nn.GELU())
        self.head=nn.Linear(128,24);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        if arm.startswith('C2_'):
            self.mix1=nn.Linear(384,32);self.mix2=nn.Linear(32,128)
            nn.init.zeros_(self.mix2.weight);nn.init.zeros_(self.mix2.bias)
        if arm.startswith('A_'):
            p=np.eye(12) if arm=='A_cv_free' else projection
            if p is None or np.shape(p)!=(12,12): raise ValueError('Locked projection required')
            self.register_buffer('projection',torch.tensor(p,dtype=torch.float32))
        expected=98552 if arm.startswith('C2_') else 82008
        assert sum(p.numel() for p in self.parameters())==expected

    def slot_features(self,x):
        h=self.trunk((x-self.mean)/self.std)[:,None].expand(-1,20,-1)
        return self.fuse(torch.cat((h,self.queries[None].expand(len(x),-1,-1)),dim=-1))

    def communicate(self,h):
        if self.arm=='C2_setmix':
            z=torch.cat((h,h.mean(1,keepdim=True).expand_as(h),torch.amax(h,dim=1,keepdim=True).expand_as(h)),-1)
        elif self.arm=='C2_isomlp': z=torch.cat((h,h,h),-1)
        else: return h
        return h+self.mix2(torch.nn.functional.gelu(self.mix1(z)))

    def forward(self,features):
        r=self.head(self.communicate(self.slot_features(features))).reshape(-1,20,12,2)
        if self.arm.startswith('A_'):
            r=torch.einsum('tu,bkud->bktd',self.projection,r)
            cv=(features[:,16:18,None]*torch.arange(1,13,device=r.device,dtype=r.dtype)[None,None]*.4).transpose(1,2)
            return r+cv[:,None]
        return r

def temperature(step):
    if not 1<=step<=592: raise ValueError('Registered steps 1..592')
    return .20*(.10**((step-1)/591))

def objective(paths,gt,arm,step):
    e=torch.linalg.vector_norm(paths-gt[:,None],dim=-1)
    q=e.mean(-1)+.5*e[:,:,-1]; best,winner=q.min(1)
    if arm=='C1_awta':
        tau=temperature(step); weights=torch.softmax(-q/tau,dim=1).detach()
        task=(weights*q).sum(1).mean()
    else:
        tau=None;weights=torch.nn.functional.one_hot(winner,20).to(q.dtype);task=best.mean()
    pair=torch.triu_indices(20,20,offset=1,device=paths.device)
    ends=paths[:,:,-1];div=torch.relu(.3-torch.linalg.vector_norm(ends[:,pair[0]]-ends[:,pair[1]],dim=-1)).mean()
    value=task+.05*div
    # Original S1/T1 A included the identically-zero no-teacher branch. Retain
    # that exact control graph for the required bitwise C1_hard reproduction.
    if arm=='C1_hard':value=value+paths.sum()*0
    return value,dict(task=task,div=div,winner=winner,weights=weights,tau=tau,
        entropy=-(weights*weights.clamp_min(torch.finfo(q.dtype).tiny).log()).sum(1),Q=q)
