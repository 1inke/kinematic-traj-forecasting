from core import *
import torch
from torch import nn
from torch.nn import functional as F
class DiagnosticEditor(nn.Module):
    def __init__(self,mean,std,arm):
        super().__init__();assert arm in ARMS
        self.arm=arm
        self.register_buffer('feature_mean',torch.as_tensor(mean,dtype=torch.float32).clone())
        self.register_buffer('feature_std',torch.as_tensor(std,dtype=torch.float32).clone())
        active=torch.ones(1284,dtype=torch.bool)
        if arm not in ('C_token','C_both'):active[324:1092]=False
        if arm not in ('C_neighbor','C_both'):active[1092:]=False
        self.register_buffer('active_mask',active)
        self.net=nn.Sequential(nn.Linear(1284,32),nn.GELU(),nn.Linear(32,32),nn.GELU(),nn.Linear(32,1,bias=False))
        nn.init.zeros_(self.net[-1].weight)
        assert sum(p.numel() for p in self.parameters())==42208
    def normalized(self,x):
        return ((x-self.feature_mean)/self.feature_std)*self.active_mask.to(x.dtype)
    def forward(self,x):return self.net(self.normalized(x)).squeeze(-1)
def weighted_loss(scores,weights,target):
    return (weights*torch.where(target,F.softplus(-scores),F.softplus(scores))).sum(-1).mean()
