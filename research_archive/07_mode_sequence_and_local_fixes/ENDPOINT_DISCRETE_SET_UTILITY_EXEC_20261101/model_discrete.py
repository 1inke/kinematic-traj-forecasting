"""6800 new parameters. No deployment GT or identity parameter."""
import torch
from torch import nn

class ActionUtilityHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(48,64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,48))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,features):return self.net(features)

def utility_loss(score,value,valid):
    error=(score.double()-value).square();numer=torch.zeros(len(score),device=score.device,dtype=torch.float64);groups=torch.zeros_like(numer)
    for sm in (value>0,value<0,value==0):
        mask=valid&sm;count=mask.sum((1,2));subtotal=torch.where(mask,error,0.).sum((1,2))
        numer=numer+subtotal/count.clamp_min(1);groups=groups+(count>0).double()
    return (numer/groups.clamp_min(1)).mean()
