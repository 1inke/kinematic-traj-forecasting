"""Only supervision arm differs; all401 scores then one window loss."""
import torch
from torch import nn
from torch.nn import functional as F
class ActionEditor(nn.Module):
    def __init__(self,mean,std):
        super().__init__();self.register_buffer('feature_mean',torch.as_tensor(mean,dtype=torch.float32).clone());self.register_buffer('feature_std',torch.as_tensor(std,dtype=torch.float32).clone())
        self.net=nn.Sequential(nn.Linear(324,32),nn.GELU(),nn.Linear(32,32),nn.GELU(),nn.Linear(32,1,bias=False))
        nn.init.zeros_(self.net[-1].weight)
        assert sum(p.numel() for p in self.parameters())==11488
    def forward(self,observable_features):
        x=observable_features
        z=torch.cat(((x[...,:323]-self.feature_mean)/self.feature_std,x[...,323:]),dim=-1)
        return self.net(z).squeeze(-1)
def loss(scores,eligible,target,arm):
    if scores.shape!=eligible.shape or target.shape!=scores.shape:raise ValueError('401 full-score contract')
    pos=eligible&target;neg=eligible&~target;hasneg=neg.any(-1)
    if not pos.any(-1).all():raise ValueError('Every window must have positive action')
    if arm=='I':
        pos_loss=(F.softplus(-scores)*pos).sum(-1)/pos.sum(-1)
        neg_loss=(F.softplus(scores)*neg).sum(-1)/neg.sum(-1).clamp_min(1)
        each=.5*(pos_loss+neg_loss)*hasneg
    elif arm=='S':
        pmax=scores.masked_fill(~pos,-torch.inf).max(-1).values
        nmax=scores.masked_fill(~neg,-torch.inf).max(-1).values
        nmax=torch.where(hasneg,nmax,torch.zeros_like(nmax))
        each=F.softplus(1+nmax-pmax)*hasneg
    else:raise ValueError('Only I and S registered')
    return each.mean()
def choose(scores,eligible):return scores.masked_fill(~eligible,-torch.inf).argmax(-1)
def numpy_loss_and_score_grad(scores,eligible,target,arm):
    import numpy as np
    s=np.asarray(scores,np.float64);gradient=np.zeros_like(s);values=[]
    for i,row in enumerate(s):
        pos=np.flatnonzero(eligible[i]&target[i]);neg=np.flatnonzero(eligible[i]&~target[i]);assert len(pos)
        if not len(neg):values.append(0.);continue
        if arm=='I':
            values.append(.5*np.logaddexp(0,-row[pos]).mean()+.5*np.logaddexp(0,row[neg]).mean())
            gradient[i,pos]=-.5/(1+np.exp(row[pos]))/len(pos)
            gradient[i,neg]=.5/(1+np.exp(-row[neg]))/len(neg)
        else:
            p=int(pos[np.argmax(row[pos])]);n=int(neg[np.argmax(row[neg])]);z=1+row[n]-row[p]
            values.append(float(np.logaddexp(0,z)));g=1/(1+np.exp(-z));gradient[i,n]=g;gradient[i,p]=-g
    return float(np.mean(values)),gradient/len(s)
