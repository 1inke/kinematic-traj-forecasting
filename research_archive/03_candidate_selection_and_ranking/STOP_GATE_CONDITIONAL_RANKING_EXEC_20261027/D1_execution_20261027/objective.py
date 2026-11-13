"""Exact original network/I/S AST plus the single registered new D objective."""
from core import *
import torch
from torch import nn
from torch.nn import functional as F
legacy={'__name__':'registered_legacy_functions','torch':torch,'nn':nn,'F':F,'np':np}
pure(OLD/'editor.py',('ActionEditor','loss','choose','numpy_loss_and_score_grad'),legacy)
ActionEditor=legacy['ActionEditor']
def decomposed(scores,eligible,target):
    edits=eligible.clone();edits[:,0]=False;positive=edits&target;negative=edits&~target
    has=edits.any(1);hp=positive.any(1);hn=negative.any(1)
    gate=scores[:,0]*0.;rank=scores[:,0]*0.
    rows=torch.where(has)[0]
    if rows.numel():
        v=scores[rows];best=v.masked_fill(~edits[rows],-torch.inf).max(1).values;margin=best-v[:,0]
        gate[rows]=torch.where(hp[rows],F.softplus(-margin),F.softplus(margin))
    rows=torch.where(hp&hn)[0]
    if rows.numel():
        v=scores[rows];p=v.masked_fill(~positive[rows],-torch.inf).max(1).values;n=v.masked_fill(~negative[rows],-torch.inf).max(1).values
        rank[rows]=F.softplus(1+n-p)
    return (.5*gate+.5*rank).mean(),dict(gate=gate.detach(),rank=rank.detach())
def loss(scores,eligible,target,arm):
    assert scores.shape==eligible.shape==target.shape and eligible[:,0].all()
    return decomposed(scores,eligible,target)[0] if arm=='D' else legacy['loss'](scores,eligible,target,arm)
def choose(scores,eligible):
    assert torch.isfinite(scores).all() and eligible[:,0].all()
    return legacy['choose'](scores,eligible)
def numpy_loss_gradient(s,e,t,arm):
    if arm!='D':return legacy['numpy_loss_and_score_grad'](s,e,t,arm)
    s=np.asarray(s,np.float64);g=np.zeros_like(s);values=[]
    sigmoid=lambda z:float(np.exp(-np.logaddexp(0,-z)))
    for i,row in enumerate(s):
        edits=np.flatnonzero(e[i]);edits=edits[edits!=0];p=edits[t[i,edits]];n=edits[~t[i,edits]];v=0.
        if len(edits):
            a=int(edits[np.argmax(row[edits])]);m=float(row[a]-row[0]);y=bool(len(p));v+=.5*float(np.logaddexp(0,-m if y else m));z=.5*(sigmoid(m)-int(y));g[i,a]+=z;g[i,0]-=z
        if len(p) and len(n):
            a=int(p[np.argmax(row[p])]);b=int(n[np.argmax(row[n])]);z=1+float(row[b]-row[a]);v+=.5*float(np.logaddexp(0,z));vgrad=.5*sigmoid(z);g[i,b]+=vgrad;g[i,a]-=vgrad
        values.append(v)
    return float(np.mean(values)),g/len(s)
def faces(scores,e,t):
    edits=e.clone();edits[:,0]=False;p=edits&t;n=edits&~t
    return tuple(x for mask in (edits,p,n,e&t,e&~t) for x in (mask.any(1).tolist(),scores.masked_fill(~mask,-torch.inf).argmax(1).tolist()))
