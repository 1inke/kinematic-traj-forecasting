import common as c
import numpy as np,torch
c.extract(c.PARENT/'model.py',('objective','objective_numpy','objective_independent_torch'),globals())

def path_J(paths,gt):
    e=torch.linalg.vector_norm(paths.double()-gt.double()[:,None],dim=-1);a=e.mean(-1)/.2;f=e[...,-1]/.4
    return torch.stack((a,f),-1).max(-1).values

def keep_components(pred,native,gt):
    with torch.no_grad():
        j0=path_J(native,gt);good=j0<=1;valid=good.any(1);ref=j0.min(1).values
    n=len(pred);values=pred.sum((1,2,3))*0.;slot=torch.full((n,),-1,device=pred.device,dtype=torch.long);active=torch.zeros(n,device=pred.device,dtype=torch.bool)
    if bool(valid.any()):
        j=path_J(pred[valid],gt[valid]);best,sl=j.masked_fill(~good[valid],float('inf')).min(1)
        penalty=.2*torch.relu(best-ref[valid]);values=values.clone();values[valid]=penalty;slot[valid]=sl;active[valid]=best>ref[valid]
    return values.mean(),dict(values=values,slot=slot,active=active,good0=good,valid=valid,reference=ref)

def loss(pred,native,gt,cell):
    assert cell in ('A','B','C','D');base,q,div,w=objective(pred,gt);keep,diag=keep_components(pred,native,gt)
    return base+(keep if cell in ('B','D') else 0.),dict(base=base,quality=q,diversity=div,keep=keep,winner=w,**diag)

def numpy_components(pred,native,gt):
    def jf(x):
        d=np.sqrt(((x-gt[:,None])**2).sum(-1));a=d.mean(-1)/.2;f=d[...,-1]/.4
        return np.maximum(a,f),a,f
    j0,_,_=jf(native);j,a,f=jf(pred);good=j0<=1;valid=good.any(1);values=np.zeros(len(pred));slot=np.full(len(pred),-1);active=np.zeros(len(pred),bool)
    for i in np.flatnonzero(valid):
        sl=np.flatnonzero(good[i]);k=sl[np.argmin(j[i,sl])];slot[i]=k;values[i]=.2*max(0,j[i,k]-j0[i].min());active[i]=j[i,k]>j0[i].min()
    base,w=objective_numpy(pred,gt)
    return dict(base=base,keep=float(values.mean()),values=values,slot=slot,active=active,good0=good,valid=valid,reference=j0.min(1),ADE_active=a>=f,component_tie=a==f,winner=w)
