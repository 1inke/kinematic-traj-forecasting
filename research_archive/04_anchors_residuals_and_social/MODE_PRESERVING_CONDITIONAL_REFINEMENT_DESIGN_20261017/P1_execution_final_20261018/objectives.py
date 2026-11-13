"""Registered 2x2, all20 Q task, full190 diversity, B*19 nonowner anchor."""
import numpy as np,torch
CELLS=('A','B','C','D')
def quality(p,gt):
    e=torch.linalg.vector_norm(p-gt.double()[:,None],dim=-1)
    return e.mean(-1)+.5*e[...,-1]
def loss(p,native,gt,cell):
    assert cell in CELLS
    q=quality(p,gt);q0=quality(native.double(),gt);fixed=q0.detach().argmin(1);dynamic=q.detach().argmin(1)
    owner=fixed if cell in ('B','D') else dynamic
    task=q.gather(1,owner[:,None]).mean()
    drift=torch.linalg.vector_norm(p-native.double(),dim=-1);drift=drift.mean(-1)+.5*drift[...,-1]
    mask=torch.ones_like(drift,dtype=torch.bool);mask.scatter_(1,owner[:,None],False)
    anchor=(drift*mask).sum()/(len(p)*19)
    pair=torch.triu_indices(20,20,1,device=p.device)
    dist=torch.linalg.vector_norm(p[:,pair[0]]-p[:,pair[1]],dim=-1).mean(-1)
    diversity=torch.relu(.3-dist).square().mean()
    total=task+.05*diversity+(anchor if cell in ('C','D') else 0.)
    return total,dict(task=task,diversity=diversity,anchor=anchor,owner=owner,dynamic=dynamic,fixed=fixed,Q=q,Q0=q0,mask=mask,drift=drift)
def numpy_components(p,native,gt,cell):
    e=np.sqrt(np.sum((p-gt[:,None])**2,axis=-1));q=e.mean(-1)+.5*e[...,-1]
    en=np.sqrt(np.sum((native-gt[:,None])**2,axis=-1));q0=en.mean(-1)+.5*en[...,-1]
    dynamic=q.argmin(1);fixed=q0.argmin(1);owner=fixed if cell in ('B','D') else dynamic
    dr=np.sqrt(np.sum((p-native)**2,axis=-1));dr=dr.mean(-1)+.5*dr[...,-1]
    mask=np.ones_like(dr,bool);mask[np.arange(len(p)),owner]=False
    task=float(q[np.arange(len(p)),owner].mean());anchor=float(np.where(mask,dr,0).sum()/(len(p)*19))
    div=[]
    for j in range(20):
        for k in range(j+1,20):div.append(np.maximum(0,.3-np.sqrt(((p[:,j]-p[:,k])**2).sum(-1)).mean(-1))**2)
    diversity=float(np.mean(div));return dict(total=task+.05*diversity+(anchor if cell in ('C','D') else 0.),task=task,anchor=anchor,diversity=diversity,owner=owner,dynamic=dynamic,fixed=fixed,Q=q,Q0=q0,drift=dr)



