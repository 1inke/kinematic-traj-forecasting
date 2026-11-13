"""Registered mean of within-group Q-WTA + LINEAR endpoint hinge."""
import numpy as np
import torch
PAIR=np.triu_indices(20,1)
def loss(p,y):
    pieces=[];info=[]
    for g in range(p.shape[1]):
        d=torch.linalg.vector_norm(p[:,g]-y[:,None],dim=-1);q=d.mean(-1)+.5*d[:,:,-1]
        task,w=q.min(1);end=p[:,g,:,-1]
        pair=torch.linalg.vector_norm(end[:,PAIR[0]]-end[:,PAIR[1]],dim=-1)
        div=torch.relu(.3-pair).mean();pieces.append(task.mean()+.05*div)
        info.append(dict(winner=w,ties=(q==q.min(1,keepdim=True).values).sum(1),hinge=pair<.3,task=task.mean(),div=div))
    return torch.stack(pieces).mean(),info
def numpy_loss_gradient(p,y):
    """Independent analytic all-path gradient, first physical tie, zero norm derivative0."""
    p=np.asarray(p,float);y=np.asarray(y,float);B,G,K,T,_=p.shape
    grad=np.zeros_like(p);values=[];faces=[]
    for g in range(G):
        e=p[:,g]-y[:,None];dist=np.sqrt(np.sum(e*e,-1));q=dist.mean(-1)+.5*dist[:,:,-1];win=np.argmin(q,1)
        unit=np.divide(e,dist[...,None],out=np.zeros_like(e),where=dist[...,None]>0)
        for i,k in enumerate(win):
            grad[i,g,k]+=unit[i,k]/(T*B*G);grad[i,g,k,-1]+=.5*unit[i,k,-1]/(B*G)
        a,b=PAIR;v=p[:,g,a,-1]-p[:,g,b,-1];r=np.sqrt(np.sum(v*v,-1));active=r<.3
        dv=np.divide(v,r[...,None],out=np.zeros_like(v),where=r[...,None]>0)*active[...,None]*(-.05/(B*190*G))
        for j,(k,l) in enumerate(zip(a,b)):grad[:,g,k,-1]+=dv[:,j];grad[:,g,l,-1]-=dv[:,j]
        values.append(q[np.arange(B),win].mean()+.05*np.maximum(.3-r,0).mean())
        faces.append(dict(winner=win,hinge=active))
    return float(np.mean(values)),grad,faces
