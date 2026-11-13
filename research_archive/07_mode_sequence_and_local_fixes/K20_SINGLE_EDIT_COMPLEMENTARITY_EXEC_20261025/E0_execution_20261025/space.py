"""Production one-edit arithmetic. Eligibility API accepts paths only."""
import numpy as np
from common import NAMES
def magnitudes(p):
    p=np.asarray(p,np.float64)
    v=np.diff(p,axis=-2,prepend=np.zeros_like(p[...,:1,:]))/.4
    a=np.diff(v,axis=-2)/.4;j=np.diff(a,axis=-2)/.4
    assert (v.shape[-2],a.shape[-2],j.shape[-2])==(12,11,10)
    return tuple(np.linalg.norm(x,axis=-1) for x in (v,a,j))
def reduce_magnitudes(s,a,j):
    axes=(-2,-1)
    return np.stack((np.quantile(s.reshape(*s.shape[:-2],-1),.95,axis=-1),a.mean(axis=axes),np.quantile(a.reshape(*a.shape[:-2],-1),.95,axis=-1),j.mean(axis=axes),np.quantile(j.reshape(*j.shape[:-2],-1),.95,axis=-1)),axis=-1)
def motion(p):return reduce_magnitudes(*magnitudes(p))
def slot_quality(p,y):
    d=np.linalg.norm(np.asarray(p,np.float64)-np.asarray(y,np.float64)[...,None,:,:],axis=-1)
    a=d.mean(-1);f=d[...,-1];return np.stack((a,f,np.maximum(a/.2,f/.4)),axis=-1)
def materialize(n,k,action):
    p=n.copy()
    if action:r,l=divmod(int(action)-1,20);p[r]=k[l]
    return p
def eligible_actions(n,k):
    nm=magnitudes(n);km=magnitudes(k);base=reduce_magnitudes(*nm)
    out=np.empty((401,5));out[0]=base
    for r in range(20):
        values=[]
        for x,z in zip(nm,km):
            remain=np.delete(x,r,axis=0)
            values.append(np.concatenate((np.broadcast_to(remain,(20,*remain.shape)),z[:,None]),axis=1))
        out[1+20*r:1+20*(r+1)]=reduce_magnitudes(*values)
    rejects=out>1.01*base;rejects[0]=False
    return ~rejects.any(-1),out,rejects
def quality_actions(n,k,y):
    nq=slot_quality(n,y);kq=slot_quality(k,y);out=np.empty((401,3));out[0]=nq.min(0)
    for r in range(20):out[1+20*r:1+20*(r+1)]=np.minimum(np.delete(nq,r,axis=0).min(0),kq)
    return out
def labels(q,eligible):
    B=eligible&(q<=q[0]).all(-1)&(q<q[0]).any(-1)&((q[0,2]>1)|(q[:,2]<=1));B[0]=False
    t=B.copy()
    if not B.any():t[0]=True
    return B,t
def oracle(q,B):
    choices=np.flatnonzero(B)
    if not len(choices):return 0
    # Stable integer action order is r then l, after J,A,F.
    return int(choices[np.lexsort((choices,q[choices,1],q[choices,0],q[choices,2]))[0]])
def enumerate_window(n,k,y):
    eligible,pm,rejects=eligible_actions(n,k);q=quality_actions(n,k,y);B,t=labels(q,eligible)
    return dict(eligible=eligible,motion=pm,rejects=rejects,quality=q,B=B,target=t,margin=q[0]-q,action=oracle(q,B))
