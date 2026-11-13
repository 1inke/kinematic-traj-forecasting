"""Independent NumPy value + path/logit gradient for the fixed B0 objective.

Detach responsibilities, soft targets and winner are computed once from the
reference point and held fixed during the finite-difference verification.
"""
import numpy as np


def softmax(x):
    e=np.exp(x-x.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)


def loss_grad(p,y,z,weights,constants=None):
    p=np.asarray(p,float)[:,0]; y=np.asarray(y,float)[:,0];z=np.asarray(z,float)[:,0]
    b,k,t,_=p.shape
    error=p-y[:,None];dist=np.linalg.norm(error,axis=-1);q=dist.mean(-1)+.25*dist[:,:,-1]
    if constants is None:
        winner=q.argmin(1);gap=q-q.min(1,keepdims=True)
        target=softmax(-gap/(.5*np.maximum(gap.mean(1,keepdims=True),.05)))
        responsibilities=softmax(-q/.1)
        constants=(winner,target,responsibilities)
    winner,target,responsibilities=constants
    gp=np.zeros_like(p);idx=np.arange(b)
    wp=p[idx,winner];er=wp-y
    def huber(x):return np.where(abs(x)<1,.5*x*x,abs(x)-.5)
    def derivative(x):return np.clip(x,-1,1)
    reg=(huber(er).sum(-1)*weights).mean()
    gp[idx,winner]+=derivative(er)*weights[None,:,None]/(b*t)
    endpoint=huber(er[:,-1]).mean();gp[idx,winner,-1]+=.25*derivative(er[:,-1])/(b*2)
    short=huber(er[:,:2]).mean();gp[idx,winner,:2]+=.2*derivative(er[:,:2])/(b*4)
    prob=softmax(z);logprob=z-np.log(np.exp(z-z.max(-1,keepdims=True)).sum(-1,keepdims=True))-z.max(-1,keepdims=True)
    classification=-(target*logprob).sum(-1).mean()-.25*logprob[idx,winner].mean()
    one=np.zeros_like(prob);one[idx,winner]=1
    gz=.5*((prob-target)+.25*(prob-one))/b
    diversity=0.
    for a in range(k):
        for c in range(a+1,k):
            delta=p[:,a,-1]-p[:,c,-1];d=np.linalg.norm(delta,axis=-1)
            diversity+=np.maximum(.4-d,0).mean()/190
            active=(d<.4)&(d>0)
            direction=np.divide(delta,d[:,None],out=np.zeros_like(delta),where=d[:,None]>0)
            value=-.05*direction*active[:,None]/(b*190)
            gp[:,a,-1]+=value;gp[:,c,-1]-=value
    coverage=(responsibilities*q).sum(-1).mean()
    direction=np.divide(error,dist[...,None],out=np.zeros_like(error),where=dist[...,None]>0)
    gp+=.2*responsibilities[:,:,None,None]*direction/(b*t)
    gp[:,:,-1]+=.2*.25*responsibilities[:,:,None]*direction[:,:,-1]/b
    value=reg+.5*classification+.25*endpoint+.05*diversity+.2*short+.2*coverage
    return value,gp[:,None],gz[:,None],constants
