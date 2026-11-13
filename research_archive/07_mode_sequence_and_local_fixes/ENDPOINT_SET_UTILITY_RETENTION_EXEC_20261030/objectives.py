"""Frozen M vs M+R vs M+R+S; float32 MSE, float64 continuous same-slot utility."""
import numpy as np
import torch
from model import window_mse

def continuous_j(native,u,gt):
    t=torch.arange(1,13,device=u.device,dtype=torch.float64)/12
    paths=native.double()+.4*t[None,None,:,None]*u.double()[:,:,None,:]
    d=torch.linalg.vector_norm(paths-gt.double()[:,None],dim=-1)
    comp=torch.stack((d.mean(-1)/.2,d[...,-1]/.4),-1)
    j,which=comp.max(-1)
    return j,dict(component=which.detach(),zero_norm=(d==0).detach())

def objective(u,teacher,native,gt,native_j,best,arm):
    mse=window_mse(u,teacher)
    j,activity=continuous_j(native,u,gt)
    nj=native_j.detach().double();best=best.detach().double().min(1).values
    support=nj<=1;covered=support.any(1)
    ret=j.new_zeros(len(j));resc=j.new_zeros(len(j))
    rw=torch.full((len(j),),-1,dtype=torch.long,device=u.device)
    sw=rw.clone();ractive=torch.zeros(len(j),dtype=torch.bool,device=u.device);sactive=ractive.clone()
    ix=torch.where(covered)[0]
    if len(ix):
        value,winner=j[ix].masked_fill(~support[ix],float('inf')).min(1)
        gap=value-nj[ix].min(1).values;ret=ret.index_copy(0,ix,torch.relu(gap).square());rw[ix]=winner;ractive[ix]=gap>0
    ix=torch.where(~covered)[0]
    if len(ix):
        value,winner=j[ix].min(1);gap=value-best[ix]
        resc=resc.index_copy(0,ix,torch.relu(gap).square());sw[ix]=winner;sactive[ix]=gap>0
    lr,ls=ret.mean(),resc.mean()
    total=mse if arm=='M' else mse+lr if arm=='R' else mse+lr+ls if arm=='RS' else None
    if total is None:raise ValueError('unknown arm')
    activity.update(retention_winner=rw.detach(),rescue_winner=sw.detach(),retention_active=ractive.detach(),rescue_active=sactive.detach(),covered=covered.detach())
    return total,dict(mse=mse,retention=lr,rescue=ls),activity

def numpy_objective(u,y,native,gt,arm):
    """Independent scalar and analytic parameter-output gradient; recompute Native support."""
    u,y,native,gt=map(lambda x:np.asarray(x,dtype=np.float64),(u,y,native,gt))
    B=len(u);t=np.arange(1,13)/12;delta=native+.4*t[None,None,:,None]*u[:,:,None]-gt[:,None]
    norms=np.sqrt((delta*delta).sum(-1));unit=np.divide(delta,norms[...,None],out=np.zeros_like(delta),where=norms[...,None]>0)
    ade=norms.mean(-1)/.2;fde=norms[...,-1]/.4;comp=(fde>ade).astype(int);j=np.maximum(ade,fde)
    djA=(unit*(.4*t)[None,None,:,None]).mean(-2)/.2;djF=unit[...,-1,:]*.4/.4
    dj=np.where(comp[...,None]==0,djA,djF)
    dn=np.sqrt(((native-gt[:,None])**2).sum(-1));nj=np.maximum(dn.mean(-1)/.2,dn[...,-1]/.4)
    support=nj<=1;good=support.any(1)
    # caller provides q* separately to avoid deriving fake safe paths here
    return j,dj,nj,support,good,comp

def numpy_loss_and_grad(u,y,native,gt,best,arm):
    u,y=np.asarray(u,float),np.asarray(y,float)
    j,dj,nj,support,good,component=numpy_objective(u,y,native,gt,arm)
    B=len(u);mse=((u-y)**2).mean();grad=2*(u-y)/u.size;lr=ls=0.
    retgrad=np.zeros_like(grad);rescgrad=np.zeros_like(grad)
    for i in range(B):
        if good[i]:
            k=int(np.argmin(np.where(support[i],j[i],np.inf)));gap=max(j[i,k]-nj[i].min(),0.)
            lr+=gap*gap/B;retgrad[i,k]=2*gap/B*dj[i,k]
        else:
            k=int(j[i].argmin());gap=max(j[i,k]-np.asarray(best)[i].min(),0.)
            ls+=gap*gap/B;rescgrad[i,k]=2*gap/B*dj[i,k]
    total=mse if arm=='M' else mse+lr if arm=='R' else mse+lr+ls
    if arm in ('R','RS'):grad+=retgrad
    if arm=='RS':grad+=rescgrad
    return total,grad,dict(mse=mse,retention=lr,rescue=ls)
