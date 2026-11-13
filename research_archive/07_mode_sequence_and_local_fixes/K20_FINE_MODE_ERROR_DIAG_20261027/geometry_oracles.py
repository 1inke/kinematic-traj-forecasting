from common import *
from scipy.spatial.distance import cdist
RATES=np.linspace(.75,1.25,21)
def native_progress(p):
    z=np.concatenate((np.zeros((len(p),1,2)),p),axis=1)
    lengths=np.linalg.norm(np.diff(z,axis=1),axis=-1);total=lengths.sum(1)
    q=np.divide(np.cumsum(lengths,axis=1),total[:,None],out=np.zeros_like(lengths),where=total[:,None]>0)
    q[:,-1]=1
    return z,lengths,total,q
def at_progress(z,origq,q):
    # Batch piecewise-linear interpolation; exact origin and endpoint are forced.
    knots=np.concatenate((np.zeros((len(origq),1)),origq),1)
    ix=(q[...,None]>=knots[:,None,:]).sum(-1)-1;ix=np.clip(ix,0,11)
    rows=np.arange(len(q))[:,None];lo=knots[rows,ix];hi=knots[rows,ix+1]
    t=np.divide(q-lo,hi-lo,out=np.zeros_like(q),where=hi>lo)
    out=z[rows,ix]+t[...,None]*(z[rows,ix+1]-z[rows,ix])
    out[:,-1]=z[:,-1]
    return out
def time_from_rates(p,a,b,valid):
    z,length,total,q=native_progress(p)
    w1=q[:,3];w2=q[:,7]-w1;w3=1-q[:,7]
    c=np.divide(1-a*w1-b*w2,w3,out=np.ones(len(p)),where=w3>0)
    qq=np.where(np.arange(12)<4,q*a[:,None],np.where(np.arange(12)<8,w1[:,None]*a[:,None]+(q-w1[:,None])*b[:,None],w1[:,None]*a[:,None]+w2[:,None]*b[:,None]+(q-q[:,7,None])*c[:,None]))
    qq[:,-1]=1
    out=at_progress(z,q,qq);out[~valid]=p[~valid]
    identity=(a==1)&(b==1);out[identity]=p[identity]
    return out,qq,c
def time_search(p,gt,chunk=128,progress=False):
    shape=p.shape;p=p.reshape(-1,12,2);targets=np.repeat(gt,shape[1],axis=0)
    z,length,total,q=native_progress(p);valid=(total>0)&np.all(length>0,axis=1)
    # Identical mathematical grid, vectorized per original polyline.
    aa,bb=np.meshgrid(RATES,RATES,indexing='ij');ra=aa.ravel();rb=bb.ravel()
    a=np.ones(len(p));b=a.copy();counts=np.zeros(len(p),np.int16)
    before=np.linalg.norm(p-targets,axis=-1).mean(1);best=before.copy()
    for i in range(len(p)):
        if not valid[i]:continue
        w1=q[i,3];w2=q[i,7]-w1;w3=1-q[i,7]
        rc=(1-ra*w1-rb*w2)/w3;ok=(rc>=.75)&(rc<=1.25)
        counts[i]=ok.sum();x=ra[ok];y=rb[ok];c=rc[ok]
        qq=np.where(np.arange(12)<4,q[i]*x[:,None],np.where(np.arange(12)<8,w1*x[:,None]+(q[i]-w1)*y[:,None],w1*x[:,None]+w2*y[:,None]+(q[i]-q[i,7])*c[:,None]));qq[:,-1]=1
        curve=np.stack([np.interp(qq,np.r_[0,q[i]],z[i,:,dim]) for dim in range(2)],-1);curve[:,-1]=p[i,-1]
        identity=(x==1)&(y==1);curve[identity]=p[i]
        ade=np.linalg.norm(curve-targets[i],axis=-1).mean(1)
        k=int(np.argmin(ade))
        if ade[k]<best[i]:best[i]=ade[k];a[i]=x[k];b[i]=y[k]
        if progress and (i+1)%8192==0:mark('O_TIME_FINITE_GRID',candidate_done=i+1,candidate_total=len(p))
    assert np.all(best<=before)
    return dict(rate_a=a.reshape(shape[:2]),rate_b=b.reshape(shape[:2]),time_feasible=valid.reshape(shape[:2]),grid_feasible_count=counts.reshape(shape[:2]),best_ADE=best.reshape(shape[:2]))
def shape_samples(p,count=33):
    p=p.reshape(-1,12,2);z,length,total,q=native_progress(p);out=np.empty((len(p),count,2));grid=np.linspace(0,1,count)
    for i in range(len(p)):
        if total[i]==0:out[i]=z[i,0];continue
        knots=np.r_[0,q[i]];keep=np.r_[True,np.diff(knots)>0]
        for c in range(2):out[i,:,c]=np.interp(grid,knots[keep],z[i,keep,c])
    return out
def motion(p):
    zero=np.zeros(p.shape[:-2]+(1,2));v=np.diff(np.concatenate((zero,p),axis=-2),axis=-2)/.4
    speed=np.linalg.norm(v,axis=-1);acc=np.linalg.norm(np.diff(v,axis=-2)/.4,axis=-1);jerk=np.linalg.norm(np.diff(v,n=2,axis=-2)/(.4**2),axis=-1)
    window=np.stack((np.quantile(speed,.95,axis=-1),acc.mean(-1),np.quantile(acc,.95,axis=-1),jerk.mean(-1),np.quantile(jerk,.95,axis=-1)),-1)
    return dict(window=window,speed=speed,acc=acc,jerk=jerk)
def pooled(m,mask):
    return [float(np.quantile(m['speed'][mask],.95)),float(m['acc'][mask].mean()),float(np.quantile(m['acc'][mask],.95)),float(m['jerk'][mask].mean()),float(np.quantile(m['jerk'][mask],.95))]
