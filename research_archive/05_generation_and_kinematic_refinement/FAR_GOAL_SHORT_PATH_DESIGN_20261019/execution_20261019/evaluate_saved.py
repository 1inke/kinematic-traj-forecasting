"""Full K20 quality, GT-free complete linkage, and 12v/11a/10j motion."""
import common as c
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster

def errors(p,gt,ind=False):
    e=np.sqrt(np.sum((p-gt[:,None])**2,-1)) if ind else np.linalg.norm(p-gt[:,None],axis=-1)
    A=e.mean(-1);F=e[:,:,-1];J=np.maximum(A/.2,F/.4);good=(A<=.2)&(F<=.4)
    j=J.argmin(1);q=(A+.5*F).argmin(1);segments=e[np.arange(len(p)),j].reshape(-1,3,4).mean(-1)
    return A,F,J,good,j,q,segments
def motion(p,ind=False):
    if ind:
        v=np.concatenate((p[:,:,:1],p[:,:,1:]-p[:,:,:-1]),2)/.4;a=(v[:,:,1:]-v[:,:,:-1])/.4;j=(a[:,:,1:]-a[:,:,:-1])/.4
    else:
        v=np.diff(p,axis=2,prepend=np.zeros_like(p[:,:,:1]))/.4;a=np.diff(v,axis=2)/.4;j=np.diff(a,axis=2)/.4
    assert v.shape[2]==12 and a.shape[2]==11 and j.shape[2]==10
    s,aa,jj=[np.sqrt(np.sum(x*x,-1)) for x in (v,a,j)]
    return dict(speed_p95=float(np.quantile(s,.95,method='linear')),accel_mean=float(aa.mean()),accel_p95=float(np.quantile(aa,.95,method='linear')),jerk_mean=float(jj.mean()),jerk_p95=float(np.quantile(jj,.95,method='linear')))
def clusters(p,good,ind=False):
    ans=np.empty((len(p),2),np.int16)
    for i,row in enumerate(p):
        if i%128==0:c.guard()
        if ind:
            dist=[]
            for a in range(20):
                for b in range(a+1,20):
                    d=np.sqrt(np.sum((row[a]-row[b])**2,-1));dist.append(max(float(d.mean()/.1),float(d[-1]/.2)))
            condensed=np.array(dist)
        else:
            d=np.linalg.norm(row[:,None]-row[None,:],axis=-1);condensed=np.maximum(d.mean(-1)/.1,d[:,:,-1]/.2)[np.triu_indices(20,1)]
        label=fcluster(linkage(condensed,method='complete'),1,criterion='distance');ans[i]=[len(np.unique(label)),len(np.unique(label[good[i]]))]
    return ans
def evaluate(p,gt,groups,native,ids,ind=False,do_clusters=True):
    A,F,J,g,j,q,seg=errors(p,gt,ind);na,nf,nj,ng,_,_,_=errors(native,gt,ind);good=g.any(1);base=ng.any(1)
    cl=clusters(p,g,ind) if do_clusters else np.zeros((len(p),2),int);metrics={}
    for name,mask in c.masks(groups):
        n=int(mask.sum())
        if not n:metrics[name]=dict(n=0);continue
        mot=motion(p[mask],ind);nm=motion(native[mask],ind);lost=(~good&base&mask);rescue=(good&~base&mask)
        metrics[name]=dict(n=n,K=20,good=int(good[mask].sum()),coverage=float(good[mask].mean()),native_good=int(base[mask].sum()),minADE=float(A[mask].min(1).mean()),minFDE=float(F[mask].min(1).mean()),J=float(J[mask].min(1).mean()),rescued=int(rescue.sum()),lost=int(lost.sum()),retained=int((good&base&mask).sum()),neither=int((~good&~base&mask).sum()),lost_over_native_good=float(lost.sum()/base[mask].sum()) if base[mask].any() else None,rescue_ids=ids[rescue],lost_ids=ids[lost],effective_distinct=float(cl[mask,0].mean()),effective_good=float(cl[mask,1].mean()),single_Jclosest_segments=seg[mask].mean(0),Jslot_Qslot_different=int((j[mask]!=q[mask]).sum()),sensitivity_03_05_good=int(((A<=.3)&(F<=.5)).any(1)[mask].sum()),motion=mot,native_motion=nm,motion_failed=[k for k in mot if mot[k]>1.01*nm[k]])
    aux=dict(ADE=A,FDE=F,J=J,good_candidate=g,Jclosest_slot=j,quality_winner=q,single_J_segments=seg,clusters=cl,vectors=np.stack((good,A.min(1),F.min(1),J.min(1)),1))
    return metrics,aux

def compare(a,b):
    # Counts and categorical arrays EXACT; saved arithmetic continuous tolerance
    # never applies to scientific coverage or motion gates.
    if isinstance(a,dict):
        assert set(a)==set(b),(set(a)^set(b))
        for k in a:compare(a[k],b[k])
    elif isinstance(a,(list,tuple,np.ndarray)):
        aa=np.asarray(a);bb=np.asarray(b)
        if aa.dtype.kind in 'fc' or bb.dtype.kind in 'fc':assert np.allclose(aa,bb,atol=2e-12,rtol=0,equal_nan=True),float(np.nanmax(abs(aa-bb)))
        else:assert np.array_equal(aa,bb)
    elif isinstance(a,(float,np.floating)):
        assert abs(a-b)<=2e-12,(a,b)
    else:assert a==b,(a,b)
