import common as c
import numpy as np
N=100000;SEEDS=(42,43,44);CELLS=('A','B','C','D')
CONTRASTS={'B-A':[-1,1,0,0],'C-A':[-1,0,1,0],'D-A':[-1,0,0,1],'D-C':[0,0,-1,1],'D-B':[0,-1,0,1],'D-C-B+A':[1,-1,-1,1]}
def interval(x,point,family):
    out=[]
    for j in range(x.shape[-1]):
        v=x[:,j];v=v[np.isfinite(v)];n=len(v)
        out.append(dict(estimate=float(point[j]),valid=n,invalid=len(x)-n,usable=n>=.95*len(x),CI95=np.quantile(v,[.025,.975]) if n else None,CI_family=np.quantile(v,[.05/(2*family),1-.05/(2*family)]) if n else None))
    return out
def resample(stats,tracks,n=N,ind=False,saved=None):
    # stats shape cell,seed,track,12 sums: 4 numerators,4 denominators,3 errors,N.
    rng=np.random.Generator(np.random.PCG64(20261017));sdraw=np.empty((n,3),np.int16);tdraw=np.empty((n,len(tracks)),np.int32)
    for first in range(0,n,128):
        b=min(128,n-first);sdraw[first:first+b]=rng.integers(0,3,(b,3));tdraw[first:first+b]=rng.integers(0,len(tracks),(b,len(tracks)))
    if saved is not None:
        assert np.array_equal(saved['seeds'],sdraw) and np.array_equal(saved['tracks'],tdraw)
    draws=np.empty((n,4,7));point=np.empty((4,3,7))
    for ci in range(4):
        s=stats[ci].sum(1);point[ci,:,:4]=np.divide(s[:,:4],s[:,4:8],out=np.full((3,4),np.nan),where=s[:,4:8]!=0);point[ci,:,4:]=s[:,8:11]/s[:,11:]
        for first in range(0,n,128):
            c.guard();ss=sdraw[first:first+128];tt=tdraw[first:first+128]
            if ind:
                # Independent explicit draw/selected-seed loop over stored track sufficient statistics.
                sums=np.array([[stats[ci,seed][tr].sum(0) for seed in seedrow] for seedrow,tr in zip(ss,tt)])
            else:sums=stats[ci][ss[:,:,None],tt[:,None,:]].sum(2)
            ratio=np.divide(sums[:,:,:4],sums[:,:,4:8],out=np.full(sums[:,:,:4].shape,np.nan),where=sums[:,:,4:8]!=0);ab=sums[:,:,8:11]/sums[:,:,11:]
            draws[first:first+len(ss),ci]=np.concatenate((ratio,ab),-1).mean(1)
    return draws,point,sdraw,tdraw
def compute(ind=False):
    stats=None;native=None;tracks=None;ids=None
    for ci,cell in enumerate(CELLS):
        for si,seed in enumerate(SEEDS):
            z=c.npz(c.H/f'runs/{cell}/seed{seed}/outer_aux.npz');sel=z['groups']==2;tr=z['track'][sel]
            if tracks is None:
                tracks=np.unique(tr);ids=z['sample_id'][sel];stats=np.zeros((4,3,len(tracks),12));native=np.zeros((3,len(tracks),4))
            assert np.array_equal(ids,z['sample_id'][sel]);new=z['good'][sel].any(1);ng=z['native_vectors4'][sel,0].astype(bool);ones=np.ones(len(tr))
            value=np.column_stack((new,ng&~new,~ng&new,z['J'][sel].min(1),ones,ng,~ng,ones,z['ADE'][sel].min(1),z['FDE'][sel].min(1),z['J'][sel].min(1),ones))
            if ind:
                for ti,t in enumerate(tracks):stats[ci,si,ti]=value[tr==t].sum(0)
            else:np.add.at(stats[ci,si],np.searchsorted(tracks,tr),value)
            if ci==0:
                nv=z['native_vectors4'][sel]
                np.add.at(native[si],np.searchsorted(tracks,tr),np.column_stack((nv[:,0],nv[:,1],nv[:,2],nv[:,3])))
    saved=c.npz(c.H/'bootstrap_draws.npz') if ind else None
    draws,per,ss,tt=resample(stats,tracks,ind=ind,saved=saved)
    counts=stats[0,:,:,4]
    npoint=np.array([native[s].sum(0)/counts[s].sum() for s in range(3)])
    ndraw=np.empty((N,4))
    for first in range(0,N,128):
        s=ss[first:first+128];t=tt[first:first+128];num=native[s[:,:,None],t[:,None,:]].sum(2);den=counts[s[:,:,None],t[:,None,:]].sum(2);ndraw[first:first+len(s)]=(num/den[:,:,None]).mean(1)
    primary={}
    for name,coef in CONTRASTS.items():
        coef=np.asarray(coef);dd=np.einsum('ncm,c->nm',draws[:,:,:4],coef);pp=np.einsum('csm,c->sm',per[:,:,:4],coef)
        primary[name]=dict(per_seed=pp,rows=interval(dd,pp.mean(0),24))
    absolute={}
    for ci in (1,2,3):
        cov=draws[:,ci,0:1]-ndraw[:,0:1];er=draws[:,ci,4:]-ndraw[:,1:]
        absolute[CELLS[ci]]=dict(coverage=interval(cov,np.array([(per[ci,:,0]-npoint[:,0]).mean()]),3),errors=interval(er,(per[ci,:,4:]-npoint[:,1:]).mean(0),9))
    if ind:
        c.close(draws,saved['cell_values']);c.close(ndraw,saved['native_values'])
    else:np.savez_compressed(c.H/'bootstrap_draws.npz',seeds=ss,tracks=tt,track_order=tracks,cell_values=draws,native_values=ndraw)
    return dict(primary=primary,absolute_native=absolute,N=N,family24=24,native_coverage_family=3,native_error_family=9,minimum_valid=95000,seed=20261017,outer_change_ID_digest=c.digest(ids.tolist()),track_count=len(tracks),same_assistant_saved_arithmetic_not_external=True)


