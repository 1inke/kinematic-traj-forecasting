import common as c
import numpy as np
ARMS=('free_position','integrated_acceleration');SEEDS=(42,43,44)
def compute(ind=False):
    rng=np.random.Generator(np.random.PCG64(20261015));a=[c.npz(c.H/f'runs/{ARMS[0]}/seed{s}/outer_aux.npz') for s in SEEDS];b=[c.npz(c.H/f'runs/{ARMS[1]}/seed{s}/outer_aux.npz') for s in SEEDS]
    base=c.npz(c.H/'native/outer_aux.npz');m=base['groups']==2;tracks,inv=np.unique(base['track'][m],return_inverse=True);T=len(tracks);counts=np.bincount(inv);diffs=[np.stack([z['vectors4'][m]-base['vectors4'][m] for z in b]),np.stack([z['vectors4'][m]-base['vectors4'][m] for z in a]),np.stack([y['vectors4'][m]-x['vectors4'][m] for x,y in zip(a,b)])];sums=[]
    for diff in diffs:
        arr=np.zeros((3,T,4))
        if ind:
            for k in range(T):arr[:,k]=diff[:,inv==k].sum(1)
        else:
            for k in range(3):np.add.at(arr[k],inv,diff[k])
        sums.append(arr)
    draws=[[],[],[]];seeds=[];trackdraw=[];N=20000
    for start in range(0,N,128):
        c.guard();B=min(128,N-start);ss=np.array([rng.integers(0,3,3) for _ in range(B)]);tt=np.array([rng.integers(0,T,T) for _ in range(B)]);seeds.append(ss);trackdraw.append(tt)
        weights=np.zeros((B,T),np.int64);np.add.at(weights,(np.repeat(np.arange(B),T),tt.ravel()),1);den=3*(weights@counts)
        for rows,sm in zip(draws,sums):
            z=np.einsum('bt,stm->bsm',weights,sm,optimize=False);rows.append(z[np.arange(B)[:,None],ss].sum(1)/den[:,None])
    saved=c.H/'bootstrap_draws.npz';ss=np.concatenate(seeds);tt=np.concatenate(trackdraw)
    if ind:
        z=c.npz(saved);assert np.array_equal(z['seed_draws'],ss) and np.array_equal(z['track_draws'],tt)
    else:np.savez_compressed(saved,seed_draws=ss,track_draws=tt,tracks=tracks)
    rows=[]
    for ref,d,parts in zip(('coupled-native','free-native','coupled-free'),diffs,draws):
        values=np.concatenate(parts);rows.append(dict(comparison=ref,metrics=['coverage','ADE','FDE','J'],n=int(m.sum()),tracks=T,estimate=d.mean((0,1)),per_seed=d.mean(1),CI95=np.quantile(values,[.025,.975],axis=0).T,CI_family12=np.quantile(values,[.05/24,1-.05/24],axis=0).T))
    return dict(status='PASS',primary='outer change final25',family=12,draws=20000,PCG64seed=20261015,rows=rows,draws_SHA=c.sha(saved),adaptive_exploratory_not_independent_wholemodel=True)
