import numpy as np
import common as c
N=20000;BLOCK=128;LABELS=('native_failure_dJ_fixed','native_success_lost_rate','native_failure_rescue_rate','allchange_deltaJ_fixed')

def features(w,sensitivity=False):
    nf=~w['C0'];ns=w['C0'];ones=np.ones(len(ns));certain=~w['uncertain_fixed'] if sensitivity else np.ones(len(ns),bool)
    num=np.stack((np.where(nf&certain,w['dJ_fixed'],0.),(w['event']==2).astype(float),(w['event']==3).astype(float),np.where(certain,w['deltaJ_fixed'],0.)),1)
    den=np.stack((nf&certain,ns,nf,certain),1).astype(float)
    return np.concatenate((num,den),1)

def ratio(sums):
    return np.divide(sums[...,:4],sums[...,4:],out=np.full(sums[...,:4].shape,np.nan),where=sums[...,4:]>0)

def sufficient(jobs,independent=False,sensitivity=False):
    populations=[]
    for split in ('FIT','outer'):
        items=[];track=None;ids=None
        for seed in c.SEEDS:
            row=jobs[f'integrated_acceleration/seed{seed}/{split}'];w=c.npz(row['window']);m=w['groups']==2
            now=w['track'][m];nid=w['sample_id'][m]
            if track is None:track=now;ids=nid
            else:assert np.array_equal(track,now) and np.array_equal(ids,nid)
            items.append(features({k:v[m] for k,v in w.items()},sensitivity))
        tracks,inv=np.unique(track,return_inverse=True);arr=np.stack(items);sm=np.zeros((3,len(tracks),8))
        if independent:
            for t in range(len(tracks)):sm[:,t]=arr[:,inv==t].sum(1)
        else:
            for seed in range(3):np.add.at(sm[seed],inv,arr[seed])
        populations.append(dict(split=split,tracks=tracks,stats=sm,n=len(track)))
    return populations

def compute(jobs,independent=False,sensitivity=False,draw_path=None):
    pops=sufficient(jobs,independent,sensitivity);rng=np.random.Generator(np.random.PCG64(20261015));draws=[];sssave=[];tsave=[[],[]]
    for first in range(0,N,BLOCK):
        c.guard();B=min(BLOCK,N-first)
        # Fixed draw order: entireblock pairedmodel seeds, FITtracks, outertracks.
        ss=rng.integers(0,3,size=(B,3));sssave.append(ss.astype(np.int8));values=[]
        for pop,saved in zip(pops,tsave):
            T=len(pop['tracks']);tt=rng.integers(0,T,size=(B,T));saved.append(tt.astype(np.int32));weights=np.zeros((B,T),np.int64)
            np.add.at(weights,(np.repeat(np.arange(B),T),tt.ravel()),1)
            totals=np.einsum('bt,stm->bsm',weights,pop['stats'],optimize=False)
            values.append(ratio(totals[np.arange(B)[:,None],ss].sum(1)))
        draws.append(values[1]-values[0])
    arrays=dict(seed_draws=np.concatenate(sssave),FIT_track_draws=np.concatenate(tsave[0]),outer_track_draws=np.concatenate(tsave[1]),FIT_tracks=pops[0]['tracks'],outer_tracks=pops[1]['tracks'])
    path=c.H/'bootstrap_draws.npz' if draw_path is None else draw_path
    if independent or sensitivity:
        old=c.npz(path)
        for k,v in arrays.items():assert np.array_equal(old[k],v),('DRAW_REPLAY_MISMATCH',k)
    else:c.save(path,**arrays)
    vals=np.concatenate(draws);points=[ratio(p['stats'].sum(1)) for p in pops];gap=points[1]-points[0];rows=[]
    for metric in range(4):
        ok=np.isfinite(vals[:,metric]);z=vals[ok,metric];valid=int(ok.sum())
        public=lambda x:[float(v) if np.isfinite(v) else None for v in x]
        rows.append(dict(metric=LABELS[metric],FIT_per_seed=public(points[0][:,metric]),outer_per_seed=public(points[1][:,metric]),gap_per_seed=public(gap[:,metric]),estimate=float(gap[:,metric].mean()) if np.isfinite(gap[:,metric]).all() else None,valid_draws=valid,invalid_draws=N-valid,CI_usable_for_branch=valid>=19000,CI95=np.quantile(z,[.025,.975]) if valid else None,CI_family4=np.quantile(z,[.05/8,1-.05/8]) if valid else None))
    return dict(status='PASS',primary_arm='integrated_acceleration',contrast='outer_change_minus_FIT_change',population_n={p['split']:p['n'] for p in pops},tracks={p['split']:len(p['tracks']) for p in pops},draws=N,PCG64seed=20261015,family=4,draw_order='block128: all_modelseeds then FITtracks then outertracks',paired_modelseeds_independent_population_tracks=True,no_invalid_resampling=True,sensitivity_excludes_fixed_j0_uncertain_for_direction_and_delta_only=sensitivity,rows=rows,draws_SHA=c.sha(path),not_causal=True)
