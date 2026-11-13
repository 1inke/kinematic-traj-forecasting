"""One registered primary comparison, whole-track paired-seed draws; no invalid redraw."""
import numpy as np
from env import *
def gather_inner():
    allq={a:[] for a in ARMS};base=[];ids0=groups0=tracks0=None
    for seed in SEEDS:
        folder=HERE/'populations'/f'seed{seed}'/'inner';d=read_json(folder/'done.json');ids=[];groups=[];tracks=[];qs={a:[] for a in ARMS};native=[]
        for file in d['chunks']:
            with np.load(folder/file) as z:
                q=z['quality'];native.append(q[:,0]);ids.append(z['sample_id']);groups.append(z['group']);tracks.append(z['track_key'])
                for arm in ARMS:
                    with np.load(folder/(arm+'_'+file[len('aux_'):])) as out:qs[arm].append(q[np.arange(len(q)),out['action']])
        ids=np.concatenate(ids);groups=np.concatenate(groups);tracks=np.concatenate(tracks)
        if ids0 is None:ids0,groups0,tracks0=ids,groups,tracks
        else:
            assert np.array_equal(ids0,ids) and np.array_equal(groups0,groups) and np.array_equal(tracks0,tracks)
        for arm in ARMS:allq[arm].append(np.concatenate(qs[arm]))
        base.append(np.concatenate(native))
    return {a:np.asarray(v) for a,v in allq.items()},np.asarray(base),ids0,groups0,tracks0
def main():
    q,base,ids,groups,tracks=gather_inner();change=groups==2;unique,idx=np.unique(tracks,return_inverse=True);nt=len(unique)
    if change.sum()!=497 or len(set(tracks[change]))<20:raise ValueError('Primary identity/change track gate')
    counts=np.bincount(idx,weights=change,minlength=nt)
    good={a:q[a][:,:,2]<=1 for a in ARMS};ng=base[:,:,2]<=1
    summed={a:np.stack([np.bincount(idx,weights=good[a][s]*change,minlength=nt) for s in range(3)]) for a in ARMS}
    absol=np.stack([np.stack([np.bincount(idx,weights=(q['S'][s,:,k]-base[s,:,k])*change,minlength=nt) for k in range(2)]) for s in range(3)])
    rng=np.random.default_rng(20261025);trackdraw=np.empty((20000,nt),np.int16);seeddraw=np.empty((20000,3),np.int8)
    boot=np.full(20000,np.nan);fixed=np.full(20000,np.nan);absolute=np.full((20000,2),np.nan)
    maxreview=0.
    for b in range(20000):
        td=rng.integers(nt,size=nt);sd=rng.integers(3,size=3);trackdraw[b]=td;seeddraw[b]=sd;w=np.bincount(td,minlength=nt);den=float(w@counts)
        if den==0:continue
        diff=(summed['S']@w-summed['I']@w)/den;boot[b]=diff[sd].mean();fixed[b]=diff.mean();absolute[b]=(absol@w)[sd].mean(0)/den
        # Second implementation indexes repeated sampled tracks rather than weight multiplication.
        ir=(summed['S'][:,td].sum(1)-summed['I'][:,td].sum(1))/counts[td].sum()
        ia=absol[:,:,td].sum(2)[sd].mean(0)/counts[td].sum()
        maxreview=max(maxreview,abs(boot[b]-ir[sd].mean()),abs(fixed[b]-ir.mean()),float(abs(absolute[b]-ia).max()))
    if maxreview>1e-12:raise ValueError('Independent complete bootstrap arithmetic')
    valid=np.isfinite(boot);insufficient=valid.sum()<19000
    ci=np.quantile(boot[valid],[.025,.975]) if valid.any() else np.array([np.nan,np.nan]);fci=np.quantile(fixed[valid],[.025,.975]) if valid.any() else np.array([np.nan,np.nan])
    per=(good['S'][:,change].mean(1)-good['I'][:,change].mean(1));vsnative=good['S'][:,change].mean(1)-ng[:,change].mean(1)
    signal=not insufficient and bool((per>0).all() and per.mean()>=.02 and ci[0]>0 and fci[0]>0 and (vsnative>=0).all() and vsnative.mean()>=.02)
    joint=np.quantile(absolute[valid],[.0125,.9875],axis=0) if valid.any() else np.full((2,2),np.nan)
    gains=1-q['S'][:,change,:2].mean(1)/base[:,change,:2].mean(1)
    np.savez_compressed(HERE/'BOOTSTRAP_DRAWS.npz',tracks=trackdraw,seeds=seeddraw,track_keys=unique,primary=boot,fixed_seed=fixed,absolute_S_native_ADE_FDE=absolute,valid=valid)
    result=dict(status='INSUFFICIENT' if insufficient else 'PASS_COMPLETE_PRIMARY_BOOTSTRAP',registered_primary='innerchange S-I coverage',signal=signal,per_seed_difference=per.tolist(),mean_difference=float(per.mean()),CI95=ci.tolist(),fixed_seed_CI95=fci.tolist(),S_native_per_seed=vsnative.tolist(),S_native_mean=float(vsnative.mean()),n=497,change_tracks=len(set(tracks[change])),all_inner_tracks=nt,draws=20000,valid=int(valid.sum()),rng=20261025,draw_order='wholeinner tracks then paired3seed',independent_all20000_maxerror=maxreview,absolute_S_native_change_ADE_FDE_gain_perseed=gains.tolist(),absolute_ADE_FDE_joint_Bonferroni2_CI=joint.tolist(),major_error_and_joint_signal=bool((gains>=.05).all() and (joint[1]<0).all()),native_reference_not_three_independent_models=True)
    write_json(HERE/'PRIMARY_COMPARISON.json',result);return result
