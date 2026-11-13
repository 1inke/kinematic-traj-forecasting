"""One fixed family3; whole track then paired seed, all20000 draws independently replayed."""
from common import *
def primary(a,b,groups,tracks,out=None):
    a=np.asarray(a,bool);b=np.asarray(b,bool);mask=np.asarray(groups)==2;keys,inv=np.unique(tracks,return_inverse=True);T=len(keys)
    denom=np.bincount(inv,weights=mask,minlength=T);num=np.stack([np.bincount(inv,weights=(a[s].astype(float)-b[s])*mask,minlength=T) for s in range(3)])
    rng=np.random.default_rng(20261101);trackdraw=np.empty((20000,T),np.int32);seeddraw=np.empty((20000,3),np.int8)
    for j in range(20000):trackdraw[j]=rng.integers(T,size=T);seeddraw[j]=rng.integers(3,size=3)
    # Fixed seeds are a sensitivity on the same whole-track draws, not new pipelines.
    n=num[:,trackdraw].sum(2).T;de=denom[trackdraw].sum(1);per=np.divide(n,de[:,None],out=np.full_like(n,np.nan),where=de[:,None]>0)
    joint=np.take_along_axis(per,seeddraw,axis=1).mean(1);fixed=per.mean(1)
    independent_joint=[];independent_fixed=[]
    for draw,sd in zip(trackdraw,seeddraw):
        weights=np.bincount(draw,minlength=T)[inv];dd=float(weights[mask].sum());vals=[]
        for s in range(3):vals.append(float(np.sum(weights[mask]*(a[s,mask].astype(float)-b[s,mask]))/dd) if dd else np.nan)
        independent_joint.append(float(np.mean([vals[int(s)] for s in sd])));independent_fixed.append(float(np.mean(vals)))
    ej=float(np.nanmax(abs(joint-independent_joint)));ef=float(np.nanmax(abs(fixed-independent_fixed)));assert max(ej,ef)<=2e-12
    def describe(x):
        good=np.isfinite(x);return dict(valid_draws=int(good.sum()),ordinary95=np.quantile(x[good],[.025,.975]).tolist() if good.any() else None,corrected_family3=np.quantile(x[good],[.0083333333,.9916666667]).tolist() if good.any() else None)
    sj,sf=describe(joint),describe(fixed);change_tracks=int(np.count_nonzero(denom));estimable=change_tracks>=20 and sj['valid_draws']>=19000 and sf['valid_draws']>=19000
    if out:npz(out,whole_track_draw=trackdraw,paired_seed_draw=seeddraw,joint=joint,fixedseed=fixed,track_keys=keys,track_change_counts=denom,track_seed_difference=num)
    return dict(status='ESTIMABLE_EXPLORATORY' if estimable else 'NOT_ESTIMABLE',N=int(mask.sum()),change_tracks=change_tracks,all_tracks=T,
        per_seed_delta=[float((a[s,mask].astype(float)-b[s,mask]).mean()) for s in range(3)],mean_delta=float((a[:,mask].astype(float)-b[:,mask]).mean()),
        family=3,draws=20000,RNG=20261101,paired_seed_track=sj,fixedseed_track_sensitivity=sf,independent_all_draw_maxerror=max(ej,ef),no_invalid_redraw=True)

def gates(primary_result,allmetrics,samples,native):
    p=primary_result;relative=[];native_rows=[]
    for s in SEEDS:
        for split in ('FIT','inner'):
            for name in ('overall','stable','transition','change'):
                a=allmetrics['Q3',s,split][name];b=allmetrics['Q1',s,split][name];n=native[split][name]
                m=samples['Q3',s,split]['groups'];mask=np.ones(len(m),bool) if name=='overall' else m==('stable','transition','change').index(name)
                cv=samples['Q3',s,split]['coverage'];ref=samples['Q1',s,split]['coverage'];den=int((ref&mask).sum());lost=int((~cv&ref&mask).sum())
                checks={k:a['motion'][k]<=b['motion'][k]*1.01 for k in ('speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95')}
                if split=='inner':checks.update(ADE=a['ADE']<=b['ADE']*1.01,FDE=a['FDE']<=b['FDE']*1.01,coverage=a['coverage']>=b['coverage']*.99,lost_reference_success=bool(den and lost/den<=.01))
                relative.append(dict(seed=s,split=split,group=name,n=a['n'],Q1_good=b['good'],Q3_good=a['good'],Q3_lost_Q1=lost,Q1_success_denominator=den,checks=checks,pass_=all(checks.values())))
                for arm in ('Q1','Q3'):
                    a=allmetrics[arm,s,split][name];checks={k:a['motion'][k]<=n['motion'][k]*1.01 for k in a['motion']}
                    if split=='inner' and name!='change':checks.update(ADE=a['ADE']<=n['ADE']*1.01,FDE=a['FDE']<=n['FDE']*1.01,coverage=a['coverage']>=n['coverage']*.99,lost_native_success=bool(a['native_good'] and a['lost']/a['native_good']<=.01))
                    native_rows.append(dict(arm=arm,seed=s,split=split,group=name,checks=checks,pass_=all(checks.values())))
    ciok=p['status']=='ESTIMABLE_EXPLORATORY' and p['paired_seed_track']['corrected_family3'][0]>0
    # The sensitivity is disclosed separately; gate text requires corrected paired CI.
    core=dict(each_seed_positive=all(x>0 for x in p['per_seed_delta']),mean_at_least_2pp=p['mean_delta']>=.02,corrected_CI_lower_positive=ciok,all_relative_protections=all(x['pass_'] for x in relative))
    ok=all(core.values())
    return dict(status='Q_MECHANISM_PASS_ALLOW_M_OWN_PREFLIGHT' if ok else 'Q_STOP_MECHANISM_GATE_FAILED',mechanism_gate=core,mechanism_pass=ok,relative_rows=relative,native_safety_rows=native_rows,
        Q3_native_safety=all(x['pass_'] for x in native_rows if x['arm']=='Q3'),M='OWN_PREFLIGHT_REQUIRED' if ok else 'NOT_TRIGGERED_Q_GATE',E='NOT_TRIGGERED_PENDING_QUALIFIED_M',baseline_promoted=False)
