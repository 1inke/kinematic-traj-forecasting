"""Fixed family18 paired target-track x DSF-seed descriptive bootstrap."""
from run_common import *
METRICS=('coverage','kin_valid_coverage','kin_violation_rate','minADE','minFDE','effective_good')

def read_values(arm,seed=None):
    folder=RUN/'populations'/('native' if arm=='native' else f'{arm}_seed{seed}')/'dev'
    with np.load(folder/'window_metrics.npz',allow_pickle=False) as f:em={k:f[k] for k in ('coverage','ADE','FDE')}
    with np.load(folder/'auxiliary.npz',allow_pickle=False) as f:a={k:f[k] for k in ('ids','groups','kin_valid_coverage','kin_violation_rate','effective_good')}
    v=np.stack([em['coverage'],a['kin_valid_coverage'],a['kin_violation_rate'],em['ADE'].min(1),em['FDE'].min(1),a['effective_good']],-1).astype(np.float64)
    return v,a['ids'],a['groups']

def bootstrap(values,ids,group):
    # values: comparison x seed x window x metric.
    tracks=np.array(['|'.join(str(i).split('|')[:2]) for i in ids]);names=np.unique(tracks);ti=np.searchsorted(names,tracks);mask=group==2
    if len(np.unique(tracks[mask]))<20:raise ValueError('UNDERPOWERED_CHANGE_TRACKS')
    counts=np.bincount(ti,weights=mask.astype(int),minlength=len(names));sums=np.zeros((values.shape[0],3,len(names),6))
    for j in range(values.shape[0]):
        for s in range(3):
            for metric in range(6):sums[j,s,:,metric]=np.bincount(ti,weights=values[j,s,:,metric]*mask,minlength=len(names))
    rng=np.random.default_rng(20261106);draws=np.empty((20000,values.shape[0],6));fixed=np.empty_like(draws);trackdraw=np.empty((20000,len(names)),np.int32);seeddraw=np.empty((20000,3),np.int8);maxerr=0.
    for k in range(20000):
        td=rng.integers(len(names),size=len(names));sd=rng.integers(3,size=3);trackdraw[k]=td;seeddraw[k]=sd
        denominator=counts[td].sum()
        if denominator==0:draws[k]=np.nan;fixed[k]=np.nan;continue
        result=sums[:,:,td].sum(2)/denominator
        draws[k]=result[:,sd].mean(1);fixed[k]=result.mean(1)
        # Independent track multiplicity arithmetic checks EVERY draw.
        w=np.bincount(td,minlength=len(names));second=np.einsum('cstm,t->csm',sums,w)/float(np.dot(w,counts));check=second[:,sd].mean(1)
        maxerr=max(maxerr,float(np.max(abs(check-draws[k]))))
    if maxerr>2e-12:raise ValueError('BOOTSTRAP_SECOND_ARITHMETIC')
    np.savez_compressed(RUN/'BOOTSTRAP_DRAWS.npz',draws=draws,fixedseed=fixed,track_draw=trackdraw,seed_draw=seeddraw,track_names=names)
    rows=[]
    for comp,label in enumerate(('kin-uniform','kin-native','uniform-native')):
        for m,metric in enumerate(METRICS):
            x=draws[:,comp,m];valid=np.isfinite(x);xf=fixed[:,comp,m];point=values[comp][:,mask,m].mean(1)
            rows.append(dict(comparison=label,metric=metric,per_seed=point.tolist(),mean=float(point.mean()),ordinary95=np.quantile(x[valid],[.025,.975]).tolist(),family18=np.quantile(x[valid],[.05/36,1-.05/36]).tolist(),fixedseed_family18=np.quantile(xf[np.isfinite(xf)],[.05/36,1-.05/36]).tolist(),valid_draws=int(valid.sum()),status='ESTIMABLE_EXPLORATORY' if valid.sum()>=19000 else 'UNDERPOWERED'))
    return dict(status='PASS_FIXED_FAMILY18_ALL20000_DRAWS_SECOND_ARITHMETIC',family=18,draws=20000,conditional_on_one_pretrained_generator=True,max_second_error=maxerr,rows=rows)

def main():
    native,ids,group=read_values('native');u=[];k=[]
    for seed in (42,43,44):
        a,ai,ag=read_values('uniform',seed);b,bi,bg=read_values('kin',seed)
        if not np.array_equal(ids,ai) or not np.array_equal(ids,bi) or not np.array_equal(group,ag) or not np.array_equal(group,bg):raise ValueError('COMPARISON_IDS')
        u.append(a);k.append(b)
    u=np.stack(u);k=np.stack(k);v=np.stack([k-u,k-native[None],u-native[None]])
    result=bootstrap(v,ids,group);write(RUN/'PRIMARY_COMPARISON.json',result)
    print('PRIMARY_COMPARISON_PASS',flush=True)

if __name__=='__main__':main()
