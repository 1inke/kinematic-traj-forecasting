from base import *
G=guard('TREE')
import sklearn,joblib
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingClassifier
from tree import parameters,training_rows
from analysis_statistics import bootstrap

def main():
    t=time.perf_counter();folder=HERE/('preflight_cpu_'+str(time.time_ns()));folder.mkdir();records=[]
    for seed in SEEDS:
        lab=opt_labels(seed)
        with np.load(HERE/f'prepared/actions_seed{seed}.npz') as z:a=z['actions'];w=z['weights'];y=z['labels'];pi=z['inclusion_probability']
        for i in range(4096):
            hasneg=(lab['eligible'][i]&~lab['target'][i]).any()
            if not hasneg:assert not w[i].any();continue
            for truth in (True,False):
                pool=np.flatnonzero(lab['eligible'][i]&(lab['target'][i]==truth));cols=np.flatnonzero((w[i]>0)&(y[i]==truth));chosen=a[i,cols];assert len(cols)==min(16,len(pool));assert set(chosen)<=set(pool)
                if 0 in pool:assert 0 in chosen
                for c in cols:
                    prob=1 if a[i,c]==0 or len(pool)<=16 else (15/(len(pool)-1) if 0 in pool else 16/len(pool))
                    assert pi[i,c]==prob and w[i,c]==1/(2*len(pool)*prob)
                assert abs(w[i,cols].sum()-.5)<1e-12
        records.append(dict(seed=seed,windows=4096,actual_rows=int((w>0).sum()),original_IPW_STOP_target_eligible_EXACT=True))
    timings={};x,y,w,ix,act=training_rows(42,256)
    for arm in ('T_full','T_geom'):
        xx=x if arm=='T_full' else x[:,234:324];m=HistGradientBoostingClassifier(**parameters(42));start=time.perf_counter()
        with threadpool_limits(limits=1):m.fit(xx,y,sample_weight=w);pred=m.decision_function(xx)
        assert m.n_iter_==64 and np.isfinite(pred).all();joblib.dump(m,folder/f'{arm}_throwaway.joblib');reload=joblib.load(folder/f'{arm}_throwaway.joblib')
        with threadpool_limits(limits=1):assert np.array_equal(pred,reload.decision_function(xx))
        timings[arm]=dict(seconds=time.perf_counter()-start,windows=256,rows=len(x),iterations=64,not_scientific_model=True)
        G.check()
    # Entire draw algorithm / zero denominator handling using all20000 registered draws.
    track=np.repeat(np.array([f't{i}' for i in range(24)]),2);group=np.full(48,2);arr=np.stack([np.arange(48)%2,np.arange(48)%3==0,np.arange(48)%5==0]).astype(float)
    rec,draw=bootstrap(dict(fixture=arr),group,track);assert rec['fixture']['valid']==20000 and rec['fixture']['change_tracks']==24
    small,_=bootstrap(dict(z=arr),np.zeros(48,int),track,draws=200);assert small['z']['invalid']==200 and small['z']['status']=='UNDERPOWERED'
    save_npz(folder/'bootstrap_fixture.npz',**draw)
    g2=guard('AUDIT');from evaluate_saved import edit,independent_check
    with np.load(HERE/'prepared/tiny.npz') as z:tiny={k:z[k] for k in z.files if k not in ('features','opt_row')}
    scores=np.zeros((128,401));ch,p=edit(tiny,scores);assert not ch.any();start=time.perf_counter();independent_check(tiny,scores,ch,p);saved128seconds=time.perf_counter()-start;g2.check()
    perms=read_json(HERE/'prepared/PERMUTATION_MAP.json');assert perms['blocked_seeds']==[42,43,44]
    resource_report(folder,G);write_json(folder/'AUDIT_RESOURCE.json',dict(last=g2.check(),observations=g2.observations));result=dict(status='PASS',at=now(),source=verify(),seconds=time.perf_counter()-t,records=records,HGB_API=parameters(42),sklearn=sklearn.__version__,tree_subset_timings=timings,CPU128_independent_save_review_seconds=saved128seconds,bootstrap_all20000_independent_counts_EXACT=True,invalid_draws_not_resampled=True,T_perm='BLOCKED_ALL3',no_Torch_imported='torch' not in sys.modules,resource=read_json(folder/'RESOURCE.json'),folder=folder.name)
    assert result['no_Torch_imported'];write_json(folder/'PREFLIGHT.json',result);write_json(HERE/'PREFLIGHT_CPU.json',result);print('PASS CPU PREFLIGHT',result['seconds'],timings,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:fail(e);raise
