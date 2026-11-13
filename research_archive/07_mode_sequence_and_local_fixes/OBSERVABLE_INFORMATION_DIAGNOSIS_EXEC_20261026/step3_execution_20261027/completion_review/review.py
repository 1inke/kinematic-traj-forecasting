"""CPU saved-only supplement. Full cluster/motion repeat is inherited by SHA,
not falsely claimed to run here again. Repeat choice/slots/counts/draws fully.
"""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from core import *
from src.f16.eval import errors
OUT=HERE/'completion_review'
def checked(folder):
    d=read_json(folder/'done.json');completed(folder,d['contract']);return d
def main():
    g=guard('CPU');source0=verify();assert read_json(HERE/'worker_state.json')['status']=='SUCCESS' and read_json(HERE/'exit_status.json')['status']=='SUCCESS'
    preserved=read_json(HERE/'PRESERVATION.json')
    for p,h in preserved['outputs'].items():assert sha256(HERE/p)==h,p
    groups=[];holds={};nwindow=0
    for arm in ('C_base','C_token'):
        holds[arm]=[]
        for seed in SEEDS:
            checked(HERE/f'runs/{arm}_seed{seed}')
            for split in ('opt','hold','inner'):
                f=HERE/f'populations/{arm}_seed{seed}/{split}';d=checked(f);ad=checked(f/'audit')
                assert ad['full_independent_complete_link'] and ad['pooled_all5']
                m=read_json(f/'audit/METRICS.json');a=read_json(f/'audit/ASSIGNMENT.json')
                with np.load(f/'audit/sample_metrics.npz') as q:sm={k:q[k] for k in q.files}
                lo=0
                for name,z in zip(d['chunks'],aux_chunks(seed,split)):
                    with np.load(f/name) as q:s={k:q[k] for k in q.files}
                    for k in ('sample_id','eligible','B','target'):assert np.array_equal(s[k],z[k])
                    n=len(z['sample_id']);sl=slice(lo,lo+n)
                    assert np.isfinite(s['scores']).all() and s['scores'].shape==(n,401)
                    choice=np.where(z['eligible'],s['scores'],-np.inf).argmax(1)
                    assert np.array_equal(choice,s['chosen']);p=z['native'].copy()
                    for i,v in enumerate(choice):
                        if v:r,l=divmod(int(v)-1,20);p[i,r]=z['kin'][i,l]
                    assert np.array_equal(p,s['paths'])
                    dist=np.linalg.norm(p-z['gt'][:,None],axis=-1);ade=dist.mean(-1);fde=dist[:,:,-1];j=np.maximum(ade/.2,fde/.4)
                    for k,v in (('ADE',ade),('FDE',fde),('J',j)):assert np.max(abs(v-sm[k][sl]))<=2e-12
                    assert np.array_equal(((ade<=.2)&(fde<=.4)).any(1),sm['good'][sl])
                    assert np.array_equal(errors(z['native'],z['gt'])['coverage'],sm['native_good'][sl])
                    assert np.array_equal(np.argmin(ade+.5*fde,1),sm['Q_slot'][sl]) and np.array_equal(np.argmin(j,1),sm['J_slot'][sl])
                    assert np.array_equal(z['sample_id'],sm['sample_id'][sl]) and np.array_equal(choice,sm['chosen'][sl]);lo+=n;g.check()
                assert lo==len(sm['sample_id']);nwindow+=lo
                for i,name in enumerate(('overall','stable','transition','change')):
                    mask=np.ones(lo,bool) if not i else sm['group']==i-1;good=sm['good'][mask];native=sm['native_good'][mask];r=m[name]
                    assert r['n']==int(mask.sum()) and r['good']==int(good.sum()) and r['native_good']==int(native.sum())
                    assert r['rescue']==int((good&~native).sum()) and r['lost']==int((native&~good).sum())
                    assert r['rescue_IDs']==sm['sample_id'][mask][good&~native].tolist() and r['lost_IDs']==sm['sample_id'][mask][native&~good].tolist()
                groups.append(dict(arm=arm,seed=seed,split=split,metrics=m,assignment=a,all_STOP=bool((sm['chosen']==0).all()),coverage_exact_native=bool(np.array_equal(sm['good'],sm['native_good']))))
                if split=='hold':holds[arm].append(sm)
    h=holds['C_base'][0];unique=np.unique(h['track_key']);ix=np.searchsorted(unique,h['track_key']);mask=h['group']==2
    den=np.bincount(ix[mask],minlength=len(unique)).astype(float)
    arrays={k:np.stack([z['good'].astype(float) for z in v]) for k,v in holds.items()};native=np.stack([z['native_good'].astype(float) for z in holds['C_base']])
    diffs=dict(Ctoken_minus_Cbase=arrays['C_token']-arrays['C_base'],C_base_minus_native=arrays['C_base']-native,C_token_minus_native=arrays['C_token']-native)
    with np.load(HERE/'statistics/BOOTSTRAP_DRAWS.npz') as q:draw={k:q[k] for k in q.files}
    with np.load(PREVIOUS/'statistics/BOOTSTRAP_DRAWS.npz') as q:
        for k in ('track_draws','seed_draws','tracks'):assert np.array_equal(q[k],draw[k])
    rng=np.random.default_rng(20261026);sums={k:np.stack([np.bincount(ix[mask],weights=row[mask],minlength=len(unique)) for row in v]) for k,v in diffs.items()}
    for i in range(20000):
        td=rng.integers(len(unique),size=len(unique));sd=rng.integers(3,size=3);assert np.array_equal(td,draw['track_draws'][i]) and np.array_equal(sd,draw['seed_draws'][i])
        count=np.bincount(td,minlength=len(unique));d=count@den
        for k,v in sums.items():
            if d:
                per=v@count/d;assert abs(per[sd].mean()-draw['values_'+k][i])<=2e-12 and abs(per.mean()-draw['fixed_'+k][i])<=2e-12
            else:assert np.isnan(draw['values_'+k][i])
    snapshots=[]
    for r in read_json(HERE/'RESOURCE.json')['records']:
        for v in r['report']['observations']+[r['report']['last']]:assert v['status']=='PASS';snapshots.append(v['snapshot'])
    resources={k:func(s[k] for s in snapshots) for k,func in [('peak_private',max),('peak_rss',max),('physical',min),('commit',min)]}
    roots=['RESULT.json','RESOURCE.json','PRESERVATION.json','ARITHMETIC_AUDIT.json','ALL_PROBE_FINALS_LOCK.json','statistics/PRIMARY_COMPARISONS.json','statistics/BOOTSTRAP_DRAWS.npz','PRODUCTION_SOURCE_LOCK.json','source_manifest.json']
    write_json(OUT/'ALL_GROUPS.json',groups)
    report=dict(status='PASS_SAVED_COMPLETION_SUPPLEMENT',at=now(),source=source0,source_count=2844,populations=18,repeated_model_windows=nwindow,
        all_choice_slot_quality_count_IDs_EXACT=True,all_20000_actual_draws_independent=True,step2_draw_order_EXACT=True,
        full_cluster_motion_inherited_worker_SHA_not_repeated_here=True,no_model_forward_backward_optimizer=True,resource_production=resources,
        root_SHA={p:sha256(HERE/p) for p in roots},supplement_source_SHA=sha256(__file__),final_source=verify())
    write_json(OUT/'REVIEW.json',report);resource_report(OUT/'RESOURCE.json',g);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
