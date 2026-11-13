"""Independent Q0 saved review: SciPy pdist distances and window reconstruction."""
from common import *
import pyarrow.parquet as pq
from saved_attribution import metrics_saved
def audit():
    guard=Guard(E);guard.check('startup');source=verify_manifest();result=read(OUT/'RESULT.json');assert source==result['source_digest']
    for n,h in result['outputs'].items():assert sha(OUT/n)==h,('OUTPUT_CHANGED',n)
    checked=0;maximum=0.;summary=[]
    transitions=read_csv('COVERAGE_TRANSITIONS.csv');motion=read_csv('MOTION_ATTRIBUTION.csv');fullmotion=read_csv('ALL_POPULATION_MOTION.csv')
    for seed in SEEDS:
        for split in ('FIT','inner'):
            print('Q0_INDEPENDENT_REVIEW',seed,split,flush=True);guard.check();d,paths=metrics_saved(seed,split)
            tab=pq.read_table(OUT/'WINDOW_TRANSITIONS.parquet',filters=[('seed','=',seed),('split','=',split)]).to_pydict()
            assert tab['sample_id']==d['sample_id'].tolist() and tab['track']==d['track'].tolist()
            independent_errors={};independent_physical={}
            with np.load(OUT/f'window_aux_seed{seed}_{split}.npz') as aux:
                cc={}
                for arm,p in paths.items():
                    # Independent coordinate formulation, no training evaluator.
                    distance=np.linalg.norm(p.astype(np.float64)-d['gt'][:,None].astype(float),axis=3)
                    a=distance.sum(2)/12;f=distance[:,:,11];j=np.maximum(a/.2,f/.4);gg=(a<=.2)&(f<=.4)
                    assert np.array_equal(gg,aux[arm+'_slot_good']);assert_close(j,aux[arm+'_slot_J'])
                    for k,v in [('ADE',a.min(1)),('FDE',f.min(1)),('J',j.min(1))]:assert_close(v,tab[arm+'_'+k])
                    assert np.array_equal(gg.any(1),tab[arm+'_covered'])
                    independent_errors[arm]={'ADE':a.min(1),'FDE':f.min(1),'J':j.min(1),'covered':gg.any(1)}
                    v=np.zeros_like(p,dtype=float);v[:,:,0]=p[:,:,0]/.4;v[:,:,1:]=(p[:,:,1:].astype(float)-p[:,:,:-1].astype(float))/.4
                    acc=(v[:,:,1:]-v[:,:,:-1])/.4;jerk=(acc[:,:,1:]-acc[:,:,:-1])/.4
                    independent_physical[arm]=[np.linalg.norm(x,axis=3) for x in (v,acc,jerk)]
                    if arm in ('iso','mix'):
                        label,distinct,good=clusters(p,gg,independent=True)
                        assert np.array_equal(label,aux[arm+'_labels']) and np.array_equal(distinct,tab[arm+'_distinct']) and np.array_equal(good,tab[arm+'_good_clusters'])
                        cc[arm]=good;checked+=len(p)
            a=np.asarray(tab['iso_covered']);b=np.asarray(tab['mix_covered']);labels=np.asarray(tab['paired_transition'])
            for key,mask in partitions(a,b).items():assert np.array_equal(mask,labels==key)
            delta=cc['mix']-cc['iso'];assert np.array_equal(delta,tab['delta_good_clusters'])
            for gi,g in enumerate(GROUPS):
                take=np.ones(len(a),bool) if gi==0 else d['groups']==gi-1
                cond=[r for r in read_csv('CLUSTER_CONTRIBUTIONS.csv') if int(r['seed'])==seed and r['split']==split and r['group']==g]
                assert len(cond)==4 and sum(int(r['n']) for r in cond)==int(take.sum())
                subtotal=0.
                for r in cond:
                    mask=take&(labels==r['cell']);con=float(delta[mask].sum()/take.sum());assert_close(con,float(r['signed_whole_population_contribution']));subtotal+=con
                    assert int(r['increased_good_clusters_windows'])==int((mask&(delta>0)).sum())
                error=abs(subtotal-float(delta[take].mean()));assert error<=2e-12;maximum=max(maximum,error)
                for r in [r for r in transitions if int(r['seed'])==seed and r['split']==split and r['group']==g]:
                    ar=independent_errors[r['reference']]['covered'];br=independent_errors[r['treatment']]['covered']
                    parts=partitions(ar,br)
                    for key,m in parts.items():assert int(r[{'iso_only':'reference_only','mix_only':'treatment_only'}.get(key,key)])==int((take&m).sum())
                for r in [r for r in motion if int(r['seed'])==seed and r['split']==split and r['group']==g]:
                    ref,tr=r['reference'],r['treatment'];ar=independent_errors[ref]['covered'];br=independent_errors[tr]['covered']
                    masks={'retained':ar&br,'lost':ar&~br,'rescue':~ar&br,'unresolved':~ar&~br};mask=take&masks[r['event']]
                    assert int(mask.sum())==int(r['n'])
                    for arm in (ref,tr,'native'):
                        for metric,val in pool(independent_physical[arm],mask).items():
                            stored=r[arm+'_'+metric]
                            if val is None:assert stored==''
                            else:assert_close(val,float(stored))
                        for metric in ('ADE','FDE','J'):
                            if mask.any():assert_close(independent_errors[arm][metric][mask].mean(),float(r[arm+'_'+metric]))
                for r in [r for r in fullmotion if int(r['seed'])==seed and r['split']==split and r['group']==g]:
                    vals=pool(independent_physical[r['arm']],take);native=pool(independent_physical['native'],take)
                    for key,v in vals.items():assert_close(v,float(r[key]));assert_close(native[key],float(r['native_'+key]))
                    assert (r['all_five_pass']=='True')==all(vals[k]<=1.01*native[k] for k in vals)
            summary.append(dict(seed=seed,split=split,rows=len(a),independent_cluster_windows=2*len(a)))
    assert checked==124590;assert pq.read_metadata(OUT/'WINDOW_TRANSITIONS.parquet').num_rows==62295
    assert verify_manifest()==source
    for n,h in result['outputs'].items():assert sha(OUT/n)==h
    guard.check();report=dict(status='PASS_INDEPENDENT_SAVED_Q0',source_digest=source,rows=62295,cluster_windows=checked,
        independent_distance='12 SciPy pdist per window vs production broadcast distances; both exact original complete-link cut1',
        coverage_transitions='ALL_WINDOWS_EXACT',cluster_signed_decomposition='ALL_GROUPS_EXACT',motion_event_and_full_population='ALL_ROWS_RECOMPUTED',maximum_reconstruction_error=maximum,
        populations=summary,no_model_forward=True,no_training=True,no_pvalues=True,resource_observations=guard.observations)
    write(OUT/'REVIEW.json',report);return report
def read_csv(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
if __name__=='__main__':audit()
