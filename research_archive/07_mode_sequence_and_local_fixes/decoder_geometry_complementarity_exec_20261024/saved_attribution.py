"""One Q0 complete saved-only supplement, not GT candidate search or training."""
from common import *
import time,gc
def metrics_saved(seed,split):
    d=load(OLD/f'native_{split}.npz');packs={}
    for arm in ('iso','mix'):
        name='C2_isomlp' if arm=='iso' else 'C2_setmix'
        run=PREVIOUS/f'direction_C_set_mechanisms/C2_execution_20261024/{name}_seed{seed}'
        done=read(run/'done.json');assert done['status']=='SUCCESS'
        for n,h in done['outputs'].items():assert sha(run/n)==h,('DONE_OUTPUT_CHANGED',run,n)
        p=load(run/f'predictions_{split}.npz')
        for k in ('sample_id','gt','native','groups','track'):assert np.array_equal(p[k],d[k]),k
        packs[arm]=p['paths']
    old=load(OLD/f'runs/A_seed{seed}/predictions_{split}.npz')
    assert np.array_equal(old['sample_id'],d['sample_id']);packs['oldA']=old['paths'];packs['native']=d['native']
    return d,packs
def produce():
    import pyarrow as pa,pyarrow.parquet as pq
    source=verify_manifest();guard=Guard(E);guard.check('startup');OUT.mkdir(exist_ok=True)
    target=OUT/'WINDOW_TRANSITIONS.parquet'
    if target.exists():raise FileExistsError('Preserve existing Q0, no duplicate full supplement')
    tables=[];contrib=[];cost=[];support=[];population=[];outputs={};writer=None
    summary=[];start=time.perf_counter()
    for seed in SEEDS:
        for split in ('FIT','inner'):
            guard.check();d,paths=metrics_saved(seed,split);n=len(d['sample_id']);ee={k:geometry(p,d['gt']) for k,p in paths.items()};ph={k:physical(p) for k,p in paths.items()}
            cluster={}
            for k in ('iso','mix'):
                print('Q0_CLUSTER',seed,split,k,flush=True);cluster[k]=clusters(paths[k],ee[k]['slot_good'])
                run=PREVIOUS/f"direction_C_set_mechanisms/C2_execution_20261024/{'C2_isomlp' if k=='iso' else 'C2_setmix'}_seed{seed}"
                stored=read(run/f'metrics_{split}.json')
                for gi,g in enumerate(GROUPS):
                    take=np.ones(n,bool) if gi==0 else d['groups']==gi-1
                    assert_close(cluster[k][1][take].mean(),stored[g]['distinct']);assert_close(cluster[k][2][take].mean(),stored[g]['effective_good'])
            delta=cluster['mix'][2]-cluster['iso'][2];paired=partitions(ee['iso']['covered'],ee['mix']['covered'])
            labels=np.empty(n,dtype='<U12')
            for key,m in paired.items():labels[m]=key
            cols=dict(sample_id=d['sample_id'],track=d['track'],group=d['groups'],seed=np.full(n,seed),split=np.full(n,split),paired_transition=labels,
                delta_good_clusters=delta,iso_good_clusters=cluster['iso'][2],mix_good_clusters=cluster['mix'][2],iso_distinct=cluster['iso'][1],mix_distinct=cluster['mix'][1])
            aux=dict(sample_id=d['sample_id'],iso_labels=cluster['iso'][0],mix_labels=cluster['mix'][0])
            for arm,e in ee.items():
                for k in ('covered','ADE','FDE','J'):cols[arm+'_'+k]=e[k]
                cols[arm+'_good_slots']=e['slot_good'].sum(1)
                cols[arm+'_good_among_native_physical_indices']=(e['slot_good']&ee['native']['slot_good']).sum(1)
                aux[arm+'_slot_good']=e['slot_good'];aux[arm+'_slot_J']=e['slot_J']
                s,a,j=ph[arm]
                for key,val in [('speed_p95',np.percentile(s,95,axis=(1,2))),('accel_mean',a.mean((1,2))),('accel_p95',np.percentile(a,95,axis=(1,2))),('jerk_mean',j.mean((1,2))),('jerk_p95',np.percentile(j,95,axis=(1,2)))]:cols[arm+'_'+key]=val
            cols['native_margin_best_J']=1-ee['native']['J']
            tab=pa.table(cols)
            if writer is None:writer=pq.ParquetWriter(target,tab.schema,compression='zstd')
            writer.write_table(tab)
            ap=OUT/f'window_aux_seed{seed}_{split}.npz';np.savez_compressed(ap,**aux);outputs[ap.name]=sha(ap)
            for gi,g in enumerate(GROUPS):
                gm=np.ones(n,bool) if gi==0 else d['groups']==gi-1;N=int(gm.sum());total=float(delta[gm].mean())
                signed={z:float(delta[gm&m].sum()/N) for z,m in paired.items()};assert_close(sum(signed.values()),total)
                positive=sum(max(v,0.) for v in signed.values())
                for cell,m in paired.items():
                    take=gm&m;count=int(take.sum());contrib.append(dict(seed=seed,split=split,group=g,cell=cell,n=count,N=N,tracks=len(np.unique(d['track'][take])),
                        delta_sum=int(delta[take].sum()),delta_mean=float(delta[take].mean()) if count else None,signed_whole_population_contribution=signed[cell],
                        positive_contribution_share=max(signed[cell],0)/positive if positive>0 else None,
                        increased_good_clusters_windows=int((take&(delta>0)).sum()),same_already_covered_more_clusters=int((take&(delta>0)&paired['both_good']).sum()),total_delta=total))
                for ref,tr in [('iso','mix'),('native','iso'),('native','mix')]:
                    a=ee[ref]['covered'];b=ee[tr]['covered'];part=partitions(a,b)
                    row=dict(seed=seed,split=split,group=g,reference=ref,treatment=tr,N=N,tracks=len(np.unique(d['track'][gm])),reference_good=int(a[gm].sum()),treatment_good=int(b[gm].sum()))
                    for key,m in part.items():row[{'iso_only':'reference_only','mix_only':'treatment_only'}.get(key,key)]=int((m&gm).sum())
                    assert sum(row[k] for k in ('both_good','reference_only','treatment_only','both_bad'))==N
                    row['net_good']=row['treatment_only']-row['reference_only'];tables.append(row)
                    for event,mask in [('retained',a&b),('lost',a&~b),('rescue',~a&b),('unresolved',~a&~b)]:
                        take=gm&mask;count=int(take.sum());r=dict(seed=seed,split=split,group=g,reference=ref,treatment=tr,event=event,n=count,N=N,tracks=len(np.unique(d['track'][take])))
                        for name in (ref,tr,'native'):
                            for metric in ('ADE','FDE','J'):r[name+'_'+metric]=float(ee[name][metric][take].mean()) if count else None
                            for metric,val in pool(ph[name],take).items():r[name+'_'+metric]=val
                        cost.append(r)
                ng=ee['native']['covered']
                for arm in ('iso','mix','oldA'):
                    for event,mask in [('native_retained',ng&ee[arm]['covered']),('native_lost',ng&~ee[arm]['covered']),('oldA_retained',ee['oldA']['covered']&ee[arm]['covered']),('oldA_lost',ee['oldA']['covered']&~ee[arm]['covered'])]:
                        take=gm&mask;count=int(take.sum());margin=1-ee['native']['J'][take]
                        support.append(dict(seed=seed,split=split,group=g,arm=arm,event=event,n=count,reference_good=int((gm&(ng if event.startswith('native') else ee['oldA']['covered'])).sum()),tracks=len(np.unique(d['track'][take])),
                            native_best_J_margin_min=float(margin.min()) if count else None,native_best_J_margin_median=float(np.median(margin)) if count else None,
                            native_good_physical_slots=int(ee['native']['slot_good'][take].sum()),good_at_native_physical_indices=int((ee[arm]['slot_good'][take]&ee['native']['slot_good'][take]).sum()),physical_indices_not_path_identity=True))
                native_m=pool(ph['native'],gm)
                for arm in paths:
                    m=pool(ph[arm],gm);r=dict(seed=seed,split=split,group=g,arm=arm,n=N,good=int(ee[arm]['covered'][gm].sum()))
                    for k,v in m.items():r[k]=v;r['native_'+k]=native_m[k]
                    r['all_five_pass']=all(m[k]<=1.01*native_m[k] for k in m);population.append(r)
            cm=d['groups']==2
            summary.append(dict(seed=seed,split=split,n_change=int(cm.sum()),iso_good=int(ee['iso']['covered'][cm].sum()),mix_good=int(ee['mix']['covered'][cm].sum()),
                both_good=int((paired['both_good']&cm).sum()),rescue=int((paired['mix_only']&cm).sum()),lost=int((paired['iso_only']&cm).sum()),both_bad=int((paired['both_bad']&cm).sum()),
                good_cluster_delta=float(delta[cm].mean()),same_covered_more_clusters=int((cm&paired['both_good']&(delta>0)).sum())))
            guard.check();del paths,ee,ph,d,cluster,tab,aux;gc.collect()
    writer.close()
    for name,rows in [('COVERAGE_TRANSITIONS.csv',tables),('CLUSTER_CONTRIBUTIONS.csv',contrib),('MOTION_ATTRIBUTION.csv',cost),('NATIVE_SUCCESS_SUPPORT.csv',support),('ALL_POPULATION_MOTION.csv',population)]:
        csvout(OUT/name,rows);outputs[name]=sha(OUT/name)
    outputs[target.name]=sha(target);guard.check();assert verify_manifest()==source
    result=dict(status='SUCCESS_SAVED_ATTRIBUTION_REVIEW_REQUIRED',source_digest=source,outputs=outputs,summary=summary,
        window_rows=3*(18909+1856),scientific_input_populations=12,formal_training=0,no_model_forward=True,no_optimizer=True,
        elapsed_seconds=time.perf_counter()-start,resource_observations=guard.observations,
        interpretation='descriptive observed path-set attribution, not internal causal mediation, not Q1/Q2 evidence')
    write(OUT/'RESULT.json',result);return result
if __name__=='__main__':produce()
