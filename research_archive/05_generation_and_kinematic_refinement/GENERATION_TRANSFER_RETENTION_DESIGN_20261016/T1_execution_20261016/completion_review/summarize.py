"""CPU/NumPy saved-summary review. Never imports model, Torch or optimizer."""
import sys,time
from pathlib import Path
H=Path(__file__).resolve().parents[1];sys.path.insert(0,str(H))
import common as c
import numpy as np
OUT=Path(__file__).resolve().parent

def main():
    started=time.perf_counter();files=c.read(H/'source_manifest.json')['files'];c.verify(files);result=c.read(H/'RESULT.json');audit=c.read(H/'ARITHMETIC_AUDIT.json');stats=c.read(H/'STATISTICS.json');resource=c.read(H/'RESOURCE.json');estimate=c.read(H/'MEMORY_ESTIMATE.json');assert result['status']=='PASS_T1_TRANSFER_RETENTION' and audit['status']=='PASS';jobs=c.read(H/'JOBS.json');runs=c.read(H/'RUNS.json');assert len(jobs)==28 and len(runs)==6
    c.verify(result['outputs_SHA']);oldjobs=c.read(c.PARENT/'JOBS.json');rows=[];repeat=[];nativeids={};err=0.;source=c.digest(files)
    for key,row in jobs.items():
        c.guard();c.valid_done(row,source);z=c.npz(row['output']);aux=c.npz(row['aux']);met=c.read(row['metrics']);p=z['paths'];base=z['native'];gt=z['gt'];n=len(p);assert p.shape==(n,20,12,2) and len(np.unique(z['sample_id']))==n
        d=np.sqrt(((p-gt[:,None])**2).sum(-1));a=d.mean(-1);f=d[...,-1];j=np.maximum(a/.2,f/.4);good=(a<=.2)&(f<=.4);gg=good.any(1);nd=np.sqrt(((base-gt[:,None])**2).sum(-1));na=nd.mean(-1);nf=nd[...,-1];ng=((na<=.2)&(nf<=.4)).any(1)
        for k,value in (('ADE',a),('FDE',f),('J',j)):err=max(err,c.close(value,aux[k]))
        assert np.array_equal(good,aux['good']);split=key.split('/')[-1]
        if key.startswith('native/'):
            nativeids[split]=z['sample_id'].copy();assert np.array_equal(p,base)
        else:
            for group,mask in c.masks(z['groups']):
                g=met[group];assert g['n']==int(mask.sum());assert g['good']==int((mask&gg).sum()) and g['rescue']==int((mask&gg&~ng).sum()) and g['lost']==int((mask&~gg&ng).sum())
                assert g['good']-g['native_good']==g['rescue']-g['lost']
                failure=[k for k,v in g['motion'].items() if v>1.01*g['native_motion'][k]];harms={k:100*(v/g['native_motion'][k]-1) if g['native_motion'][k] else None for k,v in g['motion'].items()}
                rows.append(dict(job=key,cell=row['cell'],seed=int(key.split('/')[1][4:]),split=split,group=group,**g,motion_failure_metrics=failure,motion_harm_pct=harms,protected_error_pass=group=='change' or (g['ADE']<=1.01*g['native_ADE'] and g['FDE']<=1.01*g['native_FDE']),native_success_protection_pass=g['lost_rate_native_good']<=.01))
        if key.startswith('A/'):
            seed=key.split('/')[1];old=c.npz(oldjobs['integrated_acceleration/'+seed+'/'+split]['output']);checks={k:bool(np.array_equal(z[k],old[k])) for k in z};assert all(checks.values());repeat.append(dict(job=key,all_saved_arrays_bitwise_EXACT=True,fields=list(checks),not_independent_generalization=True,model_tensor_replay_not_done=True));del old
        del z,aux,p,base,gt,d,a,f,j,good,nd,na,nf; c.gc.collect()
    assert len(rows)==96 and len(repeat)==12 and sum(map(len,nativeids.values()))==24679 and len(set(np.concatenate(list(nativeids.values()))))==24679
    traces=[]
    for key,row in runs.items():
        c.valid_done(row,source);hist=c.read(H/'runs'/key/'history.json');assert len(hist)==25 and sum(r['optimization_steps'] for r in hist)==1850
        traces.append(dict(run=key,first_total=hist[0]['total'],final_total=hist[-1]['total'],first_base=hist[0]['base'],final_base=hist[-1]['base'],first_keep=hist[0]['keep'],final_keep=hist[-1]['keep'],success_fraction=hist[-1]['success']/18909,final_keep_activation=hist[-1]['keep_active']/18909,coordinate_gradient_diagnostics=[dict(epoch=h['epoch'],**h['fixed_FIT_coordinate_gradient']) for h in hist if 'fixed_FIT_coordinate_gradient' in h],not_final_population_objective=True))
    primary=stats['primary'];assert primary['family']==24 and primary['draws']==20000;rr=primary['rows'];points=np.array([r['estimate'] for r in rr]);per=np.array(primary['per_seed'])
    gate=dict(lost_mean_drop_at_least3pp=points[1]<=-.03,all_seed_lost_strictly_drop=bool(np.all(per[:,1]<0)),lost_sim_upper_lt0=rr[1]['CI_family'][1]<0,rescue_mean_no_decrease=points[2]>=0,rescue_sim_lower_ge_minus2pp=rr[2]['CI_family'][0]>=-.02,all_seed_coverage_nondecrease=bool(np.all(per[:,0]>=0)),all_draws_usable=all(r['usable'] for r in rr))
    signal=all(gate.values());assert signal==stats['decision']['RETENTION_SIGNAL']
    lim=1.3*estimate['estimated_total_peak_bytes'];deviation=dict(status='STOP_RESOURCE_REVIEW_REQUIRED',estimated_peak_bytes=estimate['estimated_total_peak_bytes'],approved_process_limit_bytes=lim,recorded_peak_private_bytes=resource['peak_private_bytes'],recorded_peak_RSS_bytes=resource['peak_RSS_bytes'],recorded_private_peak_exceeds_limit=resource['peak_private_bytes']>lim,peak_ratio_to_estimate=resource['peak_private_bytes']/estimate['estimated_total_peak_bytes'],over_limit_bytes=resource['peak_private_bytes']-lim,original_RESOURCE_status=resource['status'],original_RESOURCE_SHA=c.sha(H/'RESOURCE.json'),guard_checks_only_current_snapshot_not_retained_lifetime_peak=True,guard_source=str(c.PARENT/'audit_memory_policy_20261015/memory_policy.py'),reason='Saved lifetime private-labelled peak exceeds registered1.3E although sampled-current guard returned PASS; scientific arrays/old sources preserved. Stop next training and review estimate/guard, not silently inflate.',not_OS_memory_exhaustion=True,no_live_resource_or_PID_query=True)
    assert deviation['recorded_private_peak_exceeds_limit'];c.verify(files);c.verify(result['outputs_SHA'])
    c.write(OUT/'SUMMARY.json',dict(status='PASS_SAVED_NUMPY_SUMMARY_WITH_RESOURCE_PROTOCOL_DEVIATION',rows=rows,training=traces,A_repeat=repeat,primary=primary,absolute=stats['absolute'],all_exploratory=stats['exploratory95'],retention_gate=gate,RETENTION_SIGNAL=signal,resource_deviation=deviation))
    c.write(H/'RESOURCE_DEVIATION_REVIEW.json',deviation)
    c.write(OUT/'REVIEW.json',dict(status='PASS_NUMERICAL_SUMMARY_RESOURCE_PROTOCOL_NOT_PASS',saved_jobs=28,source_count=len(files),models_SHA_verified=6,quality_max_error=err,A_saved_bitwise_repeat_jobs=12,source_and_all_completed_output_SHA_initial_final_unchanged=True,no_Torch_model_forward_optimizer=True,does_not_repeat_all_SciPy_clusters_or_bootstrap_draws=True,original_full_ARITHMETIC_AUDIT_supplies_cluster_motion_CI_review=True,resource_deviation_must_not_be_hidden=True,elapsed_seconds=time.perf_counter()-started,resource=c.RESOURCE,script_SHA=c.sha(__file__)))
    print('SUMMARY',dict(rows=len(rows),max_error=err,retention_signal=signal,resource_peak_GiB=resource['peak_private_bytes']/2**30,limit_GiB=lim/2**30,seconds=time.perf_counter()-started),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_COMPLETION_SUMMARY');raise
