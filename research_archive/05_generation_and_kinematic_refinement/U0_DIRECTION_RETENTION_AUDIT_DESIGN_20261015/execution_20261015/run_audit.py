import sys,time,gc
import numpy as np
import common as c,independent_review as rev,cluster_bootstrap as bs,attribution as at

def paths(key):
    prefix=c.H/'jobs'/key;prefix.parent.mkdir(parents=True,exist_ok=True)
    return [prefix.with_name(prefix.name+s) for s in ('_candidate_diagnostics.npz','_window_diagnostics.npz','_summary.json','_done.json')]

def save_job(key,row,source):
    cp,wp,jp,dp=paths(key)
    if dp.exists():done=c.read(dp);c.valid_done(done,source);return done
    assert not any(p.exists() for p in (cp,wp,jp)),'PARTIAL_ANALYSIS_PRESERVE_STOP'
    c.valid_done(row,c.digest(c.read(c.PARENT/'source_manifest.json')['files']));z=c.npz(row['output']);ref=c.native(key.split('/')[-1]);c.check_input(z,ref,row);del ref
    d,w,summary=rev.process(z)
    oldaux=c.npz(row['aux']);oldmetrics=c.read(row['metrics'])
    for name,a in (('ADE','A1'),('FDE','F1'),('J','J1')):c.close(oldaux[name],d[a])
    assert np.array_equal(oldaux['good'],d['good1']);c.close(oldaux['window_motion5'][:,1],w['refined_accel'])
    for group,mask in c.masks(z['groups']):
        out=summary[group]['overall'];old=oldmetrics[group]
        for k in ('n','native_good','good','lost','rescue','retained'):assert out[k]==old[k],(key,group,k)
        c.close(summary[group]['accel_attribution']['native_mean'],old['native_motion']['accel_mean']);c.close(summary[group]['accel_attribution']['refined_mean'],old['motion']['accel_mean'])
    c.save(cp,**d,sample_id=z['sample_id'],physical_slot=np.arange(20));c.save(wp,**w);c.write(jp,summary)
    done=dict(status='PASS',key=key,source_digest=source,input_source_digest=row['source_digest'],n=len(z['sample_id']),candidate=str(cp),window=str(wp),summary=str(jp),files={str(p):c.sha(p) for p in (cp,wp,jp)},no_model_forward=True)
    c.write(dp,done);return done

def review(jobs,source):
    rows=[];maxerror=0.;oldjobs=c.read(c.PARENT/'JOBS.json')
    for key,row in jobs.items():
        c.stage('INDEPENDENT_ALL_SAVED_DIAGNOSTICS',job=key,done=len(rows),total=24);c.valid_done(row,source)
        old=oldjobs[key];z=c.npz(old['output']);ref=c.native(key.split('/')[-1]);c.check_input(z,ref,old);del ref
        d=c.npz(row['candidate']);d.pop('sample_id');assert np.array_equal(d.pop('physical_slot'),np.arange(20))
        w=c.npz(row['window']);s=c.read(row['summary']);r=rev.review_job(z,d,w,s);maxerror=max(maxerror,r['max_error']);rows.append(dict(job=key,**r));del z,d,w,s;gc.collect()
    primary=c.read(c.H/'PRIMARY_COMPARISONS.json');new=bs.compute(jobs,True);maxerror=max(maxerror,c.compare(primary,new))
    sensitivity=c.read(c.H/'SENSITIVITY_COMPARISONS.json');new=bs.compute(jobs,True,True);maxerror=max(maxerror,c.compare(sensitivity,new))
    result=dict(status='PASS',jobs=24,records=rows,max_error=maxerror,full_saved_direction_retention_brackets_and_attribution=True,independent_bootstrap_track_sums_and_saved_draws=True,no_model_gradient_replay=True,not_repeated_SciPy_clusters=True);c.write('ARITHMETIC_AUDIT.json',result)

def branches(jobs,primary):
    flags={};rows=primary['rows']
    for index,label in ((0,'DIRECTION_STATISTICAL_CRITERION'),(1,'RETENTION_GAP_SUPPORTED')):
        r=rows[index];flags[label]=bool(r['CI_usable_for_branch'] and r['CI_family4'][0]>0 and all(x>0 for x in r['gap_per_seed']))
    patterns={}
    for split in c.SIZES:
        ss=[]
        for seed in c.SEEDS:
            job=jobs[f'integrated_acceleration/seed{seed}/{split}'];s=c.read(job['summary'])['change']['overall'];w=c.npz(job['window']);f=at.fragile_check(w)
            a=s.get('lost_G0_overshoot_slot_fraction');b=s.get('lost_G0_overshoot_windowweighted_fraction');ss.append(dict(seed=seed,slot_fraction=a,windowweighted_fraction=b,majority_agrees=(a is not None and b is not None and (a>.5)==(b>.5)),finite_step_pattern=(a is not None and b is not None and a>.5 and b>.5),fragile=f,lost_n=s['lost'],lost_track_n=len(set(w['track'][(w['groups']==2)&(w['event']==2)]))))
        patterns[split]=ss
    flags['FINITE_STEP_OVERSHOOT_PATTERN']=all(r['finite_step_pattern'] for r in patterns['outer'])
    flags['FRAGILE_MARGIN_PATTERN']=all(r['fragile']['supported'] for r in patterns['outer'])
    return dict(status='SAVED_DIAGNOSIS_REVIEW_READY',flags=flags,descriptive_patterns=patterns,primary_pattern_population='outer change; other splits descriptive counterevidence',direction_requires_manual_fixedslot_delta_and_all20_counterevidence_review=True,no_automatic_training=True,not_unique_causal_root=True)

def main():
    start=time.perf_counter();pre=c.read(c.H/'PREFLIGHT.json');assert pre['status']=='PASS' and pre['planning_minutes']<=30;c.guard('startup');lock=c.read(c.H/'source_manifest.json');assert pre['source_digest']==lock['source_digest']
    c.stage('INITIAL_SOURCE_AND_COMPLETED_U0_SHA');c.verify(lock['files']);inputs=c.input_audit();jobs={}
    for key,row in inputs.items():
        c.stage('DIRECTION_RETENTION_ATTRIBUTION_SAVE',job=key,done=len(jobs),total=24);jobs[key]=save_job(key,row,lock['source_digest']);c.write('JOBS.json',jobs);gc.collect()
    c.stage('PRIMARY_FAMILY4_BOOTSTRAP');p=bs.compute(jobs);c.write('PRIMARY_COMPARISONS.json',p);c.write('SENSITIVITY_COMPARISONS.json',bs.compute(jobs,sensitivity=True));review(jobs,lock['source_digest'])
    ceiling={k:{g:r['retention_ceiling'] for g,r in c.read(row['summary']).items()} for k,row in jobs.items()};acc={k:{g:r['accel_attribution'] for g,r in c.read(row['summary']).items()} for k,row in jobs.items()}
    c.write('RETENTION_CEILING.json',dict(status='PASS',jobs=ceiling,GT_privileged_count_only=True,not_bound_for_training_changing_rescues=True));c.write('ACCEL_ATTRIBUTION.json',dict(status='PASS',jobs=acc,quantiles_not_additive=True));c.write('NEXT_BRANCH.json',branches(jobs,p))
    c.stage('FINAL_SOURCE_AND_OUTPUT_SHA');c.verify(lock['files']);files={p:h for row in jobs.values() for p,h in row['files'].items()};c.verify(files)
    c.write('PRESERVATION.json',dict(status='PASS',source_count=len(lock['files']),inherited_sources=2092,source_digest=lock['source_digest'],initial_final_unchanged=True,old_outputs_models_failures_untouched=True));c.write('RESOURCE.json',dict(status='PASS',policy='task_memory_1p3_v1',**c.RESOURCE))
    c.write('RESULT.json',dict(status='PASS_DIAGNOSTIC_ONLY_U1_DIRECTION_RETENTION',jobs=jobs,source_digest=lock['source_digest'],source_count=len(lock['files']),elapsed_seconds=time.perf_counter()-start,new_formal_training_experiments=0,no_model_or_optimizer=True,no_new_data=True,manual_evidence_synthesis_pending=True))
    c.stage('SUCCESS',jobs=24)

if __name__=='__main__':
    try:main()
    except BaseException:c.failure('STOP_DIRECTION_RETENTION');raise
