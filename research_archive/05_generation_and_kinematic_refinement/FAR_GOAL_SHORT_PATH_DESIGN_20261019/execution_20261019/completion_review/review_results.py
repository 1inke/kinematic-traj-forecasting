"""CPU-only saved result review. No Torch/model/optimizer/solver imports.

Worker already fully recomputed clusters, pooled motion and20k bootstrap draws.
This supplement independently rechecks quality/geometry, H2 pairing/overrides,
aggregate points/denominators, all6 histories, and source/output SHA before/after.
It does NOT claim a second complete cluster/bootstrap/model replay.
"""
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(H))
import common as c
OUT=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def norm(x):return np.sqrt(np.sum(x*x,axis=-1))
def quality(p,y):
    e=norm(p-y[:,None]);a=e.mean(-1);f=e[:,:,-1];j=np.maximum(a/.2,f/.4);good=((a<=.2)&(f<=.4)).any(1)
    return a,f,j,good
def reconstruct(x):
    z=x['z'].astype(float);coef=x['process_coeff'].astype(float);v=x['v0'].astype(float)
    rows=[]
    for t in range(1,13):
        s=t/12;rows.append((t*.4)*v[:,None,:]+s*s*(z-4.8*v[:,None,:]+coef[:,:,0])+s*s*(1-s)*coef[:,:,1]+s*s*(1-s)*(2*s-1)*coef[:,:,2])
    return np.stack(rows,axis=2)

def main():
    begin=time.perf_counter();c.guard('startup');r=read(H/'RESULT.json');sm=read(H/'source_manifest.json');jobs=read(H/'JOBS.json');w=read(H/'worker_state.json');exit=read(H/'exit_status.json')
    assert w['status']==exit['status']=='SUCCESS' and r['status']=='PASS_FAR_GOAL_SHORT_PATH_EXECUTION'
    assert r['jobs']==len(jobs)==82 and r['models']==6 and r['epochs']==150 and r['steps']==11100
    for name in ('ARITHMETIC_AUDIT.json','BOOTSTRAP_AUDIT.json','PRESERVATION.json','RESOURCE.json'):assert read(H/name)['status']=='PASS'
    assert read(H/'ARITHMETIC_AUDIT.json')['jobs']==82
    inputs={str(H/p):c.sha(H/p) for p in ('RESULT.json','JOBS.json','source_manifest.json','PRIMARY_COMPARISONS.json','PRIMARY_DRAWS.npz','worker_state.json','exit_status.json','ARITHMETIC_AUDIT.json','BOOTSTRAP_AUDIT.json','PRESERVATION.json','RESOURCE.json','H0_RESULT.json','LONG_LABEL_AUDIT.json')}
    c.verify(sm['files']);c.verify(r['output_sha']);records=r['records'];rows=[];maxerror=0.;h2=[]
    for key,dd in jobs.items():
        c.guard();assert dd['source_digest']==r['source_digest'] and dd['status']=='COMPLETE';c.verify(dd['files'])
        f=H/'jobs'/key;x=c.npz(f/'paths.npz');q=c.npz(f/'quality.npz');a,b,j,good=quality(x['paths'],x['gt']);na,nf,nj,base=quality(x['native'],x['gt'])
        assert np.array_equal((a<=.2)&(b<=.4),q['good_candidate']);assert np.array_equal(good,q['vectors'][:,0].astype(bool))
        for name,v in [('ADE',a),('FDE',b),('J',j)]:
            error=float(abs(v-q[name]).max());maxerror=max(maxerror,error);assert error<=2e-12,(key,name,error)
        for group,mask in c.masks(x['groups']):
            stored=records[key]['metrics'][group];assert stored['n']==int(mask.sum())
            if not mask.any():continue
            rescue=int((good&~base&mask).sum());lost=int((~good&base&mask).sum());retained=int((good&base&mask).sum())
            assert stored['good']==int(good[mask].sum()) and stored['rescued']==rescue and stored['lost']==lost and stored['retained']==retained
            assert stored['good']-stored['native_good']==rescue-lost
            assert set(stored['rescue_ids'])==set(x['sample_id'][good&~base&mask].tolist()) and set(stored['lost_ids'])==set(x['sample_id'][~good&base&mask].tolist())
        if 'z' in x:
            err=float(abs(reconstruct(x)-x['paths']).max());assert err<=2e-12,(key,err);maxerror=max(maxerror,err)
            err=float(abs(x['paths'][:,:,-1]-x['z'].astype(float)-x['process_coeff'][:,:,0]).max());assert err<=2e-12
        if '/normal/' in key and not key.startswith('native'):
            armseed,_,split=key.split('/');arm,seed=armseed.split('_seed')
            for group,mask in c.masks(x['groups']):
                z=records[key]['metrics'][group];nm=z['native_motion'];mot=z['motion']
                native_metric=records['native/normal/'+split]['metrics'][group]
                rows.append(dict(arm=arm,seed=int(seed),split=split,group=group,n=z['n'],good=z['good'],native_good=z['native_good'],coverage=z['coverage'],rescue=z['rescued'],lost=z['lost'],lost_rate=z['lost_over_native_good'],minADE=z['minADE'],minFDE=z['minFDE'],J=z['J'],error_harm_pct={k:100*(z[k]/native_metric[k]-1) for k in ('minADE','minFDE','J')},motion_pct={k:100*(mot[k]/nm[k]-1) if nm[k]!=0 else None for k in mot},motion_failed=z['motion_failed'],effective_good=z['effective_good'],distinct=z['effective_distinct'],protected_error_gate=(group=='change' or (z['minADE']<=1.01*native_metric['minADE'] and z['minFDE']<=1.01*native_metric['minFDE'])),protected_coverage_gate=(group=='change' or z['good']>=native_metric['good']),native_good_retention_gate=(z['lost_over_native_good'] is None or z['lost_over_native_good']<=.01)))
        elif '/normal/' not in key:
            armseed,condition,split=key.split('/');arm,seed=armseed.split('_seed');Horizon=12 if arm=='A' else 20
            normal=c.npz(H/'jobs'/armseed/'normal'/split/'paths.npz');nq=c.npz(H/'jobs'/armseed/'normal'/split/'quality.npz');lookup={int(i):k for k,i in enumerate(normal['sample_id'])};index=np.array([lookup[int(i)] for i in x['sample_id']]);nb=nq['vectors'][index,0].astype(bool)
            assert np.array_equal(x['gt'],normal['gt'][index]) and np.array_equal(x['native'],normal['native'][index])
            if condition=='CV_goal':expected=np.repeat((4.8*x['v0'])[:,None],20,axis=1)
            elif condition=='GT_goal_local_diagnostic':
                assert np.all(x['m20']);target=x['gt'][:,-1] if Horizon==12 else x['y20'];expected=np.repeat(((12/Horizon)*target).astype(np.float32)[:,None],20,axis=1)
            else:
                assert np.all(x['donor_track']!=x['track']);di=np.array([lookup[int(i)] for i in x['donor_id']]);expected=normal['z'][di]
            assert np.array_equal(x['z'],expected) and np.array_equal(x['override_z'],expected),(key,'override')
            dz=x['z'].astype(float)-normal['z'][index];da=x['process_coeff'][:,:,0].astype(float)-normal['process_coeff'][index,:,0].astype(float)
            dzsq=float(np.sum(dz*dz));cancellation_projection=float(-np.sum(dz*da)/dzsq) if dzsq else None
            for group,mask in c.masks(x['groups']):
                if not mask.any():continue
                response=norm(x['paths'][mask]-normal['paths'][index][mask]);endpoint=norm(x['paths'][mask,:,-1]-normal['paths'][index][mask,:,-1]);nz=norm(dz[mask]);normal_good=int(nb[mask].sum());now_good=int(good[mask].sum())
                h2.append(dict(arm=arm,seed=int(seed),condition=condition,split=split,group=group,n=int(mask.sum()),normal_same_cohort_good=normal_good,intervention_good=now_good,delta_pp=100*(now_good-normal_good)/mask.sum(),rescue_from_normal=int((good&~nb&mask).sum()),lost_from_normal=int((~good&nb&mask).sum()),mean_path_response=float(response.mean()),mean_endpoint_response=float(endpoint.mean()),mean_z_shift=float(nz.mean()),global_dz_cancellation_projection=cancellation_projection,motion_failed=records[key]['metrics'][group]['motion_failed']))
        print('review',key,flush=True)
    # Independent primary POINTS from saved vectors; draw recheck already worker.
    boot=read(H/'PRIMARY_COMPARISONS.json');vectors={}
    for arm in ('A','B','native'):
        keys=['native/normal/outer']*3 if arm=='native' else [f'{arm}_seed{s}/normal/outer' for s in (42,43,44)]
        vectors[arm]=np.stack([c.npz(H/'jobs'/key/'quality.npz')['vectors'] for key in keys])
    outer=c.npz(H/'jobs/native/normal/outer/paths.npz');points={}
    for comp in ('B-A','B-native'):
        other=comp.split('-')[1]
        for group,mask in [('overall',np.ones(len(outer['sample_id']),bool)),('change',outer['groups']==2)]:
            point=(vectors['B'][:,mask].mean(1)-vectors[other][:,mask].mean(1)).mean(0)
            for idx,name in enumerate(('coverage','minADE','minFDE','J')):
                key=comp+'/'+group+'/'+name;error=abs(float(point[idx])-boot['metrics'][key]['point']);assert error<=2e-12;points[key]=float(point[idx])
    histories={}
    for arm in ('A','B'):
        for seed in (42,43,44):
            f=H/'runs'/f'{arm}_seed{seed}';hist=read(f/'history.json');manifest=read(f/'run_manifest.json');done=read(f/'done.json')
            assert len(hist)==25 and hist[-1]['global_step']==1850 and sum(t['batches'] for t in hist)==1850 and all(t['n']==18909 and t['m20_count']==13856 for t in hist)
            assert manifest['H']==(12 if arm=='A' else 20) and manifest['epochs']==25 and done['source_digest']==r['source_digest'];c.verify(done['files'])
            for file in ('history.json','run_manifest.json','done.json'):inputs[str(f/file)]=c.sha(f/file)
            histories[f'{arm}_seed{seed}']=dict(first=hist[0],last=hist[-1],history_length=len(hist),steps=hist[-1]['global_step'])
    gates={}
    for arm in ('A','B'):
        rr=[z for z in rows if z['arm']==arm];oc=[z for z in rr if z['split']=='outer' and z['group']=='change']
        gates[arm]=dict(all48_group_motion_pass=all(not z['motion_failed'] for z in rr),motion_groups_failing=sum(bool(z['motion_failed']) for z in rr),native_good_retention_failing=sum(not z['native_good_retention_gate'] for z in rr),protected_error_failing=sum(not z['protected_error_gate'] for z in rr),protected_coverage_failing=sum(not z['protected_coverage_gate'] for z in rr),outer_change_each_seed_ge2pp=all(z['good']-z['native_good']>=6 for z in oc),change_ADE_FDE_each_ge5pct=all(z['error_harm_pct']['minADE']<=-5 and z['error_harm_pct']['minFDE']<=-5 for z in oc))
    mci=boot['metrics'];gates['far_goal_signal']=dict(coverage_family16_lower_positive=mci['B-A/change/coverage']['CI_family16'][0]>0,three_seed_coverage_nondecreasing=all(vectors['B'][i,outer['groups']==2,0].mean()>=vectors['A'][i,outer['groups']==2,0].mean() for i in range(3)),ADE_FDE_J_family16_upper_negative=all(mci['B-A/change/'+k]['CI_family16'][1]<0 for k in ('minADE','minFDE','J')),all_motion_pass=gates['B']['all48_group_motion_pass'])
    gates['FAR_GOAL_SIGNAL_ESTABLISHED']=all(gates['far_goal_signal'].values());gates['MAJOR_SUCCESS']=False
    # Explicit A->B churn, separate from native->method churn.
    ab=[]
    for split in ('FIT','outer','inner','original_CAL'):
        for seed in (42,43,44):
            aa=c.npz(H/'jobs'/f'A_seed{seed}'/'normal'/split/'quality.npz')['vectors'][:,0].astype(bool);bb=c.npz(H/'jobs'/f'B_seed{seed}'/'normal'/split/'quality.npz')['vectors'][:,0].astype(bool);xx=c.npz(H/'jobs'/f'A_seed{seed}'/'normal'/split/'paths.npz')
            for group,mask in c.masks(xx['groups']):ab.append(dict(split=split,seed=seed,group=group,n=int(mask.sum()),A_good=int(aa[mask].sum()),B_good=int(bb[mask].sum()),rescue=int((bb&~aa&mask).sum()),lost=int((aa&~bb&mask).sum()),retained=int((aa&bb&mask).sum()),neither=int((~aa&~bb&mask).sum())))
    c.verify(sm['files']);c.verify(r['output_sha']);c.verify(inputs)
    summary=dict(status='PASS_COMPLETION_SAVED_SUMMARY_REVIEW',at=c.now(),seconds=time.perf_counter()-begin,worker_exit_both_SUCCESS=True,completed_jobs=82,completed_models=6,completed_epochs=150,steps=11100,source_files=len(sm['files']),output_files=len(r['output_sha']),max_independent_geometry_quality_error=maxerror,all_source_and_output_SHA_initial_final_exact=True,input_SHA=inputs,normal_all96_group_rows=rows,H2_all216_group_rows=h2,A_to_B=ab,primary_points=points,primary_full16=boot,gates=gates,histories=histories,resource_original=read(H/'RESOURCE.json'),review_scope='CPU saved geometry/quality/counts/allH2 overrides/primary points/history/SHA; worker already fully rechecked clustering/motion/20kdraws; NOT a second full cluster/bootstrap/model replay',not_external_review=True)
    c.write(OUT/'REVIEW.json',summary)
    print(json.dumps(c.safe(dict(status=summary['status'],seconds=summary['seconds'],gates=gates,maxerror=maxerror)),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
