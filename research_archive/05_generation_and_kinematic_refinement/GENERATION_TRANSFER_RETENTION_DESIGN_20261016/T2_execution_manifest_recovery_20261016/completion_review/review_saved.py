"""CPU-only report supplement. No Torch/model/optimizer; production files read-only."""
import hashlib, importlib.util, json, sys
from pathlib import Path
from datetime import datetime
import numpy as np

H=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
T1=H.parent/'T1_execution_20261016'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    a=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(4*1024*1024): a.update(b)
    return a.hexdigest()
def write(name,x):
    (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
spec=importlib.util.spec_from_file_location('review_memory_guard',H/'memory_policy.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
guard.H=OUT
guard._config=(read(H/'MEMORY_ESTIMATE.json')['estimated_total_peak_bytes'],sha(H/'resource_policy.json'),sha(H/'MEMORY_ESTIMATE.json'))
guard.set_stage('SAVED_CPU_SUPPLEMENT');guard.guard('startup')
result=read(H/'RESULT.json'); stats=read(H/'STATISTICS.json')
assert read(H/'worker_state.json')['status']==read(H/'exit_status.json')['status']=='SUCCESS'
assert result['status']=='PASS_T2_TRANSFER_RETENTION'
for name in ('ARITHMETIC_AUDIT.json','PRESERVATION.json'):
    assert read(H/name)['status']=='PASS'
sources=read(H/'source_manifest.json')['files']
assert len(sources)==2560
jobs=read(H/'JOBS.json')
for k,v in read(T1/'JOBS.json').items():
    if k.startswith('B/'): jobs[k]=v
locked=dict(sources);locked.update(result['outputs_SHA'])
for job in jobs.values():locked.update(job['files'])
def verify():
    for i,(p,v) in enumerate(locked.items()):
        if i%64==0:guard.guard()
        assert sha(p)==v, ('SHA',p)
verify()
errors=[]
def close(a,b,label):
    a=np.asarray(a);b=np.asarray(b)
    assert a.shape==b.shape,label
    err=float(np.max(np.abs(a-b))) if a.size else 0.
    errors.append(err);assert err<=2e-12,(label,err)
def quality(p,gt):
    e=np.sqrt(np.sum((p-gt[:,None])**2,axis=-1))
    a=e.mean(-1);f=e[:,:,-1];j=np.maximum(a/.2,f/.4)
    return a,f,j,(a<=.2)&(f<=.4)
def motion(p):
    v=np.diff(p,axis=2,prepend=np.zeros_like(p[:,:,:1]))/.4
    a=np.diff(v,axis=2)/.4;jerk=np.diff(a,axis=2)/.4
    vm=np.linalg.norm(v,axis=-1).reshape(-1);am=np.linalg.norm(a,axis=-1).reshape(-1);jm=np.linalg.norm(jerk,axis=-1).reshape(-1)
    return dict(speed_p95=float(np.quantile(vm,.95)),accel_mean=float(am.mean()),accel_p95=float(np.quantile(am,.95)),jerk_mean=float(jm.mean()),jerk_p95=float(np.quantile(jm,.95)))
rows=[];references=[];history=[]
for key,job in jobs.items():
    guard.guard()
    m=read(job['metrics'])
    with np.load(job['output'],allow_pickle=False) as z, np.load(job['aux'],allow_pickle=False) as aux:
        p=z['paths'];native=z['native'];gt=z['gt'];ids=z['sample_id'];groups=z['groups']
        assert p.shape==(len(ids),20,12,2) and len(np.unique(ids))==len(ids) and np.isfinite(p).all()
        a,f,j,g=quality(p,gt);na,nf,nj,ng=quality(native,gt)
        close(a,aux['ADE'],key+'ADE');close(f,aux['FDE'],key+'FDE');close(j,aux['J'],key+'J');assert np.array_equal(g,aux['good'])
        gg=g.any(1);bg=ng.any(1)
        for name,mask in [('overall',np.ones(len(ids),bool)),('stable',groups==0),('transition',groups==1),('change',groups==2)]:
            r=m[name];n=int(mask.sum());assert r['n']==n
            counts=dict(good=int((gg&mask).sum()),native_good=int((bg&mask).sum()),rescue=int((gg&~bg&mask).sum()),lost=int((~gg&bg&mask).sum()),retained=int((gg&bg&mask).sum()))
            for k,val in counts.items():assert r[k]==val,(key,name,k)
            assert r['good']==r['native_good']+r['rescue']-r['lost']
            assert np.array_equal(ids[mask&gg&~bg],r['rescue_IDs']) and np.array_equal(ids[mask&~gg&bg],r['lost_IDs'])
            for label,x in [('ADE',a),('FDE',f),('J',j)]:close(x[mask].min(1).mean(),r[label],key+name+label)
            mot=motion(p[mask]);nmot=motion(native[mask])
            for k in mot:close(mot[k],r['motion'][k],key+name+k);close(nmot[k],r['native_motion'][k],key+name+'native'+k)
            row=dict(job=key,cell=job['cell'],split=key.split('/')[-1],seed=None if key.startswith('native/') else int(key.split('/')[1][4:]),group=name,**r)
            row['motion_harm_pct']={k:100*(mot[k]/nmot[k]-1) if nmot[k] else None for k in mot}
            row['motion_failed_metrics']=[k for k in mot if mot[k]>1.01*nmot[k]]
            row['old_success_preservation_pass']=counts['lost']<=.01*counts['native_good']
            row['error_harm_pct']={k:100*(r[k]/r['native_'+k]-1) for k in ('ADE','FDE','J')}
            (references if key.startswith('native/') else rows).append(row)
    del p,native,gt,a,f,j,g,na,nf,nj,ng
    print('reviewed',key,flush=True)
assert len(rows)==144 and len(references)==16
for cell,root in [('A',T1),('B',T1),('C',H)]:
    for seed in (42,43,44):
        p=root/'runs'/cell/f'seed{seed}'/'history.json';hh=read(p)
        assert len(hh)==25 and sum(x['optimization_steps'] for x in hh)==1850
        assert all(x['n']==18909 for x in hh)
        history.append(dict(cell=cell,seed=seed,rows=25,steps=1850,first_total=hh[0]['total'],last_total=hh[-1]['total'],SHA=sha(p),preupdate_not_final_population=True))
verify();guard.guard()
assert 'torch' not in sys.modules
summary=dict(status='PASS_SAVED_CPU_SUPPLEMENT',at=datetime.now().astimezone().isoformat(),source_count=len(sources),locked_source_and_completed_files_checked_twice=len(locked),jobs=len(jobs),science_groups=len(rows),native_groups=len(references),max_saved_numeric_error=max(errors),rows=rows,native=references,histories=history,statistics=stats,production_audit=read(H/'ARITHMETIC_AUDIT.json'),production_resource=read(H/'RESOURCE.json'),review_resource=guard.RESOURCE,not_external_review=True,no_torch_model_forward_backward_optimizer=True,clusters_and_20k_CI_recomputed_by_production_independent_audit_not_repeated_here=True,T1_resource_deviation_preserved=True)
write('SUMMARY.json',summary)
write('REVIEW.json',{k:v for k,v in summary.items() if k not in ('rows','native','statistics')})
print('PASS MAXERROR',max(errors),'SCIENCEGROUPS',len(rows),flush=True)
