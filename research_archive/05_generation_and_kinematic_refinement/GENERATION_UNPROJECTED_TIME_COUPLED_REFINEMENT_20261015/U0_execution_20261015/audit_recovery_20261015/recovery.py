"""Frozen saved-output recovery. No training/model forward APIs are loaded."""
import common as c
import ast,types,sys,time,gc,json,math
import numpy as np,torch
torch.set_num_threads(1)

def isolated(name,filename,functions=None,extra=None):
    path=c.PARENT/filename;tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    if functions is not None:
        tree=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in functions],type_ignores=[])
        assert {n.name for n in tree.body}==set(functions)
    mod=types.ModuleType(name);mod.__dict__.update(extra or {});sys.modules[name]=mod
    exec(compile(tree,str(path),'exec'),mod.__dict__)
    return mod

m=isolated('model','model.py',('exact','reconstruct'),dict(np=np,torch=torch,ARMS=('free_position','integrated_acceleration'),SEEDS=(42,43,44)))
v=isolated('metrics','metrics.py')
stats=isolated('statistics_ci','statistics_ci.py')
audit=isolated('audit_saved','audit_saved.py')

def locks():
    original=c.read(c.PARENT/'source_manifest.json');files=dict(original['files']);assert len(files)==1916
    jobs=c.read(c.PARENT/'JOBS.json');runs=c.read(c.PARENT/'RUNS.json');assert len(jobs)==28 and len(runs)==6
    additions={p:h for row in list(jobs.values())+list(runs.values()) for p,h in row['files'].items()}
    paths=[c.PARENT/x for x in ('JOBS.json','RUNS.json','BOOTSTRAP.json','bootstrap_draws.npz','RESOURCE_FAILURE.json','FAILURE_REPORT.md','FAILURE_REPORT.json','exit_status.json','worker_state.json')]
    paths+=list((c.PARENT/'runs').glob('*/*/done.json'))+list((c.PARENT/'native').glob('*_done.json'))
    paths+=list((c.PARENT/'failures').rglob('*'))
    paths+=[*c.OUT.glob('*.py'),*c.OUT.glob('*.ps1'),c.OUT/'README.md',c.OUT/'protocol.json',c.OUT/'resource_policy.json',c.OUT/'MEMORY_ESTIMATE.json',c.POLICY/'memory_policy.py',c.ROOT/'RESOURCE_POLICY.json']
    for p in paths:
        if p.is_file():additions[str(p)]=c.sha(p)
    for p,h in additions.items():
        if p in files:assert files[p]==h,('INHERITED_DIGEST_CHANGED',p)
        files[p]=h
    return dict(status='LOCKED',parent_source_count=1916,parent_source_digest=c.digest(original['files']),files=files,source_digest=c.digest(files),no_parent_output_writes=True)

def verify_input():
    lock=c.read(c.OUT/'source_manifest.json');assert c.digest(lock['files'])==lock['source_digest'];c.verify(lock['files']);return lock

def subset(z,n):return {k:a[:n].copy() for k,a in z.items()}

def preflight():
    start=time.perf_counter();c.guard('startup');lock=locks();c.write('source_manifest.json',lock);c.verify(lock['files'])
    data=c.load();jobs=c.read(c.PARENT/'JOBS.json');runs=c.read(c.PARENT/'RUNS.json');timings=[];fixture=[]
    for arm in m.ARMS:
        row=jobs[arm+'/seed42/outer'];c.valid_done(row,lock['parent_source_digest'])
        full=c.npz(row['output']);z=subset(full,128);del full;gc.collect()
        report,aux=v.evaluate(z,False)
        path=c.OUT/('smoke_'+arm+'.npz');np.savez_compressed(path,**z)
        replay=c.npz(path);t=time.perf_counter();v.validate(replay,report,aux,arm);timings.append((time.perf_counter()-t)/128)
        c.write('smoke_'+arm+'_metrics.json',report)
        fixture.append(dict(arm=arm,n=128,status='PASS',paths_SHA=c.sha(path),no_forward_backward_optimizer=True))
        del z,replay,aux,report;gc.collect()
    key=next(iter(runs));row=runs[key];c.valid_done(row,lock['parent_source_digest'])
    state=torch.load(c.PARENT/'runs'/key/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert len(state['history'])==25 and all(r['optimization_steps']==74 and r['n']==18909 for r in state['history'])
    m.exact(state['history'],c.read(c.PARENT/'runs'/key/'history.json'))
    assert state['model']['native_identity_flag'].dtype==torch.bool and all(torch.isfinite(x).all() for x in state['model'].values())
    del state;gc.collect()
    try:c.valid_done(dict(row,source_digest='bad'),lock['parent_source_digest'])
    except AssertionError:pass
    else:raise AssertionError('BAD_SOURCE_ACCEPTED')
    bad=c.OUT/'deliberately_bad_output.fixture';bad.write_bytes(b'fixture')
    try:c.valid_done(dict(status='PASS',source_digest=lock['parent_source_digest'],files={str(bad):'0'*64}),lock['parent_source_digest'])
    except AssertionError:pass
    else:raise AssertionError('BAD_OUTPUT_ACCEPTED')
    # Same numerical guard decisions with cached versus disk configurations.
    assert c._config==c._read('resource_policy.json') and c._estimate==c._read('MEMORY_ESTIMATE.json')
    sample=c.policy.snapshot();a=c.policy.decide(sample,c._estimate['estimated_total_peak_bytes']);assert a['factor']==1.3 and not a['fixed6GiB_rule']
    count=sum(r['n'] for r in jobs.values());compute=max(timings)*count
    c.verify(lock['files']);elapsed=time.perf_counter()-start
    # Full-population sorting/motion, CPU checkpoint hashes, independent CI and
    # two complete source/output scans need headroom beyond 128-row throughput.
    planning=max(6.,(compute*2.5+2*elapsed+90)/60.)
    assert planning<=30,'STOP_AUDIT_BUDGET'
    result=dict(status='PASS',scope='SAVED_OUTPUT_AUDIT_ONLY',source_digest=lock['source_digest'],source_count=len(lock['files']),parent_source_count=1916,scientific_function_bodies='EXACT_ORIGINAL_AST',saved_smoke=fixture,registered_job_rows=count,independent_seconds_per_row=timings,analysis_contract_bad_source_output_rejected=True,checkpoint_CPU_metadata_only=True,all_parent_source_and_registered_outputs_SHA_initial_final=True,no_forward_backward_optimizer=True,elapsed_seconds=elapsed,planning_minutes=planning,ETA_minutes=[.7*planning,1.5*planning],resource=dict(c.RESOURCE))
    c.write('PREFLIGHT.json',result);return result

def decision(boot,jobs):
    # Compile exactly the original registered decision block, not a redefinition.
    tree=ast.parse((c.PARENT/'run.py').read_text(encoding='utf-8-sig'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    first=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cr' for t in n.targets))
    block=[]
    for n in fn.body[first:]:
        if isinstance(n,ast.Expr):break
        block.append(n)
    ns=dict(c=c,m=m,np=np,boot=boot,data={'split':c.SIZES})
    exec(compile(ast.Module(body=block,type_ignores=[]),'registered_parent_decision','exec'),ns)
    return ns['decision']

def main():
    start=time.perf_counter();pre=c.read(c.OUT/'PREFLIGHT.json');assert pre['status']=='PASS' and pre['planning_minutes']<=30;c.guard('startup')
    c.stage('RECOVERY_INITIAL_SOURCE_COMPLETED_OUTPUT_SHA');lock=verify_input();assert lock['source_digest']==pre['source_digest']
    audit.main()  # Exact original AST: full28 saved jobs,6 CPU checkpoints,12-CI.
    c.stage('RECOVERY_FINAL_SOURCE_COMPLETED_OUTPUT_SHA');verify_input()
    jobs=c.read(c.PARENT/'JOBS.json');runs=c.read(c.PARENT/'RUNS.json');boot=c.read(c.PARENT/'BOOTSTRAP.json');d=decision(boot,jobs)
    c.write('DECISION.json',d);c.write('PRESERVATION.json',dict(status='PASS',source_count=len(lock['files']),parent_source_count=1916,initial_final_SHA_unchanged=True,parent_FAILURE_preserved=True,no_parameter_updates=True))
    c.write('RESOURCE.json',dict(status='PASS',policy='task_memory_1p3_v1',**c.RESOURCE))
    result=dict(status='PASS_U0_UNPROJECTED_TEMPORAL_REFINEMENT',recovery='SAVED_OUTPUT_AUDIT_ONLY',decision=d,jobs=jobs,runs=runs,scientific_jobs=24,native_reference_jobs=4,histories=150,steps=11100,source_digest=lock['source_digest'],source_count=len(lock['files']),parent_source_digest=lock['parent_source_digest'],elapsed_seconds=time.perf_counter()-start,new_formal_experiments=0,parent_experiment_formal_count=1,saved_math_not_model_gradient_replay=True)
    c.write('RESULT.json',result);c.write('RECOVERY_RESULT.json',dict(status='SUCCESS',result_status=result['status'],parent_FAILURE_preserved=True,no_retraining=True));c.stage('SUCCESS',decision=d['status'])

if __name__=='__main__':
    try:
        z=preflight() if '--preflight' in sys.argv else main()
        if z:print(json.dumps(z,ensure_ascii=False,default=lambda x:x.tolist()),flush=True)
    except BaseException:c.failure('STOP_U0_AUDIT_RECOVERY');raise
