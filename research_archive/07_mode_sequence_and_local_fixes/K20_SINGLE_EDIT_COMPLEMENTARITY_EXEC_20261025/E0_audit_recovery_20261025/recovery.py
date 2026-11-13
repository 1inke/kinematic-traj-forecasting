"""Only route writes and detach oracle views. Historical scientific function bodies unchanged."""
import sys,os,json,time,hashlib,traceback,msvcrt
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'E0_execution_20261025'
sys.path.insert(0,str(PARENT))
import common as c
from data_contract import load
import review_saved as review
RAW_DIRECT=review.direct
RAW_WRITE=c.write_json
def bounded_direct(*args,**kw):
    result,path=RAW_DIRECT(*args,**kw)
    return result,path.copy()
def write(path,value):
    path=Path(path)
    try:relative=path.relative_to(PARENT)
    except ValueError:pass
    else:path=HERE/relative
    RAW_WRITE(path,value)
def stage(name,**kw):write(HERE/'runtime_state.json',dict(status='RUNNING',stage=name,at=c.now(),**kw))
def install():
    review.direct=bounded_direct;review.write_json=write;review.stage=stage
def verify():
    m=c.read_json(HERE/'source_manifest.json')
    for path,ss in m['files'].items():
        if c.sha256(c.ROOT/path)!=ss:raise ValueError('Recovery source/output SHA '+path)
    if c.digest(m['files'])!=m['digest']:raise ValueError('Recovery manifest digest')
    return m['digest']
def parent_done():
    source=c.verify();out=[]
    for seed in c.SEEDS:out.append(c.validate_done(PARENT/f'seed{seed}'/'done.json',source))
    return source,out
def prepare():
    if (HERE/'source_manifest.json').exists():raise FileExistsError('Locked version exists')
    parent=c.read_json(PARENT/'source_manifest.json');files={k:v['sha256'] for k,v in parent['files'].items()}
    source,done=parent_done()
    for seed,d in zip(c.SEEDS,done):
        for name,ss in d['outputs'].items():files[(PARENT/f'seed{seed}'/name).relative_to(c.ROOT).as_posix()]=ss
        p=PARENT/f'seed{seed}'/'done.json';files[p.relative_to(c.ROOT).as_posix()]=c.sha256(p)
    for name in ('source_manifest.json','PREFLIGHT.json','PRELAUNCH_REVIEW.json','FAILURE.json','worker_stderr.log','worker_state.json','exit_status.json','SPLIT_REGISTRY.npz','IDENTITY_AUDIT.json','INPUT_MANIFEST.json'):
        p=PARENT/name;files[p.relative_to(c.ROOT).as_posix()]=c.sha256(p)
    for p in HERE.iterdir():
        if p.suffix in ('.py','.ps1','.md'):files[p.relative_to(c.ROOT).as_posix()]=c.sha256(p)
    write(HERE/'source_manifest.json',dict(digest=c.digest(files),files=files,inherited_count=len(parent['files']),parent_source_digest=source,at=c.now()))
    print('SOURCE_LOCK',len(files),c.digest(files),flush=True)
def preflight():
    guard=c.Guard(c.ESTIMATE);guard.check('startup');start=time.perf_counter();source=verify();ps,done=parent_done();data=load(save=False)
    # Exactly demonstrate ownership defect and exact copy, not a scientific recomputation change.
    n=data['native'][0];y=data['gt'][0]
    from data_contract import kin_opt
    kin=kin_opt(42,data['opt_row']);r,view=RAW_DIRECT(n,kin[0],y);r2,copy=bounded_direct(n,kin[0],y)
    base=view
    while isinstance(base.base,np.ndarray):base=base.base
    assert view.base is not None and base.nbytes==401*20*12*2*8 and copy.base is None and np.array_equal(view,copy)
    ownership=dict(original_selected_bytes=view.nbytes,original_retained_base_bytes=base.nbytes,repaired_retained_bytes=copy.nbytes,original_is_view=True,repaired_is_owned_copy=True,values_exact=True)
    del view,base,copy
    install();maximum=0.;total=0;kept=[];t0=time.perf_counter()
    # Fixed first256 and old failure-neighbourhood1536:1792, allseed42. No GT selection.
    selected=range(0,256),range(1536,1792)
    for subset in selected:
        for offset in range(subset.start,subset.stop,128):
            file=f'chunk_{offset:05d}_{offset+128:05d}.npz'
            with np.load(PARENT/'seed42'/file,allow_pickle=False) as z:
                for i in range(128):
                    row=offset+i;ref,path=review.direct(data['native'][row],kin[row],data['gt'][row]);kept.append(path)
                    assert path.base is None
                    saved={k:z[k][i] for k in ('motion','quality','margin','rejects','eligible','B','target','action')}
                    maximum=max(maximum,review.compare(saved,ref));total+=400
            guard.check()
    elapsed=time.perf_counter()-t0
    p=np.asarray(kept);ix=np.r_[np.arange(256),np.arange(1536,1792)]
    rows,_=review.independent_population(p,data['gt'][ix],data['groups'][ix],data['native'][ix],data['ids'][ix],data['tracks'][ix])
    write(HERE/'smoke_population.json',rows)
    # Actual original completed reader and its bad-source/output branches.
    reader=HERE/'reader_fixture';reader.mkdir(exist_ok=True);np.save(reader/'array.npy',np.arange(3))
    d=dict(source_digest=source,outputs={'array.npy':c.sha256(reader/'array.npy')})
    write(reader/'valid.json',d);c.validate_done(reader/'valid.json',source)
    rejected=[]
    for kind in ('source','output'):
        bad={**d,'outputs':d['outputs'].copy()}
        if kind=='source':bad['source_digest']='bad'
        else:bad['outputs']['array.npy']='bad'
        write(reader/f'bad_{kind}.json',bad)
        try:c.validate_done(reader/f'bad_{kind}.json',source)
        except ValueError:rejected.append(kind)
        else:raise AssertionError('Bad completed reader accepted')
    assert verify()==source
    last=guard.check();planning=1.25*(elapsed*3*4096/512+35)
    result=dict(status='PASS',at=c.now(),source_digest=source,ownership_repair=ownership,actual_windows=512,actual_nonSTOP_actions=total,includes_old_failure_neighbourhood=True,max_abs_error=maximum,all_literal_masks_labels_exact=True,old_done_full_SHA_verified=True,actual_bad_reader_rejected=rejected,resource_observations=guard.observations,last_resource=last,estimate=c.ESTIMATE,resource_factor=1.3,benchmark=dict(seconds=elapsed,planning_seconds=planning,ETA_seconds=[.7*planning,1.5*planning]),elapsed_seconds=time.perf_counter()-start,no_forward_training=True,math_body_change=False)
    write(HERE/'PREFLIGHT.json',result);print(json.dumps(result['benchmark']),flush=True)
def worker():
    lock=(HERE/'worker.lock').open('a+b')
    if (HERE/'worker_state.json').exists():raise FileExistsError('Recovery already existed')
    if not lock.tell():lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    state=dict(status='RUNNING',pid=os.getpid(),parent_pid=os.getppid(),started_at=c.now());write(HERE/'worker_state.json',state);stage('RECOVERY_INITIAL_SOURCE_COMPLETED_OUTPUT_SHA')
    guard=c.Guard(c.ESTIMATE)
    try:
        guard.check('startup');source=verify();pre=c.read_json(HERE/'PREFLIGHT.json');assert pre['status']=='PASS' and pre['source_digest']==source
        ps,done=parent_done();data=load(save=False);install()
        reviews=[review.audit_seed(seed,data,ps,guard) for seed in c.SEEDS]
        write(HERE/'ARITHMETIC_AUDIT.json',dict(status='PASS_ALL_4915200_ACTIONS_AND_ORACLE',records=reviews,no_model_replay=True))
        gates=[d['gate'] for d in done]
        stage('RECOVERY_FINAL_SOURCE_COMPLETED_OUTPUT_SHA');assert verify()==source;parent_done();last=guard.check()
        write(HERE/'PRESERVATION.json',dict(status='PASS',source_digest=source,parent_source_digest=ps,all_parent_completed_outputs_and_failure_unchanged=True))
        write(HERE/'RESOURCE.json',dict(status='PASS',estimate=c.ESTIMATE,factor=1.3,observations=guard.observations,last=last,production_lifetime_peak_private=last['snapshot']['peak_private'],production_lifetime_peak_RSS=last['snapshot']['peak_rss']))
        decision='E0_PASS_ALLOW_E1_OWN_PREFLIGHT' if all(d['pass_all'] for d in gates) else 'E0_STOP'
        write(HERE/'DECISION.json',dict(status=decision,gates=gates,E1='OWN_PREFLIGHT_REQUIRED' if decision.startswith('E0_PASS') else 'NOT_TRIGGERED'))
        write(HERE/'RESULT.json',dict(status='PASS_E0_SAVED_AUDIT_RECOVERY',scientific_decision=decision,source_digest=source,nonSTOP_actions=4915200,formal_training=0,parent_FAILURE_preserved=True))
        state.update(status='SUCCESS',finished_at=c.now());write(HERE/'exit_status.json',dict(status='SUCCESS',at=c.now()));write(HERE/'worker_state.json',state);write(HERE/'runtime_state.json',dict(status='COMPLETED_REVIEW_REQUIRED',stage='RECOVERY_COMPLETE',finished_at=state['finished_at'],active_worker=False))
    except BaseException as e:
        write(HERE/'FAILURE.json',dict(type=type(e).__name__,message=str(e),at=c.now(),traceback=traceback.format_exc(),resource_observations=guard.observations));state.update(status='FAILURE',finished_at=c.now());write(HERE/'worker_state.json',state);write(HERE/'exit_status.json',dict(status='FAILURE',at=c.now()));raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
if __name__=='__main__':
    try:
        {'prepare':prepare,'preflight':preflight,'worker':worker}[sys.argv[1]]()
    except BaseException as e:
        if sys.argv[1]!='worker':write(HERE/(sys.argv[1].upper()+'_FAILURE.json'),dict(type=type(e).__name__,message=str(e),at=c.now(),traceback=traceback.format_exc()))
        raise
