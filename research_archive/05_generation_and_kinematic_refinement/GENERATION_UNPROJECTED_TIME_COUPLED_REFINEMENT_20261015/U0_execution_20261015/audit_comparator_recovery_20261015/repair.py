import common as c
import importlib.util,time,json,sys,copy
spec=importlib.util.spec_from_file_location('readonly_first_recovery',c.FIRST/'recovery.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
previous_locks=r.locks

def locks():
    lock=previous_locks();files=dict(c.read(c.FIRST/'source_manifest.json')['files'])
    assert len(files)==2057
    for p,h in lock['files'].items():
        if p in files:assert files[p]==h,('INHERITED_DIGEST_CHANGED',p)
        files[p]=h
    paths=[p for p in c.FIRST.iterdir() if p.is_file() and p.name not in {'runtime_state.json','progress.json','worker.lock'}]
    paths+=list((c.FIRST/'failures').rglob('*'))+list((c.OUT/'failures').rglob('*'))
    for p in paths:
        if p.is_file():
            h=c.sha(p)
            if str(p) in files:assert files[str(p)]==h
            files[str(p)]=h
    lock.update(files=files,source_digest=c.digest(files),inherited_first_recovery_sources=2057,comparator_repair_only=True)
    return lock
r.locks=locks

def fixture():
    start=time.perf_counter();c.guard('startup')
    old=c.read(c.PARENT/'BOOTSTRAP.json');new=r.stats.compute(True)
    try:c.prior.compare(old,new)
    except TypeError:original_type_error_reproduced=True
    else:raise AssertionError('Expected original nested-dict-list TypeError')
    error=c.compare(old,new)
    for bad in (dict(rows=[{'x':2}]),dict(rows=[{'x':1},{'x':1}]),dict(rows=[{'y':1}])):
        try:c.compare(dict(rows=[{'x':1}]),bad)
        except AssertionError:pass
        else:raise AssertionError('Comparator accepted invalid structure/value')
    a=dict(rows=[dict(s='ok',b=True,n=None,values=[1.,2.])]);assert c.compare(a,copy.deepcopy(a))==0
    # Actual12-CI and20k saved draw recomputation is unchanged, no model forward.
    c.write('COMPARATOR_FIXTURE.json',dict(status='PASS',original_nested_list_TypeError_reproduced=original_type_error_reproduced,actual_bootstrap_all_fields_max_error=error,tolerance=2e-12,bad_value_length_keys_rejected=True,original_statistics_AST_unchanged=True,no_forward_backward_optimizer=True,elapsed_seconds=time.perf_counter()-start))
    return error

if __name__=='__main__':
    try:
        if '--preflight' in sys.argv:
            fixture();print(json.dumps(r.preflight(),ensure_ascii=False,default=lambda x:x.tolist()),flush=True)
        else:r.main()
    except BaseException:c.failure('STOP_COMPARATOR_RECOVERY');raise
