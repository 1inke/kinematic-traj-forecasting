"""Actual CPU E0 stage preflight. No Torch or generator forward."""
import time,shutil,tempfile,inspect
import numpy as np
from common import *
from data_contract import load,kin_opt
from space import enumerate_window,materialize,eligible_actions,quality_actions,labels,oracle
from review_saved import direct,compare,independent_population
def fixtures():
    rng=np.random.default_rng(20261025);answer=[]
    n=rng.normal(size=(20,12,2))*.1;k=rng.normal(size=(20,12,2))*.1;y=rng.normal(size=(12,2))*.1
    cases=[('general',n,k,y),('zero_native',np.zeros_like(n),k,y),('identical_copies',np.zeros_like(n),np.zeros_like(k),np.zeros_like(y)),('duplicate_pool',n,n.copy(),y)]
    unique=np.ones_like(n)*3;unique[0]=0
    cases.append(('unique_good_deleted',unique,k,np.zeros_like(y)))
    disjoint=np.ones_like(n)*5;disjoint[0,:11]=0;disjoint[0,11]=[.6,0];disjoint[1]=[.25,0];disjoint[1,11]=0
    cases.append(('ADE_FDE_different_slots',disjoint,disjoint.copy(),np.zeros_like(y)))
    for name,nn,kk,yy in cases:
        p=enumerate_window(nn,kk,yy);r,path=direct(nn,kk,yy);err=compare(p,r)
        assert np.array_equal(materialize(nn,kk,p['action']),path)
        if name=='zero_native':assert p['eligible'].sum()==1 and np.all(p['rejects'][1:].any(0))
        if name=='identical_copies':assert not p['B'].any() and p['target'][0] and p['action']==0
        # Changing GT cannot change eligibility or action materialization.
        p2=enumerate_window(nn,kk,yy+100)
        assert np.array_equal(p['eligible'],p2['eligible']) and np.array_equal(p['motion'],p2['motion'])
        for a in (0,1,20,21,400):
            m=materialize(nn,kk,a);mask=np.ones(20,bool)
            if a:mask[(a-1)//20]=False
            assert np.array_equal(m[mask],nn[mask])
        answer.append(dict(name=name,max_error=err))
    assert tuple(inspect.signature(eligible_actions).parameters)==('n','k')
    # Multiple positives and empty B are mathematical label fixtures, not natural outcomes.
    q=np.array([[1.,1.,2.],[.9,.9,1.9],[.8,1.,1.8],[1.1,.7,2.]])
    B,t=labels(q,np.ones(4,bool));assert np.array_equal(B,[False,True,True,False]) and oracle(q,B)==2
    B,t=labels(np.ones((401,3)),np.r_[True,np.zeros(400,bool)]);assert t.sum()==1 and t[0]
    return answer
def contract_fixture(source):
    folder=HERE/'preflight_smoke'/'reader_contract';folder.mkdir(parents=True,exist_ok=True)
    np.save(folder/'output.npy',np.arange(4))
    valid=dict(source_digest=source,outputs={'output.npy':sha256(folder/'output.npy')})
    write_json(folder/'valid.json',valid);validate_done(folder/'valid.json',source)
    rejected=[]
    for kind in ('source','output'):
        broken={**valid,'outputs':valid['outputs'].copy()}
        if kind=='source':broken['source_digest']='bad'
        else:broken['outputs']['output.npy']='bad'
        write_json(folder/f'bad_{kind}.json',broken)
        try:validate_done(folder/f'bad_{kind}.json',source)
        except ValueError:rejected.append(kind)
        else:raise AssertionError('Actual reader accepted bad '+kind)
    return dict(valid_SHA_skip=True,actual_bad_reader_rejection=rejected)
def main():
    start=time.perf_counter();guard=Guard(ESTIMATE);guard.check('startup');source=verify();f=fixtures();data=load(save=True);guard.check()
    kin=kin_opt(42,data['opt_row']);saved=[];maximum=0.;t0=time.perf_counter()
    # Registered actual first128 hash-ordered FIT_opt, not GT-selected.
    for i in range(128):saved.append(enumerate_window(data['native'][i],kin[i],data['gt'][i]))
    enum=time.perf_counter()-t0
    smoke=HERE/'preflight_smoke';smoke.mkdir(exist_ok=True)
    pack={name:np.asarray([s[name] for s in saved]) for name in saved[0]};pack['sample_id']=data['ids'][:128]
    np.savez_compressed(smoke/'actual128.npz',**pack);paths=[];t0=time.perf_counter()
    with np.load(smoke/'actual128.npz',allow_pickle=False) as z:
        for i in range(128):
            ref,path=direct(data['native'][i],kin[i],data['gt'][i]);paths.append(path)
            maximum=max(maximum,compare({n:z[n][i] for n in saved[0]},ref))
    audit=time.perf_counter()-t0;t0=time.perf_counter()
    metrics,cc=independent_population(np.asarray(paths),data['gt'][:128],data['groups'][:128],data['native'][:128],data['ids'][:128],data['tracks'][:128]);population=time.perf_counter()-t0
    write_json(smoke/'POPULATION_METRICS.json',metrics)
    contract=contract_fixture(source);assert verify()==source;resource=guard.check()
    # Full workload + explicit source/IO and full twice-population review allowance.
    compute=(enum+audit)*3*4096/128;pop=population*3*4096/128
    planning=(compute+pop+60)*1.25
    free=shutil.disk_usage(HERE).free;conservative=2*1024**3
    if free<1.3*conservative:raise OSError('STOP_DISK expected new outputs2GiB plus1.3reserve')
    out=dict(status='PASS',at=now(),source_digest=source,source_count=len(read_json(HERE/'source_manifest.json')['files']),actual_128_windows=128,actual_nonSTOP_actions=51200,actual_seed42_first128_not_GT_selected=True,synthetic=f,max_abs_error=maximum,contract=contract,no_forward_training=True,identity=data['identity'],benchmark=dict(enumeration128_seconds=enum,independent51200action_seconds=audit,population128_seconds=population,planning_seconds=planning,ETA_seconds=[.7*planning,1.5*planning],not_production_guarantee=True),resource=resource,observations=guard.observations,disk_free_bytes=free,estimated_new_outputs_bytes=conservative,elapsed_seconds=time.perf_counter()-start)
    write_json(HERE/'PREFLIGHT.json',out);print(json.dumps(out['benchmark']));print('PASS E0 ACTUAL PREFLIGHT')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        import traceback
        write_json(HERE/'PREFLIGHT_FAILURE.json',dict(at=now(),type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));raise
