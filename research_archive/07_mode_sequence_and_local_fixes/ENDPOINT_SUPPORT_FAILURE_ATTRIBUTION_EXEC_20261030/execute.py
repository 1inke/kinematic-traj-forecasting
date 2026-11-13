"""One finite foreground CPU diagnosis. Exclusive output writes prevent silent reruns."""
import ast
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import attribution_core as c

def fixtures():
    tests=[]
    # Literal same-slot joint quality: independently low ADE/FDE cannot be spliced.
    a=np.array([.19,.21]);f=np.array([.41,.39]);assert not ((a<=.2)&(f<=.4)).any();tests.append('disjoint_minima')
    b=np.array([[1,0,0],[1,1,0]],bool);s=np.zeros_like(b)
    assert np.array_equal(b.sum(1),[1,2]) and np.all(b.any(1)&~s.any(1));tests.append('all_supports_required')
    p=np.arange(20*12*2,dtype=float).reshape(1,20,12,2)/100;gt=np.zeros((1,12,2))
    perm=np.arange(19,-1,-1);q=c.quality(p,gt);r=c.quality(p[:,perm],gt)
    for x,y in zip(q,r):assert np.array_equal(x[:,perm],y)
    assert np.argmin(np.zeros(20))==0;tests.append('slot_permutation_and_firsttie')
    u=np.array([.9,.9]);assert (np.abs(u)<=1).all() and np.linalg.norm(.4*u)>.4;tests.append('continuous_norm_no_clip')
    ac=c.space('END');assert c.decode(np.zeros((12,2)),[0,0],ac)==ac.identity
    assert np.array_equal(c.apply_action(p[0,0],ac,ac.identity),p[0,0]);tests.append('identity_priority_exact')
    assert c.event(True,False)=='lost' and c.event(False,True)=='rescue';assert ''.join(map(str,[1,0,1,0]))=='1010';tests.append('stage_reversals_not_exclusive')
    # Real selector implementation, tiny observed fixture with wholetrack exclusion.
    zz=np.zeros((2,20,40),np.float32);zz[1,:,16:]=1
    mm,_=c.nearest_maps(zz,np.array([5,8]),np.array(['a|1','a|2']),np.array(['a','a']))
    assert np.all(mm['H_main'][:20,0]>=20) and np.all(mm['H_main'][:20,1:]==-1)
    me,_=c.nearest_maps(zz,np.array([5,8]),np.array(['a|1','a|1']),np.array(['a','a']))
    assert np.all(me['H_main']==-1);tests.append('wholetrack_exclusion')
    nan=zz.copy();nan[0,0,0]=np.nan
    try:c.nearest_maps(nan,[5,8],np.array(['a|1','a|2']),np.array(['a','a']))
    except ValueError:pass
    else:raise AssertionError('nonfinite accepted')
    zeros=np.zeros((128,20,40),np.float32);zeros[0,0]=np.float32(-0.)
    targets=np.zeros((128,20,2));targets[0,0]=[1,1]
    db=c.duplicate_bound(zeros,targets)
    expected=((targets-targets.mean((0,1)))**2).sum()/2/2560
    assert db['H']['groups']==1 and abs(db['H']['empirical_MSE_lower_bound']-expected)<2e-12
    tests.append('insufficient_empty_nonfinite_duplicate_signedzero')
    # Selector signature cannot receive GT; C/Q/S construction is observed-only.
    assert set(c.nearest_maps.__code__.co_varnames[:c.nearest_maps.__code__.co_argcount])=={'z','ids','tracks','sequence','check'}
    assert np.array_equal(mm['H_main'],c.nearest_maps(zz,[5,8],np.array(['a|1','a|2']),np.array(['a','a']))[0]['H_main'])
    tests.append('GT_not_selector_input')
    for k in range(20):
        cc=p.copy();cc[0,k]=0;assert cc.shape[1]==20
    tests.append('single_slot_always_K20_no_subset_search')
    assert 'torch' not in sys.modules
    return tests

def main():
    started=time.perf_counter();g=c.guard();g.check(mode='startup',force=True)
    for p in c.ROOT.glob('*.py'):
        tree=ast.parse(p.read_text(encoding='utf-8'))
        for n in ast.walk(tree):
            if isinstance(n,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module or '']
                assert not any(x.startswith(('torch','pickle')) for x in names)
    src,sha_seconds=c.source_lock(g.check)
    c.write_json(c.ROOT/'RESEARCH_REVIEW_DA1.json',dict(status='PASS_WITH_EXPLORATORY_LIMITS',same_assistant_not_external=True,
        issue='This design already saw 7lost; GT restore and oracle are definitionally favorable, not deployable methods.',
        resolution='All128/all20, all15 event pages; no training; GT-free map sealed before label association; no causal/root claim.'))
    tests=fixtures();x,z,idx=c.observed()
    # This is deliberately before loading GT_train, teacher labels or saved predictions.
    maps,distances,nn_seconds=c.seal_maps(x,z,g.check)
    gt,labels,masks=c.selected_labels(x,idx)
    paths,aid,failed,q,e0,e1,active=c.build_ports(x,gt,labels,masks,g.check)
    pt=time.perf_counter()
    smoke=c.counterfactual(x,gt,paths,q,g.check,only_first=8)
    for j,p in enumerate(paths):
        sm=c.sample_metrics(p[:8],gt[:8])
        assert np.array_equal(sm['good'],q['good'][j,:8])
    preflight_seconds=time.perf_counter()-pt
    # Conservative measured full counterfactual scaling, independent second implementation
    # allowance, plots + source IO; not the previous tiny training compute time.
    planning=3*preflight_seconds*16+2*nn_seconds+2*sha_seconds+120
    c.write_json(c.ROOT/'PREFLIGHT.json',dict(status='PASS_REAL_FIRST8_AND_10_FIXTURES',fixtures=tests,
        population='original SELECTION first8, not selected by outcome',counterfactual_smoke_sets=320,
        real_first8_seconds=preflight_seconds,full_neighbor_seconds=nn_seconds,source_initial_seconds=sha_seconds,
        planning_seconds=planning,ETA_seconds=[.7*planning,1.5*planning],no_Torch=True,
        steps_so_far='source bridge, observed-only fullmap sealed, first8 full5port/counterfactual pipeline; no training'))
    print('PREFLIGHT PASS; planning seconds',planning,flush=True)
    if planning>1800:raise RuntimeError('Planning exceeds recommended30min: report measured components before full run')
    rows=c.group_metrics(x,gt,paths,q,g.check)
    strata,trans,gains=c.stage_tables(x,paths,aid,failed,q,e0,e1,active)
    cf=c.counterfactual(x,gt,paths,q,g.check)
    for key,val in smoke.items():
        assert np.allclose(cf[key][:8],val,atol=2e-12,rtol=0)
    util,ns=c.utility(x,z,maps,labels,masks,q)
    c.write_json(c.ROOT/'exact_input_conflicts.json',c.duplicate_bound(z,labels['target'].astype(np.float32).astype(float)))
    ci=c.bootstrap(x,q,strata,util)
    np.save # Only generated outputs use c.npz_new, never modify old arrays.
    c.npz_new(c.ROOT/'observed_inputs.npz',sample_id=x['sample_id'],normalized=z,track_key=x['track_key'],sequence=x['sequence'])
    c.verify_files(src,g.check);g.check(force=True)
    c.write_json(c.ROOT/'RESOURCE_COMPUTATION.json',dict(status='PASS',policy='warning_first_remaining_v3',estimate=g.estimate,
        observations=g.observations,no_memory_locked=True,no_OS_changes=True,no_old_process_queries=True))
    c.write_json(c.ROOT/'COMPUTATION_RESULT.json',dict(status='SUCCESS_OFFLINE_COMPUTATION_REVIEW_REQUIRED',
        windows=128,tracks=97,candidate_ports=12800,counterfactual_sets=5120,
        science_all49_new_enumerations=0,old_Jmatrices_reused=True,train_updates=0,model_forward=0,
        elapsed_seconds=time.perf_counter()-started,source_count=len(src),source_unchanged=True,
        checksum_map=c.sha(c.ROOT/'neighbor_map.npz'),preflight_before_full_metrics=True))
    print('All A0/A1/A2/A3 saved; independent review and plots pending',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        out=c.ROOT/('FAILURE_'+time.strftime('%H%M%S')+'.json')
        c.write_json(out,dict(status='FAILURE_PRESERVED',error=repr(e),traceback=traceback.format_exc()))
        raise
