"""CPU exact/synthetic/real GT-free matching and opt-only label smoke tests."""
from core import *
from matching import *
from independent import hungarian
from select_and_audit import audit
from posthoc import reconstruct_quality,positives,compare,independent_compare
from statistics_fixed import summarize_values
def main():
    require3();g=guard();source0=verify();start=time.perf_counter();out=HERE/f'preflight_{time.time_ns()}';out.mkdir()
    rng=np.random.default_rng(20261027)
    # 20x20 independent optimality, physical permutations and ambiguity fixtures.
    for i in range(8):
        a=rng.uniform(-2,2,(20,12,2));per=rng.permutation(20);b=a[per]
        cost=matrix(a,b);m,total=assignment(cost);mi,ti=hungarian(cost)
        assert total<=2e-12 and abs(total-ti)<=2e-12 and np.array_equal(m,np.argsort(per))
        mapping=action_map(m,m);assert len(np.unique(mapping))==401
        reverse,rt=assignment(cost,True);assert np.array_equal(reverse,m) and rt==total
    zeros=np.zeros((20,20));m,z=assignment(zeros);gap,amb,n=certificate(zeros,m,z);assert gap==0 and amb and n==20
    bad_slot=False
    try:action_map(np.zeros(20,int),np.arange(20))
    except AssertionError:bad_slot=True
    assert bad_slot
    # Strict full20 interval and target-track exclusions.
    lib=dict(sample_id=np.arange(5),track_key=np.array(['other0','other1','other2','other3','same']),sequence=np.array(['s']*5),start=np.array([0,19,20,-1,40]),known=np.array([1,1,1,0,1],bool),past=np.zeros((5,8,2)))
    query=dict(sample_id=np.array([99]),track_key=np.array(['same']),sequence=np.array(['s']),start=np.array([0]),known=np.array([True]),past=np.zeros((1,8,2)))
    assert legal_rows(query,0,lib).tolist()==[False,False,True,False,False]
    # Literal tau equality admitted, its next representable float rejected.
    ident={'opt':dict(reference_rows=np.arange(4),track_key=np.array(['a','b','c','d']),sample_id=np.arange(4)),'hold':dict(reference_rows=np.array([0]),track_key=np.array(['q']),sample_id=np.array([99]),known=np.array([True]))}
    fixture=dict(kind=np.array([0,0,0,0,1,1]),reference_position=np.array([0,1,2,3,0,0]),query_row=np.array([0,1,2,3,0,0]),library_row=np.array([1,2,3,0,1,2]),d=np.array([0,1,2,3,.75,np.nextafter(.75,np.inf)]),reverse_d=np.array([0,1,2,3,.75,np.nextafter(.75,np.inf)]))
    choose,summary=selection(fixture,ident);assert summary['primary']['tau']==.75 and choose['primary'].tolist()==[4]
    # Full multi-positive/common-domain and STOP fixture, literal loop reference.
    eq=np.ones(401,bool);ep=np.ones(401,bool);bq=np.zeros(401,bool);bp=np.zeros(401,bool);bq[1]=True;bp[2]=True
    qq=np.full((401,3),2.);pq=qq.copy();qq[1]=1;pq[2]=1
    c,v=compare(eq,ep,bq,bp,qq,pq,np.arange(401));ic=independent_compare(eq,ep,bq,bp,qq,pq,np.arange(20),np.arange(20))
    assert c['conflict'] and c['regret']==1 and c['regret']==ic['regret']
    ep[1:]=False;c,v=compare(eq,ep,bq,bp,qq,pq,np.arange(401));assert c['conflict'] is None and c['domain_count']==1 and c['regret']==0
    # Actual immutable opt4 quality/B, not held labels or selection effects.
    pools=load_npz(HERE/'prepared/pools_seed42_opt.npz');label=old_columns(42,'opt',('sample_id','quality','B','gt'))
    for i in range(4):
        q=reconstruct_quality(pools['native'][i],pools['kin'][i],label['gt'][i]);assert np.max(abs(q-label['quality'][i]))<=2e-12
        assert np.array_equal(positives(q,pools['eligible'][i]),label['B'][i])
    del pools,label;g.check()
    # Actual first2 hashrefs in EACH opt/hold: no B/GT/group read.
    t=time.perf_counter();pack,selected,actual=compute(42,g,limit_references=2);matching_seconds=time.perf_counter()-t
    save_npz(out/'MATCHES.npz',**pack);save_npz(out/'SELECTED.npz',**selected)
    t=time.perf_counter();review=audit(load_npz(out/'MATCHES.npz'),load_npz(out/'SELECTED.npz'),actual,42,g,limit_references=2);audit_seconds=time.perf_counter()-t
    # Actual negative readers: ID/extraGT/slot/source/output. Preserve fixtures.
    identity=load_npz(HERE/'prepared/identity_opt.npz');pool=load_npz(HERE/'prepared/pools_seed42_opt.npz');badids={k:v[:2].copy() for k,v in pool.items()};badids['sample_id'][0]=-1
    save_npz(out/'bad_ID.npz',**badids);reject=[]
    try:checked_pools(out/'bad_ID.npz',{k:v[:2] for k,v in identity.items() if k!='reference_rows'})
    except AssertionError:reject.append('ID')
    badextra={k:v[:2].copy() for k,v in pool.items()};badextra['gt']=np.zeros((2,12,2));save_npz(out/'bad_extra_GT.npz',**badextra)
    try:checked_pools(out/'bad_extra_GT.npz',{k:v[:2] for k,v in identity.items() if k!='reference_rows'})
    except AssertionError:reject.append('GT_input')
    f=out/'reader_good';f.mkdir();write_json(f/'payload.json',{'test':1});contract=dict(source=source0,split='opt_only',prepared_SHA=sha256(HERE/'prepared/done.json'));finish(f,contract);completed(f,contract)
    for key in ('source','split','prepared_SHA'):
        bad=dict(contract);bad[key]='wrong'
        try:completed(f,bad)
        except AssertionError:reject.append(key)
    bad=out/'reader_bad_output';bad.mkdir();write_json(bad/'payload.json',{'test':2});write_json(bad/'done.json',dict(contract=contract,outputs={'payload.json':sha256(f/'payload.json')}))
    try:completed(bad,contract)
    except AssertionError:reject.append('output')
    assert len(reject)==6
    # All20k actual draw-counts fixture, NaN/no-redraw and signed values.
    draws=np.random.default_rng(20261026).integers(16,size=(20000,16),dtype=np.int32);vals=np.random.default_rng(7).normal(size=(3,16));vals[0,::3]=np.nan
    stats,a,b=summarize_values(vals,draws);assert stats['combined_valid']>19000
    empty=vals.copy();empty[1]=np.nan;ss,aa,bb=summarize_values(empty,draws);assert ss['combined_valid']==0 and ss['combined_invalid']==20000 and np.isnan(bb).all()
    assert 'torch' not in sys.modules and verify()==source0
    report=dict(status='PASS',at=now(),source=source0,seconds=time.perf_counter()-start,
        real_pairs=len(pack['d']),real_science_matching_problems=2*len(pack['d']),matching_seconds=matching_seconds,independent_audit_seconds=audit_seconds,
        full_real_saved_readback=True,independent_Hungarian_primal_dual=review,forbidden_edge_ambiguity_fixture=True,strict_time_t0_tau_literal_fixture=True,
        full401_mapping_positive_set_conflict_opportunity_STOP=True,opt_only4_quality_B_exact=True,no_held_B_GT_group_read=True,
        reader_rejections=reject,slot_rejected=True,all20000_bootstrap_independent_zero_denominator_no_redraw=True,
        no_Torch_model_forward_optimizer_training_or_RNG_restore_claim=True)
    write_json(HERE/'PREFLIGHT.json',report);resource(out/'RESOURCE.json',g);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
