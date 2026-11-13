"""Original identity-bearing loader; no legacy train/runner is executed."""
from common import *
def main():
    setup();g=Guard(2*2**30);g.check(force=True);t=time.perf_counter();files={}
    def add(p,s):
        p=str(Path(p).resolve())
        if p in files and files[p]!=s:raise ValueError('INHERITED_DIGEST_CONFLICT '+p)
        files[p]=s
    for name in ('source_manifest.json','FINAL_PRESERVATION_V3.json'):
        z=read(PREVIOUS/name)
        for p,s in z['files'].items():add(Path(p) if Path(p).is_absolute() else PREVIOUS/p,s)
        add(PREVIOUS/name,sha(PREVIOUS/name))
    for p in [PREVIOUS/'FINAL_SEAL_CHECK_V3.json',*old.prior.PATHS.values(),OLD/'core.py',OLD/'training.py',OLD/'statistics.npz',OLD/'split_ids.npz',OLD/'input_contract.json',OLD/'source_audit/RESULT.json',PROJECT/'audit/verify.py',PROJECT/'src/f16/eval.py',PROJECT/'src/f16/checkpoint.py',PROJECT/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py',*(PROJECT/'docs/GROUP_QUERY_MODE_SEQUENCE_DESIGN_20261101').glob('*.md')]:add(p,sha(p))
    stage('Q_PREPARE_INHERITED_SOURCE_SHA',count=len(files));verify(files,g.check)
    write(HERE/'INPUT_SOURCE_LOCK.json',dict(status='PASS_ALL_ORIGINAL_DIGESTS',count=len(files),files=files,no_digest_replacement=True,seconds=time.perf_counter()-t))
    d=old.load_data();fit=d['FIT'];inn=d['inner'];s=load_npz(OLD/'statistics.npz')
    mean=fit['x'].astype(float).mean(0);std=fit['x'].astype(float).std(0);std[std<1e-6]=1
    assert np.array_equal(mean.astype(np.float32),s['mean']) and np.array_equal(std.astype(np.float32),s['std'])
    assert np.array_equal(fit['ids'],s['fit_ids'])
    # Independent Welford-style chunk population moments, no held fitting.
    nm=0;mu=np.zeros(210);M=np.zeros(210)
    for i in range(0,len(fit['x']),31):
        v=fit['x'][i:i+31].astype(float);n=len(v);a=v.mean(0);b=((v-a)**2).sum(0);delta=a-mu;new=nm+n
        M+=b+delta*delta*nm*n/new;mu+=delta*n/new;nm=new
    sd=np.sqrt(M/nm);sd[sd<1e-6]=1
    error=max(float(abs(mu-mean).max()),float(abs(sd-std).max()));assert error<=1e-11
    npz(HERE/'statistics.npz',**s);npz(HERE/'split_ids.npz',FIT=fit['ids'],inner=inn['ids'])
    oldsplit=load_npz(OLD/'split_ids.npz');assert equal(oldsplit,dict(FIT=fit['ids'],inner=inn['ids']))
    records={}
    for name,v in d.items():
        records[name]=dict(N=len(v['ids']),groups={str(k):int((v['groups']==k).sum()) for k in range(3)},tracks=len(np.unique(v['track'])),sha={k:digest(v[k]) for k in ('ids','x','past','native','gt','track','groups')})
    assert not np.intersect1d(fit['ids'],inn['ids']).size and not np.intersect1d(fit['track'],inn['track']).size
    diag=np.array(sorted(range(len(fit['ids'])),key=lambda i:hashlib.sha256(f'group-query-diagnostic-v1|{int(fit["ids"][i])}'.encode()).digest())[:128])
    npz(HERE/'diagnostic_ids.npz',indices=diag,sample_id=fit['ids'][diag])
    for seedvalue in SEEDS:
        q1=new_model(seedvalue,1,'cpu');rng=torch.get_rng_state().clone();q3=new_model(seedvalue,3,'cpu');assert torch.equal(rng,torch.get_rng_state())
        for k,v in q1.state_dict().items():assert torch.equal(v,q3.state_dict()[k]),k
        assert not torch.equal(q3.queries,q3.aux[0]) and not torch.equal(q3.aux[0],q3.aux[1])
        for arm,m in [('Q1',q1),('Q3',q3)]:save_torch(HERE/f'initial/{arm}_{seedvalue}.pt',state(m))
        o=load_npz(OLD/f'batch_orders/seed{seedvalue}.npz');assert o['orders'].shape==(8,18909)
        assert all(np.array_equal(np.sort(x),np.arange(18909)) for x in o['orders']);npz(HERE/f'orders/{seedvalue}.npz',**o)
    write(HERE/'IDENTITY_STATISTICS_AUDIT.json',dict(status='PASS',populations=records,FIT_inner_ID_track_disjoint=True,features=210,past_unit='original focal heading meters lastobserved0',dt=.4,
        old_statistics_float32_exact=True,independent_chunk31_error=error,paired_all_initial_state_exact=True,independent_aux_no_global_RNG_consumption=True,old_batch_orders_exact=True,
        original_F16_exposure=read(OLD/'source_audit/RESULT.json')['exposure'],no_new_old_F16_forward=True,no_outer_CAL_evaluation=True))
    write(HERE/'PREPARE_RESOURCE.json',dict(policy='warning_first_remaining_v3',observations=g.observations));print('PREPARE PASS',flush=True)
if __name__=='__main__':main()
