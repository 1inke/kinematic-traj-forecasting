"""Additional CPU acceptance; does not modify any of the420 locked sources."""
from core import *
import inspect

def main():
    digest=verify_sources();d=load_data();x=d['FIT']['x'].astype(float);z=np.load(HERE/'statistics.npz');means=[];stds=[]
    for chunk in [31,256]:
        sm=np.zeros(210);ss=np.zeros(210)
        for i in range(0,len(x),chunk):sm+=x[i:i+chunk].sum(0);ss+=(x[i:i+chunk]**2).sum(0)
        mu=sm/len(x);sd=np.sqrt(np.maximum(ss/len(x)-mu**2,0));sd[sd<1e-6]=1;means.append(mu);stds.append(sd)
    err=max(float(np.max(np.abs(a-z[b]))) for seq,b in [(means,'mean_float64'),(stds,'std_float64')] for a in seq)
    assert err<=1e-11,err
    assert np.array_equal(z['fit_ids'],d['FIT']['ids']) and np.array_equal(z['mean'],z['mean_float64'].astype(np.float32)) and np.array_equal(z['std'],z['std_float64'].astype(np.float32))
    seeds=[]
    for s in SEEDS:
        with np.load(HERE/f'batch_orders/seed{s}.npz') as oo:
            order=oo['orders'];assert order.shape==(8,18909) and all(np.array_equal(np.sort(v),np.arange(18909)) for v in order)
        m=new_model(s,'cpu');initial=torch.load(HERE/f'initial_states/seed{s}.pt',weights_only=True);assert equal(initial,m.state_dict())
        assert torch.count_nonzero(m(torch.from_numpy(d['FIT']['x'][:8])))==0
        for arm in ['A','B','E_bal']:
            n=new_model(s,'cpu');n.load_state_dict(initial);assert equal(n.state_dict(),initial)
        seeds.append(s)
    assert list(inspect.signature(ContextK20.forward).parameters)==['self','features']
    # Native zero/20copy degeneracy fixture retains physical literal comparisons and one complete-link cluster.
    from f16.eval import clusters
    from scipy.cluster.hierarchy import linkage,fcluster
    p=np.zeros((2,20,12,2));gg=np.ones((2,20),bool);assert np.array_equal(clusters(p,gg),np.ones((2,2),np.int32))
    assert digest==verify_sources()
    ext=dict(base_source_digest=digest,extension_files={str(Path(__file__).relative_to(ROOT)):sha256(__file__)},base_manifest_not_overwritten=True,reason='additional CPU FIT-only stats/initial/order/forward-interface fixtures after full GPU acceptance; no production source modification')
    write_json(HERE/'source_extension_prelaunch.json',ext)
    write_json(HERE/'PRELAUNCH_SUPPLEMENT.json',dict(status='PASS',FIT_statistics_chunk31_256_maxerror=err,tolerance=1e-11,initial_all_params_buffers_three_arms_three_seeds_exact=seeds,all8orders_exact_permutations=True,forward_only_features=True,twenty_copies_one_cluster=True,base_source_digest=digest,extension=ext,actual_GPU_preflight_not_replaced=True,old_prefailure_warning='PREFLIGHT_FAILURE.json is preserved first UTF8 failure, not latest preflight or production status'))
    print('PRELAUNCH SUPPLEMENT PASS',err,flush=True)

if __name__=='__main__':main()
