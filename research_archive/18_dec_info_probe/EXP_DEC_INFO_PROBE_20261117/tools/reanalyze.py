"""Recompute saved-metric summaries and fixed rules using only NumPy."""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / 'snapshot'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def arrays(p):
    with np.load(p, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def equal(x, y, path=''):
    if isinstance(x,dict):
        assert x.keys()==y.keys(),path
        for k in x:
            equal(x[k],y[k],f'{path}/{k}')
    elif isinstance(x,list):
        assert len(x)==len(y),path
        for i,(a,b) in enumerate(zip(x,y)):
            equal(a,b,f'{path}/{i}')
    elif isinstance(x,float):
        np.testing.assert_allclose(x,y,atol=2e-12,rtol=2e-12,err_msg=path)
    else:
        assert x==y,(path,x,y)


def analysis_rules():
    # Use the exact sealed NumPy rules without importing the encoder or Torch.
    p=S/'code/dec_analysis.py'
    tree=ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    names={'masks','aggregate','paired','effective','proxy','decision'}
    keep=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in keep}==names
    scope={'np':np}
    exec(compile(ast.Module(body=keep,type_ignores=[]),str(p),'exec'),scope)
    return scope


def d0_summary(w, groups):
    out={}
    for name,m in groups.items():
        n=int(np.count_nonzero(m))
        s=m&w['social']
        social=int(np.count_nonzero(s))
        c3,c4=w['c3_count'][m],w['c4_count'][m]
        def hist(v,k):
            return [int(np.count_nonzero(v==i)) for i in range(k)]
        differing=int(np.count_nonzero(w['candidate_sets_vary'][s]))
        out[name]=dict(n=n, social_n=social, social_rate=social/n if n else None,
            neighbor_count_histogram=hist(w['neighbor_count'][m],9),
            candidate_sets_vary_n=differing,
            candidate_sets_vary_among_social=differing/social if social else None,
            c3_neighbors_per_candidate=float(c3.mean()) if n else None,
            c4_neighbors_per_candidate=float(c4.mean()) if n else None,
            c3_neighbor_count_histogram=hist(c3,4),c4_neighbor_count_histogram=hist(np.repeat(c4,20),4),
            same_count_fraction=float((c3==c4[:,None]).mean()) if n else None,
            same_set_fraction=float(w['c3_and_c4_sets_equal'][m].mean()) if n else None)
    return out


def main():
    manifest=read(ROOT/'ARTIFACT_MANIFEST.json')['files']
    for p,expected in manifest.items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==expected,p
    r=read(S/'RESULT.json')
    assert read(S/'D0.json')==r['D0']
    rules=analysis_rules()
    for split,n in [('fit',21719),('dev',4753)]:
        d=arrays(S/f'D0_{split}_windows.npz')
        assert d['c3_count'].shape==(n,20) and d['c4_count'].shape==(n,)
        assert len(d['ids'])==len(np.unique(d['ids']))==n
        assert set(d)=={'ids','group','labels','neighbor_count','social','candidate_sets_vary','c3_count','c4_count','c3_and_c4_sets_equal'}
        equal(d0_summary(d,rules['masks'](d,d['social'])),r['D0'][split],f'D0/{split}')
    meta={k:d[k] for k in ('ids','group','labels')}
    social=d['social']
    s=read(S/'NEIGHBOR_SHUFFLE.json')
    donor=np.array(s['donor_row']); count=d['neighbor_count']
    np.testing.assert_array_equal(np.sort(donor),np.arange(len(donor)))
    np.testing.assert_array_equal(count,count[donor])
    buckets={}
    for k in range(9):
        ix=np.flatnonzero(count==k)
        buckets[str(k)]=dict(n=len(ix),moved=int(np.count_nonzero(ix!=donor[ix])))
    equal(buckets,s['buckets'],'neighbor_shuffle')
    assert all(k=='0' or x['n']==x['moved'] for k,x in buckets.items())
    def report(folder,src):
        w=arrays(folder/src)
        for k in meta:
            np.testing.assert_array_equal(w.pop(k),meta[k])
        np.testing.assert_array_equal(w.pop('social'),social)
        assert set(w)=={'minADE','minFDE','top1FDE','uncovered','mean_shift'}
        assert all(v.shape==(4753,) and np.isfinite(v).all() for v in w.values())
        np.testing.assert_array_equal(w['uncovered'],w['minFDE']>.4)
        return rules['aggregate'](w,meta,social)
    base=report(S/'A0','window_metrics.npz')
    equal(base,r['baseline'],'A0')
    runs={}
    orders={}
    n_pop=1
    updates=0
    for a,seeds in r['runs'].items():
        runs[a]={}
        for seed,row in seeds.items():
            p=S/'runs'/f'{a}_s{seed}'
            train=read(p/'TRAINING.json')
            epoch_receipts=sorted((p/'checkpoints').glob('epoch*.json'))
            assert [read(x)['epoch'] for x in epoch_receipts]==list(range(1,21))
            assert [b['updates'] for b in train['history']]==[85*i for i in range(1,21)]
            assert train['updates']==1700 and train['epochs']==20 and train['history'][-1]['lr']==0
            journal=[x['order_hash'] for x in train['history']]
            assert orders.setdefault(seed,journal)==journal
            updates+=train['updates']
            rebuilt=dict(row,clean=report(p,'window_metrics.npz'))
            n_pop+=1
            if 'shuffled' in row:
                rebuilt['shuffled']=report(p,'window_metrics_shuffled.npz')
                n_pop+=1
            equal(rebuilt,read(p/'RESULT.json'),f'{a}/{seed}')
            runs[a][seed]=rebuilt
    rebuilt=rules['decision'](runs,base,r['frozen_base_parameters'],buckets)
    equal(rebuilt,r['decision'],'decision')
    assert rebuilt['verdict']=='NO_INFO_GAIN' and not rebuilt['eligible_to_advance']
    assert n_pop==28 and updates==30600
    for a,groups in r['shuffle_comparisons'].items():
        for group,expected in groups.items():
            for mode in ('clean','shuffled'):
                paired=rules['paired'](runs,base,a,'C1',group,mode)
                equal(paired,expected[f'{mode}_vs_C1'],f'{a}/{group}/{mode}')
                equal(float(np.mean([v[mode][group]['minFDE'] for v in runs[a].values()])),
                      expected[f'{mode}_minFDE'])
    print(json.dumps(dict(status='PASS',verdict=rebuilt['verdict'],verified_manifest_entries=len(manifest),
          evaluation_populations=n_pop,window_records=n_pop*len(social),evaluation_group_summaries=n_pop*7,
          D0_splits=2,D0_windows=21719+4753,paired_batch_order_receipts='PASS',neighbor_permutation='PASS',
          fixed_decision_and_guards='MATCH',new_training=False,new_inference=False,raw_geometry_recomputed=False)))


if __name__=='__main__':
    main()
