"""Read-only NumPy verification from saved errors and derived observed speeds.

No raw-coordinate error or junction-acceleration recomputation is claimed here.
That original local audit is recorded in LOCAL_RECOMPUTE_RECEIPT.json.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SNAP=ROOT/'snapshot'
sys.path.insert(0,str(ROOT/'dependencies/workspace/qoder_work/FIVEFOLD_B0_EXEC_20261115/code'))
from fivefold_metrics import aggregate


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def equal(a,b,label):
    if isinstance(b,dict):
        assert a.keys()==b.keys(),(label,'keys')
        for k,v in b.items():equal(a[k],v,label+'/'+k)
    elif isinstance(b,(int,float)) and not isinstance(b,bool):assert np.isclose(a,b,rtol=1e-12,atol=1e-12),(label,a,b)
    else:assert a==b,(label,a,b)


def main():
    manifest=read(ROOT/'ARTIFACT_MANIFEST.json')['files']
    for rel,digest in manifest.items():
        p=ROOT/rel;assert p.resolve().is_relative_to(ROOT.resolve());assert sha(p)==digest,rel
    copies=read(ROOT/'COPY_PROVENANCE.json')['copies'];mapping={row['source']:row['archive_path'] for row in copies}
    for row in copies:assert sha(ROOT/row['archive_path'])==row['sha256']
    original_sources=read(SNAP/'SOURCE_MANIFEST.json')['files']
    for original,digest in original_sources.items():
        if original in mapping:assert sha(ROOT/mapping[original])==digest
    seal=read(SNAP/'FINAL_OUTPUT_SEAL.json');included=0
    for original,digest in seal['files'].items():
        if original in mapping:assert sha(ROOT/mapping[original])==digest;included+=1
        else:assert original.endswith(('.pt','_predictions.npz')),original
    assert seal['source_manifest_sha256']==sha(SNAP/'SOURCE_MANIFEST.json')
    status=read(ROOT/'STATUS.json');final=read(SNAP/'standalone/RESULT.json')
    assert status['execution_status']=='COMPLETE' and status['original_verdict']=='no_help'
    assert final['status']=='SUCCESS' and final['verdict']=='no_help'
    assert read(SNAP/'standalone/WORKER_EXIT.json')['exit_code']==0
    assert status['experiment_two_decision']=='KEEP_SHELVED' and status['experiment_two_executed'] is False
    assert not status['extra_seed_executed'] and not status['baseline_upgraded']
    primary=read(SNAP/'ETH_T1_RESULT.json');analysis=read(SNAP/'ANALYSIS_20261116.json')
    assert sha(SNAP/'ETH_T1_RESULT.json')==analysis['original_result_sha256']
    arrays={};metrics={}
    for arm in ('B0','T1'):
        folder=SNAP/'runs'/f'{arm}_eth_E64_s42'
        with np.load(folder/'test_window_metrics.npz',allow_pickle=False) as z:w={k:z[k] for k in z.files}
        assert w['ids'].shape==w['group'].shape==(364,) and len(np.unique(w['ids']))==364
        for k,v in w.items():
            if k not in ('ids','group'):assert v.shape==(364,) and np.isfinite(v).all(),(arm,k)
        expected=read(folder/'result.json')['test']
        actual=aggregate({k:v for k,v in w.items() if k not in ('ids','group')},w['group'])
        expected_without_junction={g:{k:v for k,v in block.items() if k!='junction_accel_allK20'} for g,block in expected.items()}
        equal(actual,expected_without_junction,arm+'/aggregate')
        equal(expected,analysis['metrics'][arm],arm+'/analysis')
        equal(expected['overall'],primary[arm],arm+'/primary')
        arrays[arm]=w;metrics[arm]=actual
    b=arrays['B0'];t=arrays['T1']
    for k in ('ids','group'):assert np.array_equal(b[k],t[k])
    gains={k:(metrics['B0']['overall'][k]-metrics['T1']['overall'][k])/metrics['B0']['overall'][k] for k in ('minADE','minFDE')}
    verdict='helps' if all(g>=.02 for g in gains.values()) else 'no_help' if all(g<.005 for g in gains.values()) else 'unclear'
    assert verdict==primary['verdict']=='no_help'
    equal(gains,analysis['primary_relative_improvement'],'primary/gains')
    equal(gains['minADE'],status['relative_ADE_improvement'],'status/ADE');equal(gains['minFDE'],status['relative_FDE_improvement'],'status/FDE')
    for name,mask in [('overall',np.ones(364,bool)),('stable',b['group']==0),('transition',b['group']==1),('change',b['group']==2)]:
        lost=int((b['coverage'][mask]&~t['coverage'][mask]).sum());rescued=int((~b['coverage'][mask]&t['coverage'][mask]).sum())
        expected=primary['groups'][name]
        assert expected['n']==int(mask.sum()) and expected['lost']==lost and expected['rescued']==rescued
        effects=dict(n=int(mask.sum()),rescued=rescued,lost=lost,net=rescued-lost,
            ADE_relative_worsening_pct=100*(metrics['T1'][name]['minADE']/metrics['B0'][name]['minADE']-1),
            FDE_relative_worsening_pct=100*(metrics['T1'][name]['minFDE']/metrics['B0'][name]['minFDE']-1),
            ADE_delta_contribution_to_overall=float((t['minADE'][mask]-b['minADE'][mask]).sum()/364),
            FDE_delta_contribution_to_overall=float((t['minFDE'][mask]-b['minFDE'][mask]).sum()/364))
        equal(effects,analysis['groups'][name],name+'/effects')
        equal(100*metrics['B0'][name]['coverage'],expected['B0_cov'],name+'/B0_coverage')
        equal(100*metrics['T1'][name]['coverage'],expected['T1_cov'],name+'/T1_coverage')
    partitions={}
    for name,mask in [('both_good',b['coverage']&t['coverage']),('rescued',~b['coverage']&t['coverage']),('lost',b['coverage']&~t['coverage']),('both_bad',~b['coverage']&~t['coverage'])]:
        partitions[name]=dict(n=int(mask.sum()),ADE_delta_sum=float((t['minADE'][mask]-b['minADE'][mask]).sum()),FDE_delta_sum=float((t['minFDE'][mask]-b['minFDE'][mask]).sum()))
    equal(partitions,analysis['coverage_partition'],'partition')
    quantiles={arm:{k:{str(q):float(np.quantile(arrays[arm][k],q)) for q in (.5,.9,.95,.99)} for k in ('minADE','minFDE')} for arm in ('B0','T1')}
    equal(quantiles,analysis['error_quantiles'],'quantiles')
    speedfile=SNAP/'analysis_inputs/observed_speed_diagnostics.npz';assert sha(speedfile)==analysis['derived_speed_npz_sha256']
    with np.load(speedfile,allow_pickle=False) as z:s={k:z[k] for k in z.files}
    assert np.array_equal(s['ids'],b['ids']) and np.array_equal(s['group'],b['group'])
    assert s['test_speed_mps'].shape==(364,) and s['train_speed_mps'].shape==(28791,)
    assert np.isfinite(s['test_speed_mps']).all() and np.isfinite(s['train_speed_mps']).all()
    edges=np.quantile(s['test_speed_mps'],[0,1/3,2/3,1]);assert np.array_equal(edges,s['test_speed_bin_edges'])
    bins=np.searchsorted(edges[1:-1],s['test_speed_mps'],side='right');assert np.array_equal(bins,s['test_speed_bin'])
    rows=[]
    for i in range(3):
        mask=bins==i
        rows.append(dict(bin=i,lo_mps=float(edges[i]),hi_mps=float(edges[i+1]),n=int(mask.sum()),B0_ADE=float(b['minADE'][mask].mean()),T1_ADE=float(t['minADE'][mask].mean()),B0_FDE=float(b['minFDE'][mask].mean()),T1_FDE=float(t['minFDE'][mask].mean()),B0_coverage=float(b['coverage'][mask].mean()),T1_coverage=float(t['coverage'][mask].mean())))
    assert len(rows)==3 and sum(x['n'] for x in rows)==364
    for actual,expected in zip(rows,analysis['speed_rows']):equal(actual,expected,'speed/bin')
    for key in ('train_speed','test_speed'):
        values=s[key+'_mps'];summary=dict(n=len(values),median=float(np.median(values)),p90=float(np.quantile(values,.9)))
        equal(summary,primary['speed'][key],key)
    train=read(SNAP/'runs/T1_eth_E64_s42/TRAIN_DONE.json');old=read(ROOT/'dependencies/workspace/qoder_work/FIVEFOLD_B0_EXEC_20261115/FIVEFOLD_RUNS/runs/budget_eth_E64_s42/metrics.json')
    assert train['train']['updates']==28800 and train['train']['parameters']==295474
    assert train['train']['order_hash']==old['train']['order_hash']
    assert train['contract']['train_index_sha']==old['contract']['train_index_sha']
    assert analysis['experiment_two_decision']['run_now'] is False
    print(json.dumps(dict(status='PASS',execution='COMPLETE',scientific_verdict=verdict,archive_hashes_verified=len(manifest),included_final_output_hashes_verified=included,
        metric_populations_reconstructed=2,test_windows=364,new_training_updates=28800,relative_ADE_improvement=gains['minADE'],relative_FDE_improvement=gains['minFDE'],
        coverage_rescued=31,coverage_lost=10,experiment_two='KEEP_SHELVED',
        limitation='Reconstructs from saved per-window errors and derived speed arrays. Original coordinate errors and junction geometry require omitted local targets/predictions.'),indent=2))


if __name__=='__main__':main()
