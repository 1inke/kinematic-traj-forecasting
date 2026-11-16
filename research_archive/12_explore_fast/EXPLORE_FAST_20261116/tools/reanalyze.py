"""NumPy-only read-only verification of the published short-budget study.

Reconstructs aggregates, paired exchanges, fixed calibration and effect summaries
from saved per-window metrics. It does not recompute coordinate errors or motion
geometry from the omitted predictions/targets; see LOCAL_PREDICTION_RECOMPUTE.json.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SNAP=ROOT/'snapshot'
sys.path.insert(0,str(ROOT/'dependencies/workspace/qoder_work/FIVEFOLD_B0_EXEC_20261115/code'))
from fivefold_metrics import aggregate


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def equal(a,b,label):
    if isinstance(b,dict):
        assert set(a)==set(b),(label,'keys')
        for k,v in b.items():equal(a[k],v,label+'/'+k)
    elif isinstance(b,(float,int)) and not isinstance(b,bool):
        assert np.isclose(a,b,rtol=1e-12,atol=1e-12),(label,a,b)
    else:assert a==b,(label,a,b)


def population(path,expected):
    with np.load(path,allow_pickle=False) as z:w={k:z[k] for k in z.files}
    ids=w['ids'];g=w['group']
    assert ids.ndim==g.ndim==1 and len(np.unique(ids))==len(ids)
    assert len(ids)==expected['overall']['n'] and np.isin(g,[0,1,2]).all()
    metric={k:v for k,v in w.items() if k not in ('ids','group')}
    for k,v in metric.items():assert v.shape==ids.shape and np.isfinite(v).all(),(path,k)
    equal(aggregate(metric,g),expected,str(path.relative_to(ROOT)))
    return w


def main():
    manifest=read(ROOT/'ARTIFACT_MANIFEST.json')['files']
    for rel,digest in manifest.items():
        p=ROOT/rel
        assert p.resolve().is_relative_to(ROOT.resolve())
        assert sha(p)==digest,rel
    copied=read(ROOT/'COPY_PROVENANCE.json')['copies']
    mapped={r['source']:r['archive_path'] for r in copied}
    for row in copied:assert sha(ROOT/row['archive_path'])==row['sha256']
    source=SNAP/'SOURCE_MANIFEST.json'
    for name,digest in read(source)['files'].items():
        if name in mapped:assert sha(ROOT/mapped[name])==digest
    status=read(ROOT/'STATUS.json');result=read(SNAP/'EXPLORATION_BUDGET.json')
    assert status['execution_status']=='COMPLETE' and status['calibration_status']=='NOT_CALIBRATED'
    assert result==read(SNAP/'standalone/RESULT.json') and result['status']=='SUCCESS'
    assert result['chosen_initial_exploration_budget'] is None
    assert read(SNAP/'standalone/WORKER_EXIT.json')['exit_code']==0
    assert read(SNAP/'standalone/CURRENT.json')['status']=='SUCCESS'
    assert status['completed_runs']==status['planned_runs']==8
    assert not status['E16_calibration_passed'] and not status['E32_calibration_passed']
    original_analysis=read(SNAP/'ANALYSIS_20261116.json')
    table={r['run']:r for r in original_analysis['rows']}
    saved={};windows={};verified_outputs=0
    for e in (16,32):
        for arm,seed in [('B0',42),('B0',43),('T1',42),('E3_self',42)]:
            name=f'{arm}_E{e}_s{seed}';d=SNAP/'runs'/name;j=read(d/'result.json')
            c=j['contract']
            assert j['status']=='DONE' and c['epochs']==e and c['arm']==arm and c['seed']==seed
            assert c['source']==sha(source) and c['n_train']==21719 and c['train_set']=='C0_full_fit'
            assert j['train']['updates']==e*340
            expected={k:v for k,v in j['dev'].items() if k!='motion_allK20'}
            windows[name]=population(d/'dev_window_metrics.npz',expected);saved[name]=j
            equal(j['dev'],table[name]['metrics'],name+'/analysis')
            assert sha(d/'result.json')==original_analysis['run_result_sha256'][name]
            for original,digest in j['files'].items():
                if original in mapped:
                    assert sha(ROOT/mapped[original])==digest;verified_outputs+=1
                else:assert original.endswith(('.pt','_predictions.npz')),original
    assert len(list((SNAP/'runs').glob('*/result.json')))==8
    assert sum(r['train']['updates'] for r in saved.values())==status['completed_updates']==65280
    comparisons=[];calibrations=[]
    ref64={'T1':{'overall':1.6621081422259625,'change':3.8541666666666665},
           'E3_self':{'overall':2.608878602987587,'change':4.583333333333333}}
    for e in (16,32):
        bn=f'B0_E{e}_s42';b=saved[bn]['dev'];bw=windows[bn];second=saved[f'B0_E{e}_s43']['dev']
        calibration=read(SNAP/f'CALIBRATION_E{e}.json')
        noise={g:100*(b[g]['coverage']-second[g]['coverage']) for g in ('overall','change')}
        equal(noise,calibration['B0_pair_difference_pp'],f'E{e}/noise')
        assert len({saved[f'{arm}_E{e}_s42']['train']['order_hash'] for arm in ('B0','T1','E3_self')})==1
        for arm in ('T1','E3_self'):
            name=f'{arm}_E{e}_s42';m=saved[name]['dev'];mw=windows[name]
            assert np.array_equal(mw['ids'],bw['ids']) and np.array_equal(mw['group'],bw['group'])
            paired={};effects={}
            for group,mask in [('overall',np.ones(len(bw['ids']),bool)),('stable',bw['group']==0),('transition',bw['group']==1),('change',bw['group']==2)]:
                old=bw['coverage'][mask];new=mw['coverage'][mask]
                paired[group]=dict(n=int(mask.sum()),reference_good=int(old.sum()),lost=int((old&~new).sum()),rescued=int((~old&new).sum()),
                    lost_fraction_of_reference_good=float((old&~new).sum()/old.sum()),coverage_diff_pp=100*float(new.mean()-old.mean()))
                effects[group]=dict(coverage_diff_pp=100*(m[group]['coverage']-b[group]['coverage']),
                    minADE_relative_pct=100*(m[group]['minADE']/b[group]['minADE']-1),
                    minFDE_relative_pct=100*(m[group]['minFDE']/b[group]['minFDE']-1),
                    top1_coverage_diff_pp=100*(m[group]['top1_good']-b[group]['top1_good']))
            equal(paired,saved[name]['paired'],name+'/paired')
            diff={g:effects[g]['coverage_diff_pp'] for g in ('overall','change')}
            gate=dict(diff_pp=diff,half_reference_reproduced=all(diff[g]>=.5*ref64[arm][g] for g in diff),
                exceeds_single_B0_pair_difference=all(diff[g]>abs(noise[g]) for g in diff),known_full_screen_success=False)
            equal(gate,calibration['controls'][arm],name+'/gate')
            archived=next(x for x in original_analysis['comparisons'] if x['epochs']==e and x['arm']==arm)
            equal(effects,archived['metrics'],name+'/effects');equal(paired,archived['paired'],name+'/exchange')
            motion_effects={kind:{stat:100*(m['motion_allK20'][kind][stat]/b['motion_allK20'][kind][stat]-1) for stat in ('mean','p95')}
                            for kind in ('speed','acceleration','jerk','junction_acceleration')}
            equal(motion_effects,archived['motion_relative_pct'],name+'/saved-motion-summary-deltas')
            comparisons.append(dict(epochs=e,arm=arm,coverage_diff_pp=diff))
        accepted=all(x['half_reference_reproduced'] and x['exceeds_single_B0_pair_difference'] for x in calibration['controls'].values())
        assert accepted is False and calibration['usable_as_initial_screen'] is False
        equal(calibration,original_analysis['calibrations'][str(e)],f'E{e}/analysis-calibration')
        equal(calibration,result['decisions'][0 if e==16 else 1],f'E{e}/final-calibration')
        calibrations.append(dict(epochs=e,accepted=accepted))
    reference=read(SNAP/'BUDGET_REFERENCE.json');refrows=[]
    for row in reference['rows']:
        assert row['train_fraction']==.95 and row['seed']==42 and row['epochs']==128
        population(SNAP/'budget_reference'/row['fold']/'test_window_metrics.npz',row['test'])
        refrows.append(row['test']['overall'])
    mean={k:float(np.mean([row[k] for row in refrows])) for k in ('minADE','minFDE')}
    equal(mean,reference['equal_weight'],'reference/equal-mean')
    assert len(refrows)==5 and reference['not_the_unfinished_three_seed_result'] is True
    # Make the near-threshold result explicit without changing the fixed decision.
    e3=next(x for x in original_analysis['comparisons'] if x['epochs']==32 and x['arm']=='E3_self')['paired']['change']
    assert e3['rescued']-e3['lost']==21 and e3['n']==960
    assert np.isclose(.5*ref64['E3_self']['change']/100*960,22)
    print(json.dumps(dict(status='PASS',execution='COMPLETE',calibration='NOT_CALIBRATED',
        archive_hashes_verified=len(manifest),new_run_populations_reconstructed=8,reference_populations_reconstructed=5,
        included_original_output_hashes_verified=verified_outputs,completed_updates=65280,
        chosen_initial_exploration_budget=None,calibrations=calibrations,comparisons=comparisons,
        E3_self_E32_change_net_windows=21,prespecified_half_reference_windows=22,
        limitation='Aggregate reconstruction from saved window metrics; raw-coordinate error and motion-geometry recomputation requires local-only targets/predictions.'),indent=2))


if __name__=='__main__':main()
