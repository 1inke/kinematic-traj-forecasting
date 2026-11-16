"""Read-only CPU audit of the completed finite calibration; no new training/test."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from explore_common import read, sha, verify_source, validate_result, EXEC, FIVE
sys.path.insert(0, str(FIVE/'code'))
from fivefold_metrics import per_window, aggregate


def check(a,b,label):
    if isinstance(b,dict):
        assert set(a)==set(b),(label,'keys')
        for k,v in b.items():check(a[k],v,f'{label}/{k}')
    elif isinstance(b,(float,int)) and not isinstance(b,bool):
        assert np.isclose(a,b,rtol=1e-12,atol=1e-12),(label,a,b)
    else: assert a==b,(label,a,b)


def motion(paths,past):
    points=np.concatenate((np.broadcast_to(past[:,None,-1:,:],(len(past),20,1,2)),paths.astype(np.float64)),axis=2)
    velocity=np.diff(points,axis=2)/.4
    acceleration=np.diff(velocity,axis=2)/.4
    jerk=np.diff(acceleration,axis=2)/.4
    junction=np.linalg.norm(velocity[:,:,0]-((past[:,-1]-past[:,-2])/.4)[:,None],axis=-1)/.4
    def stats(v):return dict(mean=float(v.mean()),p95=float(np.percentile(v,95)))
    return dict(speed=stats(np.linalg.norm(velocity,axis=-1)),acceleration=stats(np.linalg.norm(acceleration,axis=-1)),
                jerk=stats(np.linalg.norm(jerk,axis=-1)),junction_acceleration=stats(junction))


def main():
    verify_source()
    result=read(ROOT/'standalone/RESULT.json')
    assert result==read(ROOT/'EXPLORATION_BUDGET.json')
    assert result['status']=='SUCCESS'
    assert read(ROOT/'standalone/WORKER_EXIT.json')['exit_code']==0
    assert read(ROOT/'standalone/CURRENT.json')['status']=='SUCCESS'
    assert not (ROOT/'standalone/FAILURE.json').exists()
    assert result['chosen_initial_exploration_budget'] is None
    with np.load(EXEC/'C0/dev.npz',allow_pickle=False) as z:dev={k:z[k] for k in z.files}
    assert len(dev['ids'])==4753 and np.sum(dev['group']==2)==960
    results={};windows={};rows=[];hashes=0
    for e in (16,32):
        for arm,seed in [('B0',42),('B0',43),('T1',42),('E3_self',42)]:
            name=f'{arm}_E{e}_s{seed}';d=ROOT/'runs'/name;j=validate_result(d)
            hashes+=len(j['files']);c=j['contract']
            assert c['epochs']==e and c['seed']==seed and c['arm']==arm
            assert c['n_train']==21719 and c['train_set']=='C0_full_fit'
            assert c['fit_sha256']==sha(EXEC/'C0/fit.npz') and c['dev_sha256']==sha(EXEC/'C0/dev.npz')
            assert j['status']=='DONE' and j['train']['updates']==e*340
            assert j['train']['parameters']==(302477 if arm=='E3_self' else 295474)
            with np.load(d/'dev_predictions.npz',allow_pickle=False) as z:p={k:z[k] for k in z.files}
            with np.load(d/'dev_window_metrics.npz',allow_pickle=False) as z:w={k:z[k] for k in z.files}
            assert p['paths'].shape==(4753,20,12,2) and p['paths'].dtype==np.float32
            assert p['logits'].shape==(4753,20) and p['logits'].dtype==np.float32
            assert np.isfinite(p['paths']).all() and np.isfinite(p['logits']).all()
            for k in ('ids','group'):
                assert np.array_equal(p[k],dev[k]) and np.array_equal(w[k],dev[k]),(name,k)
            computed=per_window(p['paths'],p['logits'],dev['future'],dev['past'][:,0])
            # Direct independent calculation of the two headline oracle errors.
            distance=np.sqrt(np.sum((p['paths'].astype(np.float64)-dev['future'][:,None].astype(np.float64))**2,axis=-1))
            assert np.allclose(computed['minADE'],distance.mean(2).min(1),rtol=1e-12,atol=1e-12)
            assert np.allclose(computed['minFDE'],distance[:,:,-1].min(1),rtol=1e-12,atol=1e-12)
            for k,a in computed.items():
                assert np.allclose(a,w[k],rtol=1e-12,atol=1e-12),(name,k)
            agg=aggregate(computed,dev['group'])
            agg['motion_allK20']=motion(p['paths'],dev['past'][:,0].astype(np.float64))
            check(agg,j['dev'],name)
            windows[name]=w;results[name]=j
            rows.append(dict(run=name,arm=arm,epochs=e,seed=seed,metrics=agg,wall_seconds=j['wall_seconds']))
    comparisons=[];calibrations={}
    ref64={'T1':{'overall':1.6621081422259625,'change':3.8541666666666665},
           'E3_self':{'overall':2.608878602987587,'change':4.583333333333333}}
    for e in (16,32):
        bname=f'B0_E{e}_s42';base=results[bname]['dev'];bw=windows[bname]
        second=results[f'B0_E{e}_s43']['dev']
        calibration=read(ROOT/f'CALIBRATION_E{e}.json')
        noise={g:100*(base[g]['coverage']-second[g]['coverage']) for g in ('overall','change')}
        check(noise,calibration['B0_pair_difference_pp'],f'E{e}/noise')
        for arm in ('T1','E3_self'):
            name=f'{arm}_E{e}_s42';m=results[name]['dev'];mw=windows[name]
            paired={};metrics={}
            for group,mask in [('overall',np.ones(4753,bool)),('stable',dev['group']==0),('transition',dev['group']==1),('change',dev['group']==2)]:
                old=bw['coverage'][mask];new=mw['coverage'][mask]
                lost=int((old&~new).sum());rescued=int((~old&new).sum())
                paired[group]=dict(n=int(mask.sum()),reference_good=int(old.sum()),lost=lost,rescued=rescued,
                    lost_fraction_of_reference_good=lost/int(old.sum()),coverage_diff_pp=100*float(new.mean()-old.mean()))
                metrics[group]=dict(coverage_diff_pp=100*(m[group]['coverage']-base[group]['coverage']),
                    minADE_relative_pct=100*(m[group]['minADE']/base[group]['minADE']-1),
                    minFDE_relative_pct=100*(m[group]['minFDE']/base[group]['minFDE']-1),
                    top1_coverage_diff_pp=100*(m[group]['top1_good']-base[group]['top1_good']))
            check(paired,results[name]['paired'],name+'/paired')
            differences={g:metrics[g]['coverage_diff_pp'] for g in ('overall','change')}
            gate=dict(diff_pp=differences,half_reference_reproduced=all(differences[g]>=.5*ref64[arm][g] for g in differences),
                exceeds_single_B0_pair_difference=all(differences[g]>abs(noise[g]) for g in differences),known_full_screen_success=False)
            check(gate,calibration['controls'][arm],name+'/calibration')
            motion_deltas={kind:{stat:100*(m['motion_allK20'][kind][stat]/base['motion_allK20'][kind][stat]-1)
                          for stat in ('mean','p95')} for kind in ('speed','acceleration','jerk','junction_acceleration')}
            comparisons.append(dict(epochs=e,arm=arm,relative_to=bname,metrics=metrics,paired=paired,motion_relative_pct=motion_deltas))
        accepted=all(x['half_reference_reproduced'] and x['exceeds_single_B0_pair_difference'] for x in calibration['controls'].values())
        assert accepted==calibration['usable_as_initial_screen'] is False
        calibrations[str(e)]=calibration
    assert len(results)==8
    assert sum(j['train']['updates'] for j in results.values())==65280
    for e in (16,32):
        assert len({results[f'{arm}_E{e}_s42']['train']['order_hash'] for arm in ('B0','T1','E3_self')})==1
    report=dict(at=datetime.now(timezone(timedelta(hours=8))).isoformat(),verification='PASS',source_files_verified=350,
        result_file_hashes_verified=hashes,prediction_populations_recomputed=8,rows=rows,comparisons=comparisons,
        calibrations=calibrations,run_result_sha256={k:sha(ROOT/'runs'/k/'result.json') for k in results},
        analysis_script_sha256=sha(__file__),session_seconds=result['session_wall_seconds'],
        chosen_initial_exploration_budget=result['chosen_initial_exploration_budget'],
        scope='C0 reused development data; descriptive calibration only; methods have seed42 only; no new training, model test inference, bootstrap or source edits.')
    (ROOT/'ANALYSIS_20261116.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(verification='PASS',source_files=350,result_hashes=hashes,populations=8,updates=65280,
        calibration32=calibrations['32'],comparisons32=[x for x in comparisons if x['epochs']==32]),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
