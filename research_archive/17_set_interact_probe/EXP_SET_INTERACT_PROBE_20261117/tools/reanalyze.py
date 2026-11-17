"""NumPy-only reanalysis of saved metrics; no coordinates, models or training.

Group/decision functions are copied from the archived independent local auditor.
This tool cannot regenerate the per-window geometric inputs or replay weights.
"""
from pathlib import Path
import copy, hashlib, json
import numpy as np
ROOT = Path(__file__).resolve().parents[1]

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def equal(a,b,path=''):
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        for k in a:equal(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),path
        for j,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(j))
    elif isinstance(a,(int,float,np.number)) and not isinstance(a,(bool,np.bool_)):
        np.testing.assert_allclose(a,b,atol=3e-7 if path.endswith('/mean_shift') else 3e-11,rtol=3e-11,err_msg=path)
    else:assert a==b,(path,a,b)

def independent_groups(w,meta):
    masks=dict(all=np.ones(len(meta['group']),bool),stable=meta['group']==0,transition=meta['group']==1,change=meta['group']==2,decel_stop=np.isin(meta['labels'],['deceleration','stop']))
    output={}
    for name,mask in masks.items():
        n=int(sum(mask));b=dict(n=n,small_group=n<20)
        if n:
            for k in ('minADE','minFDE','top1FDE','uncovered','allocation_gate','redundant_pairs','effective_candidates','mean_shift'):b[k]=float(np.sum(w[k][mask])/n)
            b['minFDE_p90']=float(np.percentile(w['minFDE'][mask],90));v=mask&w['direction_valid'];b['direction_valid_n']=int(sum(v))
            for k in ('longitudinal_outside','longitudinal_gt_below_all','longitudinal_gt_above_all','longitudinal_inside_gap'):b[k]=float(np.count_nonzero(w[k][v])/sum(v)) if v.any() else None
            counts=[int(sum(w['winner'][mask]==j)) for j in range(20)];b['winner_counts']=counts;b['winner_frequency']=[v/n for v in counts];b['rarely_winning_slots_below_0p1pct']=sum(v/n<.001 for v in counts)
        output[name]=b
    return output

def independent_decision(baseline,runs):
    def gains(arm,ref,metric='minFDE',group='all'):
        values=[]
        for seed in ('0','1','2'):
            a=runs[arm][seed]['groups'][group][metric];b=baseline['groups'][group][metric] if ref=='A0' else runs[ref][seed]['groups'][group][metric];values.append((b-a)/b)
        return values
    first=gains('A1','A0');sigma=float(np.std(first,ddof=1));thr=max(.005,2*sigma)
    result=dict(Q1_proxy_gain=float(np.mean(first)),A1_paired_relative_FDE_gains=first,sigma=sigma,threshold=thr,protection_mode='separate_status')
    if result['Q1_proxy_gain']<=0:return dict(result,verdict='PROXY_INVALID',raw_verdict='PROXY_INVALID',protection_status='NOT_EVALUATED',eligible_to_advance=False,stopped_after_A1=True)
    comparisons={}
    for name,a,b in [('Q2_sparse','A3','A1'),('Q2_generic','A2','A1'),('Q3','A5','A4'),('Q4','A5','A6'),('Q5','A4','A1')]:
        f=gains(a,b);e=gains(a,b,'minADE');mf=float(np.mean(f));me=float(np.mean(e));comparisons[name]=dict(arm=a,reference=b,FDE_paired_gains=f,ADE_paired_gains=e,mean_FDE_gain=mf,mean_ADE_gain=me,passes=mf>=thr and me>=-.002)
    q=comparisons;raw='H_SUPPORTED' if q['Q3']['passes'] and q['Q4']['passes'] else 'GENERIC_ONLY' if q['Q3']['passes'] else 'LOSS_ONLY' if q['Q5']['passes'] else 'NO_SET_GAIN'
    protections={}
    for arm in runs:
        checks={}
        for key,metric,group,tol in [('top1','top1FDE','all',.02),('stable','minFDE','stable',.01),('decel_stop','minFDE','decel_stop',.01)]:
            harm=-float(np.mean(gains(arm,'A0',metric,group)));checks[key]=dict(relative_harm=harm,max_harm=tol,passes=harm<=tol)
        protections[arm]=dict(checks=checks,passes=all(b['passes'] for b in checks.values()))
    selected='A5' if raw in ('H_SUPPORTED','GENERIC_ONLY') else 'A4' if raw=='LOSS_ONLY' else None
    protection='PASS' if selected and protections[selected]['passes'] else 'PROTECTION_FAILED' if selected else 'NO_EFFECTIVE_ARM'
    return dict(result,verdict=raw,raw_verdict=raw,comparisons=comparisons,protection_status=protection,protections=protections,selected_arm=selected,eligible_to_advance=raw=='H_SUPPORTED' and protection=='PASS',stopped_after_A1=False)

def self_test():
    baseline = dict(groups={g: dict(minFDE=1., minADE=1., top1FDE=1.) for g in ('all', 'stable', 'decel_stop')})
    def make():
        return {a: {s: dict(groups={g: dict(minFDE=.99, minADE=.99, top1FDE=1.)
               for g in ('all', 'stable', 'decel_stop')}) for s in ('0', '1', '2')}
               for a in ('A1', 'A2', 'A3', 'A4', 'A5', 'A6')}
    def change(runs, arm, value):
        for row in runs[arm].values():
            for group in row['groups'].values():
                group['minFDE'] = value
                group['minADE'] = value
    runs = make()
    assert independent_decision(baseline, runs)['verdict'] == 'NO_SET_GAIN'
    change(runs, 'A4', .98)
    assert independent_decision(baseline, runs)['verdict'] == 'LOSS_ONLY'
    change(runs, 'A5', .96); change(runs, 'A6', .96)
    assert independent_decision(baseline, runs)['verdict'] == 'GENERIC_ONLY'
    change(runs, 'A6', .98)
    d = independent_decision(baseline, runs)
    assert d['verdict'] == 'H_SUPPORTED' and d['eligible_to_advance']
    for row in runs['A5'].values():
        row['groups']['all']['top1FDE'] = 1.03
    d = independent_decision(baseline, runs)
    assert d['raw_verdict'] == 'H_SUPPORTED' and d['protection_status'] == 'PROTECTION_FAILED'
    assert not d['eligible_to_advance']
    change(runs, 'A1', 1.)
    assert independent_decision(baseline, runs)['verdict'] == 'PROXY_INVALID'
    return 6


def main():
    manifest = read(ROOT / 'ARTIFACT_MANIFEST.json')['files']
    for relative, digest in manifest.items():
        assert sha(ROOT / relative) == digest, relative
    test_cases = self_test()
    result = read(ROOT / 'snapshot/RESULT.json')
    d0 = read(ROOT / 'snapshot/D0.json')
    base_saved = read(ROOT / 'snapshot/A0/RESULT.json')
    equal(base_saved, result['baseline'], 'A0/result')
    equal(d0, base_saved, 'D0/result')
    base_w = load(ROOT / 'snapshot/A0/window_metrics.npz')
    assert len(base_w['ids']) == 4753 and len(np.unique(base_w['ids'])) == 4753
    metadata = {k: base_w[k] for k in ('ids', 'group', 'labels')}
    allowed = set(metadata) | set(('minADE', 'minFDE', 'top1FDE', 'winner', 'winner_FDE',
        'uncovered', 'allocation_gate', 'redundant_pairs', 'effective_candidates',
        'longitudinal_outside', 'longitudinal_gt_below_all', 'longitudinal_gt_above_all',
        'longitudinal_inside_gap', 'direction_valid', 'mean_shift'))
    def validate(w):
        assert set(w) == allowed, set(w)
        for name, v in w.items():
            assert v.shape == (4753,), (name, v.shape)
            if v.dtype.kind in 'if':
                assert np.isfinite(v).all(), name
        assert ((w['winner'] >= 0) & (w['winner'] < 20)).all()
        assert ((w['effective_candidates'] >= 1) & (w['effective_candidates'] <= 20)).all()
        np.testing.assert_array_equal(w['uncovered'], w['minFDE'] > .4)
        np.testing.assert_array_equal(w['allocation_gate'], w['winner_FDE'] > .4)
        for key, value in metadata.items():
            np.testing.assert_array_equal(w[key], value)
        np.testing.assert_array_equal(w['direction_valid'], base_w['direction_valid'])
        return independent_groups(w, metadata)
    baseline = dict(base_saved, groups=validate(base_w))
    equal(baseline['groups'], base_saved['groups'], 'A0/groups')
    reconstructed = {}
    paired_orders = {}
    paired_initial_hashes = {}
    total_updates = 0
    for arm, items in result['runs'].items():
        reconstructed[arm] = {}
        for seed, row in items.items():
            folder = ROOT / 'snapshot/runs' / (arm + '_s' + seed)
            saved = read(folder / 'RESULT.json')
            equal(saved, row, arm + '/' + seed + '/result')
            groups = validate(load(folder / 'window_metrics.npz'))
            equal(groups, row['groups'], arm + '/' + seed + '/groups')
            reconstructed[arm][seed] = dict(row, groups=groups)
            training = read(folder / 'TRAINING.json')
            assert training['epochs'] == 20 and training['updates'] == 1700
            assert [x['epoch'] for x in training['history']] == list(range(1, 21))
            assert [x['updates'] for x in training['history']] == [85 * i for i in range(1, 21)]
            assert training['history'][-1]['lr'] == 0.
            orders = [x['order_hash'] for x in training['history']]
            if seed in paired_orders:
                assert orders == paired_orders[seed]
            else:
                paired_orders[seed] = orders
            done = read(folder / 'DONE.json')['files']
            initial = next(v for k, v in done.items() if k.replace('\\', '/').endswith('/initial.pt'))
            if seed in paired_initial_hashes:
                assert initial == paired_initial_hashes[seed]
            else:
                paired_initial_hashes[seed] = initial
            receipts = sorted((folder / 'checkpoints').glob('epoch*.json'))
            assert len(receipts) == 20
            assert [read(p)['epoch'] for p in receipts] == list(range(1, 21))
            total_updates += training['updates']
    decision = independent_decision(baseline, reconstructed)
    equal(decision, result['decision'], 'decision')
    q1 = read(ROOT / 'snapshot/Q1_GATE.json')
    for name in ('Q1_proxy_gain', 'A1_paired_relative_FDE_gains', 'sigma', 'threshold'):
        equal(q1[name], decision[name], 'Q1/' + name)
    additional = {}
    for name, old, new in [('sparse', 'A3', 'A5'), ('generic', 'A2', 'A6')]:
        values = [((reconstructed['A4'][s]['groups']['all']['minFDE'] - reconstructed[new][s]['groups']['all']['minFDE'])
                  - (reconstructed['A1'][s]['groups']['all']['minFDE'] - reconstructed[old][s]['groups']['all']['minFDE']))
                  / baseline['groups']['all']['minFDE'] for s in ('0', '1', '2')]
        additional[name] = dict(paired_additional_interaction_gain_with_new_loss=values,
                                mean=float(np.mean(values)), decision_use=False)
    equal(additional, read(ROOT / 'snapshot/ADDITIONAL_DESCRIPTIVE.json')['interaction_loss_difference_in_differences'], 'descriptive')
    status = read(ROOT / 'STATUS.json')
    assert status['scientific_verdict'] == decision['raw_verdict'] == 'NO_SET_GAIN'
    assert status['protection_status'] == decision['protection_status']
    equal(status['comparisons'], decision['comparisons'], 'STATUS/comparisons')
    assert result['completed_runs'] == 18 and result['skipped_runs'] == 0 and total_updates == 30600
    print(json.dumps(dict(status='PASS', scientific_verdict=decision['raw_verdict'],
          protection_status=decision['protection_status'], archive_files_verified=len(manifest),
          populations=19, window_records=19 * 4753, group_summaries=19 * 5,
          completed_runs=18, complete_epochs=360, optimizer_updates=total_updates,
          group_metrics='ALL_MATCH', fixed_decision_and_guards='ALL_MATCH',
          training_order_pairing='PASS', initial_checkpoint_hash_pairing='PASS_METADATA_ONLY',
          synthetic_decision_cases=test_cases, raw_geometry_recomputed=False,
          model_or_checkpoint_binary_verified=False, new_training=False), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
