"""Read-only source/data checks. No project imports, torch, models or training.

Outputs only into this new review directory; does NOT replay saved predictions.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
DATA = Path('<LOCAL_DESKTOP>/DRSE/ASTRA_ACCURACY_Q20/datasets/eth_ucy')
SOURCES = {}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def register(path):
    path = Path(path)
    SOURCES[str(path)] = sha(path)
    return path

def read_json(path):
    return json.loads(register(path).read_text(encoding='utf-8-sig'))

def row_keys(data):
    # Scene family, source time, and float32 focal 8+12 XY. Not a track ID proof,
    # full neighbor-tensor match, or a whole-model checkpoint provenance audit.
    return [hashlib.sha256(np.ascontiguousarray(row[0, :20][:, [0,1,4,5]], dtype=np.float32).tobytes()).hexdigest()
            for row in data]

def main():
    for relative in (
        'RETRAIN_SCHEMES_20261019/run_all_retrain.py',
        'RETRAIN_SCHEMES_20261019/scheme_encoder_adapter.py',
        'INTERACTION_RESIDUAL_SUPERVISION_SERIES/current_baseline_adapter.py',
        'FRONT_ENCODER_SERIES/train_front_variant.py',
        'MODE_TEMPLATE_GROUP_SERIES/grouped_mode_template_models.py',
        'MULTI_DATASET_GRAPH_NEIGHBOR_BENCHMARK_20261019/eval_scheme2_neighbor.py',
        'MULTI_DATASET_GRAPH_NEIGHBOR_BENCHMARK_20261019/audit_route_b_rescue_lost.py',
        'MULTI_DATASET_GRAPH_NEIGHBOR_BENCHMARK_20261019/SCHEME3_COADAPTATION_RESULTS.json',
        'CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018/SAMPLE_REVIEW_DIGEST_20261019.md',
    ):
        register(ROOT/relative)
    register(DATA.parent.parent/'models/stage1_graph.py')
    register(DATA.parent.parent/'data/eth_preprocessor.py')
    ret = read_json(ROOT/'RETRAIN_SCHEMES_20261019/RETRAINED_SCHEMES_BENCHMARK_RESULTS.json')
    frozen = read_json(ROOT/'MULTI_DATASET_GRAPH_NEIGHBOR_BENCHMARK_20261019/ROUTE_B_AUDIT_RESULTS.json')
    results = {'scope': 'aggregate arithmetic, split identity and scene/time/focal path overlap, selected-case longer-label availability; NO prediction/model replay', 'aggregate_checks': {}, 'split_checks': {}, 'focal_exact_overlap': {}, 'horizon_selected_cases': []}
    n = sum(d['samples'] for d in ret['datasets'].values())
    for key in ('scheme1', 'scheme3'):
        for stage, data, total, coverage_key in (
            ('retrained',ret['datasets'],ret['total'][key],'new_coverage_pct'),
            ('frozen',frozen['datasets'],frozen['total_summary'][key],'scheme_coverage_pct')):
            base = sum(round(d[key]['base_coverage_pct'] * d['samples']/100) for d in data.values())
            new = sum(round(d[key][coverage_key] * d['samples']/100) for d in data.values())
            rescued = sum(d[key]['rescued'] for d in data.values())
            lost = sum(d[key]['lost'] for d in data.values())
            assert new-base == rescued-lost == total['net_rescue']
            assert abs(100*new/n-total[coverage_key]) < 1e-10
            results['aggregate_checks'][stage+'_'+key] = dict(n=n,base_good=base,new_good=new,rescue=rescued,lost=lost,delta_pp=100*(new-base)/n,lost_over_base_good_pct=100*lost/base)
    suffix = 'f16_temporal_social_cross/seed_1/val_scene_-1'
    baseline = ROOT/'INTERACTION_RESIDUAL_SUPERVISION_SERIES/results/general/baseline'/suffix
    bm = read_json(baseline/'metrics.json')
    with np.load(register(baseline/'split_indices.npz')) as z:
        base_indices = {k:z[k].copy() for k in z.files}
    for name in ('scheme1_conflict_gated','scheme3_coadaptation'):
        folder = ROOT/'RETRAIN_SCHEMES_20261019/results/general'/name/suffix
        m = read_json(folder/'metrics.json')
        with np.load(register(folder/'split_indices.npz')) as z:
            same = {k: bool(np.array_equal(z[k],base_indices[k])) for k in z.files}
        c1,c2 = dict(bm['locked_backend']),dict(m['locked_backend'])
        diffs = {k:[c1.get(k),c2.get(k)] for k in sorted(set(c1)|set(c2)) if c1.get(k)!=c2.get(k)}
        hist = list(csv.DictReader(register(folder/'history.csv').open(encoding='utf-8-sig')))
        results['split_checks'][name] = dict(indices_exact=same,contract_differences=diffs,epochs=len(hist),seconds=m['elapsed_seconds'],val_score_by_epoch={r['epoch']:float(r['val_checkpoint_score']) for r in hist if r['val_checkpoint_score']})
    train_path = register(DATA/'processed_data_diverse/eth_data_train.npy')
    train = np.load(train_path, mmap_mode='r')
    train_keys = row_keys(train)
    keys_full = set(train_keys)
    keys_fit = {train_keys[int(i)] for i in base_indices['train']}
    keys_val = {train_keys[int(i)] for i in base_indices['validation']}
    results['training_raw_shape'] = list(train.shape)
    for ds in ret['datasets']:
        path = register(DATA/f'processed_data_diverse/{ds}_data_test.npy')
        d = np.load(path, mmap_mode='r')
        keys = row_keys(d)
        results['focal_exact_overlap'][ds] = dict(n=len(keys),in_training_pool=sum(k in keys_full for k in keys),in_actual_fit=sum(k in keys_fit for k in keys),in_validation=sum(k in keys_val for k in keys))
        del d
    idea = read_json(ROOT/'CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018/ENDPOINT_HORIZON_IDEA_SIMILAR_SAMPLES_24102.json')
    cases = [idea['reference_24102']]+idea['samples']
    label_cache={}
    for case in cases:
        seq=case['sequence']
        if seq not in label_cache:
            files=list((DATA/'labels').glob('*/'+seq+'.txt'))
            if len(files)!=1:
                results['horizon_selected_cases'].append({'id':case['sample_id'],'error':'ambiguous full label source','paths':[str(p) for p in files]})
                continue
            rows=np.loadtxt(register(files[0]),usecols=(0,1,13,15))
            label_cache[seq]={(int(r[0]),int(r[1])):r[2:] for r in rows}
        table=label_cache[seq]
        ped,start=int(case['pedestrian_id']),int(case['start_frame'])
        old_keys=[(f,ped) for f in range(start,start+20)]
        verified=all(k in table for k in old_keys)
        err=float(np.max(np.abs(np.array([table[k] for k in old_keys])-train[case['sample_id'],0,:20,-2:]))) if verified else None
        item=dict(id=case['sample_id'],sequence=seq,ped=ped,start=start,old20_label_match_max_abs=err)
        consecutive=0
        for f in range(start+8,start+8+60):
            if (f,ped) not in table:
                break
            consecutive+=1
        item['consecutive_future_available_capped60']=consecutive
        for horizon in (20,24):
            wanted=[(f,ped) for f in range(start,start+8+horizon)]
            item[f'full8plus{horizon}_present']=all(k in table for k in wanted)
        results['horizon_selected_cases'].append(item)
    results['source_sha_before'] = dict(SOURCES)
    results['source_sha_after_unchanged'] = all(sha(Path(p))==h for p,h in SOURCES.items())
    assert results['source_sha_after_unchanged']
    (OUT/'EVIDENCE_CHECK.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in results.items() if k!='source_sha_before'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
