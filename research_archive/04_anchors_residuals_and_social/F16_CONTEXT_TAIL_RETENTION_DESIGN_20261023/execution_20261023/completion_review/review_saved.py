"""Read-only completion provenance and reporting extraction; no model or training."""
from pathlib import Path
import sys, json, hashlib
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import source_contract as c

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def h(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    digest = c.verify()
    before = {}
    result = read(HERE/'D0_D1/RESULT.json')
    assert result['status'] == 'PASS_DIAGNOSTIC_ONLY_TAIL_AND_MOTION'
    assert result['jobs'] == 6 and result['source_digest'] == digest
    assert read(HERE/'worker_state.json')['status'] == 'SUCCESS'
    assert read(HERE/'exit_status.json')['status'] == 'SUCCESS'
    records = []
    for seed in (42,43,44):
        for split in ('FIT','inner'):
            p = HERE/'D0_D1'/f'seed{seed}_{split}'
            done = read(p/'done.json')
            assert done['source_digest'] == digest
            for rel, sha in done['outputs'].items():
                fp = HERE/rel
                assert h(fp) == sha, str(fp)
                before[str(fp)] = sha
            audit = read(p/'AUDIT.json')
            assert audit['status'] == 'PASS'
            assert audit['max_abs_difference'] <= audit['tolerance']
            assert audit['all_window_slots_and_geometry_and_motion']
            assert audit['summary_contributions_tails_partitions_recomputed']
            summary = read(p/'SUMMARY.json')
            registered = read(p/'REGISTERED_SET_METRICS.json')
            assert registered['status'] == 'SOURCE_BOUND_REUSE'
            for group in summary['groups']:
                parts = group['parts']
                assert sum(v['n'] for v in parts.values()) == group['n']
                for metric in ('minADE','minFDE','Jmin'):
                    assert abs(sum(v['contribution'][metric] for v in parts.values())-group['delta'][metric]) < 1e-10
                ma = group['motion_alignment']
                metrics = registered['metrics'][group['group']]
                assert metrics['good'] == parts['retained']['n']+parts['rescue']['n']
                assert metrics['native_good'] == parts['retained']['n']+parts['lost']['n']
                records.append(dict(seed=seed,split=split,group=group['group'],n=group['n'],tracks=group['tracks'],
                    partitions={k:dict(n=v['n'],tracks=v['tracks'],contribution=v['contribution']) for k,v in parts.items()},
                    delta=group['delta'],positive_tail=group['positive_tail'],
                    motion_alignment=ma,
                    neither_diagnostics=parts['neither']['diagnostics'],
                    registered={k:metrics[k] for k in ('good','native_good','rescue','lost','ADE','FDE','motion','native_motion','motion_pass')},
                    strata=summary['strata'],audit_maxerror=audit['max_abs_difference']))
    assert c.verify() == digest
    assert all(h(k) == v for k,v in before.items())
    resource = read(HERE/'D0_D1/RESOURCE.json')
    assert resource['status'] == 'PASS'
    snapshots = [x['snapshot'] for x in resource['observations'] if 'snapshot' in x]
    review = dict(status='PASS_SAVED_COMPLETION_PROVENANCE_AND_EXTRACTION',
        source_count=len(read(HERE/'source_manifest.json')['files']),source_digest=digest,
        scientific_jobs=6,output_SHA_verified=len(before),source_and_outputs_unchanged=True,
        max_saved_independent_error=max(r['audit_maxerror'] for r in records),
        no_model_forward_optimizer=True,no_new_full_geometry_replay=True,
        production_private_peak_bytes=max(x['peak_private'] for x in snapshots),
        production_RSS_peak_bytes=max(x['peak_rss'] for x in snapshots),
        decision=read(HERE/'D0_D1/DECISION.json'),records=records)
    target = Path(__file__).with_name('REVIEW.json')
    target.write_text(json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in review.items() if k!='records'},ensure_ascii=False))
    for r in records:
        if r['group'] != 'change': continue
        ma=r['motion_alignment']
        print(json.dumps(dict(seed=r['seed'],split=r['split'],n=r['n'],delta=r['delta'],
            parts={k:v['contribution'] for k,v in r['partitions'].items()},
            tail=r['positive_tail'],
            selected={k:ma[k]['mean'] for k in ma if k.startswith(('student_J_','native_J_','student_Q_E_','native_Q_E_','student_ALL20_E_','native_ALL20_E_')) and isinstance(ma[k]['mean'],(float,int))}),ensure_ascii=False))

if __name__ == '__main__': main()
