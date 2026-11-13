"""Read-only scientific evidence review; writes only this new review namespace.

No torch/model/optimizer imports, training, new costs, or old task queries.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, hashlib, csv
import numpy as np

OUT = Path(__file__).resolve().parent
RUN = OUT.parent
def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()
def write(name, value):
    with (OUT/name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
files = {}
def verify(mapping):
    for name, expected in mapping.items():
        p = Path(name)
        actual = sha(p)
        if actual != expected: raise ValueError('SHA_MISMATCH '+name)
        files[str(p)] = actual

source = read(RUN/'PRODUCTION_SOURCE_LOCK.json')
verify(source['files'])
lock = read(RUN/'ALL6_FINALS_LOCK.json')
verify(lock['checkpoint_files'])
training=[]
for seed in (42,43,44):
    pair=[]
    for arm in ('uniform','kin'):
        p=RUN/'runs'/f'{arm}_seed{seed}'
        d=read(p/'done.json'); verify(d['files'])
        history=read(p/'history.json'); orders=read(p/'epoch_orders.json')
        assert len(history)==13580 and d['updates']==13580 and d['epochs']==20
        assert len(orders)==20 and history[-1]['epoch']==20
        pair.append(orders)
        training.append(dict(arm=arm,seed=seed,epochs=20,updates=len(history),
                             final_sha=sha(p/'final.pt')))
    assert pair[0]==pair[1]

rows=[]; np_error=0.; population_receipts=[]
for split in ('fit','dev'):
    nativep=RUN/'populations/native'/split
    with np.load(nativep/'window_metrics.npz',allow_pickle=False) as f:
        native_good=((f['ADE']<=.2)&(f['FDE']<=.4)).any(1)
    native_metrics=read(nativep/'metrics.json')
    for arm in ('native','uniform','kin'):
        for seed in ((None,) if arm=='native' else (42,43,44)):
            name='native' if seed is None else f'{arm}_seed{seed}'
            p=RUN/'populations'/name/split
            d=read(p/'done.json'); verify(d['files'])
            review=read(p/'independent_review.json')
            assert review['status'].startswith('PASS_')
            metrics=read(p/'metrics.json'); support=read(p/'support_and_quality.json')
            with np.load(p/'window_metrics.npz',allow_pickle=False) as f:
                ade=f['ADE']; fde=f['FDE']; good=(ade<=.2)&(fde<=.4)
                coverage=good.any(1)
                assert np.array_equal(good,f['good']) and np.array_equal(coverage,f['coverage'])
            with np.load(p/'auxiliary.npz',allow_pickle=False) as f:
                groups=f['groups']; ids=f['ids']; valid=f['kin_valid']; q=f['qkin']
            for ix,g in enumerate(('overall','stable','transition','change')):
                mask=np.ones(len(ids),bool) if ix==0 else groups==ix-1
                n=int(mask.sum()); a=metrics[g]; s=support[g]; b=native_metrics[g]
                lost=mask&native_good&~coverage; rescue=mask&~native_good&coverage
                assert n==a['n']==s['N']
                assert int(coverage[mask].sum())==a['good']==s['good']
                assert int(lost.sum())==s['lost'] and int(rescue.sum())==s['rescue']
                assert set(map(str,ids[lost]))==set(s['lost_IDs'])
                assert set(map(str,ids[rescue]))==set(s['rescue_IDs'])
                actual=[float(ade[mask].min(1).mean()),float(fde[mask].min(1).mean()),
                        float((~valid[mask]).mean()),float(q[mask].mean())]
                expected=[a['minADE'],a['minFDE'],s['kin_violation_rate'],s['qkin_mean']]
                error=float(np.max(np.abs(np.array(actual)-expected)))
                np_error=max(np_error,error); assert error<=2e-12
                rows.append(dict(split=split,arm=arm,seed=seed,group=g,N=n,
                    good=a['good'],coverage=a['coverage'],native_good=b['good'],
                    rescue=s['rescue'],lost=s['lost'],lost_rate=s['lost_rate'],
                    minADE=a['minADE'],minFDE=a['minFDE'],distinct=a['distinct'],
                    effective_good=a['effective_good'],EC=s['EC'],qkin_mean=s['qkin_mean'],
                    kin_violation_rate=s['kin_violation_rate'],motion=a['motion']))
            population_receipts.append(dict(name=name,split=split,N=len(ids),
                canonical_status=d['status'],independent_status=review['status']))
            for n in ('done.json','independent_review.json'): files[str(p/n)]=sha(p/n)

gates=read(RUN/'PROTECTION_GATES.json'); primary=read(RUN/'PRIMARY_COMPARISON.json')
assert len(gates['rows'])==48 and len(primary['rows'])==18 and primary['draws']==20000
gate_summary=dict(rows=48,lost_protection_failed=sum(not r['native_good_protection_PASS'] for r in gates['rows']),
    all_five_motion_failed=sum(not r['all_five_motion_PASS'] for r in gates['rows']),
    motion_cells=240,motion_failed=sum(not v['literal_PASS'] for r in gates['rows'] for v in r['five_motion'].values()),
    protected_nonchange_rows=36,protected_nonchange_failed=sum(not r['protected_ADEFDE_PASS'] for r in gates['rows'] if r['group']!='change'))
summary={}
for split in ('fit','dev'):
    for group in ('overall','stable','transition','change'):
        for arm in ('native','uniform','kin'):
            rr=[r for r in rows if r['split']==split and r['group']==group and r['arm']==arm]
            summary[f'{split}/{group}/{arm}']={k:float(np.mean([r[k] for r in rr])) for k in
                ('coverage','minADE','minFDE','distinct','effective_good','EC','qkin_mean','kin_violation_rate','lost_rate','rescue','lost')}
            summary[f'{split}/{group}/{arm}']['motion']={k:float(np.mean([r['motion'][k] for r in rr])) for k in rr[0]['motion']}
            summary[f'{split}/{group}/{arm}']['N']=rr[0]['N']
            summary[f'{split}/{group}/{arm}']['good_per_seed']=[r['good'] for r in rr]
root_names=('PRODUCTION_SOURCE_LOCK.json','ALL6_FINALS_LOCK.json','ARITHMETIC_AUDIT.json',
            'PRIMARY_COMPARISON.json','BOOTSTRAP_DRAWS.npz','PROTECTION_GATES.json','RESULT_NUMERIC_SUMMARY.md')
for n in root_names: files[str(RUN/n)]=sha(RUN/n)
missing=[n for n in ('COST_ACTUAL_ARRAY_PIPELINE.json','RESOURCE.json','PRESERVATION.json','RESULT.json','FINAL_SEAL.json','WORKER_SUCCESS.json','WORKER_EXIT.json') if not (RUN/n).exists()]
status=dict(status='PASS_SAVED_SCIENTIFIC_RESULTS_NOT_FULL_EXECUTION_COMPLETION',
    at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
    source_count=len(source['files']),verified_unique_files=len(files),
    training=training,total_DSF_epochs=120,total_DSF_updates=81480,
    population_receipts=population_receipts,all_saved_samecandidate_quality_and_support_recomputed=True,
    max_saved_scalar_error=np_error,existing_full_cluster_motion_receipts_checked_not_reexecuted=True,
    all_source_and_saved_output_SHA_unchanged=True,gate_summary=gate_summary,
    missing_finalization=missing,observed_task_last_result=1,observed_task_state='Ready',
    active_worker_observed=False,last_saved_stage='ACTUAL_ARRAY_PIPELINE_COST',
    termination_cause='UNRESOLVED_NO_WORKER_FAILURE_OR_EXIT_RECEIPT',
    no_memory_OS_cause_inferred=True,no_training_or_cost_rerun=True,
    new_report_only_no_old_source_or_result_edits=True,files=files)
write('REVIEW.json',status)
write('ALL_GROUPS.json',rows);write('SUMMARY.json',summary)
with (OUT/'ALL_GROUPS.csv').open('x',encoding='utf-8-sig',newline='') as f:
    fields=[k for k in rows[0] if k!='motion']
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for row in rows:w.writerow({k:row[k] for k in fields})
print(json.dumps(dict(status=status['status'],source_count=status['source_count'],verified_unique_files=len(files),gate_summary=gate_summary,summary=summary),ensure_ascii=False))
