"""Read-only saved-evidence analysis. CPU checkpoint reads only; no model instantiation, forward, training, process checks or deletion."""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
PKG = ROOT.parents[1]
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(name, value):
    with (OUT/name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''):
            h.update(b)
    return h.hexdigest()

verified = {}
def verify(mapping):
    assert mapping
    for p, expected in mapping.items():
        if p not in verified:
            actual = sha(p)
            assert actual == expected, ('SHA_MISMATCH', p, actual, expected)
            verified[p] = expected
        else:
            assert verified[p] == expected, ('CONFLICTING_SHA', p)

success, exit_receipt = read(ROOT/'WORKER_SUCCESS.json'), read(ROOT/'WORKER_EXIT.json')
assert success['status'] == 'SUCCESS_SAVED_SCIENCE_COST_UNMEASURED'
assert exit_receipt['status'] == 'SUCCESS' and exit_receipt['exit_code'] == 0
assert sha(ROOT/'FINAL_WORKER_SEAL.json') == success['seal_sha']
for name in ('FINAL_WORKER_SEAL.json', 'FINAL_SCIENCE_SEAL.json', 'SOURCE_LOCK.json', 'INFRASTRUCTURE_LOCK.json'):
    receipt = read(ROOT/name)
    verify(receipt['files'])
    if 'infrastructure' in receipt:
        verify(receipt['infrastructure'])
print('SOURCE_AND_SEAL_SHA_PASS', len(verified), flush=True)

locks = read(ROOT/'ALL12_FINALS_LOCK.json')
names = sorted(locks['runs'])
assert len(names) == 12
for name in names:
    run = ROOT/'runs'/name
    done = read(run/'done.json')
    assert done == locks['runs'][name]
    assert done['epochs'] == 20 and done['updates'] == 13580
    verify(done['files'])
    history = read(run/'history.json')
    assert len(history) == 13580
    assert [h['update'] for h in history] == list(range(1,13581))
    assert all(sum(h['epoch'] == ep for h in history) == 679 for ep in range(1,21))
    assert len(read(run/'epoch_orders.json')) == 20
    assert datetime.fromisoformat(locks['at']) <= datetime.fromisoformat(read(ROOT/'WORKER_SUCCESS.json')['at'])
for family in ('existing', 'cvae'):
    for seed in (42,43,44):
        a = ROOT/'runs'/f'{family}_accuracy_only_seed{seed}'
        b = ROOT/'runs'/f'{family}_accuracy_dpp_seed{seed}'
        assert read(a/'epoch_orders.json') == read(b/'epoch_orders.json')
        # Whole checkpoint SHA differs because contract.arm is intentionally different.
        # Read source-SHA-verified local checkpoint payload on CPU; never instantiate a model.
        import torch
        left = torch.load(a/'initial.pt',map_location='cpu',weights_only=False)
        right = torch.load(b/'initial.pt',map_location='cpu',weights_only=False)
        assert left['model'].keys() == right['model'].keys()
        assert all(torch.equal(left['model'][k],right['model'][k]) for k in left['model'])
        assert left['optimizer'] == right['optimizer']
        assert torch.equal(left['rng']['loader'],right['rng']['loader'])
        del left, right
print('12_FINALS_240_EPOCH_162960_UPDATES_PAIRED_INIT_ORDER_PASS', flush=True)

native = {}
rows, values, population_checks = [], {}, []
max_geometry_error = 0.
for split in ('fit','dev'):
    p = ROOT/'populations/native'/split
    with np.load(p/'predictions.npz', allow_pickle=False) as f:
        native[split] = dict(ids=f['ids'], groups=f['groups'])
    with np.load(p/'window_metrics.npz', allow_pickle=False) as f:
        native[split]['coverage'] = f['coverage'].astype(bool)
    for name in ['native'] + names:
        p = ROOT/'populations'/name/split
        verify(read(p/'done.json')['files'])
        assert read(p/'audit.json')['status'].startswith('PASS_ALL_WINDOWS')
        independent = read(p/'independent_review.json')
        assert independent['status'].startswith('PASS_SAME_SLOT_FULLK')
        m, s = read(p/'metrics.json'), read(p/'support.json')
        with np.load(p/'predictions.npz', allow_pickle=False) as f:
            ids, g, paths, future = f['ids'], f['groups'], f['paths'], f['future']
        assert np.array_equal(ids,native[split]['ids']) and np.array_equal(g,native[split]['groups'])
        distance = np.sqrt(np.sum((paths-future[:,None])**2, axis=-1))
        ade, fde = distance.mean(-1), distance[:,:,-1]
        j = np.maximum(ade/.2, fde/.4)
        good = (ade<=.2)&(fde<=.4)
        coverage = good.any(1)
        with np.load(p/'window_metrics.npz',allow_pickle=False) as f:
            for key, calculated in (('ADE',ade),('FDE',fde),('J',j)):
                err = float(np.max(np.abs(f[key]-calculated)))
                max_geometry_error = max(max_geometry_error,err)
                assert err <= 2e-12
            assert np.array_equal(good,f['good']) and np.array_equal(coverage,f['coverage'])
        with np.load(p/'auxiliary.npz',allow_pickle=False) as f:
            v = np.stack([coverage,ade.min(1),fde.min(1)] + [f[k] for k in ('kin_valid_coverage','kin_violation_rate','effective_good')],-1)
        if split == 'dev':
            values[name] = v
        for group, code in (('overall',None),('stable',0),('transition',1),('change',2)):
            mask = np.ones(len(g),bool) if code is None else g==code
            old = native[split]['coverage']
            nc, ng = int(mask.sum()),int(old[mask].sum())
            actual = dict(N=nc,native_good=ng,good=int(coverage[mask].sum()),
                          rescue=int((mask&coverage&~old).sum()),lost=int((mask&~coverage&old).sum()))
            assert all(s[group][k] == actual[k] for k in actual)
            assert m[group]['good'] == actual['good'] and m[group]['n'] == nc
            assert abs(m[group]['coverage']-coverage[mask].mean())<=2e-12
            assert abs(m[group]['minADE']-ade[mask].min(1).mean())<=2e-12
            assert abs(m[group]['minFDE']-fde[mask].min(1).mean())<=2e-12
            if name != 'native':
                assert set(s[group]['rescue_IDs']) == set(ids[mask&coverage&~old])
                assert set(s[group]['lost_IDs']) == set(ids[mask&~coverage&old])
            rows.append(dict(run=name, split=split, group=group, **actual,
                             lost_rate=actual['lost']/ng if ng else None,
                             **{k:m[group][k] for k in ('coverage','minADE','minFDE','effective_good','motion')}))
        population_checks.append(dict(run=name,split=split,N=len(ids),independent_review=independent))
        del paths, future, distance, ade, fde, j, good
    print('FULL_SAVED_GEOMETRY_SUPPORT_PASS',split,flush=True)

# Different implementation: draw multiplicities in fixed batches, matrix aggregation.
order = [('existing_accuracy_dpp','existing_accuracy_only'),('existing_accuracy_dpp','native'),
         ('cvae_accuracy_dpp','cvae_accuracy_only'),('cvae_accuracy_dpp','native'),
         ('cvae_accuracy_dpp','existing_accuracy_dpp')]
blocks = lambda x: np.repeat(values['native'][None],3,axis=0) if x=='native' else np.stack([values[x+f'_seed{s}'] for s in (42,43,44)])
v = np.stack([blocks(a)-blocks(b) for a,b in order])
ids,g = native['dev']['ids'],native['dev']['groups']
tracks = np.array(['|'.join(x.split('|')[:2]) for x in ids])
tracknames = np.unique(tracks)
ti = np.searchsorted(tracknames,tracks)
mask = g==2
count = np.bincount(ti,weights=mask,minlength=len(tracknames))
sums = np.zeros((5,3,len(tracknames),6))
for c in range(5):
    for ss in range(3):
        for mm in range(6):
            sums[c,ss,:,mm] = np.bincount(ti,weights=v[c,ss,:,mm]*mask,minlength=len(tracknames))
with np.load(ROOT/'BOOTSTRAP_DRAWS.npz',allow_pickle=False) as f:
    assert np.array_equal(f['names'],tracknames)
    draw,fixed,td,sd = [f[k] for k in ('draw','fixed','track','seed')]
second, fixed_second = np.empty_like(draw),np.empty_like(fixed)
for start in range(0,20000,128):
    end = min(start+128,20000)
    weights = np.stack([np.bincount(t,minlength=len(tracknames)) for t in td[start:end]])
    den = weights@count
    aggregate = np.einsum('bt,cstm->bcsm',weights,sums)/den[:,None,None,None]
    fixed_second[start:end] = aggregate.mean(2)
    for bi, seedrow in enumerate(sd[start:end]):
        second[start+bi] = aggregate[bi][:,seedrow,:].mean(1)
bootstrap_error = max(float(np.max(abs(second-draw))),float(np.max(abs(fixed_second-fixed))))
assert bootstrap_error <= 2e-12
primary = read(ROOT/'PRIMARY_COMPARISON.json')
for index,row in enumerate(primary['rows']):
    c,m = divmod(index,6)
    assert np.allclose(row['per_seed'],v[c][:,mask,m].mean(1),rtol=0,atol=2e-12)
    for key,array,q in (('ordinary95',second,[.025,.975]),('family30',second,[.05/60,1-.05/60]),
                        ('fixedseed_family30',fixed_second,[.05/60,1-.05/60])):
        assert np.allclose(row[key],np.quantile(array[:,c,m],q),rtol=0,atol=2e-12)
print('INDEPENDENT_ALL_20000_FAMILY30_PASS',bootstrap_error,flush=True)

gate_rows = read(ROOT/'PROTECTION_GATES.json')['rows']
gates, aggregate_rows = {}, []
for family in ('existing','cvae'):
    for arm in ('accuracy_only','accuracy_dpp'):
        label = f'{family}_{arm}'
        subset = [x for x in gate_rows if x['family']==family and x['arm']==arm]
        gates[label] = {k:dict(passed=sum(x[k] is True for x in subset),
                            total=sum(x[k] is not None for x in subset)) for k in
                            ('old_good_PASS','protected_errors_PASS','all_five_PASS','change5pp','change_ADEFDE5percent')}
        gates[label]['motion_cells'] = dict(passed=sum(sum(x['five_motion'].values()) for x in subset),total=120)
for label in ['native']+[f'{f}_{a}' for f in ('existing','cvae') for a in ('accuracy_only','accuracy_dpp')]:
    for split in ('fit','dev'):
        for group in ('overall','stable','transition','change'):
            subset = [x for x in rows if x['split']==split and x['group']==group and
                      (x['run']=='native' if label=='native' else x['run'].startswith(label+'_seed'))]
            aggregate_rows.append(dict(method=label,split=split,group=group,N=subset[0]['N'],
                **{k:float(np.mean([x[k] for x in subset])) for k in
                   ('good','native_good','rescue','lost','lost_rate','coverage','minADE','minFDE','effective_good')},
                motion={k:float(np.mean([x['motion'][k] for x in subset])) for k in subset[0]['motion']}))
resource = read(ROOT/'RESOURCE.json')
snapshots = [x['snapshot'] for x in resource['observations']]
resource_summary = dict(E=resource['estimate'],peak_private=max(x['peak_private'] for x in snapshots),
    peak_rss=max(x['peak_rss'] for x in snapshots),
    minimum_physical_retained_last128=min(x['physical'] for x in snapshots),
    minimum_commit_retained_last128=min(x['commit'] for x in snapshots),
    retained_observations=len(snapshots),
    warning_count_retained=sum(x['severity']=='WARNING' for x in resource['observations']),
    note='Lifetime Windows peaks; minima/warning counts only retained last128, not whole-run minimum. Warning is not pressure STOP.')
review = dict(status='PASS_READONLY_26POP_GEOMETRY_SUPPORT_ALL20000_STATS_SOURCE_SHA',
    at=datetime.now().astimezone().isoformat(),runs=12,epochs=240,updates=162960,
    populations=26,rows104_including_native=True,model_group_rows=96,
    unique_verified_files=len(verified),source_files=247,infrastructure_files=54,
    max_saved_geometry_error=max_geometry_error,bootstrap_error=bootstrap_error,
    geometry_windows=sum(x['N'] for x in population_checks),
    no_model_or_gradient_replay=True,CPU_checkpoint_tensor_read_only=True,no_forward_no_training=True,
    lifecycle_resource=resource_summary,worker_elapsed_seconds=success['elapsed_seconds'],
    baseline_upgrade=False,raw_cost='UNMEASURED_NO_DEPLOYMENT_PASS',same_assistant_ARS_not_external=True)
write('VERIFIED_SOURCE_OUTPUT_SHA.json',verified)
write('REVIEW.json',review)
write('ALL_SEED_SPLIT_GROUPS.json',rows)
write('METHOD_MEANS.json',aggregate_rows)
write('GATE_COUNTS.json',gates)
write('PRIMARY_COMPARISON_VERIFIED.json',primary)
write('POPULATION_AUDIT_INDEX.json',population_checks)
for filename,data in (('ALL_SEED_SPLIT_GROUPS.csv',rows),('METHOD_MEANS.csv',aggregate_rows)):
    with (OUT/filename).open('x',encoding='utf-8-sig',newline='') as f:
        fields=[k for k in data[0] if k!='motion']+list(data[0]['motion'])
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for item in data:
            writer.writerow({**{k:v for k,v in item.items() if k!='motion'},**item['motion']})
print(json.dumps(review,ensure_ascii=False),flush=True)
print('GATES',json.dumps(gates),flush=True)
