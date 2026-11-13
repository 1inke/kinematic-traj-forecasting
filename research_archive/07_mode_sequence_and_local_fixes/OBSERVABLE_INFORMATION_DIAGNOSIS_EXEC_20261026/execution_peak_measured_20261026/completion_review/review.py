"""Post-completion saved-data supplement. No model forward, solver, or training.

Canonical full complete-link/pdist and pooled-motion audits are inherited through
their immutable done contracts; this supplement repeats saved slot arithmetic,
choice/fixed19, all counts, all 20k draws, and adds explicit root-output hashes.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import bootstrap
from base import *
from src.f16.eval import errors

OUT = HERE / 'completion_review'

def checked_done(folder):
    d = read_json(folder / 'done.json')
    completed(folder, d['contract'])
    return d

def geometry(z, saved, sm, lo):
    n = len(z['sample_id']); sl = slice(lo, lo+n)
    assert np.array_equal(saved['sample_id'], z['sample_id'])
    for k in ('eligible', 'B', 'target'):
        assert np.array_equal(saved[k], z[k])
    s = saved['scores']; assert s.shape == (n,401) and np.isfinite(s).all()
    chosen = np.where(z['eligible'], s, -np.inf).argmax(1)
    assert np.array_equal(chosen, saved['chosen'])
    expected = z['native'].copy()
    for i, a in enumerate(chosen):
        if a:
            r,l = divmod(int(a)-1,20); expected[i,r] = z['kin'][i,l]
    assert np.array_equal(expected, saved['paths'])
    # Second saved-data implementation; no coordinate epsilon in good gate.
    d = np.linalg.norm(expected-z['gt'][:,None],axis=-1)
    a = d.mean(-1); f = d[:,:,-1]; j = np.maximum(a/.2, f/.4)
    good = ((a<=.2)&(f<=.4)).any(1)
    assert np.array_equal(sm['sample_id'][sl],z['sample_id'])
    for k,v in (('ADE',a),('FDE',f),('J',j)):
        assert np.max(np.abs(v-sm[k][sl])) <= 2e-12
    assert np.array_equal(good,sm['good'][sl])
    assert np.array_equal(sm['chosen'][sl],chosen)
    assert np.array_equal(np.argmin(a+.5*f,1),sm['Q_slot'][sl])
    assert np.array_equal(np.argmin(j,1),sm['J_slot'][sl])
    native = errors(z['native'],z['gt'])['coverage']
    assert np.array_equal(native,sm['native_good'][sl])

def counts(sm, metrics):
    for i,name in enumerate(('overall','stable','transition','change')):
        mask = np.ones(len(sm['good']),bool) if not i else sm['group']==i-1
        r = metrics[name]; good=sm['good'][mask]; native=sm['native_good'][mask]
        assert r['n']==int(mask.sum()) and r['good']==int(good.sum())
        assert r['native_good']==int(native.sum())
        assert r['rescue']==int((good&~native).sum()) and r['lost']==int((native&~good).sum())
        assert r['rescue_IDs']==sm['sample_id'][mask][good&~native].tolist()
        assert r['lost_IDs']==sm['sample_id'][mask][native&~good].tolist()

def main():
    OUT.mkdir(exist_ok=True); g=guard('AUDIT'); initial=verify()
    assert read_json(HERE/'worker_state.json')['status']=='SUCCESS'
    assert read_json(HERE/'exit_status.json')['status']=='SUCCESS'
    preserved=read_json(HERE/'PRESERVATION.json')
    for p,h in preserved['outputs'].items(): assert sha256(HERE/p)==h,p
    done_count=0
    for p in HERE.rglob('done.json'):
        if 'completion_review' in p.parts or any(x.startswith('preflight_') for x in p.parts): continue
        checked_done(p.parent); done_count+=1
    groups=[]; tiny=[]; hold={}; saved_windows=0
    for arm in ('I','S'):
        for seed in SEEDS:
            folder=HERE/f'step1/{arm}_seed{seed}'
            history=read_json(folder/'history.json'); assert len(history)==512
            assert [r['update'] for r in history]==list(range(1,513))
            for reading in (0,32,128,512):
                audit=folder/f'audit{reading:03d}'; checked_done(audit)
                m=read_json(audit/'METRICS.json'); a=read_json(audit/'ASSIGNMENT.json')
                with np.load(audit/'sample_metrics.npz') as q: sm={k:q[k] for k in q.files}
                counts(sm,m)
                tiny.append(dict(arm=arm,seed=seed,reading=reading,metrics=m,assignment=a))
            g.check()
    for arm in ('M_full','T_full','T_geom'):
        hold[arm]=[]
        for seed in SEEDS:
            run=HERE/f'step2/{arm}_seed{seed}'; manifest=read_json(run/'manifest.json')
            if arm=='M_full': assert len(read_json(run/'history.json'))==1024
            else:
                assert read_json(run/'TRAIN.json')['iterations']==64
                assert manifest['parameters']['max_iter']==64 and not manifest['parameters']['early_stopping']
            for split in ('opt','hold','inner'):
                folder=HERE/f'populations/{arm}_seed{seed}/{split}'
                done=checked_done(folder); audit=folder/'audit'; ad=checked_done(audit)
                assert ad['full_independent_complete_link'] and ad['pooled_all5']
                m=read_json(audit/'METRICS.json'); a=read_json(audit/'ASSIGNMENT.json')
                with np.load(audit/'sample_metrics.npz') as q: sm={k:q[k] for k in q.files}
                counts(sm,m); lo=0
                for name,z in zip(done['chunks'],aux_chunks(seed,split)):
                    with np.load(folder/name) as q: saved={k:q[k] for k in q.files}
                    geometry(z,saved,sm,lo); lo+=len(z['sample_id']); g.check()
                assert lo==len(sm['good']); saved_windows+=lo
                groups.append(dict(arm=arm,seed=seed,split=split,metrics=m,assignment=a,
                    all_STOP=bool((sm['chosen']==0).all()),coverage_exact_native=bool(np.array_equal(sm['good'],sm['native_good']))))
                if split=='hold':hold[arm].append(sm)
    # Independent actual draw and sufficient-statistic recomputation, no redraw.
    h=hold['M_full'][0]; tracks=h['track_key']; group=h['group']; unique=np.unique(tracks)
    tridx=np.searchsorted(unique,tracks); mask=group==2
    den=np.bincount(tridx[mask],minlength=len(unique)).astype(float)
    arrays={arm:np.stack([r['good'].astype(float) for r in rows]) for arm,rows in hold.items()}
    native=np.stack([r['native_good'].astype(float) for r in hold['M_full']])
    diffs=dict(Tfull_minus_M=arrays['T_full']-arrays['M_full'],Tfull_minus_geom=arrays['T_full']-arrays['T_geom'],
               Tfull_minus_native=arrays['T_full']-native,Tgeom_minus_native=arrays['T_geom']-native,M_minus_native=arrays['M_full']-native)
    with np.load(HERE/'statistics/BOOTSTRAP_DRAWS.npz') as q: draws={k:q[k] for k in q.files}
    assert np.array_equal(unique,draws['tracks']); rng=np.random.default_rng(20261026)
    sums={k:np.stack([np.bincount(tridx[mask],weights=row[mask],minlength=len(unique)) for row in v]) for k,v in diffs.items()}
    for i in range(20000):
        td=rng.integers(len(unique),size=len(unique)); sd=rng.integers(3,size=3)
        assert np.array_equal(td,draws['track_draws'][i]) and np.array_equal(sd,draws['seed_draws'][i])
        c=np.bincount(td,minlength=len(unique)); d=float(c@den)
        for k,v in sums.items():
            if d:
                per=v@c/d
                assert abs(per[sd].mean()-draws['values_'+k][i])<=2e-12
                assert abs(per.mean()-draws['fixed_'+k][i])<=2e-12
            else:assert np.isnan(draws['values_'+k][i])
    roots=['RESULT.json','RESOURCE.json','ARITHMETIC_AUDIT.json','PRESERVATION.json','ALL_PROBE_FINALS_LOCK.json',
           'FIT_DIAGNOSTICS.json','ALL_PROBE_GROUPS.json','INPUT_COLLISIONS.json','statistics/PRIMARY_COMPARISONS.json',
           'statistics/BOOTSTRAP_DRAWS.npz','source_manifest.json','PRODUCTION_SOURCE_LOCK.json']
    resource=read_json(HERE/'RESOURCE.json'); snapshots=[]
    for item in resource['records']:
        report=item['report']; records=report.get('observations',[])+([report['last']] if 'last' in report else [])
        for row in records:
            assert row['status']=='PASS'; snapshots.append(row['snapshot'])
    resource_summary={k:fn(s[k] for s in snapshots) for k,fn in [('peak_private',max),('peak_rss',max),('physical',min),('commit',min)]}
    write_json(OUT/'ALL_GROUPS.json',groups);write_json(OUT/'ALL_TINY_READINGS.json',tiny)
    report=dict(status='PASS_SAVED_COMPLETION_SUPPLEMENT',at=now(),source=initial,source_entries=1660,
                done_contracts=done_count,tiny_readings=24,probe_populations=27,probe_windows=saved_windows,
                all_20000_actual_draws_independent=True,root_output_SHA={p:sha256(HERE/p) for p in roots},
                resource_production=resource_summary,source_final=verify(),
                boundaries='No model forward/backward/optimizer. Full clusters/motion inherited through SHA-verified worker independent audits; saved choice/slot quality/counts and actual bootstrap repeated here.',
                supplement_source_SHA=sha256(__file__))
    write_json(OUT/'REVIEW.json',report);resource_report(OUT,g)
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
