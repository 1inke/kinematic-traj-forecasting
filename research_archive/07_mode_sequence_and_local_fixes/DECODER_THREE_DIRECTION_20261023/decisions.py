"""Registered local mechanism gates, not baseline promotion or future launch."""
from common import *

def analyze(package,execution,data):
    from training import PACKAGES
    control,test=PACKAGES[package];d=data['inner'];change=d['groups']==2
    track=d['track'][change];samples={};rows={}
    for arm in (control,test):
        samples[arm]=[];rows[arm]=[]
        for seed in SEEDS:
            run=execution/f'{arm}_seed{seed}'
            with np.load(run/'predictions_inner.npz') as z:
                assert np.array_equal(z['sample_id'],d['ids'])
                p=z['paths'].copy();ee=errors(p,d['gt'])
                # Per-window mean over ALL20x11 acceleration magnitudes.
                v=np.diff(p.astype(float),axis=2,prepend=np.zeros_like(p[:,:,:1],dtype=float))/.4
                aa=np.linalg.norm(np.diff(v,axis=2)/.4,axis=-1).mean((1,2))
                samples[arm].append(dict(coverage=ee['coverage'],ADE=ee['ADE'].min(1),FDE=ee['FDE'].min(1),accel=aa))
            rows[arm].append(read_json(run/'metrics_inner.json'))
    native=errors(d['native'],d['gt']);ng=native['coverage'];nchange=int(ng[change].sum())
    ci={};fixed={}
    for metric in ('coverage','ADE','FDE','accel'):
        delta=np.array([samples[test][i][metric][change].astype(float)-samples[control][i][metric][change].astype(float) for i in range(3)])
        ci[metric]=paired_ci(delta,track);fixed[metric]=paired_ci(delta,track,resample_seeds=False)
    coverage_native=np.array([s['coverage'][change].astype(float)-ng[change] for s in samples[test]])
    native_ci=paired_ci(coverage_native,track)
    good=lambda arm,group:np.array([r[group]['good'] for r in rows[arm]])
    cg,tg=good(control,'change'),good(test,'change')
    ade=np.array([r['change']['ADE'] for r in rows[test]])
    fde=np.array([r['change']['FDE'] for r in rows[test]])
    ca=np.array([r['change']['ADE'] for r in rows[control]])
    cf=np.array([r['change']['FDE'] for r in rows[control]])
    lost=lambda arm:np.array([r['change']['lost'] for r in rows[arm]])
    baseline={arm:[] for arm in (control,test)}
    for arm in (control,test):
        for seed in SEEDS:
            for split in SIZES:
                m=read_json(execution/f'{arm}_seed{seed}/metrics_{split}.json')
                for group,r in m.items():baseline[arm].append(dict(seed=seed,split=split,group=group,passed=r['G_base'],motion_pass=r['motion_pass']))
    def upper(name):return ci[name].get('status')=='OK' and ci[name]['CI95'][1]<0
    def lower(name,f=False):
        b=fixed[name] if f else ci[name]
        return b.get('status')=='OK' and b['CI95'][0]>0
    if package=='C1':
        gain=cg-nchange
        coverage_retained=np.all((tg>=cg)|((gain>0)&((tg-nchange)>=.95*gain)))
        gates=dict(coverage=bool(coverage_retained and native_ci.get('status')=='OK' and native_ci['CI95'][0]>0),
            retention=bool(lost(test).mean()<lost(control).mean() and np.all(lost(test)-lost(control)<=2)),
            errors=bool(np.all(ade<ca) and np.all(fde<cf) and upper('ADE') and upper('FDE')))
        label='AWTA_SCHEDULE_SUPPORTED_LOCALLY' if all(gates.values()) else 'AWTA_SCHEDULE_NOT_SUPPORTED'
    elif package=='C2':
        old=[]
        for seed in SEEDS:
            ids,p=old_paths(seed,'inner');assert np.array_equal(ids,d['ids'])
            old.append(int(errors(p,d['gt'])['coverage'][change].sum()))
        groups_ok=all(rows[test][i][g]['ADE']<=1.01*rows[control][i][g]['ADE'] and
            rows[test][i][g]['FDE']<=1.01*rows[control][i][g]['FDE'] and
            rows[test][i][g]['lost']<=rows[control][i][g]['lost'] for i in range(3) for g in ('overall','change'))
        gates=dict(coverage=bool(np.mean((tg-cg)/497)>=.02 and np.all(tg>cg) and lower('coverage') and lower('coverage',True)),
            old_baseline=bool(np.all(tg>=old)),
            effective_good=bool(np.mean([r['change']['effective_good'] for r in rows[test]])>
                                np.mean([r['change']['effective_good'] for r in rows[control]])),
            errors_retention=bool(groups_ok))
        label='SETMIX_RESIDUAL_SUPPORTED_LOCALLY' if all(gates.values()) else 'SETMIX_RESIDUAL_NOT_SUPPORTED'
    else:
        old=[]
        for seed in SEEDS:
            ids,p=old_paths(seed,'inner');assert np.array_equal(ids,d['ids'])
            old.append(int(errors(p,d['gt'])['coverage'][change].sum()))
        reference=np.maximum(cg,old)-nchange
        gates=dict(coverage=bool(np.all(reference>0) and np.all(tg-nchange>=.9*reference) and
            all(r[g]['good']>=.99*r[g]['native_good'] for r in rows[test] for g in ('overall','change'))),
            acceleration=bool(all(rows[test][i]['change']['motion']['accel_mean']<=.8*rows[control][i]['change']['motion']['accel_mean'] for i in range(3)) and upper('accel')),
            errors=bool(all(rows[test][i][g][metric]<=1.01*rows[control][i][g][metric] for i in range(3) for g in ('overall','change') for metric in ('ADE','FDE'))))
        label='TIME_BASIS_SUPPORTED_LOCALLY' if all(gates.values()) else 'TIME_BASIS_NOT_SUPPORTED'
    ready=all(v['passed'] for v in baseline[test])
    if all(gates.values()) and not ready:
        label='TIME_BASIS_IMPROVES_ACCEL_BUT_BASELINE_NOT_READY' if package=='A' else 'MECHANISM_SUPPORTED_BASELINE_NOT_READY'
    result=dict(local_gates=gates,mechanism_supported=all(gates.values()),baseline_ready=ready,conclusion=label,
        primary_and_secondary_unadjusted_CI=ci,fixed_seed_sensitivity=fixed,native_coverage_CI=native_ci,
        baseline_all_24_groups_per_arm=baseline,automatic_next=False,
        interpretation='Fixed encoder/repeated inner exploratory mechanisms; not independent whole-model generalization')
    write_json(execution/'DECISION.json',result);return result
