"""CPU receipt/source, full production-statistic fixture, disk, and OS wiring."""
from pathlib import Path
import sys,ast,shutil,time
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from joint_common import *

def main():
    authorize('preflight',True);base=verify_package();verify(base['dependencies'])
    launch=Path(__file__).resolve().parent;root=HERE/'executions'/'authorized_20261106'
    pre=HERE/'receipts/preflight_attempt1_20261106/scientific/PREFLIGHT.json'
    from run_series import prepare_execution
    root,receipt,lock=prepare_execution(root,pre)
    for p in launch.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    fixture=launch/'statistics_fixture';fixture.mkdir(exist_ok=False)
    n=100;ids=np.array([f'synthetic|{i//4}|{i}' for i in range(n)]);group=np.arange(n)%3
    rng=np.random.default_rng(66);arrays={}
    for name in ['native']+run_names():
        p=fixture/'populations'/name/'dev';p.mkdir(parents=True,exist_ok=False)
        ade=.05+.5*rng.random((n,20));fde=.1+rng.random((n,20))
        good=((ade<=.2)&(fde<=.4)).any(1);valid=good&(rng.random(n)>.1);violation=rng.random(n)
        effective=good.astype(float)*(1+rng.integers(0,3,n))
        np.savez_compressed(p/'window_metrics.npz',coverage=good,ADE=ade,FDE=fde)
        np.savez_compressed(p/'auxiliary.npz',ids=ids,groups=group,kin_valid_coverage=valid,
            kin_violation_rate=violation,effective_good=effective)
        write(p/'done.json',dict(status='SYNTHETIC_STATISTICS_NOT_SCIENTIFIC_POPULATION',files=seal(p.iterdir())))
        arrays[name]=np.stack([good,ade.min(1),fde.min(1),valid,violation,effective],-1)
    import joint_statistics
    t=time.perf_counter();joint_statistics.compare(fixture);stats=read(fixture/'PRIMARY_COMPARISON.json')
    native=arrays['native'];blocks={}
    for family in ('existing','cvae'):
        for arm in ('accuracy_only','accuracy_dpp'):
            blocks[family,arm]=np.stack([arrays[f'{family}_{arm}_seed{s}'] for s in (42,43,44)])
    ea,ej,ca,cj=(blocks[k] for k in (('existing','accuracy_only'),('existing','accuracy_dpp'),('cvae','accuracy_only'),('cvae','accuracy_dpp')))
    v=np.stack([ej-ea,ej-native[None],cj-ca,cj-native[None],cj-ej]);tracks=np.array(['|'.join(x.split('|')[:2]) for x in ids])
    with np.load(fixture/'BOOTSTRAP_DRAWS.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    ti=np.searchsorted(d['names'],tracks);mask=group==2;error=0.
    for i in range(20000):
        w=np.bincount(d['track'][i],minlength=len(d['names']))[ti]*mask
        den=w.sum()
        if not den:raise ValueError('UNEXPECTED_SYNTHETIC_EMPTY')
        expanded=(v[:,:,np.repeat(np.arange(n),w.astype(int)),:].mean(2))
        x=expanded[:,d['seed'][i]].mean(1);xf=expanded.mean(1)
        error=max(error,float(np.max(abs(x-d['draw'][i]))),float(np.max(abs(xf-d['fixed'][i]))))
    if error>2e-12 or len(stats['rows'])!=30:raise ValueError('PRODUCTION_STATISTICS_FIXTURE')
    for c in range(5):
        for m in range(6):
            row=stats['rows'][c*6+m]
            if len(row['per_seed'])!=3 or max(abs(np.array(row['per_seed'])-v[c][:,mask,m].mean(1)))>2e-12:
                raise ValueError('BOOTSTRAP_POINT_POPULATION')
    files=read(next((HERE/'receipts/preflight_attempt1_20261106/scientific').glob('saved_*/done.json')))['files']
    verify(files);bad=dict(files);bad[next(iter(bad))]='0'*64
    try:verify(bad)
    except ValueError:bad_output_rejected=True
    else:raise ValueError('BAD_OUTPUT_MANIFEST_ACCEPTED')
    write(fixture/'REVIEW.json',dict(status='PASS_ALL30_PRODUCTION_20K_WITH_EXPANDED_WINDOW_SECOND_IMPLEMENTATION',
        error=error,rows=30,draws=20000,elapsed=time.perf_counter()-t,
        bad_actual_output_receipt_rejected=bad_output_rejected,no_training_or_real_GT=True))
    disk=shutil.disk_usage(HERE).free;estimate=10*1024**3
    if disk<1.3*estimate:raise OSError('DISK_INSUFFICIENT_NO_DELETE')
    review=dict(status='PASS_FINITE12RUN_PRELAUNCH',at=now(),preflight_sha=sha(pre),
        SCRIPT_manifest_sha=sha(HERE/'SCRIPT_MANIFEST.json'),results=receipt['results'],
        science_code_unchanged=True,source_count=len(lock['files']),
        statistics_fixture_error=error,statistics_fixture_not_science=True,
        disk_free_bytes=disk,conservative_output_bytes=estimate,
        planned_runs=12,epochs=240,updates=162960,populations=26,
        planning_seconds=receipt['planning_seconds'],ETA_seconds=receipt['ETA_seconds'],
        raw_stream_cost='NOT_MEASURED_NO_DEPLOYMENT_PASS',no_automatic_more_variants=True,
        ARS_DA1='Joint accuracy does not guarantee retention; all original gates unchanged. Shared observed encoder is not three fresh whole-model pipelines.',
        no_old_deletes=True)
    write(root/'PRELAUNCH_REVIEW.json',review)
    deps=list(launch.glob('*.py'))+list(launch.glob('*.md'))+list(launch.glob('*.ps1'))
    deps.extend([root/'PRELAUNCH_REVIEW.json',fixture/'REVIEW.json',fixture/'PRIMARY_COMPARISON.json',fixture/'BOOTSTRAP_DRAWS.npz',
        HERE/'receipts/preflight_attempt1_20261106/PRE_POPEN_RESOURCE.json',HERE/'receipts/preflight_attempt1_20261106/LAUNCH_EXIT.json'])
    deps.extend(p for p in (HERE/'receipts/preflight_attempt1_20261106/scientific').rglob('*') if p.is_file())
    write(root/'INFRASTRUCTURE_LOCK.json',dict(status='EXPLICIT_CPU_LAUNCH_FINALIZATION_EXTENSION_SCIENCE_UNCHANGED',
        at=now(),preflight=str(pre),prepared_script_manifest_sha=sha(HERE/'SCRIPT_MANIFEST.json'),files=seal(set(deps))))
    print('PRELAUNCH_PASS',root,'planning_minutes',receipt['planning_seconds']/60,'fixture_error',error,flush=True)

if __name__=='__main__':main()
