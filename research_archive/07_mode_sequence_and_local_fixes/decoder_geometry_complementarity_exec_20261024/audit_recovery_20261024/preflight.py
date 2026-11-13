"""Actual failed population regression and immutable-source admission."""
from adapter import *
from saved_attribution import metrics_saved
import time
def fingerprint():
    source=verify_manifest();r=read(OUT/'RESULT.json');files={str(OUT/n):h for n,h in r['outputs'].items()}
    files[str(OUT/'RESULT.json')]=sha(OUT/'RESULT.json')
    for p,h in files.items():assert sha(p)==h
    return source,files
def check():
    timer=time.perf_counter();guard=Guard(E);guard.check('startup');ns,equiv=compiled();source,outputs=fingerprint()
    d,paths=metrics_saved(42,'FIT');ee={k:geometry(p,d['gt']) for k,p in paths.items()}
    failed_population=ee['iso']['covered']&ee['mix']['covered'];deltas={};maximum=0.
    for arm,p in paths.items():
        wrong=np.zeros_like(p,dtype=float);wrong[:,:,0]=p[:,:,0]/.4;wrong[:,:,1:]=(p[:,:,1:].astype(float)-p[:,:,:-1].astype(float))/.4
        fixed=wrong.copy();fixed[:,:,0]=p[:,:,0].astype(float)/.4
        def pack(v):
            a=np.diff(v,axis=2)/.4;j=np.diff(a,axis=2)/.4
            return [np.linalg.norm(x,axis=3) for x in (v,a,j)]
        gold=physical(p);actual=pack(fixed)
        assert all(np.array_equal(a,b) for a,b in zip(gold,actual))
        a=pool(gold,failed_population);b=pool(pack(wrong),failed_population);deltas[arm]={k:b[k]-a[k] for k in a}
        for k,v in pool(actual,failed_population).items():assert_close(v,a[k]);maximum=max(maximum,abs(v-a[k]))
    assert any(abs(v)>2e-12 for d in deltas.values() for v in d.values())
    for p,h in outputs.items():assert sha(p)==h
    assert verify_manifest()==source;guard.check()
    files={str(p.resolve()):dict(path=str(p.resolve()),sha256=sha(p)) for p in CHILD.glob('*.py')}
    files[str(CHILD/'README.md')]=dict(path=str(CHILD/'README.md'),sha256=sha(CHILD/'README.md'))
    for p,h in outputs.items():files[p]=dict(path=p,sha256=h)
    files[str(PARENT/'source_manifest.json')]=dict(path=str(PARENT/'source_manifest.json'),sha256=sha(PARENT/'source_manifest.json'))
    write(CHILD/'source_manifest.json',dict(files=files,digest=digest(files),inherited_parent_digest=source,completed_parent_outputs=len(outputs),parent_files_not_modified=True))
    write(CHILD/'SOURCE_EQUIVALENCE.json',equiv)
    report=dict(status='PASS_ACTUAL_FAILED_POPULATION_AUDITOR_REGRESSION',original_float32_discrepancies=deltas,corrected_maxerror=maximum,
        parent_source_digest=source,recovery_source_digest=digest(files),saved_outputs=len(outputs),
        resource_observations=guard.observations,elapsed_seconds=time.perf_counter()-timer,
        planning_seconds=180,ETA_seconds=[126,270],no_model_forward=True,no_Q0_reproduction=True,no_training=True)
    write(CHILD/'PREFLIGHT.json',report);print(json.dumps(report))
if __name__=='__main__':check()
