from core import *
import ast,shutil
def main():
    source0=verify();g=guard('CPU');gpu=read_json(HERE/'PREFLIGHT_GPU.json');cpu=read_json(HERE/'PREFLIGHT_CPU_SAVED.json')
    assert gpu['status']=='PASS_GPU_AND_RESUME_PENDING_CPU_SAVED_REVIEW' and gpu['source']==cpu['source']==source0 and cpu['status']=='PASS'
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    checked=read_json(HERE/'FEATURE_PROVENANCE.json');assert checked['admissible_arms']==[r['arm'] for r in gpu['records']]
    train_seconds=sum(r['actual32window_sampled_step_seconds'] for r in gpu['benchmark'])*3*1024
    windows=4096+3825+1856;perwindow=max(r['seconds']/128 for r in cpu['records'])
    # Actual full128 double-implementation audit throughput, prediction/feature
    # construction + compressed scores/paths/SHA given additional IO margin.
    audit_seconds=perwindow*windows*3*len(checked['admissible_arms'])
    planning=train_seconds+audit_seconds*1.5+180
    free=shutil.disk_usage(HERE).free;estimated_new_bytes=2*1024**3;assert free>=1.3*estimated_new_bytes
    verify();write_json(HERE/'PRELAUNCH_REVIEW.json',dict(status='PASS',source=source0,source_count=len(read_json(HERE/'source_manifest.json')['files']),
        all_current_GPU_CPU_FD_resume_saved128_source_resource_pass=True,admissible_arms=checked['admissible_arms'],blocked_arms=checked['blocked_arms'],
        train_seconds=train_seconds,independent_saved_audit_seconds=audit_seconds,planning_seconds=planning,ETA_seconds=[.7*planning,1.5*planning],
        no_runtime_budget_gate=True,benchmark_not_production_guarantee=True,disk_free=free,conservative_new_bytes=estimated_new_bytes,
        DA1_same_assistant_not_external=dict(verdict='PASS_LIMITED',issues=['inactive blocks change effective connections','cached token future-curation exposure','71 ambiguous observed-time rows block neighbor arms, do not delete them'])))
    write_json(HERE/'PRODUCTION_SOURCE_LOCK.json',dict(source=source0,at=now(),source_count=len(read_json(HERE/'source_manifest.json')['files']),immutable_after_lock=True))
    resource_report(HERE/'PRELAUNCH_RESOURCE.json',g);print('PASS CURRENT PRELAUNCH',planning,flush=True)
if __name__=='__main__':main()
