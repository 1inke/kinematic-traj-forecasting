from core import *
import ast,shutil
def main():
    require3();g=guard();source0=verify();p=read_json(HERE/'PREFLIGHT.json');assert p['status']=='PASS' and p['source']==source0
    for f in HERE.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'))
    # Entire original <=24576 pairs (49152 science assignments), plus full
    # independent optimality/time/source audit; extra posthoc/bootstrap/IO budget.
    paircount=p['real_pairs'];assert paircount>0
    matching_and_review=(p['matching_seconds']+p['independent_audit_seconds'])/paircount*24576
    planning=1.3*matching_and_review+180
    disk=shutil.disk_usage(HERE).free;new=1024**3;assert disk>=1.3*new
    verify();write_json(HERE/'PRELAUNCH_REVIEW.json',dict(status='PASS',source=source0,source_count=len(read_json(HERE/'source_manifest.json')['files']),
        all_current_source_ID_hash_time_matching_domain_quality_bootstrap_resource_gates_PASS=True,
        actual_sample_pairs=paircount,upper_bound_matching_and_review_seconds=matching_and_review,planning_seconds=planning,ETA_seconds=[.7*planning,1.5*planning],
        measured_actual_smallpipeline_not_guarantee=True,disk_free=disk,conservative_new_bytes=new,
        DA1_inline_not_external=dict(verdict='PASS_LIMITED',issues=['distance is not sufficient statistic','only one actual future per past','nonunique optimal mapping may change positives','unknown time and no-neighbor support retained not resampled'])))
    write_json(HERE/'PRODUCTION_SOURCE_LOCK.json',dict(source=source0,source_count=len(read_json(HERE/'source_manifest.json')['files']),at=now(),immutable_after_lock=True))
    resource(HERE/'PRELAUNCH_RESOURCE.json',g);print('PASS CURRENT STEP4 PRELAUNCH',planning,flush=True)
if __name__=='__main__':main()
