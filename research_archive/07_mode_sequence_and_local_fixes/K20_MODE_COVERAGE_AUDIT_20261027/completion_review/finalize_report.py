from pathlib import Path
import sys, subprocess, json, hashlib, datetime, csv
R=Path(__file__).resolve().parents[1]
ROOT=R.parents[1]
sys.path.insert(0,str(R))
from common import verify_sources, verify_done, read, write, sha, now
def main():
    digest=verify_sources()
    assert digest=='d6bad2c267a9d15e2b58bc537ab1d564bbd4a359c383ba93c3576cd46ecc85c7'
    original_man=read(R/'config/sources_manifest.json')
    for part in ('E1_mode_distribution','E2_oracle_coverage','statistics'):verify_done(R/part)
    review=read(R/'completion_review/REVIEW.json')
    assert review['status']=='PASS_CPU_COMPLETE_SAVED_REVIEW'
    assert review['all_single_deletions_scalar_checked']==195540
    assert len(review['bootstrap_full20000_records'])==72
    assert review['bootstrap_max_error']<2e-12
    assert read(R/'resume_v2_state.json')['status']=='SUCCESS_REVIEW_REQUIRED'
    assert read(R/'resume_v2_exit.json')['status']=='SUCCESS_REVIEW_REQUIRED'
    report=read(R/'RESULT_ANALYSIS.json')
    assert report['source_count']==len(original_man['files'])==4237
    assert report['status']=='ANALYZED_NATIVE_MODE_COVERAGE_AUDIT_COMPLETE'
    rows=read(R/'statistics/GROUP_STATISTICS_REVIEWED.json')
    for p,n,g,ng in [('hold',3825,1023,340),('inner',1856,497,166)]:
        rr=next(x for x in rows if x['population']==p and x['variant']=='coarse20' and x['group']=='change')
        assert rr['N']==g and rr['native_good_count']==ng
    validator=Path('<USER_HOME>/.codex/skills/nature-figure/scripts/validate_figure.py')
    qa=[]
    for name in ('plot_final.py','class_intervals_v2.py'):
        p=R/'completion_review'/name
        process=subprocess.run([sys.executable,'-B',str(validator),str(p),'--json'],capture_output=True,text=True,encoding='utf-8')
        assert process.returncode==0,process.stdout+process.stderr
        out=json.loads(process.stdout)
        assert all(x['level']!='FAIL' for x in out['findings'])
        write(R/'completion_review'/('VALIDATOR_'+p.stem+'.json'),out)
        qa.append(dict(source=p.relative_to(R).as_posix(),summary=out['summary'],SHA256=sha(p)))
    figure_files=[]
    for family in ('E1/E1_distribution','E1/E1_class_track_CI','E1/E1_GT_free_cases','E2/E2_coverage_oracle','E2/E2_GT_posthoc_cases'):
        for ext in ('svg','pdf','png'):
            p=R/'figures_final'/(family+'.'+ext)
            assert p.exists() and p.stat().st_size>0
            figure_files.append(dict(file=p.relative_to(R).as_posix(),SHA256=sha(p)))
    write(R/'completion_review/FIGURE_QA_FINAL.json',dict(status='PASS_STATIC_AND_ACTUAL_PNG_VISUAL_REVIEW',at=now(),validators=qa,outputs=figure_files,visual_review_by='same assistant, not external peer review',visual_checks=['all five final PNG families inspected','no clipped labels or overlapping legends','units and denominators visible','bin-count colorbars shown','wrapped labels and minimum5pt source fonts','empty absent-shape cases explicitly retained; no substituted favorable examples'],PNG_dpi=600,vector_editable_text=True,warnings_resolved_by_explanation={'TIFF':'Not final journal submission; PNG600 plus SVG/PDF delivered','LOG_GUARD':'Only native failure J>1 is plotted on log axis; no zeros, epsilon, pseudocount or data deletion'},no_simulated_data=True,all_quantitative_points_no_subsampling=True))
    stages=[]
    for rel in ['audit/PREPARE_RESOURCE.json','audit/PREFLIGHT_RESOURCE.json','E1_mode_distribution/RESOURCE.json','E2_oracle_coverage/RESOURCE.json','statistics/RESOURCE.json','completion_review/RESOURCE.json','figures_final/RESOURCE.json']:
        j=read(R/rel);obs=j['observations']
        stages.append(dict(file=rel,peak_private=max(o['snapshot']['peak_private'] for o in obs),peak_RSS=max(o['snapshot']['peak_rss'] for o in obs),min_available_physical=min(o['snapshot']['physical'] for o in obs),min_available_commit=min(o['snapshot']['commit'] for o in obs),warning_observations=sum(bool(o.get('warnings',[])) for o in obs),pressure_observations=sum(bool(o.get('pressure',False)) for o in obs)))
    write(R/'completion_review/RESOURCE_SUMMARY.json',dict(policy='warning_first_remaining_v3',historical_saved_observations_not_live_poll=True,estimated_CPU_peak=2*1024**3,stages=stages,maximum_single_process_retained_private=max(s['peak_private'] for s in stages),maximum_single_process_retained_RSS=max(s['peak_RSS'] for s in stages),no_OS_pagefile_user_process_changes=True))
    control=dict(status=report['status'],analyzed_at=now(),active_worker=False,no_training=True,new_formal_training=0,automatic_next=False,automation_created=False,OS_task_created=False,no_delete=True,old_PAUSED_untouched=True,successful_channel='resume_v2_state.json + resume_v2_exit.json',original_FAILURE_preserved=True,source_digest=digest,source_count=len(original_man['files']),report='RESULT_ANALYSIS.md')
    write(R/'runtime_control.json',control)
    ledger=read(ROOT/'AUTONOMOUS_GENERATION_CYCLE.json')
    assert ledger['K20_mode_coverage_audit_20261027']['status']==control['status']
    assert ledger['K20_mode_coverage_audit_20261027']['new_formal_training']==0
    # Seal all artifacts including root analysis, figures and explicit correction receipts.
    excluded={'FINAL_PRESERVATION.json'}
    files={p.relative_to(R).as_posix():sha(p) for p in sorted(R.rglob('*')) if p.is_file() and p.name not in excluded and '__pycache__' not in p.parts}
    assert 'RESULT_ANALYSIS.md' in files and 'completion_review/FIGURE_QA_FINAL.json' in files
    assert 'statistics/GROUP_STATISTICS.json' in files and 'statistics/GROUP_STATISTICS_REVIEWED.json' in files
    verify_sources()
    for rel,h in files.items():assert sha(R/rel)==h,rel
    write(R/'completion_review/FINAL_PRESERVATION.json',dict(status='PASS_FINAL_SOURCE_OUTPUT_REPORT_SEAL',at=now(),source_count=len(original_man['files']),source_digest=digest,inherited_source_count=original_man['inherited_count'],all_source_SHA_rechecked=True,all_output_SHA_readback=True,files=files,excluded_self_receipt=True,no_delete=True,original_failures_preserved=True,root_results_explicitly_included=True,controls_sha={'memory':sha(ROOT/'.memory'),'ledger':sha(ROOT/'AUTONOMOUS_GENERATION_CYCLE.json')},no_model_forward_backward_optimizer=True))
    print(json.dumps(dict(status=control['status'],source_count=len(original_man['files']),sealed_artifacts=len(files),source_digest=digest,resource_private=max(s['peak_private'] for s in stages),all_final_checks='PASS'),ensure_ascii=False))
if __name__=='__main__':main()

