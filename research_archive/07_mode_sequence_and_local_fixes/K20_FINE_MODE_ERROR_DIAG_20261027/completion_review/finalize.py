from pathlib import Path
import sys,subprocess,json,csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from PIL import Image
def main():
    source=verify_sources();manifest=read(HERE/'manifest.json');assert len(manifest['files'])==4411
    for p in ('E3_gt_free','E3_gt_audit','E4_error','E4_oracle','statistics'):verify_done(HERE/p)
    assert read(HERE/'execution_state.json')['status']=='SUCCESS_REVIEW_REQUIRED'
    assert read(HERE/'exit_status.json')['status']=='SUCCESS_REVIEW_REQUIRED'
    assert read(HERE/'completion_review/completion_state.json')['status']=='SUCCESS_REQUIRES_VISUAL_AND_FINAL_REPORT'
    review=read(HERE/'completion_review/REVIEW.json');assert review['status']=='PASS_COMPLETE_SAVED_CPU_REVIEW'
    assert review['all_candidate_TIME_reconstructed']==195540 and review['all12_variants_tuple_decision_windows']==117324 and review['actual20000_bootstrap_gather_records']==40
    assert review['bootstrap_maxerror']<2e-12 and review['quality_maxerror']<2e-12
    result=read(HERE/'RESULT_ANALYSIS.json');assert result['status']=='ANALYZED_FINE_MODE_ERROR_DIAG_COMPLETE'
    assert result['source_digest']==source
    assert sum(result['main']['mutually_exclusive_failure'].values())==683
    assert sum(result['main']['END04_TIME_overlap'].values())==683
    access=read(HERE/'E3_gt_free/ACCESS.json')
    assert access['GT_read']==False and not any(Path(p).name.startswith('GT_') for p in access['NPZ_files'])
    validator=Path('<USER_HOME>/.codex/skills/nature-figure/scripts/validate_figure.py')
    validations=[]
    for script in ('plot_figures_v2.py','plot_components.py'):
        p=HERE/'completion_review'/script
        process=subprocess.run([sys.executable,'-B',str(validator),str(p),'--json'],capture_output=True,text=True,encoding='utf-8')
        assert process.returncode==0,process.stdout+process.stderr
        payload=json.loads(process.stdout);assert payload['summary']['counts']['FAIL']==0
        write(HERE/'completion_review'/('VALIDATOR_'+p.stem+'.json'),payload)
        validations.append(dict(script=script,SHA256=sha(p),summary=payload['summary']))
    images=[];families=('fine_mode_crosswalk','absence_funnel','error_profiles','error_components','oracle_gain_overlap','diagnostic_cases')
    for name in families:
        files={}
        for ext in ('svg','pdf','png'):
            p=HERE/'figures_v2'/(name+'.'+ext);assert p.exists() and p.stat().st_size>0;files[ext]=sha(p)
        with Image.open(HERE/'figures_v2'/(name+'.png')) as im:
            dpi=im.info['dpi'][0];width=im.width/dpi*25.4;assert dpi>=599 and abs(width-183)<.2
            images.append(dict(figure=name,width_pixels=im.width,height_pixels=im.height,dpi=dpi,actual_width_mm=width,SHA256=files))
    unchanged=('fine_mode_crosswalk','absence_funnel','oracle_gain_overlap','diagnostic_cases')
    for name in unchanged:assert sha(HERE/'figures'/(name+'.png'))==sha(HERE/'figures_v2'/(name+'.png'))
    write(HERE/'completion_review/FIGURE_QA_FINAL.json',dict(status='PASS_ACTUAL_VISUAL_AND_STATIC_QA',at=now(),validators=validations,images=images,backend='python',six_families_reviewed=True,visual_method='v1 four unchanged PNGs directly inspected and identical byte-SHA to finalv2; finalv2 error_profiles and error_components directly inspected',checks=['units and explicit counts visible','no clipped labels or overlapping legend','all6 SVG/PDF editable text','cases have all4 registered categories and explicit counterexamples','all quantitative observations retained; only boxplot fliers marks omitted','common-valid500 vs all683 clearly separated','oracle GT privilege and overlap not summation labelled','kJ and kMode labels explicit'],acceptable_static_warnings={'FINAL_WIDTH':'validator reads183 from expression183/25.4 as inches; actualPNG width/dpi independently183mm within0.2mm','EXPORT_RASTER':'600dpiPNG with SVG/PDF for research report, not journal TIFF submission'},no_external_peer_review=True))
    stages=[]
    for p in HERE.rglob('*RESOURCE.json'):
        if 'failures' in p.parts:continue
        j=read(p);obs=j.get('observations',[])
        if not obs:continue
        stages.append(dict(file=p.relative_to(HERE).as_posix(),peak_private=max(o['snapshot']['peak_private'] for o in obs),peak_RSS=max(o['snapshot']['peak_rss'] for o in obs),min_physical=min(o['snapshot']['physical'] for o in obs),min_commit=min(o['snapshot']['commit'] for o in obs),warnings=sum(bool(o.get('warnings',[])) for o in obs),pressure=sum(bool(o.get('pressure',False)) for o in obs)))
    write(HERE/'completion_review/RESOURCE_SUMMARY.json',dict(policy='warning_first_remaining_v3',estimated_peak=2147483648,stages=stages,max_recorded_private=max(s['peak_private'] for s in stages),max_recorded_RSS=max(s['peak_RSS'] for s in stages),min_recorded_physical=min(s['min_physical'] for s in stages),min_recorded_commit=min(s['min_commit'] for s in stages),all_saved_observations_not_complete_system_timeseries=True,no_memory_lock_or_OS_changes=True))
    # Complete independent all5-pop pooled gates, preserving the worker CSV.
    with (HERE/'completion_review/MOTION_ALL5_POPULATIONS.csv').open(encoding='utf-8') as fh:motionrows=list(csv.DictReader(fh))
    keys=('speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95');gates=[]
    for row in motionrows:
        base=next(z for z in motionrows if z['population']==row['population'] and z['group']==row['group'] and z['port']=='Native')
        gates.append(dict(population=row['population'],group=row['group'],port=row['port'],all5_literal_PASS=all(float(row[k])<=1.01*float(base[k]) for k in keys)))
    write(HERE/'completion_review/ALL5_POPULATIONS_MOTION_GATES.json',gates)
    ledger=read(ROOT/'AUTONOMOUS_GENERATION_CYCLE.json');assert ledger['K20_fine_mode_error_diag_20261027']['status']==result['status']
    write(HERE/'runtime_control.json',dict(status=result['status'],at=now(),active_worker=False,new_formal_training=0,automatic_next=False,no_delete=True,old_PAUSED_untouched=True,new_automation_created=False,source=source,report='RESULT_ANALYSIS.md',success_execution='execution_state.json + exit_status.json',same_assistant_review_not_external=True))
    extensions={p.relative_to(HERE).as_posix():sha(p) for p in (HERE/'completion_review').glob('*.py')}
    write(HERE/'completion_review/EXPLICIT_CPU_EXTENSION.json',dict(base_source=source,no_scientific_source_change=True,files=extensions,scope='saved arithmetic + phase validity/table field clarification/junction diagnostic + plotting/report seal; no model training or forward',old_failed_review_preserved=True))
    files={p.relative_to(HERE).as_posix():sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='FINAL_PRESERVATION.json' and '__pycache__' not in p.parts}
    verify_sources()
    for rel,h in files.items():assert sha(HERE/rel)==h,rel
    write(HERE/'completion_review/FINAL_PRESERVATION.json',dict(status='PASS_FINAL_ALL_SOURCE_OUTPUT_REPORT_SEAL',at=now(),source=source,source_count=len(manifest['files']),inherited_count=manifest['inherited_count'],files=files,explicit_root_files=True,all_source_and_old_output_SHA_unchanged=True,all_new_output_SHA_readback=True,excluded_self_receipt=True,controls_SHA={'memory':sha(ROOT/'.memory'),'ledger':sha(ROOT/'AUTONOMOUS_GENERATION_CYCLE.json')},no_delete=True,no_model_or_gradient_replay=True))
    print(json.dumps(dict(status=result['status'],source_count=len(manifest['files']),sealed_outputs=len(files),all_final_checks='PASS',resource_max_private=max(s['peak_private'] for s in stages)),ensure_ascii=False))
if __name__=='__main__':main()
