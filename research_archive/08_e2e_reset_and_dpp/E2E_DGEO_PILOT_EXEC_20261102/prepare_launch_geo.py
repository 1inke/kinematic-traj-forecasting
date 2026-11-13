"""Align old contracts, seal new actual dependencies, conservatively admit finite four pilots."""
import ast,math,shutil,sys,time
from geo_common import *
import baseline_factory as bf


def main():
    start=time.perf_counter();p=read_json(RUN/'PREFLIGHT.json');s=read_json(RUN/'PRELAUNCH_SUPPLEMENT.json');cost=read_json(RUN/'COST_ACTUAL_PREDICTOR.json');late=read_json(RUN/'LATE_CONSUMER_PREFLIGHT.json')
    if p['status']!='PASS_ALL4_ACTUAL_GEO_PREFLIGHT' or s['status']!='PASS_GEO_ALL4_FULL_EPOCH_AND_INITIAL128' or late['status']!='PASS_ACTUAL_SAVED_COMMON_SUPPORT_TAIL_AND_READER':raise ValueError('INCOMPLETE_CURRENT_ADMISSION')
    if (RUN/'WORKER_START.json').exists():raise ValueError('DUPLICATE_WORKER')
    for py in RUN.glob('*.py'):ast.parse(py.read_text(encoding='utf-8'),filename=str(py))
    verify(read_json(BASE/'source_manifest.json')['files']);verify(read_json(UD/'source_manifest.json')['files'])
    old=read_json(BASE/'protocol.json');keys=('optimizer','scheduler','EMA','clip','drop_last','selection','normalization','data_neighbor','augmentation','base_loss','oracle_coverage','additional_auxiliary_losses','GT_in_prediction_forward')
    protocol=dict(name='E2E_DGEO_EXEC_V1_20261102',authorization='Direct user execute supplied two direction documents; resolve protocol conflicts in favour of prior actual confirmed settings',
        original_design=sha(DESIGN),supplied_analysis=sha(ANALYSIS),actual_alignment=sha(RUN/'AUTHORIZATION_AND_ALIGNMENT.md'),
        original_B0_invariants={k:old[k] for k in keys},arms=list(ARMS),seed=42,batch=64,epochs_per_model=8,updates_per_model=2720,total_updates=10880,
        fit=21719,dev=4753,context_dim=64,mode_dim=64,proj_dim=16,heads=4,coord_output_gain=.1,
        architecture='Supplied geo_modules reference: target a0 only, DGBase no geo; GeoGate adds12D+QK centering; GeoDiff A1-.5A2; GeoScore adds exact1418 post-relational scorer. Original B0 children/params/buffers preserved.',
        geometry_units='Coordinates meters, u/junc/turn per-source-sample displacement, radius meter and angle/pi; no GT; tail evaluation dt.4 world float64 unchanged',
        initial='Fresh seed42 original weights/buffers EXACT B0; new constructor fork_rng seed20261102+seed, original RNG restored',
        paired='Every epoch all RNG states and full batchID order EXACT against B0',
        model_parameters={a:p['reports'][a]['parameters'] for a in ARMS},gradient_gate='Actual initial exactness and finite connected new tensors by third backward per prior; all4 actually connected on second. Advertised numeric norm/cos/negative% not hard gates.',
        comparator='Existing matched B0 outputs only, no old trained warmstart or test effects',
        screening='Reuse exact old compare_ud.pilot_screen: overall/changeADE and protectedFDE plus all-group allK20 five tails, zero-positive-excess or small-track NOT_ESTIMABLE, no floor; new allgroupADE/estimable-tail summaries are diagnostic only',
        mechanism_gain_rules={'GeoGate_vs_Base':'overall ADE lower and change ADE <=1.01 control', 'GeoDiff_vs_GeoGate':'overall ADE lower and change ADE <=1.01 control','GeoScore_vs_GeoGate':'common-covered change Top1 numerator higher, overall/change coverage nonlower and their ADEFDE<=1.01 control'},
        promotion='Full prior screen AND prospective adjacent point gain; only eligible to request own three-seed admission, never automatic',
        empirical_causality='Base engineering bundle, Geo feature+centering bundle, same nominal capacity not equal effective capacity. Signed attention not probability/intent. One seed not significance.',
        evaluation_contract=sha(BASE/'C0/EVALUATION_CONTRACT.json'),mechanism_contract=sha(RUN/'MECHANISM_CONTRACT.json'),
        full_workflow_planning_budget_seconds=2700,updated_planning='45min explicitly supplied current direction; oldUD30min deviation unchanged; no scientific gate relaxed',
        resource='warning_first_remaining_v3; 1.3remaining warning, current low-reserve persistent emergency checkpoint stop',no_delete=True,old_PAUSED_untouched=True,no_following_seeds_V_F=True,
        runtime=dict(python=sys.version,torch=bf.torch.__version__))
    write_json(RUN/'protocol.json',protocol)
    deps=set(map(Path,read_json(BASE/'source_manifest.json')['files']))
    deps.update(RUN.glob('*.py'));deps.update(RUN.glob('*.ps1'))
    deps.update([DESIGN,ANALYSIS,UD/'source_manifest.json',UD/'PRESERVATION.json',UD/'analysis_review_20261102/ANALYSIS.json',UD/'analysis_review_20261102/REVIEW.json',BASE/'source_manifest.json',BASE/'B0_FINAL_LOCK.json'])
    deps.update(map(Path,read_json(UD/'source_manifest.json')['files']))
    deps.update(BASE/'B0_seed42'/n for n in ('initial.pt','final.pt','manifest.json','batch_orders.json','history.csv'))
    deps.update((BASE/'B0_seed42').glob('epoch*.pt'))
    for split in ('fit','dev'):
        receipt=read_json(BASE/'populations'/split/'done.json');verify(receipt['files']);deps.update(map(Path,receipt['files']));deps.add(BASE/'populations'/split/'done.json')
    deps.update(RUN/n for n in ('AUTHORIZATION_AND_ALIGNMENT.md','README.md','protocol.json','PREFLIGHT.json','PRELAUNCH_SUPPLEMENT.json','LATE_CONSUMER_PREFLIGHT.json','MEMORY_ESTIMATE.json','COST_ACTUAL_PREDICTOR.json','MECHANISM_CONTRACT.json'))
    for arm in ARMS:
        receipt=read_json(RUN/'initial_mechanisms'/arm/'done.json');verify(receipt['files']);deps.update(map(Path,receipt['files']));deps.add(RUN/'initial_mechanisms'/arm/'done.json')
    files=seal(sorted(deps,key=str));sha_elapsed=time.perf_counter()-start
    current=max(0,sum(v['max_seconds']*2720+v['saved128_and_independent_arithmetic_seconds']/128*26472 for v in p['benchmark'].values()))
    current+=sum(v['measurements']['cuda_batch64']['p95_ms']/1000*math.ceil(26472/64) for v in cost['arms'].values())+30
    prior=sum(read_json(UD/'runs'/a/'done.json')['elapsed_seconds'] for a in ('U-GRU','U-Delta','D-Gate','D-DiffGate'))
    parts=dict(science_train_fullsave_coreaudit_mechanism=max(current,prior)*1.1,
        actual_numeric_preflight=p['elapsed_seconds'],actual_raw_cost=cost['elapsed_seconds'],actual_pairing_regression=s['elapsed_seconds'],actual_late_saved_consumer=late['elapsed_seconds'],
        current_source_SHA=sha_elapsed,final_SHA_allowance=max(10,sha_elapsed*2),CPU_checkpoint_and_conditional_extra_review=90,unconditional_full_sorted_quantile_extra=90,serial_load_variation_reserve=90)
    planning=sum(parts.values());disk=shutil.disk_usage(RUN);output_estimate=int(2.2*1024**3)
    if disk.free<1.3*output_estimate:raise ValueError('DISK_INSUFFICIENT_NO_DELETE')
    if planning>2700:
        write_json(RUN/'BUDGET_BLOCK.json',dict(planning=planning,parts=parts,prior_actual_fullscience=prior,current_maxstep_pipeline=current))
        raise ValueError('FULL_CONSERVATIVE_DGEO_BUDGET_GT2700')
    write_json(RUN/'source_manifest.json',dict(status='LOCKED_ACTUAL_DGEO_DEPENDENCIES_BEFORE_SCIENCE',at=now(),files=files,count=len(files)))
    write_json(RUN/'PRELAUNCH_REVIEW.json',dict(status='PASS_AUTHORIZED_ALL4_PILOTS',at=now(),source_count=len(files),planning_seconds=planning,budget_seconds=2700,parts=parts,
        timing_calibration=dict(current_maxstep_fullpipeline=current,prior_UD_actual_science_core_audit=prior,uses_max_not_optimistic_cache=True),disk_free=disk.free,new_output_estimate=output_estimate,
        DA1_same_assistant_not_external=dict(verdict='PASS_WITH_DECLARED_LIMITATIONS',critical='No unresolved source/ID/GTforward/numeric/resume issue; original contract priority applied',
            major='Input report contains mismatched tables and excessive causal/observability claims, not adopted as facts; geometry and centering are bundled, scorer only partially decoupled',
            minor='Cos/negativeweight/norm predicted values descriptive; units perstep and scalarFLOP exclusions explicit; zero-relative tail remains not estimable',
            strongest_counterargument='New benefit could be capacity/optimization or support redistribution, not mechanistic geometry/negative-attention proof'),
        fixture_updates=dict(core=256,fullfit_pairing=1360),scientific_models_started=0,no_new_following_training=True))
    write_json(RUN/'runtime_control.json',dict(status='PREPARED_DGEO_ALL4',planning_seconds=planning,active_worker=False,first_progress_query_done=False,first_progress_query_count=0,startup_count=0,no_delete=True,at=now()))
    verify(files);print('DGEO_PRELAUNCH_PASS',len(files),'planning_seconds',planning,'minutes',planning/60,flush=True)


if __name__=='__main__':main()
