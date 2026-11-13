"""Align old contracts, seal new actual dependencies, conservatively admit finite four pilots."""
import ast,math,shutil,sys,time
from geo_common import *
import baseline_factory as bf


def main():
    start=time.perf_counter();p=read_json(RUN/'PREFLIGHT.json');s=read_json(RUN/'PRELAUNCH_SUPPLEMENT.json');cost=read_json(RUN/'COST_ACTUAL_PREDICTOR.json');late=read_json(RUN/'LATE_CONSUMER_PREFLIGHT.json')
    if p['status']!='PASS_ALL4_ACTUAL_GEO_PREFLIGHT' or s['status']!='PASS_GEO_ALL4_FULL_EPOCH_AND_INITIAL128' or late['status']!='PASS_ACTUAL_SAVED_COMMON_SUPPORT_TAIL_AND_READER':raise ValueError('INCOMPLETE_CURRENT_ADMISSION')
    if (RUN/'WORKER_START.json').exists():raise ValueError('DUPLICATE_WORKER')
    for py in RUN.glob('*.py'):ast.parse(py.read_text(encoding='utf-8'),filename=str(py))
    verify(read_json(BASE/'source_manifest.json')['files']);verify(read_json(V1/'source_manifest.json')['files'])
    old=read_json(BASE/'protocol.json');keys=('optimizer','scheduler','EMA','clip','drop_last','selection','normalization','data_neighbor','augmentation','base_loss','oracle_coverage','additional_auxiliary_losses','GT_in_prediction_forward')
    protocol=dict(name='E2E_DGEO_V2_FACTORIAL_20261103',authorization='Direct user execute supplied two direction documents; resolve protocol conflicts in favour of prior actual confirmed settings',
        original_design=sha(DESIGN),prior_reviewed_analysis=sha(ANALYSIS),taylor_matrix=[[1,0,0],[2,1,1],[3,3,0]],taylor_coefficients=[.16,.14,.04],zero_jerk_screen_note='proposed <=1pp rate tolerance is not zero count; strict zero flag separate',actual_alignment=sha(RUN/'AUTHORIZATION_AND_ALIGNMENT.md'),
        original_B0_invariants={k:old[k] for k in keys},arms=list(ARMS),seed=42,batch=64,epochs_per_model=8,updates_per_model=2720,total_updates=10880,
        fit=21719,dev=4753,context_dim=64,mode_dim=64,proj_dim=16,heads=4,coord_output_gain=.1,
        architecture='Fixed centered differential coordinator all4. A Maneuver12D/no geometry; B original scaled independent residual/metric Taylor. Common coordinator tensors paired all4; no scorer residual.',
        geometry_units='Maneuver displacement/sample, causal past4 trend, smooth scalar interaction; Taylor .16/.14/.04 metric residual and unchanged LINEAR interpolation, not safety guarantee. float64 dt.4 evaluation unchanged',
        initial='Fresh seed42 original weights/buffers EXACT B0; new constructor fork_rng seed20261103+seed, original RNG restored',
        paired='Every epoch all RNG states and full batchID order EXACT against B0',
        model_parameters={a:p['reports'][a]['parameters'] for a in ARMS},gradient_gate='Actual initial exactness and finite connected active new tensors and all6 proposal channels by third backward, actual step recorded; no norm/cos target.',
        comparator='Existing matched B0 outputs only, no old trained warmstart or test effects',
        screening='Reuse exact old compare_ud.pilot_screen: overall/changeADE and protectedFDE plus all-group allK20 five tails, zero-positive-excess or small-track NOT_ESTIMABLE, no floor; new allgroupADE/estimable-tail summaries are diagnostic only',
        mechanism_gain_rules={'A':'mean of maneuver-center and full-kin','B':'mean of kin-center and full-maneuver','interaction':'full-maneuver-kin+center'},
        promotion='No automatic promotion or more seeds; original strict and new separately labeled estimable/rate diagnostics; request-only after human review',
        empirical_causality='Factor B bundles basis envelope and scale; V1 module seed differs. No single-seed significance or unique root. Source diagnosis not proved. Signed attention not probability/intent.',
        evaluation_contract=sha(BASE/'C0/EVALUATION_CONTRACT.json'),mechanism_contract=sha(RUN/'MECHANISM_CONTRACT.json'),
        full_workflow_planning_budget_seconds=2700,updated_planning='45min scheduling target with measured-overrun disclosure, not a scientific stop; exact finite4 unchanged; see SCHEDULING_CLARIFICATION',
        resource='warning_first_remaining_v3; 1.3remaining warning, current low-reserve persistent emergency checkpoint stop',no_delete=True,old_PAUSED_untouched=True,no_following_seeds_V_F=True,
        runtime=dict(python=sys.version,torch=bf.torch.__version__))
    write_json(RUN/'protocol.json',protocol)
    deps=set(map(Path,read_json(BASE/'source_manifest.json')['files']))
    deps.update(RUN.glob('*.py'));deps.update(RUN.glob('*.ps1'))
    deps.update([DESIGN,ANALYSIS,UD/'source_manifest.json',UD/'PRESERVATION.json',UD/'analysis_review_20261102/ANALYSIS.json',UD/'analysis_review_20261102/REVIEW.json',BASE/'source_manifest.json',BASE/'B0_FINAL_LOCK.json'])
    deps.update(map(Path,read_json(V1/'source_manifest.json')['files']))
    deps.update([V1/'source_manifest.json',V1/'RESULT_ANALYSIS_REVIEWED.md',V1/'analysis_review_20261103/REVIEW.json',V1/'WORKER_EXIT.json',V1/'GEO_EXTRA_REVIEW.json'])
    for split in ('fit','dev'):
        vd=V1/'populations/DG-GeoDiff'/split/'done.json';receipt=read_json(vd);verify(receipt['files']);deps.update(map(Path,receipt['files']));deps.add(vd)
    deps.update(BASE/'B0_seed42'/n for n in ('initial.pt','final.pt','manifest.json','batch_orders.json','history.csv'))
    deps.update((BASE/'B0_seed42').glob('epoch*.pt'))
    for split in ('fit','dev'):
        receipt=read_json(BASE/'populations'/split/'done.json');verify(receipt['files']);deps.update(map(Path,receipt['files']));deps.add(BASE/'populations'/split/'done.json')
    deps.update(RUN/n for n in ('AUTHORIZATION_AND_ALIGNMENT.md','SCHEDULING_CLARIFICATION.md','README.md','protocol.json','PREFLIGHT.json','PRELAUNCH_SUPPLEMENT.json','LATE_CONSUMER_PREFLIGHT.json','MEMORY_ESTIMATE.json','COST_ACTUAL_PREDICTOR.json','MECHANISM_CONTRACT.json'))
    for arm in ARMS:
        receipt=read_json(RUN/'initial_mechanisms'/arm/'done.json');verify(receipt['files']);deps.update(map(Path,receipt['files']));deps.add(RUN/'initial_mechanisms'/arm/'done.json')
    files=seal(sorted(deps,key=str));sha_elapsed=time.perf_counter()-start
    current=max(0,sum(v['max_seconds']*2720+v['saved128_and_independent_arithmetic_seconds']/128*26472 for v in p['benchmark'].values()))
    current+=sum(v['measurements']['cuda_batch64']['p95_ms']/1000*math.ceil(26472/64) for v in cost['arms'].values())+30
    prior=sum(read_json(V1/'runs'/a/'done.json')['elapsed_seconds'] for a in ('DG-Base','DG-GeoGate','DG-GeoDiff','DG-GeoScore'))
    parts=dict(science_train_fullsave_coreaudit_mechanism=max(current,prior)*1.1,
        actual_numeric_preflight=p['elapsed_seconds'],actual_raw_cost=cost['elapsed_seconds'],actual_pairing_regression=s['elapsed_seconds'],actual_late_saved_consumer=late['elapsed_seconds'],
        current_source_SHA=sha_elapsed,final_SHA_allowance=max(10,sha_elapsed*2),CPU_checkpoint_and_conditional_extra_review=90,unconditional_full_sorted_quantile_extra=90,serial_load_variation_reserve=90)
    planning=sum(parts.values());disk=shutil.disk_usage(RUN);output_estimate=int(2.2*1024**3)
    if disk.free<1.3*output_estimate:raise ValueError('DISK_INSUFFICIENT_NO_DELETE')
    if planning>2700:
        write_json(RUN/'SCHEDULING_TARGET_OVERRUN.json',dict(status='DISCLOSED_ESTIMATE_OVER45_NO_SCIENTIFIC_CHANGE',planning=planning,parts=parts,prior_actual_fullscience=prior,current_maxstep_pipeline=current,clarification=sha(RUN/'SCHEDULING_CLARIFICATION.md')))
    write_json(RUN/'source_manifest.json',dict(status='LOCKED_ACTUAL_DGEO_DEPENDENCIES_BEFORE_SCIENCE',at=now(),files=files,count=len(files)))
    write_json(RUN/'PRELAUNCH_REVIEW.json',dict(status='PASS_AUTHORIZED_ALL4_PILOTS',at=now(),source_count=len(files),planning_seconds=planning,budget_seconds=2700,target_estimate_within45=planning<=2700,time_target_not_scientific_gate=True,parts=parts,
        timing_calibration=dict(current_maxstep_fullpipeline=current,prior_DGeo_actual_science_core_audit=prior,uses_max_not_optimistic_cache=True),disk_free=disk.free,new_output_estimate=output_estimate,
        DA1_same_assistant_not_external=dict(verdict='PASS_WITH_DECLARED_LIMITATIONS',critical='No unresolved source/ID/GTforward/numeric/resume issue; original contract priority applied',
            major='Supplied causal root assertions unproved; Taylor residual not full safety/smoothness; historical V1 constructor seed differs; all corrected in explicit alignment',
            minor='Cos/negativeweight/norm predicted values descriptive; units perstep and scalarFLOP exclusions explicit; zero-relative tail remains not estimable',
            strongest_counterargument='Reachable residual envelope, scale decoupling and optimization can drive results; no single-seed semantic/causal proof'),
        fixture_updates=dict(core=256,fullfit_pairing=1360),scientific_models_started=0,no_new_following_training=True))
    already_spent=p['elapsed_seconds']+cost['elapsed_seconds']+s['elapsed_seconds']+late['elapsed_seconds']+sha_elapsed
    remaining=planning-already_spent
    write_json(RUN/'runtime_control.json',dict(status='PREPARED_DGEO_ALL4',planning_seconds=remaining,full_workflow_planning_seconds=planning,prelaunch_measured_already_spent_seconds=already_spent,remaining_worker_planning_seconds=remaining,active_worker=False,first_progress_query_done=False,first_progress_query_count=0,startup_count=0,no_delete=True,at=now()))
    verify(files);print('DGEO_PRELAUNCH_PASS',len(files),'planning_seconds',planning,'minutes',planning/60,flush=True)


if __name__=='__main__':main()
