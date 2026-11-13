"""Seal approved contract and actual dependencies; admit conservative finite pilot budget."""
import ast
import math
import shutil
import sys
import time
from ud_common import *
import baseline_factory as bf


def main():
    start=time.perf_counter()
    p=read_json(RUN/'PREFLIGHT.json');s=read_json(RUN/'PRELAUNCH_SUPPLEMENT.json');cost=read_json(RUN/'COST_ACTUAL_PREDICTOR.json')
    if p['status']!='PASS_ALL4_ACTUAL_UD_PREFLIGHT' or s['status']!='PASS_ACTUAL_ALL4_PAIRED_FULL_EPOCH_AND_MECHANISM_CONSUMERS':raise ValueError('INCOMPLETE_CURRENT_PREFLIGHT')
    if (RUN/'WORKER_START.json').exists():raise ValueError('DUPLICATE_WORKER')
    verify(read_json(BASE/'source_manifest.json')['files']);verify(read_json(BASE/'B0_FINAL_LOCK.json')['files'])
    for py in RUN.glob('*.py'):ast.parse(py.read_text(encoding='utf-8'),filename=str(py))
    inherited=read_json(BASE/'protocol.json')
    keys=('optimizer','scheduler','EMA','clip','drop_last','selection','normalization','data_neighbor','augmentation','base_loss','oracle_coverage','additional_auxiliary_losses','GT_in_prediction_forward')
    protocol=dict(name='E2E_UD_AUTHORIZED_V1_20261102',authorization='Latest direct human agrees ARS narrow amendments, consistent with actual B0',
        approved_amendment=sha(RUN/'AUTHORIZED_AMENDMENT.md'),original_design=sha(DESIGN),
        arms=list(ARMS),seed=42,epochs_per_model=8,steps_per_model=2720,total_scientific_updates=10880,
        fit=21719,dev=4753,context_dimension=64,mode_dimension=64,batch=64,
        inherited_B0_invariants={k:inherited[k] for k in keys},
        new_module_init_RNG='fork_rng seed20261102+seed, restores all original RNG; whole-model fresh B0 seed42 shared params EXACT',
        shared_random_stream='Every epoch original Torch CPU/CUDA/Python/NumPy/loader RNG and all batchIDs checked against saved B0; no old trained warmstart',
        updated_gradient_gate='Initial exact paths/score; Step2 output projection gradient>0; Step3 every internal parameter tensor gradient>0; actual Step1 zero recorded',
        model_parameters={a:p['reports'][a]['parameters'] for a in ARMS},
        D_over305000='Approved explicit cost-target overage; do not shrink architecture',
        U='target observed8 6D v/a/unitheading; 32D GRU vs gated innovation, LN(m8-m1), zero 32->64 output',
        D='all original agent contexts + K20 embeddings before shared original mode_mlp; 32D4head single vs A1-.5A2; same gate and zero32->64 output',
        main_candidates=['U-Delta','D-DiffGate'],no_U_D_stacking=True,no_new_loss_or_scorer=True,
        evaluation_contract=sha(BASE/'C0/EVALUATION_CONTRACT.json'),mechanism_contract=sha(RUN/'MECHANISM_CONTRACT.json'),
        screening='Exact full-precision B0 comparator; overall or change ADE decrease and other<=1%harm, protected overall/stable/transition FDE<=1%harm, full all-group five allK tail +1pp/risk magnitude<=1%; zero/lowtrack reference NOT_ESTIMABLE_NO_FLOOR',
        support_audit='Every rescue/lost ID, original-good denominator and Jmargin damage, conditional score recall; loss scalar not same as final-population metric',
        single_seed_no_CI=True,conditional_next='Own objective gate/preflight/budget review. No subjective fallback or autoV/F by this worker.',
        resource='warning_first_remaining_v3',no_delete=True,old_PAUSED_untouched=True,
        full_conservative_budget_seconds=1800,baseline_historycycle11_goal_unchanged=True,
        runtime=dict(python=sys.version,torch=bf.torch.__version__))
    write_json(RUN/'protocol.json',protocol)
    deps=set(map(Path,read_json(BASE/'source_manifest.json')['files']))
    deps.update(RUN.glob('*.py'));deps.update(RUN.glob('*.ps1'))
    deps.update(PROTOTYPE/n for n in ('ud_common.py','decoder_adapter.py','motion_state_update.py','mode_interaction.py','alignment_check/ACTUAL_GRADIENT_CHECK.json'))
    deps.add(DESIGN);deps.add(BASE/'source_manifest.json');deps.add(BASE/'B0_FINAL_LOCK.json')
    deps.update(BASE/'B0_seed42'/n for n in ('initial.pt','final.pt','manifest.json','batch_orders.json','history.csv'))
    deps.update((BASE/'B0_seed42').glob('epoch*.pt'))
    for split in ('fit','dev'):
        d=read_json(BASE/'populations'/split/'done.json');verify(d['files']);deps.update(map(Path,d['files']));deps.add(BASE/'populations'/split/'done.json')
    deps.update(RUN/n for n in ('AUTHORIZED_AMENDMENT.md','README.md','protocol.json','PREFLIGHT.json','PRELAUNCH_SUPPLEMENT.json','MEMORY_ESTIMATE.json','COST_ACTUAL_PREDICTOR.json','MECHANISM_CONTRACT.json'))
    deps.add(ROOT/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py')
    files=seal(sorted(deps,key=str));sha_elapsed=time.perf_counter()-start
    parts=dict(train_all4_max_actual_step=sum(v['max_seconds']*2720 for v in p['benchmark'].values()),
        raw_forward_all4=sum(v['measurements']['cuda_batch64']['p95_ms']/1000*math.ceil(26472/64) for v in cost['arms'].values()),
        save_full_population_dual_arithmetic=sum(v['saved128_and_independent_arithmetic_seconds']/128*26472 for v in p['benchmark'].values()),
        actual_all4_preflight=p['elapsed_seconds'],actual_all4_raw_cost=cost['elapsed_seconds'],
        actual_full_epoch_and_mechanism_regression=s['elapsed_seconds'],actual_source_SHA=sha_elapsed,
        final_SHA_allowance=max(10,2*sha_elapsed),checkpoint_IO_and_CPU_metadata_review=80,
        full_fit_mechanisms_and_B0_perturb_forward_allowance=45,serial_loads_and_variation_reserve=60)
    planning=sum(parts.values())
    disk=shutil.disk_usage(RUN);output_estimate=int(2.2*1024**3)
    if disk.free<1.3*output_estimate:raise ValueError('DISK_INSUFFICIENT_NO_DELETE')
    if planning>1800:
        write_json(RUN/'PRELAUNCH_BUDGET_BLOCK.json',dict(status='BLOCKED_CONSERVATIVE_SERIES_GT1800',planning_seconds=planning,parts=parts))
        raise ValueError('CONSERVATIVE_BUDGET_GT1800_NO_SCIENCE_LAUNCH')
    write_json(RUN/'source_manifest.json',dict(status='LOCKED_AUTHORIZED_ALL4_ACTUAL_DEPENDENCIES_BEFORE_SCIENCE',at=now(),files=files,count=len(files)))
    write_json(RUN/'PRELAUNCH_REVIEW.json',dict(status='PASS_AUTHORIZED_ALL4_PILOTS',at=now(),source_count=len(files),
        planning_seconds=planning,parts=parts,budget_seconds=1800,disk_free=disk.free,new_output_estimate=output_estimate,
        DA1_same_assistant_not_external=dict(verdict='PASS_WITH_ACKNOWLEDGED_LIMITATIONS',critical_resolved='Actual64/clip5/three-stage/D cost disclosure explicitly authorized and independently evidenced',
            major='Extra capacity/optimization and specific mechanisms confounded; paired GRU/Gate controls required. Single seed not significance; future-only maneuvers separate.',
            minor='Initial fixed-density cost not universal; scalar nonlinear FLOPs excluded but latency included. Positive-excess zero cannot yield relative noninferiority.',
            strongest_counterargument='A successful fresh extra branch may help optimization without validating innovation/differential attention or generalization.'),
        repairs_preserved=['failures/import_namespace_v1','failures/cost_factory_tuple_v1'],
        fixture_updates=dict(previous_alignment=12,core_preflight=256,current_full_epoch_prelaunch=1360),science_models_started=0,
        no_V_F_or_more_seeds_by_current_worker=True))
    write_json(RUN/'runtime_control.json',dict(status='PREPARED_ALL4_AUTHORIZED',planning_seconds=planning,active_worker=False,
        first_progress_query_done=False,first_progress_query_count=0,startup_count=0,no_delete=True,at=now()))
    verify(files)
    print('ALL4_PRELAUNCH_PASS',len(files),'planning',planning,'minutes',planning/60,'disk_free',disk.free,flush=True)


if __name__=='__main__':main()
