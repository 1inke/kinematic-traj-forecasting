"""Actual dependency lock and prelaunch admission, no scientific training."""
import ast
import shutil
import sys
import time
from common import *
from baseline_factory import *


def main():
    started=time.perf_counter()
    p=read_json(HERE/'PREFLIGHT.json');cost=read_json(HERE/'COST_ACTUAL_PREDICTOR.json')
    if p['status']!='PASS_B0_ACTUAL_PREFLIGHT' or not p['budget_admitted']:raise ValueError('B0_NOT_PREFLIGHT_ADMITTED')
    if read_json(HERE/'C0_completion_review/REVIEW.json')['status']!='PASS_C0_INDEPENDENT_SPLIT_TARGET_MOTION_AND_OFFSET_ALIAS_CHECK':
        raise ValueError('C0_REVIEW_INCOMPLETE')
    for py in HERE.glob('*.py'):ast.parse(py.read_text(encoding='utf-8'),filename=str(py))
    if (HERE/'WORKER_START.json').exists():raise ValueError('WORKER_ALREADY_EXISTS')
    c0=read_json(HERE/'C0/C0_RESULT.json');verify(c0['outputs'])
    c0sources=read_json(HERE/'C0/SOURCE_MANIFEST.json')['files'];verify(c0sources)
    protocol=dict(name='E2E_RESET_C0_B0_V1',authorization='2026-11-02 user execute supplied 02 design, repeated path and continue',
        stage='C0_COMPLETE_THEN_ONE_FRESH_B0',seed=42,epochs=8,batch=64,fit=21719,dev=4753,
        official_test='SHA_AND_SOURCE_INVENTORY_ONLY_NO_GT_PARSE',
        model='Existing FrontEncoderASTRA(f16_temporal_social_cross) + h3_anchor_proposal',
        anchors=[4,8,12],K=20,use_refiner=False,whole_model_parameters=295474,trainable_parameters=294897,
        initialization='All fresh seed42, no checkpoint or tokens. Temporary discarded linear constructor avoids unrelated PCA asset load.',
        normalization='Past-only focal heading lag3 speed>.03 else[1,0], peragent eight-observation mean; no fitted input stats',
        data_neighbor='Full8 observed only, closest t0 stableID tie, focal+8, no neighbor future-life mask',
        augmentation={'reflection':True,'rotation':False,'speed':[.9,1.1],'probability':.5,'curvature':0},
        optimizer={'name':'AdamW','lr':.00025,'weight_decay':.0005,'betas':[.9,.999],'eps':1e-8,'foreach':False},
        scheduler='monotonic cosine; warmup1epoch(340updates); total2720; per-update step',
        EMA=.995,clip=5.,drop_last=False,selection='FIXED_FINAL_EPOCH8_EMA; no dev best checkpoint or earlystop',
        base_loss='Existing locked ModeQueryLoss smoothL1 exp weights1..2, winner ADE+.25FDE, classification.5*(softCE temp.5 gapscale>=.05+.25hardCE), endpoint.25, endpoint190pair relu(.4-distance)linear*.05, shortfirst2smoothL1*.2',
        oracle_coverage={'weight':.20,'responsibility':'detach softmax(-(ADE+.25FDE)/.10)','normalized':False,'warmup':0},
        additional_auxiliary_losses=0,GT_in_prediction_forward=False,
        training_side_reference=sha(HERE/'C0/MOTION_REFERENCE.json'),evaluation_contract=sha(HERE/'C0/EVALUATION_CONTRACT.json'),
        later_U_D_V_F='NOT_STARTED_BY_THIS_WORKER',old_Q_gates='UNCHANGED',baseline_promoted=False,
        resource='Latest warning_first_remaining_v3, measured E, no fixed6GiB hard gate, no OS/pagefile/userkill',
        no_delete=True,old_PAUSED_untouched=True,full_conservative_budget_seconds=1800,
        statistical_inference='Single seed development; no significance or fresh blind benchmark claim',
        runtime_versions={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__},
        preflight_repairs=[{'issue':'motion history broadcast1->20 interface','source_preserved':'failures/preflight_v1','full_v2_recheck':'PASS'},
                           {'issue':'GraphStats dataclass serialization','source_preserved':'failures/cost_v1','recheck':'Full CPU/GPU50warm500 trace rerun, PASS'}])
    write_json(HERE/'protocol.json',protocol)
    dependencies=set(map(Path,c0sources))
    dependencies.update(HERE.glob('*.py'));dependencies.update(HERE.glob('*.ps1'))
    dependencies.update(HERE/'C0'/n for n in ('fit.npz','dev.npz','C0_RESULT.json','DATA_LINEAGE.json','SPLIT_AUDIT.json','CAUSAL_NEIGHBOR_AUDIT.json','MOTION_REFERENCE.json','EVALUATION_CONTRACT.json'))
    dependencies.update(HERE/n for n in ('protocol.json','README.md','RESOLVED_CONFIG.json','PREFLIGHT.json','MEMORY_ESTIMATE.json','COST_ACTUAL_PREDICTOR.json','C0_completion_review/REVIEW.json'))
    dependencies.add(CONFIG)
    dependencies.add(ROOT/'src/f16/eval.py')
    dependencies.add(ROOT/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py')
    # Actual imported local model implementation modules, not the whole installed environment.
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if not value:continue
        path=Path(value).resolve()
        if path.suffix=='.py' and (ASTRA in path.parents or ROOT/'FRONT_ENCODER_SERIES' in path.parents or ROOT/'HIERARCHICAL_DECODER_SERIES' in path.parents):
            dependencies.add(path)
    files=seal(sorted(dependencies,key=str))
    write_json(HERE/'source_manifest.json',{'status':'LOCKED_ACTUAL_DEPENDENCIES_ONLY_BEFORE_B0',
        'at':now(),'files':files,'count':len(files),'no_old_weights_or_tokens':True})
    sha_time=time.perf_counter()-started
    train_max=p['benchmark']['max_step_seconds']*2720
    infer=cost['latency']['cuda_batch64']['p95_ms']/1000*math.ceil(26472/64)
    save=p['benchmark']['saved128_double_implementation_seconds']/128*26472
    measured_parts={'train_max_step_extrapolation':train_max,'population_forward_transfer':infer,
        'save_plus_independent_full_quality_once':save,'source_prepare_SHA_measured':sha_time,
        'final_SHA_allowance':max(5.,sha_time*2),'checkpoint_IO_and_CPU_metadata_review_allowance':90.,
        'actual_cost_measurement':cost['elapsed_seconds'],'actual_numerical_preflight':p['elapsed_seconds'],
        'additional_fixed_load_and_timing_variation_reserve':180.}
    planning=max(p['planning_seconds'],sum(measured_parts.values())*1.25)
    if planning>1800:
        write_json(HERE/'PRELAUNCH_REVIEW.json',{'status':'SKIPPED_BUDGET','planning_seconds':planning,'parts':measured_parts})
        raise ValueError('BUDGET_GREATER_THAN1800')
    disk=shutil.disk_usage(HERE)
    if disk.free<1.3*1024**3:raise ValueError('INSUFFICIENT_CURRENT_DISK_NO_DELETE')
    write_json(HERE/'PRELAUNCH_REVIEW.json',{'status':'PASS_C0_B0_ALLOW_SOLE_WORKER','at':now(),
        'source_count':len(files),'planning_seconds':planning,'parts':measured_parts,'budget_seconds':1800,
        'disk_free':disk.free,'conservative_new_output_bytes':1024**3,
        'DA1_same_assistant_not_external':[
            'Sister recording physical footage identity not fully independently verified. Explicit UNKNOWN qualification retained, no claim of whole-scene context independence.',
            'Existing registered unused/fixed modules count toward total; zero first-step upstream gradients are expected with zero anchor/score projections, eight phased batches verify active encoder/anchor/score connections.',
            'One seed8epoch is only a pipeline/development comparator, not fully converged or statistically positive. No old native matched output on this split, legacy shadow NOT_COMPARABLE.',
            'Attention contractions and actual operator list included; scalar nonlinear/NumPy preprocessing excluded from FLOP arithmetic but included in raw-observation pipeline latency. Initial fixed cost population not a universal density bound.'],
        'repairs_before_production_preserved':True,'full_saved_audit_preflight':'PASS',
        'next':'Sole XML0 OS worker, one startup; do not spawn U/D/V/formal models.'})
    write_json(HERE/'runtime_control.json',{'status':'PREPARED_C0_COMPLETE_B0_ADMITTED','planning_seconds':planning,
        'first_progress_query_done':False,'first_progress_query_count':0,'startup_count':0,
        'no_old_task_queries':True,'no_delete':True,'at':now()})
    verify(files)
    print('PRELAUNCH_PASS',len(files),'planning_seconds',planning,'disk_free',disk.free,flush=True)


if __name__=='__main__':main()
