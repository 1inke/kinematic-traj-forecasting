"""Publish evidence-grounded finite-series decision, without changing locked science."""
from pathlib import Path
import sys,json,ast
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[2];sys.path.insert(0,str(HERE))
from env import *
OUT=Path(__file__).resolve().parent
def main():
    review=read_json(OUT/'REVIEW.json');assert review['status']=='PASS_COMPLETION_SOURCE_OUTPUT_AND_SAVED_DATA'
    primary=read_json(HERE/'PRIMARY_COMPARISON.json');cost=read_json(HERE/'cost_review/COST_END_TO_END_V3.json');assert primary['signal'] is False
    cohorts=[]
    for seed in SEEDS:
        for split in ('opt','hold','inner'):
            for arm in ARMS:
                folder=HERE/f'populations/seed{seed}/{split}'
                cohorts.append(dict(seed=seed,split=split,arm=arm,metrics=read_json(folder/f'{arm}_metrics.json'),assignment=read_json(folder/f'{arm}_assignment.json')))
    cost_summary=dict(status='COST_NOT_DEPLOYABLE_PARAMETER_BUDGET_AND_CPU_PARITY',actual_benchmark=cost['status'],parameter_counts=cost['parameter_counts'],total_unique_parameters=cost['total_unique_parameters'],budget=305000,budget_pass=False,timing=cost['timing'],per_device_parity={})
    for device in ('cpu','cuda'):
        rows=[x for x in cost['identity32_each_device'] if x['device']==device]
        cost_summary['per_device_parity'][device]=dict(n=len(rows),failures=sum(not x['cache_parity_PASS'] for x in rows),tolerance=2e-5,max_errors={k:max(x['errors'][k] for x in rows) for k in ('past','tokens','native','kin')})
    assert cost_summary['per_device_parity']['cuda']['failures']==0 and cost_summary['per_device_parity']['cpu']['failures']==32
    costfiles=[p for p in (HERE/'cost_review').iterdir() if p.is_file()]
    cost_summary.update(source_SHA={p.name:sha256(p) for p in costfiles},all_old_weights_bool_buffers_exact=cost['allweights_buffers_exact'],no_backward_optimizer=cost['no_backward_optimizer'],CPU_numerical_gate_not_relaxed=True,source_changes='new helper versions only; V2 descriptive parity failure retained, V3 device attribute adapter; no old model source/equation/cache change')
    write_json(HERE/'cost_review/REVIEW.json',cost_summary)
    state=read_json(HERE/'worker_state.json')
    result=dict(status='ANALYZED_NO_PRIMARY_EFFECT',deployment='NOT_DEPLOYABLE',scientific_execution='PASS_EXECUTION_E1_E2_REVIEW_REQUIRED',E0='ANALYZED_PASS',E1_formal_experiments_completed=1,new_fresh_runs=6,epochs=48,optimizer_steps=6144,E2_saved_audit='PASS',E2_actual_cost='MEASURED_GPU_CACHE_PARITY_PASS_CPU_PARITY_BLOCKED_PARAMETER_BUDGET_FAIL',primary=primary,motion_groups=72,motion_items=360,motion_failures=review['motion_failures'],protected_error_failures=review['protected_error_failures'],native_success_lost_failures=review['lost_failures'],source_count=700,extension_count=1,preservation_output_count=837,extra_primary_output_SHA=review['primary_output_SHA'],all_seed_split_groups=cohorts,cost_review=cost_summary,worker=state,halfway=read_json(HERE/'halfway_check.json'),automatic_next_training=False,no_delete=True,baseline_historycycle11_unfinishedgoal_unchanged=True,limitations=['F16/KIN pre-exposure','target disjoint not context disjoint','three seeds and correlated windows','adaptive reuse of inner','legacy neighbor sidecar future-availability curation','CPU raw-cache numerical parity unresolved'],ARS={'DA2':'PASS_LIMITED_NEGATIVE_CLAIM_WITH_POSITIVE_COUNTEREXAMPLES','DA3':'PASS_NO_DEPLOYMENT_OR_GENERALIZATION_OVERCLAIM','same_assistant_not_external_review':True})
    write_json(HERE/'RESULT_ANALYSIS.json',result)
    evidence=[HERE/'RESULT_ANALYSIS.md',HERE/'RESULT_ANALYSIS.json']+[p for p in OUT.iterdir() if p.is_file()]+[p for p in (HERE/'cost_review').iterdir() if p.is_file()]
    write_json(HERE/'COMPLETION_EXTENSION.json',dict(status='PASS_EXPLICIT_NEW_COMPLETION_AND_COST_EVIDENCE_BINDING',base_source_digest=review['source'],prelaunch_extension_SHA=sha256(HERE/'source_extension_prelaunch.json'),files={p.relative_to(HERE).as_posix():sha256(p) for p in evidence},not_replacing_locked_source_manifest=True,scientific_decision='ANALYZED_NO_PRIMARY_EFFECT',no_delete=True))
    print(json.dumps(dict(status=result['status'],primary_signal=primary['signal'],motion_failures=len(review['motion_failures']),protected_error_failures=len(review['protected_error_failures']),lost_failures=len(review['lost_failures']),total_parameters=cost['total_unique_parameters'],cost_status=cost['status'],evidence_files=len(evidence)),ensure_ascii=False))
if __name__=='__main__':main()
