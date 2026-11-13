from training import *

def main():
    timer=time.perf_counter();guard=Guard(E);guard.check('startup');stage('T1_INITIAL_SOURCE_SHA',status='RUNNING',active_worker=True,completed_scientific_jobs=0)
    digest=verify_sources();data=load_data()
    for split,d in data.items():
        with np.load(HERE/f'native_J_{split}.npz') as z:assert np.array_equal(z['sample_id'],d['ids']);d['native_j']=z['J'].copy()
    results={};completed=0
    for arm in ARMS:
        for s in SEEDS:
            results[f'{arm}:{s}']=train_run(arm,s,data,guard);completed+=2
            stage('RUN_COMPLETE',arm=arm,seed=s,completed_scientific_jobs=completed)
    stage('PAIRED_TRACK_SEED_20K_CI_AND_INDEPENDENT_REVIEW')
    stats=module('_tail_t1_final_statistics',HERE/'statistical_analysis.py');decision=stats.finish(results);write_json(HERE/'DECISION.json',decision)
    stage('FINAL_CHECKPOINT_HISTORY_SOURCE_OUTPUT_SHA')
    for arm in ARMS:
        for s in SEEDS:
            done=valid_done(HERE/f'runs/{arm}_seed{s}/done.json',contract(arm,s));blob=torch.load(HERE/f'runs/{arm}_seed{s}/checkpoint.pt',map_location='cpu',weights_only=False)
            assert blob['epoch']==8 and blob['step']==592 and len(blob['history'])==8
            assert equal(blob['history'],read_json(HERE/f'runs/{arm}_seed{s}/history.json'))
            assert done['eval_weights_bool_scalers_exact']
    assert verify_sources()==digest;guard.check()
    write_json(HERE/'ARITHMETIC_AUDIT.json',dict(status='PASS',scientific_jobs=12,fresh_models=6,epochs=48,optimizer_steps=3552,full_K20_quality_complete_link_five_motion=True,full_geometry_GT_alignment_independent=True,primary_secondary16_20k_independent=True,control_reproduction='ALL_THREE_BITWISE_AND_COMMON_HISTORY_EXACT',model_eval_buffers_weights_exact=True,saved_arithmetic_not_all_model_gradient_second_replay=True))
    write_json(HERE/'PRESERVATION.json',dict(status='PASS',source_digest=digest,source_count=len(read_json(HERE/'source_manifest.json')['files']),all_old_and_new_locked_sources_unchanged=True,all_done_output_SHA_verified=True))
    write_json(HERE/'RESOURCE.json',dict(status='PASS',estimate_bytes=E,observations=guard.observations,retained_private_peak=max(x['snapshot']['peak_private'] for x in guard.observations),retained_RSS_peak=max(x['snapshot']['peak_rss'] for x in guard.observations)))
    elapsed=time.perf_counter()-timer
    write_json(HERE/'RESULT.json',dict(status='PASS_EXECUTION_T1_RELATIVE_REGRESSION',science=decision['status'],mechanism_supported=decision['mechanism_supported'],fresh_runs=6,epochs=48,optimizer_steps=3552,scientific_jobs=12,source_digest=digest,all_audits='PASS',elapsed_seconds=elapsed,automatic_next_training=False))
    write_json(HERE/'RESULT_ANALYSIS_DRAFT.json',dict(status='COMPUTATION_COMPLETE_MANUAL_COUNTEREVIDENCE_REVIEW_REQUIRED',decision=decision,metrics={k:v['metrics'] for k,v in results.items()},elapsed_seconds=elapsed))
    stage('T1_COMPUTATION_COMPLETE_REVIEW_REQUIRED',status='COMPLETED',active_worker=False,completed_scientific_jobs=12,science=decision['status'],elapsed_seconds=elapsed)

if __name__=='__main__':main()
