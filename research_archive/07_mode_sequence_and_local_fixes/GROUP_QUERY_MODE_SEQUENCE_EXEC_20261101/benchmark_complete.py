"""Real saved pipeline scaling, all20k bootstrap replay and inherited SHA IO."""
from common import *
from train_stage import predict
from evaluate_saved import save_population
from statistics_fixed import primary
def main():
    setup();d=data();fit=d['FIT'];g=Guard(4*2**30);timings=[];start=time.perf_counter()
    m=new_model(42,3,'cpu');p=torch.load(HERE/'preflight/Q3/interrupted.pt',map_location='cpu',weights_only=False);m.load_state_dict(p['model'])
    target=HERE/'benchmark_saved';target.mkdir(exist_ok=False)
    # Existing real preflight state, no new benchmark model selected by quality.
    for n in (128,512,1024):
        sub={k:v[:n] for k,v in fit.items()};paths=predict(m,sub['x']);t=time.perf_counter();save_population(target/str(n),paths,sub,g)
        timings.append(dict(N=n,seconds=time.perf_counter()-t));del paths
    n=len(d['inner']['ids']);cover=np.stack([((np.arange(n)+s)%3)==0 for s in range(3)]);t=time.perf_counter()
    stat=primary(cover,np.roll(cover,1,axis=1),d['inner']['groups'],d['inner']['track'],HERE/'benchmark_bootstrap_draws.npz');bst=time.perf_counter()-t
    t=time.perf_counter();verify(read(HERE/'INPUT_SOURCE_LOCK.json')['files'],g.check);hashsec=time.perf_counter()-t
    old=read(HERE/'BENCHMARK_PARTIAL.json');tr=old['training'];train=1776*(tr['Q1']['mean']+tr['Q3']['mean'])
    # Upper envelope of observed saved128/512/1024 per-window rates includes
    # compression, all main clusters twice, all group quality and pooled physics.
    rate=max(x['seconds']/x['N'] for x in timings+old['saved_pipeline128'])
    windows=14*(18909+1856)/2 #12 model jobs +2 native jobs, same full population total
    saved=windows*rate
    # CPU completion re-read/full cluster second algorithm is bounded by a full
    # additional pipeline pass; 4 inherited hash passes and bootstrap+IO explicit.
    review=saved;hashwork=4*hashsec;misc=90.
    nominal=train+saved+review+hashwork+2*bst+misc;conservative=1.2*nominal
    rec=dict(status='PASS_BUDGET' if conservative<=1800 else 'SKIPPED_BUDGET',training_seconds=train,saved_population_seconds=saved,independent_completion_seconds=review,
        source_hash_seconds=hashwork,one_source_pass_seconds=hashsec,bootstrap_twice_seconds=2*bst,IO_export_diagnostics_margin=misc,nominal_seconds=nominal,
        conservative_full_seconds=conservative,ETA_seconds=[.7*conservative,1.5*conservative],real_pipeline=timings,real_bootstrap_seconds=bst,
        real_bootstrap_independent_error=stat['independent_all_draw_maxerror'],scientific_windows_main_including_native=windows,rate_upper_envelope=rate,
        throughput=tr,implementation_person_time_not_runtime=True)
    write(HERE/'BENCHMARK.json',rec);write(HERE/'BENCHMARK_COMPLETE_RESOURCE.json',dict(observations=g.observations,own_private_peak=max(x['snapshot']['peak_private'] for x in g.observations)))
    pfpeak=read(HERE/'PREFLIGHT_RESOURCE.json')['own_private_peak'];cpupeak=max(x['snapshot']['peak_private'] for x in g.observations)
    # Q3 exact observed preflight private + full FIT all-group output + full
    # double evaluator arrays and serialization/allocator reserve. Not available-driven.
    payload=18909*3*20*12*2*4;extra=18909*20*12*2*8*3+256*2**20
    E=int(pfpeak+payload+extra);cpuE=int(cpupeak+payload+extra)
    write(HERE/'MEMORY_ESTIMATE.json',dict(policy='warning_first_remaining_v3',GPU_E=E,CPU_E=cpuE,inventory=dict(actual_GPU_pref_private=pfpeak,actual_CPU_pipeline_private=cpupeak,fullFIT_threegroup_float32_output=payload,double_evaluator_three_path_arrays=extra-256*2**20,allocator_serialization_reserve=256*2**20),no_fixed6GiB=True,no_OS_or_pagefile_or_userkill=True))
    if conservative>1800:
        write(HERE/'BUDGET_STOP.json',dict(status='SKIPPED_BUDGET',benchmark=rec,no_scientific_models_started=True,M_E='NOT_TRIGGERED_BUDGET'));print('SKIPPED_BUDGET',conservative,flush=True);return
    write(HERE/'PREFLIGHT.json',dict(status='PASS_ACTUAL_COMPLETE_PREFLIGHT',GPU='PREFLIGHT_GPU.json',CPU='IDENTITY_STATISTICS_AUDIT.json',saved='actual both128 plus128/512/1024 complete independent clusters+five+junction',readers='actual sourceIDstatgroupinitialordercheckpointoutput reject',full_budget='PASS',original_contract_unchanged=True,scientific_updates=0))
    print(json.dumps(rec),flush=True)
if __name__=='__main__':main()
