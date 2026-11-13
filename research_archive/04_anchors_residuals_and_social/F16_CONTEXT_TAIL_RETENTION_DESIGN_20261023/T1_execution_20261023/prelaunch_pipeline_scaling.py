"""Post-lock explicit CPU saved-path timing extension; no model, optimizer or science change."""
from evaluate_saved import *

def main():
    guard=Guard(E);guard.check('startup');digest=verify_sources();data=load_data()['FIT']
    with np.load(OLD/'runs/A_seed42/predictions_FIT.npz') as z:
        assert np.array_equal(z['sample_id'],data['ids']);paths=z['paths'].copy()
    order=sorted(range(len(data['ids'])),key=lambda i:hashlib.sha256(f'tail-20261023::{int(data["ids"][i])}'.encode()).digest())
    records=[]
    for n in (128,512,1024):
        ix=np.array(order[:n]);sub={k:v[ix] for k,v in data.items()};dest=HERE/f'prelaunch_pipeline_scaling/n{n}';dest.mkdir(parents=True,exist_ok=True)
        start=time.perf_counter();save_evaluate(dest,'FIT',paths[ix],sub,guard);elapsed=time.perf_counter()-start
        records.append(dict(n=n,seconds=elapsed,full_quality_complete_link_motion_diagnostics_independent=True))
    # Upper-envelope affine model: include measured fixed source-read cost per population,
    # rather than charging it once per128 production windows.
    slopes=[(b['seconds']-a['seconds'])/(b['n']-a['n']) for a,b in zip(records,records[1:])]
    beta=max(slopes+[records[-1]['seconds']/records[-1]['n']*.5,0.])
    alpha=max([r['seconds']-beta*r['n'] for r in records]+[0.])
    assert all(alpha+beta*r['n']>=r['seconds']-1e-10 for r in records)
    bench=read_json(HERE/'BENCHMARK.json');train=bench['train_compute_seconds'];planning=(train+12*alpha+6*(18909+1856)*beta+180)*1.5
    assert verify_sources()==digest;guard.check()
    report=dict(status='PASS_CPU_SAVED_PIPELINE_ETA_EXTENSION',base_source_digest=digest,source_code_sha256=sha256(__file__),records=records,fixed_IO_seconds_per_population=alpha,incremental_seconds_per_window=beta,planning_seconds=planning,ETA_seconds=[.7*planning,1.5*planning],original128_linear_planning_seconds=bench['planning_seconds'],original_benchmark_not_overwritten=True,no_model_forward_backward_optimizer=True,no_scientific_protocol_change=True,resource_observations=guard.observations)
    write_json(HERE/'PRELAUNCH_PIPELINE_SCALING.json',report)
    write_json(HERE/'source_extension_prelaunch.json',dict(base_source_digest=digest,extension_files={Path(__file__).relative_to(ROOT).as_posix():sha256(__file__)},scope='CPU saved-data timing/acceptance extension only; not production source replacement'))
    print(json.dumps({k:v for k,v in report.items() if k!='resource_observations'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
