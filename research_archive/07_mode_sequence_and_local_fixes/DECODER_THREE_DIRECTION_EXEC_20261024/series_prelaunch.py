"""Actual source/ID, CPU projection+saved audit and B metadata prelaunch.

No formal training and no production screen gates are evaluated here. Frozen
encoder smoke is a separate subprocess so its namespace cannot affect decoders.
"""
import sys,time,shutil,traceback,subprocess
from pathlib import Path
sys.dont_write_bytecode=True
from common import *
from f16.resource import Guard
sys.path.insert(0,str(ROOT));from audit.verify import independent_metrics,compare
sys.path.insert(0,str(HERE/'direction_A_time_basis'))
sys.path.insert(0,str(HERE/'direction_B_scene_prior'))
from basis import knot3,svd4,project
from prior import Record,EndpointPrior,ARMS
from prepare_records import prepare as records_prepare

def run():
    out=HERE/'prelaunch_tests';out.mkdir(exist_ok=True);guard=Guard(3*2**30);guard.check('startup')
    source=verify_sources(TAIL/'source_manifest.json');data=load_data();fit=data['FIT']
    raw=Path(r'<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\processed_data_diverse\eth_data_train.npy')
    inherited=read_json(TAIL/'source_manifest.json')['files']
    matches=[v for v in inherited.values() if Path(v['path'])==raw]
    assert len(matches)==1 and sha256(raw)==matches[0]['sha256']
    records=HERE/'direction_B_scene_prior/records_20261024.json'
    if not records.exists():records_prepare(raw,records,matches[0]['sha256'],3*2**30)
    bundle=read_json(records);rows=bundle['records'];bank=[]
    for row in rows:
        if row['split']=='FIT':
            r={k:v for k,v in row.items() if k!='split'};r['cell']=tuple(r['cell']);r['endpoint']=tuple(r['endpoint']);bank.append(Record(**r))
    prior=EndpointPrior(bank,data['FIT']['ids']);queries=[r for r in rows if r['split']=='inner'][:128]
    timer=time.perf_counter()
    for q in queries:
        api={k:q[k] for k in ('sample_id','track','sequence','scene','t0','cell','heading','verified_time')}
        for arm in ARMS:
            a=prior.query(arm=arm,**api)
            if a.get('status')=='time_unverifiable':continue
            assert q['track'] not in a['source_tracks']
            assert abs(a['probability'].sum()-1)<2e-12
            if arm.endswith('causal') and not a['cold_start'] and not arm.startswith('XSCENE'):
                assert max(a['source_end_indices'],default=-1)<=q['t0']
    seconds_B_per_query=(time.perf_counter()-timer)/len(queries)
    ids,old=old_paths(42,'FIT');assert np.array_equal(ids,fit['ids'])
    take=np.arange(128);sample={k:v[take] for k,v in fit.items()};p=project(old[take],fit['past'][take],knot3())
    np.savez_compressed(out/'projection128.npz',paths=p,gt=sample['gt'],native=sample['native'],groups=sample['groups'],sample_id=sample['ids'],track=sample['track'])
    timer=time.perf_counter();m,e=report_paths(p,sample,out/'projection128_metrics.json')
    with np.load(out/'projection128.npz') as z:ind=independent_metrics(z)
    audit=compare({g:{k:m[g][k] for k in ind[g]} for g in ind},ind)
    assert audit['status']=='PASS',audit
    seconds_A_per_window=(time.perf_counter()-timer)/128
    assert verify_sources(TAIL/'source_manifest.json')==source;guard.check()
    if shutil.disk_usage(ROOT).free<1.3*4*2**30:raise OSError('INSUFFICIENT_DISK_FOR_FINITE_SERIES')
    result=dict(status='PASS_ACTUAL_CPU_SOURCES_IDS_METADATA_SAVED128',
        inherited_sources=len(inherited),parent_source_digest=source,records_SHA=sha256(records),
        world_local_max=bundle['provenance']['past_alignment_max'],B128queries8arms='PASS_NO_TRAINING',
        A128projection_saved_metrics_audit=audit,seconds_A_pipeline_per_window=seconds_A_per_window,
        seconds_B_eight_arms_per_query=seconds_B_per_query,
        A_screen_pipeline_nominal_seconds=seconds_A_per_window*20765*6,
        B_probe_nominal_seconds=seconds_B_per_query*1856,
        CPU_estimate_bytes=3*2**30,GPU_estimate_bytes=6*2**30,
        disk_free_bytes=shutil.disk_usage(ROOT).free,resource_observations=guard.observations,
        no_formal_training=True,no_scientific_stage_gate_selected=True)
    write_json(HERE/'PRELAUNCH.json',result);print(json.dumps(result,default=str),flush=True)

if __name__=='__main__':
    try:run()
    except BaseException as error:
        write_json(HERE/'PRELAUNCH_FAILURE.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        raise
