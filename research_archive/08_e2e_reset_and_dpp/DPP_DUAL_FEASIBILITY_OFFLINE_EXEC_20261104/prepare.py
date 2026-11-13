"""Actual preparation is intentionally inaccessible until a later human start request."""
import argparse,hashlib,shutil
from common import *
import numpy as np
from scoring import pair_squared

def hash128(ids,tag):
    return np.array(sorted(range(len(ids)),key=lambda i:hashlib.sha256((tag+'|'+str(ids[i])).encode()).hexdigest())[:128])

def calibrate(past,future):
    p=np.asarray(past,np.float64);y=np.asarray(future,np.float64)
    vh=(p[:,-1]-p[:,-2])/.4
    v=np.diff(np.concatenate([p[:,-1:],y],axis=1),axis=1)/.4
    a=np.diff(np.concatenate([vh[:,None],v],axis=1),axis=1)/.4
    values=dict(v=np.linalg.norm(v,axis=-1).reshape(-1),a=np.linalg.norm(a,axis=-1).reshape(-1),join=np.linalg.norm(v[:,0]-vh,axis=-1))
    c={}
    for name,limit,sigma in [('v','vmax','sigma_v'),('a','amax','sigma_a'),('join','joinmax','sigma_join')]:
        x=values[name];c[limit]=float(np.quantile(x,.995,method='linear'))
        c[sigma]=float(np.quantile(x,.75)-np.quantile(x,.25))
        if not c[sigma]>0:raise ValueError('DEGENERATE_FIT_SCALE '+name)
    return c

def main(execute=False):
    if not authorize(execute,'prepare'):return
    if (RUN/'source_manifest.json').exists():raise RuntimeError('PREPARATION_ALREADY_SEALED_NO_OVERWRITE')
    write(RUN/'PREPARATION_STARTED.json',dict(at=now(),no_training=True))
    rg=guard_module();guard=rg.Guard(2*1024**3);guard.check(force=True)
    draft=read(RUN/'protocol.draft.json')
    with np.load(BASE/'C0/fit.npz',allow_pickle=False) as f:
        if len(f['ids'])!=21719:raise ValueError('FIT_COUNT')
        calibration=calibrate(f['past'][:,0],f['future'])
        ids=f['ids'].copy()
    baseline,_=checked_population('fresh-B0','fit')
    index=hash128(ids,'dpp-kernel-calibration-v1')
    distances=pair_squared(baseline['paths'][index])[:,np.triu_indices(20,1)[0],np.triu_indices(20,1)[1]]
    median=float(np.median(distances))
    if median<=0:raise ValueError('DPP_KERNEL_DEGENERATE_DO_NOT_RETUNE')
    calibration.update(k=1/median,epsilon_q=draft['dpp']['epsilon_q'],source_split='FIT_ONLY',hash128_ids=ids[index].tolist(),kernel_median_squared_distance=median,fit_calibration_no_dev=True)
    write(RUN/'CALIBRATION.json',calibration)
    registry=[];inputs=set([DESIGN,MAP,BASE/'C0/fit.npz',BASE/'C0/dev.npz',BASE/'C0/EVALUATION_CONTRACT.json',V2/'WORKER_EXIT.json',V2/'FINAL_SEAL.json'])
    if read(V2/'WORKER_EXIT.json')['status']!='SUCCESS':raise ValueError('V2_NOT_COMPLETE_NO_SOURCE_USE')
    verify(read(V2/'FINAL_SEAL.json')['files'])
    for name in MODELS:
        for split in ('fit','dev'):
            folder=population(name,split);done=read(folder/'done.json');verify(done['files'])
            inputs.update(map(Path,done['files']));inputs.add(folder/'done.json')
            registry.append(dict(model=name,split=split,folder=str(folder),expected_N=21719 if split=='fit' else 4753,seed=42,candidate_budget=20))
            guard.check()
    known_map=read(MAP);by_recording={r['recording_id']:r for r in known_map['records']}
    current_ids=[]
    for split in ('fit','dev'):
        with np.load(BASE/'C0'/f'{split}.npz',allow_pickle=False) as f:current_ids.extend(f['ids'].tolist())
    recording_counts={}
    for identity in current_ids:
        recording=identity.split('|')[0];recording_counts[recording]=recording_counts.get(recording,0)+1
    admission=[dict(recording=rec,n=number,status='BLOCKED_NO_ADMITTED_METRIC_SCENE_MAP',prior_status=by_recording.get(rec,{}).get('admission','NOT_IN_PRIOR_MAP_REGISTRY'),qscene_default=None) for rec,number in sorted(recording_counts.items())]
    write(RUN/'INPUT_MANIFEST.json',dict(populations=registry,no_new_candidates=True,no_model_forward=True))
    write(RUN/'MAP_ADMISSION.json',dict(status='BLOCKED_NO_ADMITTED_REAL_MAP',records=admission,total_rows=len(current_ids),stageB='BLOCKED_MAP_AND_DSF_INTERFACE',old_inventory_sha=sha(MAP),no_scene_effect_negative_claim=True))
    inputs.update(RUN.glob('*.py'));inputs.update(RUN.glob('*.md'));inputs.update(RUN.glob('*.ps1'))
    inputs.update(RUN/n for n in ('protocol.draft.json','CALIBRATION.json','INPUT_MANIFEST.json','MAP_ADMISSION.json','map_registry.template.json'))
    write(RUN/'source_manifest.json',dict(status='SEALED_STAGE_A_ACTUAL_DEPENDENCIES',at=now(),files=seal(sorted(inputs,key=str)),no_production_model_training=True))
    guard.check(force=True);write(RUN/'PREPARATION_RESOURCE.json',dict(snapshot=rg.snapshot(),observations=guard.observations,disk_free=shutil.disk_usage(RUN).free))
    print('PREPARED_STAGE_A_REAL_DIAGNOSIS_ONLY')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');main(p.parse_args().execute)

