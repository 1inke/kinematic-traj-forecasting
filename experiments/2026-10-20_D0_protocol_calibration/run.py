"""Read-only old EXP03 replay, new-pipeline verification and D-0 precision.
Not an S3 experiment, no new model, optimizer, training or CAL evaluation.
"""
import os,sys,json,subprocess,time
from pathlib import Path
import numpy as np

ROOT=Path(os.environ.get('F16_ROOT',Path(__file__).resolve().parents[2]));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from f16.data import load_identities
from f16.paths import resolve_paths
from f16.io import sha256,write_json,read_json
from f16.eval import evaluate,errors
from f16.statistics import paired_bootstrap,mde_normal_approx
from f16.resource import Guard
from audit.verify import audit_file,compare
from audit.statistics_verify import independent_draws

OUT=Path(__file__).resolve().parent

def main():
    started=time.monotonic();p=resolve_paths();data=load_identities(p);ix=data['split']['outer'];ids=data['sample_id'][ix];tracks=data['track'][ix];groups=data['groups'][ix];before={};resources=Guard(2*1024**3);resources.check('startup')
    allmetrics={};covered={};replay=[];frozen_kwargs=dict(frozen_cache=p['cache'],frozen_labels=p['labels'],frozen_metadata=p['metadata'],frozen_manifest=ROOT/'config/data_manifest.v1.json',root=ROOT)
    with np.load(p['cache'],allow_pickle=False) as z:native=z['candidate'][ix,:,:24].reshape(-1,20,12,2).astype(float)
    with np.load(p['labels'],allow_pickle=False) as z:
        if not np.array_equal(z['sample_id'][ix],ids):raise ValueError('Frozen GT ID mismatch')
        truth=z['future'][ix].astype(float)
    for name in ['cache','labels','metadata','splits']:before[p[name].relative_to(ROOT).as_posix()]=sha256(p[name])
    nativefile=OUT/'native_outer.npz';np.savez_compressed(nativefile,paths=native,native=native,gt=truth,groups=groups,sample_id=ids,track=tracks)
    metrics,e=evaluate(native,truth,groups,native,ids,tracks);write_json(OUT/'native_metrics.json',metrics)
    audit=audit_file(nativefile,OUT/'native_metrics.json',**frozen_kwargs);write_json(OUT/'native_AUDIT.json',audit)
    if audit['status']!='PASS':raise RuntimeError('Native independent audit failed')
    covered['native']=np.repeat(e['coverage'][None],3,0);allmetrics['native']=metrics
    for arm in ['A','B']:
        runs=[];cover=[]
        for seed in [42,43,44]:
            resources.check();old=ROOT/f'GEMINI_EXP/EXP03_KINEMATIC_DAMPING_FACTORIAL/runs/{arm}_seed{seed}';prediction=old/'outer_predictions.npz';prior=old/'metrics.json'
            for fp in [prediction,prior,old/'checkpoint.pt',old/'history.json']:before[fp.relative_to(ROOT).as_posix()]=sha256(fp)
            with np.load(prediction,allow_pickle=False) as z:
                if not np.array_equal(z['sample_id'],ids) or not np.array_equal(z['native'],native) or not np.array_equal(z['gt'],truth):raise ValueError('Old saved output/frozen source mismatch')
                paths=z['paths'].copy()
            bundle=OUT/f'{arm}_seed{seed}_outer.npz';np.savez_compressed(bundle,paths=paths,native=native,gt=truth,groups=groups,sample_id=ids,track=tracks)
            metrics,e=evaluate(paths,truth,groups,native,ids,tracks);result=OUT/f'{arm}_seed{seed}_metrics.json';write_json(result,metrics)
            verified=audit_file(bundle,result,**frozen_kwargs);write_json(OUT/f'{arm}_seed{seed}_AUDIT.json',verified)
            if verified['status']!='PASS':raise RuntimeError('Independent saved metrics mismatch')
            oldmetrics=read_json(prior)['outer'];sub_old={g:{k:oldmetrics[g][k] for k in m if k in oldmetrics[g]} for g,m in metrics.items()};sub_new={g:{k:metrics[g][k] for k in oldmetrics[g] if k in metrics[g]} for g in metrics}
            historical=compare(sub_old,sub_new);replay.append(dict(arm=arm,seed=seed,independent_audit=verified['status'],historical_common_fields=historical,new_Q_winner_error_fields=True))
            if historical['status']!='PASS':raise RuntimeError('Historical common-field regression failed')
            runs.append(metrics);cover.append(e['coverage'])
        allmetrics[arm]=runs;covered[arm]=np.stack(cover)
    change=groups==2;calibration={}
    for contrast,a,b in [('A-native','A','native'),('B-native','B','native'),('B-A','B','A')]:
        stats,draws=paired_bootstrap(covered[a][:,change],covered[b][:,change],tracks[change]);independent=independent_draws(covered[a][:,change],covered[b][:,change],tracks[change]);difference=float(np.max(np.abs(draws-independent)))
        if difference>2e-12:raise RuntimeError('Independent full bootstrap mismatch')
        stats['independent_full_draw_maxdiff']=difference;stats['normal_approx_MDE80_pp']=100*mde_normal_approx(stats['standard_error']);stats['power_method']='two-sided normal approximation, alpha .05, power .80; NOT empirical power or confirmatory effect';calibration[contrast]=stats;np.save(OUT/f'{contrast}_bootstrap.npy',draws)
    resources.check();after={name:sha256(ROOT/name) for name in before}
    if before!=after:raise RuntimeError('Historical input changed during replay')
    result=dict(status='COMPLETE_CPU_ENGINEERING_CALIBRATION_NOT_NEW_S3',historical_replay=replay,calibration=calibration,native_counts={g:{k:allmetrics['native'][g][k] for k in ['n','tracks','good','coverage']} for g in allmetrics['native']},historical_inputs_sha256=before,preservation=before==after,formal_training_runs=0,gpu_training_hours=0,author_MDE_confirmation='OPEN',protocol_freeze='BLOCKED_BY_C3_AND_C6',duration_seconds=time.monotonic()-started,resource_observations=resources.observations)
    write_json(OUT/'RESULT.json',result)
    print(json.dumps({'duration_seconds':result['duration_seconds'],'MDE80_pp':{k:v['normal_approx_MDE80_pp'] for k,v in calibration.items()},'native_change':result['native_counts']['change'],'audits':7,'historical_comparisons':6}))

if __name__=='__main__':main()
