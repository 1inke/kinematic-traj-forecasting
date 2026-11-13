"""All-population stageA; paths/slots/scores are never replaced or selected."""
import argparse,time
from common import *
from scoring import *
import numpy as np
from threadpoolctl import threadpool_limits

def finite_summary(x):
    x=np.asarray(x,np.float64);v=x[np.isfinite(x)]
    return dict(n=int(x.size),n_finite=int(v.size),mean=float(v.mean()) if v.size else None,
        p05=float(np.quantile(v,.05)) if v.size else None,p50=float(np.quantile(v,.5)) if v.size else None,
        p95=float(np.quantile(v,.95)) if v.size else None)

def group_rows(name,split,p,m,s):
    ade=m['ADE'];fde=m['FDE'];good=(ade<=.2)&(fde<=.4);coverage=good.any(1)
    if not np.array_equal(coverage,m['coverage']):raise ValueError('COVERAGE_CONTRACT')
    gt_angle=turning(p['past'].astype(np.float64),p['future'][:,None].astype(np.float64))[:,0]
    groups=[('overall',np.ones(len(coverage),bool)),('stable',p['groups']==0),('transition',p['groups']==1),('change',p['groups']==2),
        ('GT_turn_straight',gt_angle<15),('GT_turn_moderate',(gt_angle>=15)&(gt_angle<60)),('GT_turn_large',gt_angle>=60),('GT_turn_undefined',~np.isfinite(gt_angle))]
    oracle=np.argmax(-np.maximum(ade/.2,fde/.4),axis=1)
    rows=[];details={}
    for group,mask in groups:
        n=int(mask.sum());allq=s['qkin'][mask];oracleq=s['qkin'][np.arange(len(coverage)),oracle][mask]
        asd=finite_summary(s['valid_ASD'][mask]);fsd=finite_summary(s['valid_FSD'][mask])
        rows.append(dict(model=name,seed=42,split=split,group=group,N=n,target_tracks=len(set('|'.join(str(x).split('|')[:2]) for x in p['ids'][mask])),
            good=int(coverage[mask].sum()),coverage=float(coverage[mask].mean()) if n else None,
            minADE=float(ade[mask].min(1).mean()) if n else None,minFDE=float(fde[mask].min(1).mean()) if n else None,
            kin_valid_candidates=int(s['kin_valid'][mask].sum()),candidate_denominator=n*20,kinematic_violation_rate=float((~s['kin_valid'][mask]).mean()) if n else None,
            kin_valid_good_windows=int((s['kin_valid']&good).any(1)[mask].sum()),
            qkin_mean=float(allq.mean()) if n else None,qkin_below_point1=int((allq<.1).sum()),qkin_numerical_underflow=int((allq==0).sum()),
            oracle_qkin_mean=float(oracleq.mean()) if n else None,good_candidates_qkin_below_point1=int(((s['qkin']<.1)&good)[mask].sum()),
            good_candidate_denominator=int(good[mask].sum()),
            EC_uniform_mean=float(s['EC_uniform'][mask].mean()) if n else None,EC_kin_mean=float(s['EC_kin'][mask].mean()) if n else None,
            valid_ASD_mean=asd['mean'],valid_FSD_mean=fsd['mean'],windows_with_at_least2_valid=asd['n_finite'],
            scene_assessed_candidates=0,joint_valid_candidates=None,scene_branch='BLOCKED_NO_ADMITTED_MAP',same_paths_and_oracle_errors_unchanged=True))
        details[group]=dict(qkin=finite_summary(allq),oracle_qkin=finite_summary(oracleq),
            candidate_turn_large_good_qkin=finite_summary(s['qkin'][mask][(s['candidate_turn_degrees'][mask]>=60)&good[mask]]),
            valid_ASD=asd,valid_FSD=fsd,undefined_candidate_turns=int((~np.isfinite(s['candidate_turn_degrees'][mask])).sum()))
    return rows,details

def main(execute=False):
    if not authorize(execute,'evaluate'):return
    t=time.perf_counter();source=read(RUN/'source_manifest.json');verify(source['files'])
    pre=read(RUN/'PREFLIGHT_REAL.json')
    if pre['status']!='PASS_ACTUAL_SAVED128_KIN_DPP_QUALITY_READBACK' or pre['source_manifest']!=sha(RUN/'source_manifest.json'):raise ValueError('REAL_PREFLIGHT_NOT_COMPATIBLE')
    with (RUN/'stage_a.lock').open('x') as f:f.write('CPU_OFFLINE_NO_TRAINING')
    c=read(RUN/'CALIBRATION.json');rg=guard_module();guard=rg.Guard(2*1024**3);guard.check(force=True)
    summaries=[];outputs=[];audits={}
    with threadpool_limits(limits=1):
        for name in MODELS:
            for split in ('fit','dev'):
                p,m=checked_population(name,split);n=len(p['ids'])
                folder=RUN/'populations'/name/split;folder.mkdir(parents=True,exist_ok=False)
                write(folder/'START.json',dict(at=now(),N=n,no_model_or_optimizer=True))
                s={k:np.empty((n,20),np.float64) for k in ('qkin','logqkin','penalty_v','penalty_a','penalty_join','marginal_uniform','marginal_kin','candidate_turn_degrees')}
                s.update(kin_valid=np.empty((n,20),bool),EC_uniform=np.empty(n),EC_kin=np.empty(n),valid_ASD=np.empty(n),valid_FSD=np.empty(n),valid_pair_count=np.empty(n,np.int64))
                errors=[];quality_error=0.
                for start in range(0,n,128):
                    sl=slice(start,min(start+128,n));past=p['past'][sl].astype(np.float64);y=p['paths'][sl].astype(np.float64)
                    a=kinematic(past,y,c)
                    for key,value in a.items():s[key][sl]=value
                    s['candidate_turn_degrees'][sl]=turning(past,y)
                    for label,q in [('uniform',np.ones((len(y),20))),('kin',a['qkin'])]:
                        d=dpp(y,q,c['k'],c['epsilon_q']);s['EC_'+label][sl]=d['ec'];s['marginal_'+label][sl]=d['marginal'];errors.append(d['solve_spectral_error'])
                    vA,vF,pairs=asd_fsd(y,a['kin_valid']);s['valid_ASD'][sl]=vA;s['valid_FSD'][sl]=vF;s['valid_pair_count'][sl]=pairs
                    distance=np.linalg.norm(y-p['future'][sl,None].astype(np.float64),axis=-1)
                    quality_error=max(quality_error,float(np.max(abs(distance.mean(-1)-m['ADE'][sl]))),float(np.max(abs(distance[:,:,-1]-m['FDE'][sl]))))
                    guard.check()
                if quality_error>2e-12:raise ValueError('FULL_SAME_CANDIDATE_QUALITY')
                # qscene/qjoint stay unknown; do not pretend qscene=1 is a joint-quality ablation.
                s.update(qscene=np.full((n,20),np.nan),qjoint=np.full((n,20),np.nan),scene_known_fraction=np.zeros((n,20)))
                np.savez_compressed(folder/'diagnostics.npz',ids=p['ids'],groups=p['groups'],**s)
                rows,details=group_rows(name,split,p,m,s);summaries.extend(rows);write(folder/'SUMMARY.json',details)
                files=seal([folder/'diagnostics.npz',folder/'SUMMARY.json',folder/'START.json'])
                audit=dict(status='PASS_FULL_KIN_DPP_SPECTRAL_AND_OLD_QUALITY_NOT_FULL_MODEL_REPLAY',N=n,score_only_oracle_quality_max_error=quality_error,max_solve_spectral_error=max(errors),source_predictions_sha=sha(population(name,split)/'predictions.npz'),files=files)
                write(folder/'done.json',audit);audits[name+'/'+split]=audit
                outputs.extend([*map(Path,files),folder/'done.json'])
                print('OFFLINE_POPULATION_DONE',name,split,n,flush=True)
                del p,m,s
    table(RUN/'ALL_GROUPS.csv',summaries);write(RUN/'ARITHMETIC_AUDIT.json',dict(status='PASS_FULL_STAGE_A_KERNEL_SPECTRAL_AND_UNCHANGED_K20_QUALITY',populations=audits,no_full_model_forward_gradient_or_cluster_replay=True))
    verify(source['files']);guard.check(force=True)
    write(RUN/'RESOURCE.json',dict(policy='warning_first_remaining_v3',estimate_bytes=2*1024**3,snapshot=rg.snapshot(),observations=guard.observations,elapsed_seconds=time.perf_counter()-t))
    write(RUN/'STAGE_A_COMPUTED.json',dict(status='COMPUTED_AWAIT_INDEPENDENT_SAVED_REVIEW',at=now(),populations=10,windows=5*(21719+4753),correlated_candidate_rows=20*5*(21719+4753),trained_models=0,source_manifest=sha(RUN/'source_manifest.json'),outputs=seal(outputs+[RUN/'ALL_GROUPS.csv',RUN/'ARITHMETIC_AUDIT.json',RUN/'RESOURCE.json'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');main(p.parse_args().execute)
