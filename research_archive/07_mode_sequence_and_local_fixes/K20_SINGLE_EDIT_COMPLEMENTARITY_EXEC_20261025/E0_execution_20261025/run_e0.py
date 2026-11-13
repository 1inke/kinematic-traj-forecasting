"""Finite all-action E0 enumeration, separate science gates from arithmetic success."""
import numpy as np
from common import *
from data_contract import load,kin_opt
from space import enumerate_window,materialize
from review_saved import independent_population
def seed_run(seed,data,source,guard):
    folder=HERE/f'seed{seed}';folder.mkdir(exist_ok=True)
    if (folder/'done.json').exists():return validate_done(folder/'done.json',source)['gate']
    if list(folder.glob('chunk_*.npz')):raise RuntimeError('Partial E0 output exists: preserve; require explicit compatible saved-stage recovery, not blind rewrite')
    kin=kin_opt(seed,data['opt_row']);n=len(data['ids']);allB=[];actions=[];paths=[];chunks=[];outfiles=[];near_zero=np.zeros(3,np.int64)
    np.savez_compressed(folder/'input_pools.npz',sample_id=data['ids'],native=data['native'],kin=kin,gt=data['gt'],group=data['groups'],track_key=data['tracks'])
    outfiles.append('input_pools.npz')
    for start in range(0,n,128):
        end=min(start+128,n);pack=[]
        for i in range(start,end):
            s=enumerate_window(data['native'][i],kin[i],data['gt'][i]);pack.append(s)
            allB.append(int(s['B'].sum()));actions.append(s['action']);paths.append(materialize(data['native'][i],kin[i],s['action']))
            near_zero+=((s['margin']>0)&(s['margin']<=1e-10)).sum(0)
        arrays={name:np.asarray([s[name] for s in pack]) for name in pack[0]}
        file=f'chunk_{start:05d}_{end:05d}.npz'
        np.savez_compressed(folder/file,sample_id=data['ids'][start:end],start=np.array(start),**arrays)
        chunks.append(file);outfiles.append(file)
        guard.check();stage('E0_ENUMERATE_SAVE',seed=seed,completed_windows=end,total_windows=n,completed_nonSTOP_actions=(SEEDS.index(seed)*n+end)*400)
    paths=np.asarray(paths);acts=np.asarray(actions,np.int32);positive=np.asarray(allB)>0
    np.savez_compressed(folder/'oracle.npz',sample_id=data['ids'],paths=paths,action=acts,positive_count=np.asarray(allB,np.int32))
    rows,cc=independent_population(paths,data['gt'],data['groups'],data['native'],data['ids'],data['tracks'])
    write_json(folder/'POPULATION_METRICS.json',rows)
    change=data['groups']==2;cr=rows['change']
    checks=dict(positive_windows=100*int(positive.sum())>=5*n,positive_tracks=len(set(data['tracks'][positive]))>=50,change_windows=int(change.sum())>=300,change_tracks=len(set(data['tracks'][change]))>=30,oracle_change_coverage=100*(cr['good']-cr['native_good'])>=2*cr['n'],native_good_lost_zero=all(v['lost']==0 for v in rows.values()),allgroups_five_pooled_motion=all(v['motion_pass'] for v in rows.values()))
    gate=dict(seed=seed,pass_all=all(checks.values()),checks=checks,positive_windows=int(positive.sum()),total_windows=n,positive_tracks=len(set(data['tracks'][positive])),positive_count_quantiles=np.quantile(allB,[0,.25,.5,.75,.95,1]).tolist(),change=cr,near_zero_positive_metric_margins_le_1e_10=near_zero.tolist(),note='Near-zero bucket is descriptive, no label epsilon; gate counts exact integers. GT oracle is not deployment.')
    write_json(folder/'GATE.json',gate)
    outfiles+=['oracle.npz','POPULATION_METRICS.json','GATE.json']
    done=dict(status='SAVED_REQUIRES_ALL_ACTION_AUDIT',source_digest=source,seed=seed,windows=n,nonSTOP_actions=400*n,chunks=chunks,gate=gate,outputs={f:sha256(folder/f) for f in outfiles})
    write_json(folder/'done.json',done);return gate
def main(guard):
    source=verify();data=load(save=False)
    pre=read_json(HERE/'PREFLIGHT.json');assert pre['status']=='PASS' and pre['source_digest']==source
    assert data['identity']['opt_id_sha']==pre['identity']['opt_id_sha']
    gates=[]
    for seed in SEEDS:gates.append(seed_run(seed,data,source,guard))
    return data,gates,source
