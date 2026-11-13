"""Nine fresh tiny mechanism models, no formal admission until all audited gates."""
import time
import traceback
from pathlib import Path
import numpy as np
import common as c
import torch
from trainer import Engine
from preflight import contract
from train import infer,checkpoint_new,checkpoint_load_trusted
from preflight_checks import trees_equal
from inference import predict_paths
from evaluate import evaluate_population

def evaluate(d,u,out,g):
    pred=predict_paths(d['x']['past'],d['x']['native'],u,'END',g.check)
    C=d['x']['native']+.4*(np.arange(1,13)/12)[None,None,:,None]*u.astype(float)[:,:,None]
    saved=dict(sample_id=d['x']['sample_id'],**pred,continuous_paths=C);rows=[]
    for port,p in [('C',C),('Q',pred['raw_paths']),('S',pred['guarded_paths'])]:
        sm,rr=evaluate_population(d['x'],d['gt'],p,g.check)
        saved.update({port+'_'+key:value for key,value in sm.items()});rows.extend([dict(port=port,**r) for r in rr])
    err=((u.astype(float)-d['target'].astype(float))**2).mean(-1);initial=(d['target'].astype(float)**2).mean(-1);active=np.any(d['target']!=0,axis=-1)
    oracle=d['labels']['best_J'];chosen=np.take_along_axis(d['labels']['J_matrix'],pred['executed_action_id'][...,None],axis=2)[...,0]
    meta=dict(all_MSE=err.mean(),active_MSE=err[active].mean(),identity_MSE=err[~active].mean(),
        active_MSE_drop=1-err[active].mean()/initial[active].mean(),old90_gate='PASS_DESCRIPTIVE_ONLY' if 1-err[active].mean()/initial[active].mean()>=.9 else 'FAIL_ORIGINAL90_DESCRIPTIVE',
        exact_teacher_action_hit=(pred['executed_action_id']==d['labels']['action_id']).mean(),safe_J_regret=(chosen-oracle).mean(),
        nonidentity=(pred['executed_action_id']!=c.space('END').identity).mean(),continuous_outside_circle=int((np.linalg.norm(u,axis=-1)>1).sum()),
        guard_rejections=int(pred['rejection_reason'].any(-1).sum()),guard_failed_bits=pred['rejection_reason'].sum((0,1)),groups=rows)
    c.npz_new(out/'final.npz',**saved);c.write(out/'METRICS.json',meta)
    return meta

def main():
    torch.set_num_threads(1)
    assert c.json_read(c.ROOT/'PREFLIGHT.json')['status']=='PASS_ACTUAL_THREE_OBJECTIVE_PREFLIGHT'
    bench=c.json_read(c.ROOT/'BENCHMARK.json')
    if bench['tiny_planning_seconds']>1800:raise RuntimeError('StageT >30min planning; report before launching')
    source=c.production_lock();estimate=c.json_read(c.ROOT/'MEMORY_ESTIMATE.json')['estimate_bytes'];g=c.resource(estimate);g.check(force=True)
    d0=c.data();ix=c.load_npz(c.OLD/'tiny_v2/indices.npz')['indices'];d=c.subset(d0,ix)
    c.write(c.ROOT/'INPUT_AUDIT.json',dict(status='PASS_FIXED_ORIGINAL_TINY_IDS_STAT_TEACHER_SUPPORT',
        IDs=d['x']['sample_id'],tracks=len(np.unique(d['x']['track_key'])),active=int(np.any(d['target']!=0,axis=-1).sum()),
        binding=c.binding(d),train3103_stats_unchanged=True,no_new_held_GT_access=True))
    tiny=c.ROOT/'tiny';tiny.mkdir(exist_ok=False)
    summaries=[];initials={};orders={};started=time.perf_counter()
    for seed in (42,43,44):
        for arm in ('M','R','RS'):
            folder=tiny/f'{arm}_{seed}';folder.mkdir()
            con=contract(d,arm,'tiny256')|dict(source_manifest_sha=c.sha(c.ROOT/'source_manifest.json'),seed=seed)
            e=Engine(d,con,arm,seed=seed,epochs=64)
            before=e.snapshot();checkpoint_new(folder/'initial_checkpoint.pt',before)
            if seed not in initials:initials[seed]=before['initial_state']
            else:assert trees_equal(initials[seed],before['initial_state'])
            start=time.perf_counter()
            for step in range(256):
                g.check();e.step()
                if (step+1)%64==0:print('TINY',arm,seed,'updates',step+1,flush=True)
            torch.cuda.synchronize();train_seconds=time.perf_counter()-start
            after=e.snapshot();checkpoint_new(folder/'checkpoint.pt',after)
            if seed not in orders:orders[seed]=after['batch_order']
            else:assert orders[seed]==after['batch_order']
            # Explicit reproduction, never treated as a new independent success replicate.
            if arm=='M' and seed==42:
                old=checkpoint_load_trusted(c.OLD/'tiny_v2/checkpoint.pt')
                keys=('model','initial_state','optimizer','sampler','python_rng','numpy_rng','torch_rng','cuda_rng','epoch','cursor','updates','order','loss_sum','history','batch_order')
                checks={key:trees_equal(after[key],old[key]) for key in keys}
                c.write(c.ROOT/'M42_CONTROL_REPRODUCTION.json',dict(status='PASS_BITWISE_ORIGINAL_M42' if all(checks.values()) else 'CONTROL_REPRODUCTION_UNRESOLVED',checks=checks,
                    old_checkpoint_sha=c.sha(c.OLD/'tiny_v2/checkpoint.pt'),new_checkpoint_sha=c.sha(folder/'checkpoint.pt'),not_independent_generalization=True))
                if not all(checks.values()):raise AssertionError('CONTROL_REPRODUCTION_UNRESOLVED; preserve model, do not retrain')
            u=infer(e.model,d['z'],'cuda:0',g.check)
            if arm=='M' and seed==42:assert np.array_equal(u,c.load_npz(c.OLD/'tiny_v2/final.npz',('continuous_param',))['continuous_param'])
            state_before=e.model.state_dict();metrics=evaluate(d,u,folder,g)
            assert trees_equal(state_before,e.model.state_dict())
            c.write(folder/'history.json',after['history'])
            c.write(folder/'TRAINING_SUMMARY.json',dict(updates=256,epochs=64,train_compute_seconds=train_seconds,
                components_mean={key:np.mean([v['components'][key] for v in e.diagnostics]) for key in ('mse','retention','rescue')},
                clipped_updates=sum(v['clipped'] for v in e.diagnostics),fixed_final=True,source_contract=con))
            c.npz_new(folder/'training_trace.npz',prediction=np.stack([v['preupdate_prediction'] for v in e.diagnostics]),
                batch_indices=np.array(e.batch_order),**{key:np.stack([v['activity'][key] for v in e.diagnostics]) for key in e.diagnostics[0]['activity']},
                **{'component_'+key:np.array([v['components'][key] for v in e.diagnostics]) for key in ('mse','retention','rescue')})
            c.write(folder/'done.json',dict(status='FIXED_FINAL_OUTPUTS_COMPLETE_REVIEW_REQUIRED',contract=con,updates=256,epochs=64,
                files={p.name:c.sha(p) for p in folder.iterdir() if p.is_file()}))
            summaries.append(dict(seed=seed,arm=arm,**metrics));del e,after,before
    c.verify(source,g.check);g.check(force=True)
    c.write(c.ROOT/'TINY_COMPUTATION.json',dict(status='NINE_TINY_COMPLETE_REVIEW_REQUIRED',models=9,updates=2304,epochs=576,
        paired_initials_and_batchIDs=True,elapsed_seconds=time.perf_counter()-started,source_unchanged=True,held_GT_opened=False))
    c.write(c.ROOT/'RESOURCE_TINY.json',dict(policy='warning_first_remaining_v3',estimate_bytes=estimate,observations=g.observations))
    print('Nine tiny complete. Independent audit and gate decision required.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        c.write(c.ROOT/('FAILURE_TINY_'+time.strftime('%H%M%S')+'.json'),dict(traceback=traceback.format_exc(),code_sha=c.code_sha()))
        raise
