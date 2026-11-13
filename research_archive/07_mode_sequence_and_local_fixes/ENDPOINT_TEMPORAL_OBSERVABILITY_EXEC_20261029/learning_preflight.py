"""Actual registered-data G/H GPU/resume/tiny preflight, never formal fits.

Requires the completed P0 saved-output review. A scientific tiny STOP is final
for that branch in this series; it is not repaired by changing the training.
"""
import copy
import hashlib
import importlib.util
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
from io_contract import json_new,json_read,npz_new,load_npz,sha,verify_manifest,checked_seal,code_digest
from features import transform
from build_teacher import read_targets
from preflight_checks import gradient_check,real_subset_resume,trees_equal
from p0_driver import emit,resource


def GPU_guard(feature_bytes):
    import torch
    # Current task/context measurement, never inferred from available RAM.
    torch.zeros(1,device='cuda:0');torch.cuda.synchronize()
    spec=importlib.util.spec_from_file_location('learning_resource_v3',ROOT.parents[1]/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    baseline=module.snapshot()
    graph=128*20*(40+64+32+2)*8*4
    io_payload=32*20*441*8+128*20*12*2*8*2
    estimated=baseline['private']+feature_bytes+graph+io_payload+512*1024**2
    inventory=dict(measured_context_private=baseline['private'],feature_and_target_bytes=feature_bytes,
                   conservative_graph_bytes=graph, bounded_saved_payload_bytes=io_payload,
                   allocator_and_checkpoint_reserve=512*1024**2,estimate_bytes=estimated,
                   snapshot=baseline,not_computed_from_available=True)
    g=module.Guard(estimated);g.check(force=True)
    return g,inventory


def actual_gpu(x,y,device):
    import torch
    from model import ParameterRegressor,window_mse
    from train import seed_everything
    states=[];reports=[]
    for arm in ('G','H'):
        seed_everything(42,True);model=ParameterRegressor().to(device)
        states.append(copy.deepcopy(model.state_dict()))
        features=x[arm][:8];target=y[:8]
        xt=torch.from_numpy(features).to(device);yt=torch.from_numpy(target).to(device)
        if not torch.equal(model(xt),torch.zeros((8,20,2),device=device)):raise AssertionError('initial not identity')
        loss=window_mse(model(xt),yt);loss.backward()
        phase0={n:float(p.grad.abs().max().cpu()) for n,p in model.named_parameters()}
        # A fixed isolated nonzero final layer makes hidden gradients testable.
        with torch.no_grad(): model.layers[-1].weight.copy_(torch.randn_like(model.layers[-1].weight)*.01)
        model.zero_grad(set_to_none=True);window_mse(model(xt),yt).backward()
        grads={n:float(p.grad.abs().max().cpu()) for n,p in model.named_parameters()}
        if any(not torch.isfinite(p.grad).all() or not (p.grad.abs().max()>0) for p in model.parameters()):raise AssertionError('nonfinite/disconnected tensor')
        if arm=='G' and not torch.equal(model.layers[0].weight.grad[:,:16],torch.zeros_like(model.layers[0].weight.grad[:,:16])):raise AssertionError('masked historical gradient')
        independent=gradient_check(features,target,device)
        reports.append(dict(arm=arm,phase_zero_last=phase0,phase_nonzero_last=grads,independent=independent))
        del model
    if not trees_equal(states[0],states[1]):raise AssertionError('G/H initial weights not exact')
    return dict(status='PASS_ACTUAL_GPU8_PAIRED_PHASED_AND_ALL_PARAMETER_GRADIENTS',arms=reports)


def tiny_indices(target,ids):
    active=np.any(target!=0,axis=(1,2))
    def select(mask):return sorted(np.flatnonzero(mask),key=lambda i:hashlib.sha256(('endpoint-temporal-tiny-v1|'+str(ids[i])).encode()).hexdigest())[:64]
    a,b=select(active),select(~active)
    return np.array(a+b,dtype=np.int64),len(a),len(b)


def label_subset(labels,indices):
    keys=('target','action_id','best_J','identity_J','J_matrix')
    output={key:[] for key in keys};sample=[]
    # Selected rows only; no full held/J bank remains resident.
    wanted={int(index):pos for pos,index in enumerate(indices)};offset=0;rows={}
    for file in sorted(Path(labels).glob('[0-9]*.npz')):
        z=load_npz(file,('sample_id',)+keys)
        for local in range(len(z['sample_id'])):
            if offset+local in wanted:rows[wanted[offset+local]]={key:z[key][local] for key in keys}
        offset+=len(z['sample_id'])
    return {key:np.stack([rows[i][key] for i in range(len(indices))]) for key in keys}


def tiny_metrics(x,parameters,labels,branch,g):
    from inference import predict_paths
    from audit_saved import check_saved
    prediction=predict_paths(x['past'],x['native'],parameters,branch,g.check)
    check_saved(x,dict(sample_id=x['sample_id'],**prediction),branch,check=g.check)
    yy=labels['target'].astype(float);mse=((parameters.astype(float)-yy)**2).mean(-1)
    active=np.any(yy!=0,axis=-1)
    chosen=np.take_along_axis(labels['J_matrix'],prediction['executed_action_id'][...,None],axis=2)[...,0]
    regret=chosen-labels['best_J'];identity=labels['identity_J']-labels['best_J']
    if (regret < -2e-12).any():raise AssertionError('guarded regret below safe oracle')
    return prediction,dict(all_candidate_MSE=float(mse.mean()),active_candidate_MSE=float(mse[active].mean()),
        raw_exact_teacher_action_hit=float((prediction['raw_action_id']==labels['action_id']).mean()),
        executed_exact_teacher_action_hit=float((prediction['executed_action_id']==labels['action_id']).mean()),
        guarded_J_regret=float(regret.mean()),identity_J_regret=float(identity.mean()),
        nonidentity_target_candidates=int(active.sum()),candidate_denominator=int(active.size),
        rejected_candidates=int(prediction['rejection_reason'].any(-1).sum()))


def tiny(x,features,y,labels,branch,out,g,device):
    from train import Engine,infer,checkpoint_new
    indices,active,other=tiny_indices(y,x['sample_id'])
    if active!=64 or other!=64:
        return dict(status='STOP_TINY_STRATUM_INSUFFICIENT',active_windows=active,inactive_windows=other)
    subset={key:value[indices] for key,value in x.items()}
    target=label_subset(labels,indices)
    if not np.array_equal(target['target'].astype(np.float32),y[indices]):raise AssertionError('tiny label IDs/order')
    engine=Engine(features[indices],y[indices],dict(tiny_ids=x['sample_id'][indices].tolist(),branch=branch),device=device,epochs=64)
    before=infer(engine.model,features[indices],device,g.check)
    initial,initial_metrics=tiny_metrics(subset,before,target,branch,g)
    for step in range(256):
        g.check();engine.step()
        if step%64==63:emit('ACTUAL_TINY_128_FIXED_256',branch=branch,updates=step+1)
    after=infer(engine.model,features[indices],device,g.check)
    final,final_metrics=tiny_metrics(subset,after,target,branch,g)
    # Gate follows production float32 target tensors exactly. Diagnostic target
    # arrays retain original float64 precision for action/utility reporting.
    a=np.any(y[indices]!=0,axis=-1)
    initial_mse=float(((before.astype(float)-y[indices])**2).mean(-1)[a].mean())
    final_mse=float(((after.astype(float)-y[indices])**2).mean(-1)[a].mean())
    drop=1-final_mse/initial_mse
    npz_new(out/'TINY_SAVED.npz',sample_id=x['sample_id'][indices],target=y[indices],
        **{'before_'+key:value for key,value in initial.items()},**{'after_'+key:value for key,value in final.items()})
    checkpoint_new(out/'tiny_fixture_checkpoint.pt',engine.snapshot())
    json_new(out/'tiny_history.json',engine.history)
    return dict(status='PASS_TINY_FIT' if drop>=.9 else 'STOP_TINY_FIT',updates=256,
        initial_active_candidate_MSE=initial_mse,final_active_candidate_MSE=final_mse,relative_drop=drop,
        before=initial_metrics,after=final_metrics,selected_sample_ids=x['sample_id'][indices].tolist(),
        only_training_tiny_no_generalization=True)


def reader_rejection(features,target,contract,out,device):
    from train import Engine,checkpoint_new,checkpoint_load_trusted
    e=Engine(features[:8],target[:8],contract,device=device,epochs=1)
    e.step();ck=out/'reader_fixture.pt';checkpoint_new(ck,e.snapshot())
    manifest=out/'reader_good.json';json_new(manifest,dict(files={'reader_fixture.pt':sha(ck)}))
    verify_manifest(manifest);good=checkpoint_load_trusted(ck)
    rejected={}
    for name,key in [('source','source_lock_sha'),('stats','stats_sha'),('split','split_sha')]:
        changed=dict(contract);changed[key]='bad fixture value'
        fresh=Engine(features[:8],target[:8],changed,device=device,epochs=1)
        try:fresh.restore(good)
        except ValueError:rejected[name]=True
        else:raise AssertionError('bad checkpoint contract accepted '+name)
    bad=out/'reader_bad_output.json';json_new(bad,dict(files={'reader_fixture.pt':'0'*64}))
    try:verify_manifest(bad)
    except ValueError:rejected['output_SHA']=True
    else:raise AssertionError('bad output accepted')
    fresh=Engine(features[:8],target[:8],contract,device=device,epochs=1);fresh.restore(good)
    if not trees_equal(fresh.snapshot(),good):raise AssertionError('actual compatible checkpoint reader not exact')
    return dict(status='PASS_ACTUAL_CHECKPOINT_READER_AND_BAD_SOURCE_STATS_SPLIT_OUTPUT_REJECTED',rejected=rejected)


def main():
    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('required actual CUDA not available')
    device='cuda:0'
    # P0 REVIEW explicitly lists paths relative to the execution root (it lists
    # nested review artifacts); source extensions use absolute paths. Do not
    # silently reinterpret the root-relative list as review-folder-relative.
    review=json_read(ROOT/'p0_completion_review/REVIEW.json')
    for name,digest in review['files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('P0 review output SHA changed: '+name)
    if review['status']!='PASS_INDEPENDENT_SAVED_P0_ALL_ACTIONS_AND_ORACLES':raise ValueError('P0 not reviewed')
    verify_manifest(ROOT/'source_manifest.json');verify_manifest(ROOT/'learning_source_extension.json')
    decisions=json_read(ROOT/'p0_completion_review/DECISION.json')['branches']
    registry=json_read(ROOT/'P0_STAGE_REGISTRY.json')
    x=load_npz(ROOT/'prepared/X_train.npz');v=load_npz(ROOT/'prepared/X_val.npz');stats=load_npz(ROOT/'prepared/statistics.npz')
    actualhash=sorted(range(len(x['sample_id'])),key=lambda i:hashlib.sha256(('endpoint-temporal-gpu-v1|'+str(x['sample_id'][i])).encode()).hexdigest())
    vi=sorted(range(len(v['sample_id'])),key=lambda i:hashlib.sha256(('endpoint-temporal-val-v1|'+str(v['sample_id'][i])).encode()).hexdigest())[:12]
    features={arm:transform(x['past'],x['native'],stats,arm) for arm in ('G','H')}
    val={arm:transform(v['past'],v['native'],stats,arm) for arm in ('G','H')}
    g,inventory=GPU_guard(sum(z.nbytes for z in list(features.values())+list(val.values()))+len(x['sample_id'])*20*2*4)
    final={}
    for branch in ('END','TIME'):
        if decisions[branch]['next_stage']!='OWN_GPU_RESUME_TINY_PREFLIGHT_REQUIRED':
            final[branch]=dict(status=decisions[branch]['next_stage'],formal_training=False);continue
        out=ROOT/(branch+'_learning_preflight');out.mkdir(exist_ok=False)
        json_new(out/'MEMORY_INVENTORY.json',inventory)
        labels=ROOT/registry[branch]['train']['labels'];val_labels=ROOT/registry[branch]['val']['labels']
        y=read_targets(labels,x['sample_id']);vy=read_targets(val_labels,v['sample_id'])
        binding=dict(code_digest=code_digest(),observed_sha=sha(ROOT/'prepared/X_train.npz'),stats_sha=sha(ROOT/'prepared/statistics.npz'),
            teacher_done_sha=sha(labels/'done.json'),preparation_done_sha=sha(ROOT/'prepared/done.json'),
            split_sha=json_read(ROOT/'prepared/INPUT_AUDIT.json')['split_sha'],source_lock_sha=sha(ROOT/'source_manifest.json'),branch=branch)
        ix=np.array(actualhash[:513]);gpu=actual_gpu({a:features[a][ix] for a in features},y[ix],device)
        json_new(out/'GPU8.json',gpu);resumes={}
        for arm in ('G','H'):
            resumes[arm]=real_subset_resume(features[arm][ix],y[ix],val[arm][vi],vy[vi],binding,device,g.check)
            emit('ACTUAL_513_12_EXACT_RESUME_PASS',branch=branch,arm=arm)
        json_new(out/'RESUME.json',dict(status='PASS_BOTH_REAL513_12_SUBSET_ALL_RNG_EXACT',arms=resumes,
            train_sample_id=x['sample_id'][ix].tolist(),val_sample_id=v['sample_id'][vi].tolist()))
        rejection=reader_rejection(features['H'],y,binding,out,device)
        json_new(out/'READER_REJECTION.json',rejection)
        small=tiny(x,features['H'],y,labels,branch,out,g,device)
        json_new(out/'TINY.json',small)
        status=small['status']
        if status=='PASS_TINY_FIT':
            # A further saved-quality/RAM/benchmark preflight is still required.
            status='TINY_PASS_REQUIRES_FINAL_SAVED_PIPELINE_AND_BENCHMARK_PREFLIGHT'
        final[branch]=dict(status=status,binding=binding,tiny=small,formal_training=False)
        verify_manifest(ROOT/'source_manifest.json');verify_manifest(ROOT/'learning_source_extension.json')
        json_new(out/'RESOURCE.json',resource(g))
        json_new(out/'RESULT.json',dict(status=status,binding=binding,actual_gpu=gpu['status'],resume=resumes,
            tiny_status=small['status'],formal_training_started=False,files={str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()}))
        emit('LEARNING_PREFLIGHT_BRANCH_RESULT',branch=branch,status=status)
    json_new(ROOT/'LEARNING_PREFLIGHT_RESULT.json',dict(branches=final,formal_training_started=False,held_GT_read=False))


if __name__=='__main__':main()
