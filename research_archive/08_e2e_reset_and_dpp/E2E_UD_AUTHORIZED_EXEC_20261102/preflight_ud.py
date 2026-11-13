"""All four arms: amended staged gates, true resume, math, saved QA and timing."""
import copy
import time
import traceback
from ud_common import *
from train_core_ud import Trainer,cpu_tree,exact,save_checkpoint,infer
from ud_model_factory import new_branch
import baseline_factory as bf
from preflight import guard_module,hash_indices,reject
from evaluate_ud import save_population
from loss_independent import loss_grad
from math_reference import torch_reference,numpy_reference
import torch
import numpy as np


def branch_math(arm,module):
    m=copy.deepcopy(module).cpu().double()
    rng=np.random.default_rng(20261102)
    x=torch.tensor(rng.normal(size=(2,8,6) if arm.startswith('U') else (2,20,64)),dtype=torch.float64)
    context=None if arm.startswith('U') else torch.tensor(rng.normal(size=(2,64)),dtype=torch.float64)
    # Fixed nonzero fixture output, not a production initialization change.
    with torch.no_grad():m.out.weight.copy_(torch.tensor(rng.normal(0,.01,size=m.out.weight.shape),dtype=torch.float64))
    target=torch.tensor(rng.normal(size=(2,64) if arm.startswith('U') else (2,20,64)),dtype=torch.float64)
    actual=m(x) if context is None else m(x,context)
    actual_loss=(actual*target).mean()
    grads=torch.autograd.grad(actual_loss,tuple(m.parameters()))
    params={k:v.detach().clone().requires_grad_(True) for k,v in m.named_parameters()}
    ref=torch_reference(arm,params,x,context)
    rg=torch.autograd.grad((ref*target).mean(),tuple(params.values()))
    gradient_error=max(float((a-b).abs().max()) for a,b in zip(grads,rg))
    p={k:v.detach().numpy().copy() for k,v in m.named_parameters()}
    nr=numpy_reference(arm,p,x.numpy(),None if context is None else context.numpy())
    value_error=float(np.max(abs(actual.detach().numpy()-nr)))
    if max(gradient_error,value_error)>1e-10:raise ValueError('BRANCH_REFERENCE_MISMATCH '+arm)
    fd=[]
    for direction_id in range(2):
        direction={k:rng.normal(size=v.shape) for k,v in p.items()}
        length=np.sqrt(sum((v*v).sum() for v in direction.values()))
        direction={k:v/length for k,v in direction.items()}
        reference=sum(float((g.detach().numpy()*direction[k]).sum()) for (k,_),g in zip(m.named_parameters(),grads))
        for eps in (1e-4,1e-5):
            plus={k:v+eps*direction[k] for k,v in p.items()};minus={k:v-eps*direction[k] for k,v in p.items()}
            xp=numpy_reference(arm,plus,x.numpy(),None if context is None else context.numpy())
            xm=numpy_reference(arm,minus,x.numpy(),None if context is None else context.numpy())
            numerical=float(((xp-xm)*target.numpy()).mean()/(2*eps))
            error=abs(numerical-reference)
            if error>1e-6+1e-3*abs(reference):raise ValueError('BRANCH_FIXED_FD_MISMATCH '+arm)
            fd.append(dict(direction=direction_id,eps=eps,error=error))
    permutation=None
    if arm.startswith('D'):
        pi=torch.arange(19,-1,-1)
        permutation=float((m(x[:,pi],context)-actual[:,pi]).abs().max())
        if permutation>1e-10:raise ValueError('MODE_PERMUTATION_NOT_EQUIVARIANT')
    return dict(numpy64_value_error=value_error,functional_torch_all_parameter_gradient_error=gradient_error,
                fixed_two_epsilon_parameter_FD=fd,permutation_error=permutation,
                fixture_not_production_initialization=True)


def main():
    started=time.perf_counter();torch.set_num_threads(1)
    out=RUN/'preflight';out.mkdir(exist_ok=False)
    verify(read_json(BASE/'source_manifest.json')['files'])
    verify(read_json(BASE/'B0_FINAL_LOCK.json')['files'])
    data=dict(np.load(BASE/'C0/fit.npz'))
    indices=hash_indices(data['ids'],'ud-preflight-v1',513)
    rg=guard_module();estimate=read_json(BASE/'MEMORY_ESTIMATE.json')['GPU_E']+128*1024**2
    guard=rg.Guard(estimate);guard.check(force=True)
    reports={};benchmark={}
    for arm in ARMS:
        print('PREFLIGHT_ARM_START',arm,flush=True)
        folder=out/arm;folder.mkdir()
        trainer=Trainer(data,np.arange(len(data['ids'])),arm,check_B0_order=True)
        fresh=cpu_tree(trainer.model.state_dict())
        p=torch.tensor(data['past'][indices[:8]],device='cuda');n=torch.tensor(data['num_valid'][indices[:8]],device='cuda')
        y=torch.tensor(data['future'][indices[:8]],device='cuda')
        trainer.model.eval()
        base,cfg=bf.make_model();base.eval()
        with torch.no_grad():
            a,z,_,_=bf.prediction(base,p,n,y)
            b,zz,_,_=bf.prediction(trainer.model,p,n,y)
            exact(a,b,'ZERO_RESIDUAL_INITIAL_PATHS');exact(z,zz,'ZERO_RESIDUAL_INITIAL_SCORE')
            pp,_,graph,_,_,_=bf.prepare(p,y)
            x=trainer.model(pp,None,None,mode='test',num_valid=n,graph_loc=graph)[2]
            xx=trainer.model(pp,torch.ones((8,9,12,2),device='cuda')*999,None,mode='test',num_valid=n,graph_loc=graph)[2]
            exact(x,xx,'ARBITRARY_GT_FORWARD_INDEPENDENCE')
        del base
        # Full actual model, same first three production batches and scheduler.
        phases=[]
        for _ in range(3):
            row=trainer.step();module=new_branch(trainer.model)
            norms={k:float(v.grad.norm()) if v.grad is not None else None for k,v in module.named_parameters()}
            phases.append(dict(step=trainer.updates,norms=norms))
        if not phases[1]['norms']['out.weight']>0:raise ValueError('STAGE2_WO_NOT_CONNECTED '+arm)
        if not all(v is not None and v>0 for k,v in phases[2]['norms'].items() if k!='out.weight'):
            raise ValueError('STAGE3_INTERNAL_NOT_CONNECTED '+arm)
        math=branch_math(arm,new_branch(trainer.model))
        # Existing full loss and score/path derivatives independently recalculated.
        trainer.model.eval()
        a,z,target,_=bf.prediction(trainer.model,p,n,y)
        a=a.detach().double().requires_grad_(True);z=z.detach().double().requires_grad_(True)
        criterion=bf.ModeQueryLoss(trainer.cfg).cuda().double()
        value,_=bf.loss_value(a,target.double(),z,criterion)
        ga,gz=torch.autograd.grad(value,(a,z))
        val,nga,ngz,_=loss_grad(a.detach().cpu().numpy(),target.double().cpu().numpy(),z.detach().cpu().numpy(),criterion.regression.weights.cpu().numpy())
        loss_errors=[abs(float(value)-val),float(np.max(abs(ga.cpu().numpy()-nga))),float(np.max(abs(gz.cpu().numpy()-ngz)))]
        if max(loss_errors)>1e-10:raise ValueError('FULL_LOSS_INDEPENDENT_MISMATCH')
        contract={'arm':arm,'source':sha(DESIGN),'split':sha(BASE/'C0/fit.npz'),
            'normalization':'B0_PAST_ONLY_UNCHANGED','IDs':data['ids'][indices].tolist()}
        continuous=Trainer(data,indices,arm,epochs=2)
        for _ in range(11):continuous.step();guard.check()
        mid=continuous.snapshot(contract);save_checkpoint(folder/'resume_mid.pt',mid)
        for _ in range(7):continuous.step();guard.check()
        final=continuous.snapshot(contract)
        restored=Trainer(data,indices,arm,epochs=2)
        restored.load(torch.load(folder/'resume_mid.pt',map_location='cpu',weights_only=False),contract)
        for _ in range(7):restored.step();guard.check()
        exact(final,restored.snapshot(contract),'NEW_PARAMS_OLD_BUFFERS_EMA_ADAM_SCHED_ALLRNG_ORDER_HISTORY')
        save_checkpoint(folder/'resume_final.pt',final)
        for key in ('arm','source','split','normalization'):
            wrong=dict(contract);wrong[key]='BAD'
            reject(lambda wrong=wrong:restored.load(mid,wrong))
        wrong=dict(contract);wrong['IDs']=list(contract['IDs']);wrong['IDs'][0]='BAD'
        reject(lambda:restored.load(mid,wrong))
        selected={k:v[indices[:128]] for k,v in data.items()}
        paths,logits=infer(restored.ema.module,selected)
        t=time.perf_counter();save_population(folder/'saved128',paths,logits,selected,guard);saved_seconds=time.perf_counter()-t
        done=read_json(folder/'saved128/done.json');verify(done['files'])
        bad=dict(done['files']);bad[str(folder/'saved128/metrics.json')]='0'*64
        reject(lambda:verify(bad))
        bench=Trainer(data,indices[:64],arm,epochs=36)
        times=[]
        for step in range(36):
            torch.cuda.synchronize();t=time.perf_counter();bench.step();torch.cuda.synchronize()
            if step>=4:times.append(time.perf_counter()-t)
            guard.check()
        reports[arm]=dict(status='PASS_ACTUAL_GPU_MATH_RESUME_SAVED_QA',
            parameters=sum(v.numel() for v in trainer.model.parameters()),
            initial_B0_all_original_weights_buffers_RNG_and_paths_exact=True,
            actual_production_first3_batchIDs_exact=True,three_stage_gradient_norms=phases,
            branch_math=math,unchanged_full_loss_value_path_score_gradient_errors=loss_errors,
            arbitrary_GT_forward_independent=True,
            actual_resume=dict(real_fit_windows=513,epochs=2,updates=18,interrupt_after=11,all_states_exact=True,not_full_fit_two_epochs=True),
            completed_model_source_output_readers_passed=True,bad_contracts_rejected=True,
            saved128_complete_quality_full_original_cluster_pdist_motion='PASS')
        benchmark[arm]=dict(batch=64,warmup=4,measured=32,mean_seconds=float(np.mean(times)),max_seconds=max(times),
            saved128_and_independent_arithmetic_seconds=saved_seconds)
        write_json(folder/'PREFLIGHT.json',reports[arm])
        # Drop all cached gradient graphs and models before next arm.
        del trainer,continuous,restored,bench,module,a,z,value,criterion,ga,gz
    guard.check(force=True);snap=rg.snapshot()
    payload=int(data['past'].nbytes+data['future'].nbytes+data['neighbor_ids'].nbytes)
    E=int(snap['peak_private']+payload+384*1024**2)
    write_json(RUN/'MEMORY_ESTIMATE.json',dict(GPU_E=E,CPU_E=3*1024**3,
        derivation=dict(actual_preflight_lifetime_private=snap['peak_private'],fit_payload=payload,extra_IO_reserve=384*1024**2),
        policy='warning_first_remaining_v3_no_fixed6GiB',preflight_not_production_peak=True,snapshot=snap))
    verify(read_json(BASE/'source_manifest.json')['files'])
    write_json(RUN/'PREFLIGHT.json',dict(status='PASS_ALL4_ACTUAL_UD_PREFLIGHT',at=now(),reports=reports,
        elapsed_seconds=time.perf_counter()-started,benchmark=benchmark,resource=guard.observations,
        no_production_models_trained=True,same_assistant_second_implementation_not_external_review=True))
    print('ALL4_PREFLIGHT_PASS',time.perf_counter()-started,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        if not (RUN/'PREFLIGHT_FAILURE.json').exists():write_json(RUN/'PREFLIGHT_FAILURE.json',dict(at=now(),status='FAILURE',traceback=traceback.format_exc()))
        raise
