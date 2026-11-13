import time,traceback,math
from run_common import *
from train_core import Trainer,infer
from independent import loss_and_path_gradient,decoder_np

def reject(fn):
    try:fn()
    except (ValueError,FileNotFoundError,KeyError,RuntimeError):return True
    raise AssertionError('BAD_INPUT_ACCEPTED')
def subset(cache,ix):return {k:v[ix] for k,v in cache.items()}

def main():
    t=time.perf_counter();torch.set_num_threads(1);out=RUN/'preflight_v2';out.mkdir(exist_ok=False)
    check_files(read(RUN/'PRE_SOURCE.json')['files']);cache=cache_read('fit');dev=cache_read('dev')
    rg=rg_module();guard=rg.Guard(4*1024**3);guard.check(mode='startup',force=True)
    adapter,_=load_current('cuda');old=cpu(adapter.model.state_dict());ix=hash_indices(cache['ids'],'current-dsf-preflight-v1',513)
    paired=[];gradients={};resumes={};saved={};benchmark={}
    for arm in ('uniform','kin'):
        progress('PREFLIGHT_PHASED_GRADIENT',arm=arm)
        tr=Trainer(adapter,cache,ix[:256],arm,42,epochs=1)
        paired.append(cpu(tr.sampler.state_dict()))
        for _ in range(8):tr.step();guard.check()
        if tr.connected!=set(n for n,p in tr.sampler.named_parameters()):raise ValueError('NOT_ALL_DSF_TENSORS_PHASED_CONNECTED')
        gradients[arm]=dict(steps=8,batch=32,all_parameter_tensors_nonzero_after_phases=sorted(tr.connected))
        # Full REAL first hash513 FIT subset, 2 epochs / 34 steps. Never dev training.
        progress('PREFLIGHT_REAL_RESUME',arm=arm)
        tr=Trainer(adapter,cache,ix,arm,42,epochs=2)
        for _ in range(19):tr.step();guard.check()
        mid=tr.snapshot();save_pt(out/f'{arm}_mid.pt',mid)
        for _ in range(15):tr.step();guard.check()
        end=tr.snapshot();restored=Trainer(adapter,cache,ix,arm,42,epochs=2)
        restored.load(torch.load(out/f'{arm}_mid.pt',map_location='cpu',weights_only=False))
        for _ in range(15):restored.step();guard.check()
        exact(end,restored.snapshot(),'REAL513_TWOEPOCH_PARAMS_OPT_ALLRNG_CURSOR_HISTORY_'+arm)
        resumes[arm]=dict(N=513,epochs=2,updates=34,interrupt_after=19,ALL_DSF_params_optimizer_CPU_CUDA_loader_Python_NumPy_rng_order_history_EXACT=True,not_full21719_two_epochs=True)
        for key in ('pre_source_sha','cache_receipt','protocol_sha','normalization','frozen_sha','indices'):
            bad=copy.deepcopy(mid);bad['contract'][key]='BAD';reject(lambda bad=bad:restored.load(bad))
        # 12 dev observations only, NO labels or optimizer updates.
        small_dev=subset(dev,hash_indices(dev['ids'],'current-dsf-preflight-dev-observation-v1',12));infer(adapter,restored.sampler,small_dev,guard)
        selection=subset(cache,ix[:128]);paths,logits,z=infer(adapter,restored.sampler,selection,guard)
        # Load old canonical evaluator definitions after baseline_api installed namespace.
        from evaluate import evaluate_population,independent_saved_review
        ts=time.perf_counter();folder=out/'saved'/f'{arm}_seed42'/'fit';evaluate_population(folder,paths,logits,z,selection,'fit',guard,preflight=True)
        review=independent_saved_review(adapter,selection,folder,arm,guard);saved[arm]=dict(review=review,elapsed=time.perf_counter()-ts)
        bad=dict(read(folder/'done.json')['files']);bad[next(iter(bad))]='0'*64;reject(lambda:check_files(bad))
        path_bad=paths.copy();path_bad[0,0,0,0]=np.nan
        from evaluate_saved import evaluate as ev
        f=read(BASE/'C0/C0_RESULT.json') # only source receipt, not GT used as input.
        reject(lambda:ev(path_bad,logits,{'ids':selection['ids']}))
        # Production checkpoint reader and completed SHA contract branch.
        completed=out/f'{arm}_completed.pt';save_pt(completed,end);receipt={'files':seal([completed]),'contract':end['contract']};check_files(receipt['files']);verify_contract(torch.load(completed,map_location='cpu',weights_only=False)['contract'],restored.contract)
        wrong=copy.deepcopy(receipt['contract']);wrong['arm']='BAD';reject(lambda:verify_contract(wrong,restored.contract))
        bench=Trainer(adapter,cache,ix[:32],arm,42,epochs=36);times=[]
        for j in range(36):
            torch.cuda.synchronize();ts=time.perf_counter();bench.step();torch.cuda.synchronize()
            if j>=4:times.append(time.perf_counter()-ts)
            guard.check()
        benchmark[arm]=dict(warmup=4,measured=32,batch=32,mean=float(np.mean(times)),max=max(times))
    exact(paired[0],paired[1],'UNIFORM_KIN_PAIRED_ALL_INITIAL_WEIGHTS_BUFFERS')
    # Independent objective scalar, all path grads and full parameter-chain.
    progress('PREFLIGHT_INDEPENDENT_GRADIENT_AND_FIXED_FD')
    double_adapter=type(adapter)(copy.deepcopy(adapter.model).cpu().double(),adapter.api);b=batch_bundle(cache,ix[:8],'cpu',torch.float64)
    fd=[];independent={};base_sampler=new_sampler(42,'cpu').double()
    with torch.no_grad():
        for j,p in enumerate(base_sampler.parameters()):p.add_(.01*torch.sin(torch.arange(p.numel(),dtype=torch.double).reshape_as(p)+j+1))
    for arm in ('uniform','kin'):
        sampler=copy.deepcopy(base_sampler);loss,diag=objective(double_adapter,sampler,b,arm,read(STAGE_A/'CALIBRATION.json'))
        y=diag['paths'];tg=torch.autograd.grad(loss,y,retain_graph=True)[0]
        value,npgrad,faces=loss_and_path_gradient(b['raw_past'][:,0].numpy(),y.detach().numpy(),read(STAGE_A/'CALIBRATION.json'),arm)
        if abs(value-float(loss.detach()))>1e-10 or np.max(abs(npgrad-tg.detach().numpy()))>1e-10:raise ValueError('INDEPENDENT_ALL_PATH_GRADIENT')
        params=list(sampler.parameters());allg=torch.autograd.grad(loss,params,retain_graph=True);chain=torch.autograd.grad(y,params,grad_outputs=torch.tensor(npgrad,dtype=torch.double),retain_graph=True)
        maxchain=max(float((a-c).abs().max()) for a,c in zip(allg,chain))
        if maxchain>1e-10:raise ValueError('INDEPENDENT_ALL_PARAMETER_CHAIN')
        original=[p.detach().clone() for p in params]
        for direction in range(2):
            ds=[torch.sin(torch.arange(p.numel(),dtype=torch.double).reshape_as(p)+direction+1) for p in params];norm=torch.sqrt(sum((d*d).sum() for d in ds));ds=[d/norm for d in ds];ref=sum(float((a*d).sum()) for a,d in zip(allg,ds))
            for eps in (1e-4,1e-5):
                vals=[]
                for sign in (1,-1):
                    with torch.no_grad():
                        for p,o,d in zip(params,original,ds):p.copy_(o+sign*eps*d)
                    lv,dg=objective(double_adapter,sampler,b,arm,read(STAGE_A/'CALIBRATION.json'));vals.append(float(lv.detach()))
                    _,_,active=loss_and_path_gradient(b['raw_past'][:,0].numpy(),dg['paths'].detach().numpy(),read(STAGE_A/'CALIBRATION.json'),arm)
                    for key in ('v_active','a_active','j_active'):
                        if not np.array_equal(active[key],faces[key]):raise ValueError('FIXED_FD_ACTIVE_FACE_SWITCH_NO_EPS_SEARCH')
                error=abs((vals[0]-vals[1])/(2*eps)-ref)
                if error>1e-6+1e-3*abs(ref):raise ValueError('FIXED_FD_ERROR')
                fd.append(dict(arm=arm,direction=direction,eps=eps,error=error,active_faces_exact=True))
            with torch.no_grad():
                for p,o in zip(params,original):p.copy_(o)
        independent[arm]=dict(scalar_error=abs(value-float(loss.detach())),all_path_gradient_error=float(np.max(abs(npgrad-tg.detach().numpy()))),full_parameter_chain_error=maxchain)
    exact(old,cpu(adapter.model.state_dict()),'ALL_OLD_FROZEN_WEIGHTS_BOOL_BUFFERS')
    check_files(read(RUN/'PRE_SOURCE.json')['files']);guard.check(force=True);snap=rg.snapshot()
    largest_payload=max(x['bytes_payload'] for x in read(RUN/'CACHE_RECEIPT.json')['splits'].values())
    E=int(snap['peak_private']+largest_payload+1024**3) # largest loaded cache + explicit output/control/motion/allocator reserve.
    windows=21719+4753;steps=679*20
    train_seconds=3*steps*sum(x['max'] for x in benchmark.values())
    saved_seconds=3*windows/128*sum(x['elapsed'] for x in saved.values())
    planning=1.25*(train_seconds+saved_seconds+read(RUN/'CACHE_RECEIPT.json')['elapsed_seconds']+240+120)
    write(RUN/'MEMORY_ESTIMATE.json',dict(policy='warning_first_v3',GPU_E=E,CPU_E=2*1024**3,measured_peak_private=snap['peak_private'],largest_cache_payload=largest_payload,explicit_extra_output_control_motion_allocator_reserve=1024**3,not_available_memory_backsolved=True))
    write(RUN/'PREFLIGHT.json',dict(status='PASS_REAL_CURRENT_DSF_PAIRED_GRAD_RESUME_SAVED_AUDIT',at=now(),elapsed_seconds=time.perf_counter()-t,pre_source_sha=sha(RUN/'PRE_SOURCE.json'),phased_gradients=gradients,paired_initial_exact=True,resume=resumes,independent_objective=independent,fixed_FD=fd,saved128=saved,bad_source_split_normalization_ID_checkpoint_output_rejected=True,benchmark=benchmark,planning_seconds=planning,ETA_seconds=[.7*planning,1.5*planning],all_old_parameters_buffers_exact=True,old_generator_no_grad=True,new_scientific_training_runs=0,resource=snap,observations=guard.observations))
    print('PREFLIGHT_PASS',planning/60,'minutes',time.perf_counter()-t,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(RUN/'PREFLIGHT_V2_FAILURE.json',dict(at=now(),error=repr(e),traceback=traceback.format_exc(),production_started=False));raise
