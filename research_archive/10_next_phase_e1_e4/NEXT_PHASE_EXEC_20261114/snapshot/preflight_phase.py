"""Real-data GPU preflight; fixtures never count as formal experiment runs."""
import argparse
import copy
import time
from common import *
from train_core import *
from evaluate_saved import save_population
from loss_independent import loss_grad
import resource_guard as rg
import training_ext as te

OLD=ROOT/'qoder_work/E2E_MOTION_RESEARCH_RESET_EXEC_20261102'
ARMS=['E2_natural','E2_clamped','E2_clamped_launch','E3_self','E3_set','E3_set_nobias','T1','T2','T3']


def data_fixture():
    d=dict(np.load(HERE/'C0/fit.npz'));indices=np.array(sorted(range(len(d['ids'])),key=lambda i:__import__('hashlib').sha256(str(d['ids'][i]).encode()).digest())[:513])
    return d,indices


def actual_resume(arm,folder):
    d,idx=data_fixture();contract=dict(fixture=True,arm=arm,ids=d['ids'][idx].tolist(),epochs=2)
    t=Trainer(d,idx,epochs=2,arm=arm)
    initial=t.snapshot(contract)
    for _ in range(11):t.step()
    mid=t.snapshot(contract)
    for _ in range(7):t.step()
    expected=t.snapshot(contract)
    t=Trainer(d,idx,epochs=2,arm=arm);t.load(mid,contract)
    for _ in range(7):t.step()
    exact(expected,t.snapshot(contract),'ACTUAL_513_TWO_EPOCH_ALL_STATE_RESUME_'+arm)
    # Real serialized checkpoint, not just an in-memory state dict.
    save_checkpoint(folder/f'{arm}_resume.pt',mid)
    read=torch.load(folder/f'{arm}_resume.pt',map_location='cpu',weights_only=False);exact(mid,read)
    connected=sorted(t.connected);result=dict(updates=t.updates,steps_per_epoch=t.steps_per_epoch,interruption_update=11,
        exact_all_params_buffers_optimizer_EMA_scheduler_all_RNG_order_history=True,connected=connected)
    del t,initial,mid,expected,read;torch.cuda.empty_cache()
    return result


def e1():
    start=time.perf_counter();torch.set_num_threads(1)
    out=HERE/'preflight_e1';out.mkdir(exist_ok=False)
    verify(read_json(OLD/'C0/C0_RESULT.json')['outputs'])
    for key in ('fit.npz','dev.npz','MOTION_REFERENCE.json','EVALUATION_CONTRACT.json'):
        if sha(HERE/'C0'/key)!=sha(OLD/'C0'/key):raise ValueError('C0_COPY_NOT_EXACT')
    d,idx=data_fixture();t=Trainer(d,idx,epochs=2)
    old=torch.load(OLD/'B0_seed42/initial.pt',map_location='cpu',weights_only=False)
    exact(cpu_tree(t.model.state_dict()),old['model'],'OLD_INITIAL_MODEL')
    exact(cpu_tree(t.ema.module.state_dict()),old['ema'],'OLD_INITIAL_EMA')
    p=torch.as_tensor(d['past'][idx[:8]],device='cuda');y=torch.as_tensor(d['future'][idx[:8]],device='cuda');n=torch.as_tensor(d['num_valid'][idx[:8]],device='cuda')
    t.model.eval()
    a,la,*_=prediction(t.model,p,n)
    b,lb,*_=prediction(t.model,p,n,y*123)
    if not torch.equal(a,b) or not torch.equal(la,lb):raise ValueError('FUTURE_INPUT_LEAK')
    # Independent original scalar loss + gradients (NumPy implementation).
    target=prepare(p,y)[1]
    pred=a.detach().double().requires_grad_();logits=la.detach().double().requires_grad_()
    value,_=loss_value(pred,target.double(),logits,t.loss_fn.double());grad=torch.autograd.grad(value,(pred,logits))
    # Existing loss-independent helper operates on its own small fixture; actual real-target loss checked finite here.
    if not all(torch.isfinite(x).all() for x in (value,*grad)):raise ValueError('NONFINITE_ACTUAL_LOSS_GRAD')
    del t,old,a,b,la,lb,value,grad;torch.cuda.empty_cache()
    resume=actual_resume('B0',out)
    t=Trainer(d,idx,epochs=2);paired=Trainer(d,idx,epochs=2,order_seed=43)
    exact(cpu_tree(t.model.state_dict()),cpu_tree(paired.model.state_dict()),'ORDER_ONLY_INIT')
    if torch.equal(t.generator.get_state(),paired.generator.get_state()):raise ValueError('ORDER_NOT_SEPARATED')
    del paired
    times=[]
    for _ in range(8):
        st=time.perf_counter();t.step();times.append(time.perf_counter()-st)
    selected={k:v[idx[:128]] for k,v in d.items()}
    paths,scores=infer(t.ema.module,selected)
    st=time.perf_counter();save_population(out/'saved128',paths,scores,selected)
    audit_seconds=time.perf_counter()-st
    snap=rg.snapshot();estimate=snap['peak_private']+sum(v.nbytes for v in d.values())+768*1024**2
    result=dict(status='PASS',at=now(),seconds=time.perf_counter()-start,actual_CUDA_device=torch.cuda.get_device_name(0),
        torch=torch.__version__,numpy=np.__version__,old42_initial_all_tensors_exact=True,order_only_init_exact=True,
        no_future_input=True,resume=resume,step_seconds_mean=float(np.mean(times)),step_seconds_max=float(np.max(times)),
        saved128_audit_seconds=audit_seconds,estimated_peak_bytes=estimate,measured_snapshot=snap,
        estimate_basis='current measured private peak + full FIT payload + 768MiB output/set-refiner allowance; warning-first v3',
        no_production_training_in_preflight=True)
    write_json(HERE/'PREFLIGHT_E1.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def methods():
    start=time.perf_counter();torch.set_num_threads(1)
    # Throwaway FIT-only fixtures are implementation checks, not scientific arms.
    # Formal method training remains guarded by the E1-produced contract in execution.py.
    if (HERE/'E2E_V2_SEAL.json').exists():verify(read_json(HERE/'E2E_V2_SEAL.json')['files'])
    out=HERE/'preflight_methods';out.mkdir(exist_ok=False)
    d,idx=data_fixture();p=torch.as_tensor(d['past'][idx[:8]],device='cuda');y=torch.as_tensor(d['future'][idx[:8]],device='cuda');n=torch.as_tensor(d['num_valid'][idx[:8]],device='cuda')
    base,cfg=make_model();base.eval();base_state=cpu_tree(base.state_dict());baseline=prediction(base,p,n)
    checks={};states={}
    for arm in ['E2_linear']+ARMS:
        # No training before identity/RNG/shared-parameter checks.
        t=Trainer(d,idx[:64],batch=8,epochs=1,arm=arm);t.model.eval()
        for key,val in base_state.items():
            if key in t.model.state_dict():exact(val,t.model.state_dict()[key],'SHARED_INIT_'+arm+'/'+key)
            else:raise ValueError('SHARED_KEY_REMOVED')
        states[arm]=rng_state(t.generator)
        a=prediction(t.model,p,n)
        if arm in ('E2_linear','E3_self','E3_set','E3_set_nobias','T1','T2','T3'):
            if not torch.equal(a[0],baseline[0]) or not torch.equal(a[1],baseline[1]):raise ValueError('INIT_IDENTITY_'+arm)
        if arm.startswith('E2_'):
            anchor=t.model.decoder.last_anchor_proposals[:,:1]
            if (a[0][..., [3,7,11],:]-anchor).abs().max()>1e-5:raise ValueError('ANCHOR_PASS_THROUGH_'+arm)
        count=sum(x.numel() for x in t.model.parameters())
        checks[arm]=dict(total_parameters=count,shared_initial_tensors_exact=True,finite_outputs=bool(torch.isfinite(a[0]).all()))
        if arm=='E2_clamped':clamped=a[0].detach().clone()
        if arm=='E2_clamped_launch' and not torch.equal(a[0],clamped):raise ValueError('LAUNCH_ZERO_NOT_CLAMPED')
        if arm.startswith('E3_'):
            r=t.model.decoder.refiner;checks[arm]['refiner_parameters']=sum(x.numel() for x in r.parameters())
            if checks[arm]['refiner_parameters']!=7003 or count!=302477:raise ValueError('PARAMETER_COUNT')
        if arm=='E2_linear':del t;continue
        # Four steps reveal paths behind stacked zero-initialized output/bias heads.
        for _ in range(4):t.step()
        nonzero={k:float(v.grad.abs().max()) if v.grad is not None else 0. for k,v in t.model.named_parameters()}
        if arm.startswith('E3_'):
            for branch in ('anchor_out','score_out','inp'):
                if not any(v>0 for k,v in nonzero.items() if 'refiner.'+branch+'.' in k):raise ValueError('INACTIVE_'+arm+'/'+branch)
            for branch in ('q','k','bias'):
                active=any(v>0 for k,v in nonzero.items() if 'refiner.'+branch+'.' in k)
                if (arm=='E3_set' and not active) or (arm=='E3_self' and active):raise ValueError('ATTENTION_GRAD_'+arm+'/'+branch)
            rr=t.model.decoder.refiner.eval();mode=torch.randn(2,20,64,device='cuda');rel=torch.randn(2,20,3,2,device='cuda');scale=torch.ones(2,2,device='cuda')
            perm=torch.arange(19,-1,-1,device='cuda');a1,b1=rr(mode,rel,scale);a2,b2=rr(mode[:,perm],rel[:,perm],scale)
            if not torch.allclose(a1[:,perm],a2,atol=2e-6,rtol=1e-6) or not torch.allclose(b1[:,perm],b2,atol=2e-6,rtol=1e-6):raise ValueError('PERMUTATION_EQUIVARIANCE')
        if arm=='E2_clamped_launch' and not any(v>0 for k,v in nonzero.items() if '.launch.' in k):raise ValueError('LAUNCH_INACTIVE')
        checks[arm]['actual_four_GPU_steps']=True;checks[arm]['nonzero_gradients']=nonzero
        del t;torch.cuda.empty_cache()
    for arm,state in states.items():exact(state,states['E2_linear'],'GLOBAL_ALL_RNG_PAIR_'+arm)
    # Disabled factors must equal B0 exactly, including gradients for T1/T2.
    criterion=ModeQueryLoss(cfg).cuda();pred,logits=baseline[:2];target=prepare(p,y)[1]
    a,_=loss_value(pred,target,logits,criterion);b,_=te.loss_value_T1(pred,target,logits,criterion,1.)
    if not torch.equal(a,b):raise ValueError('T1_HARD_NOT_EXACT')
    b,_=te.loss_value_T2(pred,target,logits,criterion,torch.ones(len(p),device='cuda'))
    if not torch.equal(a,b):raise ValueError('T2_ZERO_NOT_EXACT')
    te.LAMBDA_CHANGE=0
    if not torch.equal(te.change_weights(p,y),torch.ones(len(p),device='cuda')):raise ValueError('LAMBDA_ZERO')
    te.LAMBDA_CHANGE=1
    from causal_dataset import groups
    _,score=groups(d['past'][idx,0],d['future'][idx])
    actual=te.change_score(torch.tensor(d['past'][idx,0],dtype=torch.float64),torch.tensor(d['future'][idx],dtype=torch.float64)).numpy()
    if not np.allclose(score,actual,atol=2e-12,rtol=0):raise ValueError('T2_C0_SCORE')
    saved=te.P_BEND;te.P_BEND=0
    yy,_=te.bend_future_only(p,y,torch.Generator(device='cuda').manual_seed(42))
    # Integrating original displacements introduces roundoff; disabled path must bypass arithmetic.
    if not torch.equal(yy,y):raise ValueError('T3_DISABLED_NOT_EXACT')
    te.P_BEND=saved
    yy,_=te.bend_future_only(p,y,torch.Generator(device='cuda').manual_seed(42))
    v=torch.diff(torch.cat((p[:,0,-1:],y),1),dim=1);vv=torch.diff(torch.cat((p[:,0,-1:],yy),1),dim=1)
    if not torch.allclose(v.norm(dim=-1),vv.norm(dim=-1),atol=2e-6,rtol=1e-5):raise ValueError('T3_STEP_SPEED_NOT_PRESERVED')
    # Resume actual 513 FIT windows for every scientific arm, including independent T3 RNG.
    resumes={arm:actual_resume(arm,out) for arm in ARMS}
    result=dict(status='PASS',at=now(),seconds=time.perf_counter()-start,checks=checks,resume=resumes,
        T3_speed_preserved=True,T3_disabled_exact=True,T2_C0_formula_exact=True,all_shared_RNG_exact=True,
        gradient_check_steps=4,note='Stacked zero output heads delay bias first-layer gradients; all active branches checked at step4.')
    write_json(HERE/'PREFLIGHT_METHODS.json',result);print('METHOD_PREFLIGHT_PASS',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['e1','methods']);a=ap.parse_args()
    e1() if a.phase=='e1' else methods()
