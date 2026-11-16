"""Actual CUDA, independent objective arithmetic, all-RNG recovery and saved QA."""
import importlib.util
import traceback
import time
from common import *
from train_core import *
from loss_independent import loss_grad
from evaluate_saved import save_population


def guard_module():
    path=ROOT/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py'
    spec=importlib.util.spec_from_file_location('e2e_resource_guard',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def hash_indices(ids,tag,count):
    import hashlib
    return np.array(sorted(range(len(ids)),key=lambda i:hashlib.sha256((tag+'|'+str(ids[i])).encode()).digest())[:count])


def reject(operation):
    try:operation()
    except (ValueError, FileNotFoundError, KeyError):return True
    raise AssertionError('BAD_CONTRACT_WAS_ACCEPTED')


def main():
    started=time.perf_counter()
    out=HERE/'preflight_v2'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    rg=guard_module();guard=rg.Guard(6*1024**3)
    guard.check(force=True)
    verify(read_json(HERE/'C0/C0_RESULT.json')['outputs'])
    data=dict(np.load(HERE/'C0/fit.npz'))
    indices=hash_indices(data['ids'],'e2e-b0-preflight-v1',513)
    contract={'source':sha(HERE/'C0/C0_RESULT.json'),'split':sha(HERE/'C0/fit.npz'),
              'normalization':'past_only_heading_and_peragent_mean_no_fitted_stats',
              'epoch':2,'indices':data['ids'][indices].tolist()}
    trainer=Trainer(data,indices,epochs=2)
    initial=trainer.snapshot(contract)
    paired=Trainer(data,indices,epochs=2)
    exact(initial['model'],paired.snapshot(contract)['model'],'FRESH_PAIRED_ALL_WEIGHTS_BUFFERS')
    del paired
    # Actual 8 GPU batches, each with 8 real fit windows, before the resume test.
    phased=Trainer(data,indices[:64],batch=8,epochs=1)
    grad_rows=[]
    for _ in range(8):
        grad_rows.append(phased.step());guard.check()
    if not any(k.startswith('agent_aware_trans_encoder') for k in phased.connected):raise ValueError('NO_ENCODER_GRADIENT')
    if not any(k.startswith('decoder.score_head') for k in phased.connected):raise ValueError('NO_SCORE_GRADIENT')
    if not any(k.startswith('decoder.proposal_delta') for k in phased.connected):raise ValueError('NO_ANCHOR_GRADIENT')
    gradients={k:dict(numel=p.numel(),requires_grad=p.requires_grad,
        observed_nonzero=k in phased.connected,final_connected=p.grad is not None)
        for k,p in phased.model.named_parameters()}
    write_json(out/'GRADIENTS.json',gradients)
    # FUTURE/GT absent from the API; additionally test native source forward with arbitrary fut_loc.
    model=phased.model.eval()
    p=torch.tensor(data['past'][indices[:8]],device='cuda')
    y=torch.tensor(data['future'][indices[:8]],device='cuda')
    n=torch.tensor(data['num_valid'][indices[:8]],device='cuda')
    pnorm,ynorm,graph,_,_,_=prepare(p,y)
    with torch.no_grad():
        a=model(pnorm,None,None,mode='test',num_valid=n,graph_loc=graph)[2]
        b=model(pnorm,torch.ones((8,9,12,2),device='cuda')*9876,None,mode='test',num_valid=n,graph_loc=graph)[2]
        if not torch.equal(a,b):raise ValueError('GT_DEPENDENT_FORWARD')
    # Real nonzero model paths/logits, independent complete objective and derivatives.
    with torch.no_grad():
        op,lz,ty,_=prediction(model,p,n,y)
    pd=op.detach().double().requires_grad_(True);zd=lz.detach().double().requires_grad_(True)
    criterion=ModeQueryLoss(phased.cfg).cuda().double()
    loss,_=loss_value(pd,ty.double(),zd,criterion)
    gp,gz=torch.autograd.grad(loss,(pd,zd))
    value,ngp,ngz,constants=loss_grad(pd.detach().cpu().numpy(),ty.double().cpu().numpy(),zd.detach().cpu().numpy(),criterion.regression.weights.cpu().numpy())
    errs={'scalar':abs(value-float(loss.detach())),
          'all_path_gradient':float(np.max(abs(gp.cpu().numpy()-ngp))),
          'all_score_gradient':float(np.max(abs(gz.cpu().numpy()-ngz)))}
    if max(errs.values())>1e-10:raise ValueError('INDEPENDENT_OBJECTIVE_MISMATCH '+str(errs))
    # Fixed directions and eps; detached targets held fixed, no cherry-picking active masks.
    rng=np.random.default_rng(20261102);fd=[]
    p0=pd.detach().cpu().numpy();z0=zd.detach().cpu().numpy();yy=ty.double().cpu().numpy();weights=criterion.regression.weights.cpu().numpy()
    for k in range(2):
        direction=rng.normal(size=p0.shape);direction/=np.linalg.norm(direction)
        zdirection=rng.normal(size=z0.shape);zdirection/=np.linalg.norm(zdirection)
        ref=float((ngp*direction).sum()+(ngz*zdirection).sum())
        for eps in (1e-4,1e-5):
            vp=loss_grad(p0+eps*direction,yy,z0+eps*zdirection,weights,constants)[0]
            vm=loss_grad(p0-eps*direction,yy,z0-eps*zdirection,weights,constants)[0]
            err=abs((vp-vm)/(2*eps)-ref)
            if err>1e-6+1e-3*abs(ref):raise ValueError('FD_MISMATCH')
            fd.append(dict(direction=k,eps=eps,error=err))
    # Full actual 513 fit / 2 epochs / 18 updates, interrupted after update11.
    trainer=Trainer(data,indices,epochs=2)
    for _ in range(11):trainer.step();guard.check()
    mid=trainer.snapshot(contract);save_checkpoint(out/'resume_mid.pt',mid)
    for _ in range(7):trainer.step();guard.check()
    final=trainer.snapshot(contract)
    resumed=Trainer(data,indices,epochs=2)
    restored=torch.load(out/'resume_mid.pt',map_location='cpu',weights_only=False)
    resumed.load(restored,contract)
    for _ in range(7):resumed.step();guard.check()
    exact(final,resumed.snapshot(contract),'TWO_EPOCH_ALL_PARAMS_EMA_OPT_SCHED_ALL_RNG_ORDER_HISTORY')
    save_checkpoint(out/'resume_final.pt',final)
    for key,bad in [('source','BAD'),('split','BAD'),('normalization','BAD')]:
        wrong=dict(contract);wrong[key]=bad
        reject(lambda wrong=wrong:resumed.load(restored,wrong))
    wrong=dict(contract);wrong['indices']=list(contract['indices']);wrong['indices'][0]='BAD_ID'
    reject(lambda:resumed.load(restored,wrong))
    # Production-path saved readback and independent complete-link/physics on 128 real windows.
    selected={k:v[indices[:128]] for k,v in data.items()}
    paths,logits=infer(resumed.ema.module,selected)
    t=time.perf_counter();metrics=save_population(out/'saved128',paths,logits,selected,guard)
    saved_seconds=time.perf_counter()-t
    done=read_json(out/'saved128/done.json');verify(done['files'])
    bad=dict(done['files']);bad[str(out/'saved128/metrics.json')]='0'*64
    reject(lambda:verify(bad))
    wrongpaths=paths.copy();wrongpaths[0,0,0,0]=float('nan')
    from evaluate_saved import evaluate
    reject(lambda:evaluate(wrongpaths,logits,selected))
    # Throwaway actual-batch throughput, no outcome/model selected from benchmark.
    bench=Trainer(data,indices[:64],epochs=36)
    times=[]
    for i in range(36):
        torch.cuda.synchronize();t=time.perf_counter();bench.step();torch.cuda.synchronize()
        if i>=4:times.append(time.perf_counter()-t)
        guard.check()
    peak=rg.snapshot();guard.check(force=True)
    # Explicit extra array, graph/allocator reserves above measured Windows lifetime private.
    payload=int(data['past'].nbytes+data['future'].nbytes+data['neighbor_ids'].nbytes)
    E=int(peak['peak_private']+payload+512*1024**2)
    write_json(HERE/'MEMORY_ESTIMATE.json',{'GPU_E':E,'CPU_E':3*1024**3,
        'derivation':{'measured_preflight_lifetime_private':peak['peak_private'],'causal_fit_array_payload':payload,'additional_graph_io_reserve':512*1024**2},
        'policy':'warning_first_v3_remaining; 1.3 is warning only, sustained current pressure stop',
        'preflight_snapshot':peak,'not_production_peak':True})
    estimate_steps=math.ceil(len(data['ids'])/64)*8
    # Saved review already double-implements clusters and physics. No second complete-save estimate.
    population_windows=len(data['ids'])+len(np.load(HERE/'C0/dev.npz')['ids'])
    planning=1.25*(max(times)*estimate_steps+saved_seconds/128*population_windows+180+180+90)
    # 180 cost measurement, 180 fixed profile/load/SHA allowance, 90 IO/checkpoint.
    report={'status':'PASS_B0_ACTUAL_PREFLIGHT','at':now(),'elapsed_seconds':time.perf_counter()-started,
        'parameters':sum(p.numel() for p in resumed.model.parameters()),'trainable':sum(p.numel() for p in resumed.model.parameters() if p.requires_grad),
        'gradient_phases':'8 actual FIT8 updates; all finite; connected names in GRADIENTS, unused registered modules explicitly retained',
        'paired_initial_all_weights_buffers_exact':True,'GT_arbitrary_vs_none_exact':True,
        'independent_objective':errs,'fixed_FD':fd,'resume':{'windows':513,'epochs':2,'updates':18,'interrupt_after':11,
             'all_model_buffer_optimizer_EMA_scheduler_RNG_sampler_history_exact':True,'not_full_fit_two_epochs':True},
        'bad_source_split_normalization_ID_output_rejected':True,'saved128_full_quality_pdist_motion':'PASS',
        'benchmark':{'warmup':4,'measure':32,'batch':64,'mean_step_seconds':float(np.mean(times)),
                     'max_step_seconds':max(times),'saved128_double_implementation_seconds':saved_seconds,
                     'full_train_steps':estimate_steps,'population_windows':population_windows},
        'planning_seconds':planning,'budget_seconds':1800,'budget_admitted':planning<=1800,
        'resource':guard.observations,'same_assistant_arithmetic_not_external_review':True}
    write_json(HERE/'PREFLIGHT.json',report)
    write_json(HERE/'RESOLVED_CONFIG.json',{k:v for k,v in resumed.cfg.items() if k!='device'})
    print('PREFLIGHT_PASS',report['elapsed_seconds'],'planning',planning,'step',np.mean(times),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        path=HERE/'PREFLIGHT_FAILURE.json'
        if not path.exists():write_json(path,{'status':'FAILURE','at':now(),'traceback':traceback.format_exc()})
        raise
