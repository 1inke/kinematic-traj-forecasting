"""Actual four-arm admission. No fabricated numbers from supplied design."""
import copy,time,traceback
from geo_common import *
from train_core_geo import *
from geo_factory import make_geo_model
from geo_modules import compute_anchor_geo_features_12d
from geo_math import check_component,geom
from v2_numerical_checks import additional_checks
from evaluate_geo import save_population
from preflight import guard_module,hash_indices,reject
from loss_independent import loss_grad


def main():
    start=time.perf_counter();torch.set_num_threads(1);out=RUN/'preflight';out.mkdir(exist_ok=False)
    verify(read_json(BASE/'source_manifest.json')['files']);verify(read_json(V1/'source_manifest.json')['files'])
    data=dict(np.load(BASE/'C0/fit.npz'));ids=hash_indices(data['ids'],'dgeo-preflight-v1',513)
    rg=guard_module();guard=rg.Guard(read_json(UD/'MEMORY_ESTIMATE.json')['GPU_E']);guard.check(force=True)
    additional=additional_checks(out);reports={};benchmarks={}
    # Fixed geometry fixture away from atan2's negative-axis cut, no fixture/epsilon search.
    a=torch.linspace(.1,2.,8*20*3*2,dtype=torch.float64).reshape(8,20,3,2)
    past=torch.linspace(-.7,.2,8*8*2,dtype=torch.float64).reshape(8,8,2)
    ge=compute_anchor_geo_features_12d(a,past)
    if np.max(abs(ge.numpy()-geom(a.numpy(),past.numpy(),np)))>1e-12:raise ValueError('GEOMETRY_VALUE')
    for arm in ARMS:
        print('GEO_PREFLIGHT_START',arm,flush=True);folder=out/arm;folder.mkdir()
        t=Trainer(data,np.arange(len(data['ids'])),arm,check_B0_order=True)
        old,_=bf.make_model();old.eval();t.model.eval()
        p=torch.tensor(data['past'][ids[:8]],device='cuda');n=torch.tensor(data['num_valid'][ids[:8]],device='cuda');y=torch.tensor(data['future'][ids[:8]],device='cuda')
        with torch.no_grad():
            b,z,_,_=bf.prediction(old,p,n);v,w,_,_=bf.prediction(t.model,p,n)
            exact(b,v,'INITIAL_WORLD_PATHS');exact(z,w,'INITIAL_SCORE')
            pp,_,graph,_,_,_=bf.prepare(p,y)
            b=t.model(pp,None,None,mode='test',num_valid=n,graph_loc=graph)[2]
            v=t.model(pp,torch.ones((8,9,12,2),device='cuda')*999,None,mode='test',num_valid=n,graph_loc=graph)[2]
            exact(b,v,'ARBITRARY_FUTURE_FORWARD')
        initial={k:float(v) if v.ndim==0 else None for k,v in t.model.decoder.coordinator.diagnostics.items() if k not in ('attention_weights','update_norm')}
        del old
        phases=[]
        for _ in range(3):
            t.step();norms={k:float(v.grad.norm()) if v.grad is not None else None for k,v in t.model.decoder.named_parameters() if k.startswith(('coordinator.','proposal_delta.'))}
            phases.append(dict(step=t.updates,norms=norms,proposal_delta_rows=[float(v) for v in t.model.decoder.proposal_delta.weight.grad.norm(dim=1)]))
        first_connected=next((r['step'] for r in phases if all(v is not None and v>0 for v in r['norms'].values())),None)
        if first_connected is None:raise ValueError('NO_REAL_NEW_BRANCH_CONNECTIVITY_BY_STEP3 '+arm)
        if not any(all(v>0 and np.isfinite(v) for v in row['proposal_delta_rows']) for row in phases):raise ValueError('ALL6_PROPOSAL_CHANNELS_NOT_CONNECTED '+arm)
        # First2 equality is observation, not a target of initialization or epsilon tuning.
        zfix=torch.linspace(-.4,.6,8*20*64,dtype=torch.float64).reshape(8,20,64)
        cfix=torch.linspace(-.2,.4,8*64,dtype=torch.float64).reshape(8,64)
        maths={'coordinator':check_component(t.model.decoder.coordinator,zfix,cfix,ge,'coord',True)}
        if t.model.decoder.rel_scorer is not None:maths['scorer']=check_component(t.model.decoder.rel_scorer,zfix,a,past,'score')
        t.model.eval();v,w,target,_=bf.prediction(t.model,p,n,y)
        v=v.detach().double().requires_grad_(True);w=w.detach().double().requires_grad_(True)
        criterion=bf.ModeQueryLoss(t.cfg).cuda().double();value,_=bf.loss_value(v,target.double(),w,criterion)
        gv,gw=torch.autograd.grad(value,(v,w));nv,ngv,ngw,_=loss_grad(v.detach().cpu().numpy(),target.double().cpu().numpy(),w.detach().cpu().numpy(),criterion.regression.weights.cpu().numpy())
        loss_error=max(abs(float(value)-nv),float(np.max(abs(gv.cpu().numpy()-ngv))),float(np.max(abs(gw.cpu().numpy()-ngw))))
        if loss_error>1e-10:raise ValueError('INHERITED_LOSS_GRADIENT')
        contract=dict(arm=arm,source=sha(DESIGN),split=sha(BASE/'C0/fit.npz'),normalization='B0_UNCHANGED',IDs=data['ids'][ids].tolist())
        c=Trainer(data,ids,arm,epochs=2)
        for _ in range(11):c.step();guard.check()
        mid=c.snapshot(contract);save_checkpoint(folder/'resume_mid.pt',mid)
        for _ in range(7):c.step();guard.check()
        final=c.snapshot(contract);save_checkpoint(folder/'resume_final.pt',final)
        rr=Trainer(data,ids,arm,epochs=2);rr.load(torch.load(folder/'resume_mid.pt',map_location='cpu',weights_only=False),contract)
        for _ in range(7):rr.step();guard.check()
        exact(final,rr.snapshot(contract),'FULL_MODEL_EMA_ADAM_SCHED_ALLRNG_HISTORY_ORDER')
        for key in contract:
            wrong=dict(contract);wrong[key]='BAD';reject(lambda wrong=wrong:rr.load(mid,wrong))
        selected={k:v[ids[:128]] for k,v in data.items()};paths,logits=infer(rr.ema.module,selected,guard=guard)
        ts=time.perf_counter();save_population(folder/'saved128',paths,logits,selected,guard);save_time=time.perf_counter()-ts
        done=read_json(folder/'saved128/done.json');verify(done['files']);wrong=dict(done['files']);wrong[next(iter(wrong))]='0'*64;reject(lambda:verify(wrong))
        bench=Trainer(data,ids[:64],arm,epochs=36);times=[]
        for i in range(36):
            torch.cuda.synchronize();ts=time.perf_counter();bench.step();torch.cuda.synchronize()
            if i>=4:times.append(time.perf_counter()-ts)
            guard.check()
        reports[arm]=dict(status='PASS_ACTUAL_GEO_GPU_MATH_RESUME_SAVED',parameters=sum(v.numel() for v in t.model.parameters()),
            initial_B0_all_original_weights_buffers_RNG_paths_logits_exact=True,initial_actual_diagnostics=initial,
            actual_first3_batchIDs_exact=True,phased_gradients=phases,first_all_new_tensor_positive_step=first_connected,
            advertised_numeric_gradient_targets_not_hard_gates=True,independent_maths=maths,inherited_full_loss_error=loss_error,
            arbitrary_future_forward_independence=True,actual_resume=dict(windows=513,epochs=2,updates=18,interrupt_after=11,all_states_exact=True,not_full_fit_two_epochs=True),
            bad_source_split_stats_ID_arm_output_readers_reject=True,saved128_original_fullK20_complete_link_pdist_quality_pooled_junction='PASS')
        if reports[arm]['parameters']>305000:reports[arm]['cost_target']='OVER_TARGET_DISCLOSED_NOT_SHRUNK'
        benchmarks[arm]=dict(batch=64,warmup=4,measured=32,mean_seconds=float(np.mean(times)),max_seconds=max(times),saved128_and_independent_arithmetic_seconds=save_time)
        write_json(folder/'PREFLIGHT.json',reports[arm]);print('GEO_PREFLIGHT_PASS',arm,reports[arm]['parameters'],first_connected,flush=True)
        del t,c,rr,bench,criterion,v,w,value,gv,gw
    guard.check(force=True);snap=rg.snapshot();payload=int(data['past'].nbytes+data['future'].nbytes+data['neighbor_ids'].nbytes)
    measured=max(snap['peak_private'],5837217792) # Prior same whole-model actual production peak, not available-backfit.
    E=int(measured+payload+384*1024**2)
    write_json(RUN/'MEMORY_ESTIMATE.json',dict(GPU_E=E,CPU_E=3*1024**3,derivation=dict(current_preflight_lifetime_private=snap['peak_private'],prior_UD_actual_production_private=5837217792,selected_measured_baseline=measured,fit_payload=payload,extra_IO_reserve=384*1024**2),policy='warning_first_remaining_v3'))
    verify(read_json(BASE/'source_manifest.json')['files']);verify(read_json(UD/'source_manifest.json')['files'])
    write_json(RUN/'PREFLIGHT.json',dict(status='PASS_ALL4_ACTUAL_GEO_PREFLIGHT',at=now(),reports=reports,additional_checks=additional,benchmark=benchmarks,elapsed_seconds=time.perf_counter()-start,resource=guard.observations,no_scientific_pilots=True,same_assistant_not_external=True))
    print('GEO_ALL_PREFLIGHT_PASS',time.perf_counter()-start,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        if not (RUN/'PREFLIGHT_FAILURE.json').exists():write_json(RUN/'PREFLIGHT_FAILURE.json',dict(status='FAILURE',at=now(),traceback=traceback.format_exc()))
        raise
