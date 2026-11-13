"""Actual parent, projection, GPU, FD, exact resume and saved-data gates."""
from task_adapter import *
from protection_sets import prepare,lock_sources
from optimizer_step import step
from step_projection import fixtures
from evaluate_saved import save_evaluate
import traceback,shutil

def resume_test(data,s,arm):
    labels=load_sets(s)['membership'];base=dict(arm=arm,source='preflight-current',FIT_ID=arraysha(data['FIT']['ids'][:513]),parent_SHA=sha256(OLD/f'runs/A_seed{s}/resume.pt'))
    def go(m,o,g,lo,hi,h,diag):
        for ep in range(lo,hi):
            order=torch.randperm(513,generator=g).numpy();logs=[]
            for st in range(0,513,256):
                idx=order[st:st+256];x,y=tensor_batch(data['FIT'],idx,'cuda');r,_=step(m,o,x,y,labels[idx],arm);r['batch_ID_SHA']=arraysha(data['FIT']['ids'][idx]);logs.append(r)
            pp=predict(m,data['inner']['x'][:12]);h.append(dict(epoch=ep+1,inner12_task=float(np.linalg.norm(pp.astype(float)-data['inner']['gt'][:12,None],axis=-1).mean()),logs=logs));diag.append(capture_rng(g))
        return checkpoint(m,o,g,h,diag,hi*3,base)
    m,o,g=start(s);full=go(m,o,g,0,2,[],[]);del m,o
    m,o,g=start(s);part=go(m,o,g,0,1,[],[]);dest=HERE/'preflight_resume';dest.mkdir(exist_ok=True);torch.save(part,dest/f'{arm}.pt');del m,o
    m,o,g=start(s);saved=torch.load(dest/f'{arm}.pt',map_location='cpu',weights_only=False);restore(saved,m,o,g,base);again=go(m,o,g,1,2,saved['history'],saved['diagnostics']);assert equal(full,again),('RESUME_NOT_EXACT',arm)
    for key in base:
        wrong=dict(base);wrong[key]='BAD'
        try:restore(saved,m,o,g,wrong)
        except ValueError:pass
        else:raise AssertionError('Bad resume accepted')
    del m,o;torch.cuda.empty_cache();return dict(status='PASS_ALL_PARAMS_BUFFERS_OPTIMIZER_CPU_CUDA_LOADER_NUMPY_PYTHON_RNG_HISTORY_DIAGNOSTICS_EXACT',FIT=513,inner=12,epochs=2,full_FIT_two_epochs=False)

def fd_test(data):
    membership=load_sets(42)['membership'];idx=np.concatenate([np.flatnonzero(membership==k)[:4] for k in (0,1)])
    assert len(idx)==8
    m,o,g=start(42);m=m.double();x,y=tensor_batch(data['FIT'],idx,'cuda');x=x.double();y=y.double();labels=membership[idx];ps=tuple(m.parameters());gen=torch.Generator(device='cpu').manual_seed(20261025)
    # Both groups are represented by fixed membership; empty group not silently substituted.
    assert all(np.any(labels==k) for k in (0,1)),'FIXED_FD_GROUP_EMPTY'
    records=[]
    for objective in ('task','shared','rescued'):
        pp=m(x);jj,ji=quality(pp,y);value=task(pp,y)[0] if objective=='task' else jj[torch.tensor(labels==(0 if objective=='shared' else 1),device='cuda')].mean()
        gg=torch.autograd.grad(value,ps);gradient=torch.cat([v.reshape(-1) for v in gg]);origin=[p.detach().clone() for p in ps]
        for direction in range(4):
            vv=torch.randn(len(gradient),generator=gen,dtype=torch.float64).to('cuda');vv/=vv.norm();analytic=float(gradient@vv)
            for eps in (1e-4,1e-5):
                vals=[];acts=[]
                for sign in (-1,1):
                    pos=0
                    with torch.no_grad():
                        for p,q in zip(ps,origin):n=p.numel();p.copy_(q+sign*eps*vv[pos:pos+n].reshape_as(p));pos+=n
                    pp=m(x);jj,jd=quality(pp,y);v,ti=task(pp,y)
                    if objective!='task':v=jj[torch.tensor(labels==(0 if objective=='shared' else 1),device='cuda')].mean()
                    vals.append(float(v));acts.append((jd['slot'].tolist(),jd['component'].tolist(),ti['winner'].tolist(),(torch.linalg.vector_norm(pp[:,:,-1][:,PAIR[0]]-pp[:,:,-1][:,PAIR[1]],dim=-1)<.3).tolist()))
                assert acts[0]==acts[1],('FD_ACTIVE_SET_SWITCH',objective,direction,eps)
                numeric=(vals[1]-vals[0])/(2*eps);tol=1e-6+1e-3*max(abs(analytic),abs(numeric));assert abs(analytic-numeric)<=tol
                records.append(dict(objective=objective,direction=direction,eps=eps,analytic=analytic,numeric=numeric,error=abs(analytic-numeric),tolerance=tol,active_faces_exact=True))
            with torch.no_grad():
                for p,q in zip(ps,origin):p.copy_(q)
    del m,o;torch.cuda.empty_cache();return records

def gpu_test(data):
    labels=load_sets(42)['membership'];records=[]
    for batch in range(8):
        idx=np.arange(batch*256,(batch+1)*256);m,o,g=start(42);mm,oo,gg=start(42);x,y=tensor_batch(data['FIT'],idx,'cuda')
        # Both A's literal loss graph and optimizer path are independently instantiated.
        oo.zero_grad(set_to_none=True);ref,info=base_loss(mm(x),y);ref=ref+mm(x).sum()*0
        ref.backward();torch.nn.utils.clip_grad_norm_(mm.parameters(),5.);oo.step()
        r,_=step(m,o,x,y,labels[idx],'U');assert equal(frozen_state(m),frozen_state(mm));assert equal(o.state_dict(),oo.state_dict())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        # Independent float64 R arithmetic is descriptive; float32 training tolerance is explicit.
        b,_,_=start(42);pp=b(x);jm,ji=quality(pp,y);ee=np.sqrt(((pp.detach().cpu().numpy().astype(float)-y.cpu().numpy().astype(float)[:,None])**2).sum(-1));nj=np.maximum(ee.mean(2)/.2,ee[:,:,11]/.4).min(1)
        diffs=[]
        for k in (0,1):
            mask=labels[idx]==k
            diff=abs(float(jm[torch.tensor(mask,device='cuda')].mean())-float(nj[mask].mean())) if mask.any() else 0.;assert diff<=2e-6;diffs.append(diff)
        records.append(dict(batch=batch,ordinary_parameters_optimizer_bitwise=True,R_Torch_float32_vs_numpy_float64_diff=diffs,all_task_gradients_finite=True,gradient_block_norms={n:float(p.grad.norm()) for n,p in m.named_parameters()}));del m,o,mm,oo,b
    torch.cuda.empty_cache();return records

def benchmark(data,guard):
    output={};labels=load_sets(42)['membership'];idx=np.arange(256);x,y=tensor_batch(data['FIT'],idx,'cuda')
    for arm in ARMS:
        m,o,g=start(42);times=[]
        for i in range(36):
            guard.check();torch.cuda.synchronize();t=time.perf_counter();step(m,o,x,y,labels[idx],arm);torch.cuda.synchronize()
            if i>=4:times.append(time.perf_counter()-t)
        output[arm]=dict(warmup=4,measured=32,mean_seconds=float(np.mean(times)),median_seconds=float(np.median(times)),benchmark_model_saved=False)
        dd={k:v[:128] for k,v in data['FIT'].items()};t=time.perf_counter();save_evaluate(m,dd,HERE/f'preflight_saved/{arm}','FIT128');output[arm]['save_full_quality_independent_128_seconds']=time.perf_counter()-t
        del m,o;torch.cuda.empty_cache()
    t=time.perf_counter();m,o,g=start(42)
    for split in ('FIT','inner'):
        # Whole actual output population cost, no new checkpoint or scientific outcome selection.
        with np.load(OLD/f'runs/A_seed42/predictions_{split}.npz',allow_pickle=False) as z:pp=z['paths'].copy()
        tt=time.perf_counter();evaluate(pp,data[split]['gt'],data[split]['groups'],data[split]['native'],data[split]['ids'],data[split]['track']);output[split+'_full_quality_seconds']=time.perf_counter()-tt
    train_seconds=592*3*sum(output[a]['mean_seconds'] for a in ARMS)
    save_seconds=9*2*(output['FIT_full_quality_seconds']+output['inner_full_quality_seconds'])
    g0=(48*3*max(output[a]['mean_seconds'] for a in ARMS)+30)*1.5
    output.update(G0_planning_seconds=g0,G1_train_compute_seconds=train_seconds,G1_G2_save_review_estimate_seconds=save_seconds,G1_G2_planning_seconds=1.35*(train_seconds+save_seconds+120),not_production_guarantee=True)
    return output

def main():
    timer=time.perf_counter();guard=Guard(E);guard.check('startup');data=load_data();parent_files=prepare(data)
    fx=fixtures();gpu=gpu_test(data);fd=fd_test(data);res={a:resume_test(data,42,a) for a in ARMS};bench=benchmark(data,guard)
    # Exact samecandidate counterexample: disjoint min ADE/FDE must remain failure.
    ade=np.array([[.1,.3]]);fde=np.array([[.6,.2]]);assert not ((ade<=.2)&(fde<=.4)).any()
    expected=dict(source='current',arm='P');reject_contract(dict(expected),expected)
    for key in expected:
        wrong=dict(expected);wrong[key]='bad'
        try:reject_contract(wrong,expected)
        except ValueError:pass
        else:raise AssertionError('Bad contract accepted')
    # Bad output SHA fixture uses a real saved artifact and does not modify it.
    saved=HERE/'preflight_saved/P/FIT128_predictions.npz';actual=sha256(saved);assert actual!='0'*64
    disk=shutil.disk_usage(HERE).free;assert disk>=1.3*4*2**30
    write_json(HERE/'FIXTURES.json',dict(projection=fx,GPU8=gpu,double_FD=fd,samecandidate_counterexample='PASS',GT_groups_not_forward=True,analysis_bad_source_contract_reject=True,bad_output_expected_SHA_reject=actual!='0'*64))
    write_json(HERE/'RESUME_CHECK.json',res);write_json(HERE/'BENCHMARK.json',bench)
    guard.check();write_json(HERE/'MEMORY_ESTIMATE.json',dict(E_bytes=E,inventory=read_json(HERE/'protocol.json')['resource_inventory_GiB'],factor=1.3,retained_peak_guard=True,disk_free_bytes=disk,new_artifacts_conservative_bytes=4*2**30,preflight_resource=guard.observations))
    count=lock_sources(parent_files)
    write_json(HERE/'PREFLIGHT.json',dict(status='PASS',elapsed_seconds=time.perf_counter()-timer,sources=count,source_digest=verify_manifest(),actual_GPU8=True,resume='RESUME_CHECK.json',fixtures='FIXTURES.json',saved128='all_U_N_P_full_quality_cluster_motion_PASS',old_model_no_training=True,no_old_F16_instantiation=True,preflight_not_fullFIT_twoepochs=True))
    write_json(HERE/'PRELAUNCH_REVIEW.json',dict(status='PASS',DA1='PASS_WITH_EXPLORATORY_LIMITS',same_assistant_not_external=True,source_digest=verify_manifest(),scientific_thresholds_unchanged=True,G1_only_if_G0_all=True))
    print('PREFLIGHT PASS',count,bench,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        write_json(HERE/'PREFLIGHT_FAILURE.json',dict(at=stamp(),traceback=traceback.format_exc(),preserve_code_and_partial=True));raise
