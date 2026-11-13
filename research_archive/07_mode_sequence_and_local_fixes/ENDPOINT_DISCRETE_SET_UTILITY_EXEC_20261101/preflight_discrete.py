"""Actual G/H GPU, full gradients, exact recovery and BEFORE-T saved review."""
import copy
import hashlib
import shutil
import time
import numpy as np
import contracts_discrete as c
import torch
from scipy.special import erf
from model_discrete import ActionUtilityHead,utility_loss
from engine_discrete import Engine,infer
from targets_discrete import loss_numpy,targets
from decode_discrete import select,mask_observed
from evaluate_discrete import save,read
from audit_core_discrete import saved
from features_discrete import transform,geometry8
from preflight_checks import trees_equal
from train import seed_everything

def independently(params,x,dout=None):
    x=np.asarray(x,float).reshape(-1,48)
    w0,b0,w1,b1,w2,b2=[params['net.'+k] for k in ('0.weight','0.bias','2.weight','2.bias','4.weight','4.bias')]
    gelu=lambda z:.5*z*(1+erf(z/np.sqrt(2)))
    prime=lambda z:.5*(1+erf(z/np.sqrt(2)))+z*np.exp(-z*z/2)/np.sqrt(2*np.pi)
    z0=x@w0.T+b0;a0=gelu(z0);z1=a0@w1.T+b1;a1=gelu(z1);out=a1@w2.T+b2
    if dout is None:return out.reshape(-1,20,48)
    g2=np.asarray(dout).reshape(-1,48);g1=(g2@w2)*prime(z1);g0=(g1@w1)*prime(z0)
    return dict(zip(('net.0.weight','net.0.bias','net.2.weight','net.2.bias','net.4.weight','net.4.bias'),(g0.T@x,g0.sum(0),g1.T@a0,g1.sum(0),g2.T@a1,g2.sum(0))))

def nonzero(m):
    with torch.no_grad():
        m.net[-1].weight.copy_(torch.linspace(-.01,.01,m.net[-1].weight.numel(),device='cuda',dtype=m.net[-1].weight.dtype).reshape_as(m.net[-1].weight))
        m.net[-1].bias.copy_(torch.linspace(-.03,.03,48,device='cuda',dtype=m.net[-1].bias.dtype))

def math_checks(d):
    rows=[];initial=[]
    for arm in ('G','H'):
        seed_everything(42,True);m=ActionUtilityHead().cuda();initial.append(copy.deepcopy(m.state_dict()))
        x=torch.from_numpy(d['z'+arm]).cuda();y=torch.from_numpy(d['value']).cuda();v=torch.from_numpy(d['valid']).cuda()
        zero=m(x);assert torch.equal(zero,torch.zeros_like(zero)) and not select(zero.detach().cpu().numpy(),d['valid']).any()
        phases=[]
        for phase in ('zero_last','nonzero_last'):
            if phase=='nonzero_last':nonzero(m)
            m.zero_grad(set_to_none=True);score=m(x);loss=utility_loss(score,y,v);loss.backward()
            grads={k:float(p.grad.abs().max().cpu()) for k,p in m.named_parameters()}
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            if phase=='nonzero_last':assert all(g>0 for g in grads.values())
            if arm=='G':assert torch.equal(m.net[0].weight.grad[:,:16],torch.zeros_like(m.net[0].weight.grad[:,:16]))
            phases.append(dict(phase=phase,gradient_max_abs=grads))
        md=ActionUtilityHead().cuda().double();md.load_state_dict(m.state_dict());xx=x.double();md.zero_grad(set_to_none=True)
        score=md(xx);loss=utility_loss(score,y,v);loss.backward()
        ps={k:p.detach().cpu().numpy().copy() for k,p in md.named_parameters()};out=independently(ps,d['z'+arm])
        ref,dout,_=loss_numpy(out,d['value'],d['valid']);gr=independently(ps,d['z'+arm],dout)
        maxgrad=max(float(np.max(np.abs(p.grad.cpu().numpy()-gr[k]))) for k,p in md.named_parameters())
        assert abs(float(loss.detach().cpu())-ref)<=1e-10 and maxgrad<=1e-10
        base=[p.detach().clone() for p in md.parameters()];analytic=[p.grad.detach().clone() for p in md.parameters()]
        rng=np.random.default_rng(20261101);fd=[]
        for repeat in range(4):
            direction=[torch.tensor(rng.normal(size=tuple(p.shape)),device='cuda',dtype=torch.float64) for p in md.parameters()]
            norm=torch.sqrt(sum((a*a).sum() for a in direction));direction=[a/norm for a in direction]
            an=float(sum((a*b).sum() for a,b in zip(analytic,direction)).cpu())
            for eps in (1e-4,1e-5):
                values=[];choices=[]
                for sign in (1,-1):
                    with torch.no_grad():
                        for p,b,dr in zip(md.parameters(),base,direction):p.copy_(b+sign*eps*dr)
                        so=md(xx);values.append(float(utility_loss(so,y,v).cpu()));choices.append(select(so.cpu().numpy(),d['valid']))
                num=(values[0]-values[1])/(2*eps);assert abs(num-an)<=1e-6+1e-3*abs(an)
                fd.append(dict(direction=repeat,eps=eps,analytic=an,numeric=num,error=abs(num-an),argmax_switch=not np.array_equal(*choices),argmax_not_in_loss_graph=True))
        rows.append(dict(arm=arm,phases=phases,scalar_error=abs(float(loss.detach().cpu())-ref),all6800_gradient_max_error=maxgrad,FD=fd,model_buffer_count=len(list(m.buffers()))))
    assert trees_equal(initial[0],initial[1])
    # All-invalid stays in batch denominator, zero loss AND finite zero score gradient.
    score=torch.ones((2,20,48),device='cuda',requires_grad=True);loss=utility_loss(score,torch.zeros_like(score,dtype=torch.float64),torch.zeros_like(score,dtype=torch.bool));loss.backward()
    assert loss==0 and torch.equal(score.grad,torch.zeros_like(score))
    return dict(status='PASS_ACTUAL_GPU8_ALL6800_INDEPENDENT_GRADIENTS_FIXED_FD',arms=rows,paired_initial_EXACT=True,all_invalid_denominator_and_zero_gradient=True)

def hash_indices(ids,n,pop):
    return np.array(sorted(range(len(ids)),key=lambda i:(hashlib.sha256(f'discrete-preflight-v1|{pop}|{int(ids[i])}'.encode()).hexdigest(),int(ids[i])))[:n])

def resume(d,val,out,g):
    reports=[]
    for arm in ('G','H'):
        a=Engine(d,arm,epochs=2,stage='preflight513');
        while not a.complete:g.check();a.step()
        expected=a.snapshot();pred=infer(a.model,val['z'+arm]);del a
        a=Engine(d,arm,epochs=2,stage='preflight513')
        for _ in range(19):a.step()
        p=out/f'{arm}_interrupt19.pt';c.atomic_checkpoint(p,a.snapshot());sha=c.sha(p);del a
        state=c.trusted_checkpoint(p,sha);b=Engine(d,arm,epochs=2,stage='preflight513');b.restore(state)
        while not b.complete:g.check();b.step()
        assert trees_equal(expected,b.snapshot()) and np.array_equal(pred,infer(b.model,val['z'+arm]))
        bad=[]
        for key in ('code','protocol','data'):
            corrupt=copy.deepcopy(state);corrupt['contract'][key]='BAD'
            try:b.restore(corrupt)
            except ValueError:bad.append(key)
            else:raise AssertionError('bad checkpoint contract accepted')
        for key in ('IDs','old_stats','new_stats','action_order','nonidentity_order','zG'):
            corrupt=copy.deepcopy(state);corrupt['contract']['data'][key]='BAD'
            try:b.restore(corrupt)
            except ValueError:bad.append(key)
            else:raise AssertionError('bad binding accepted')
        try:c.trusted_checkpoint(p,'0'*64)
        except ValueError:bad.append('checkpoint_SHA')
        else:raise AssertionError('bad checkpoint SHA')
        reports.append(dict(arm=arm,status='PASS',updates=34,interrupted=19,allparams_optimizer_ALLRNG_sampler_history_scores_EXACT=True,bad_reader_rejections=bad,
            train_IDs=d['x']['sample_id'].tolist(),val_IDs=val['x']['sample_id'].tolist()))
    return reports

def main():
    start=time.perf_counter();out=c.ROOT/'preflight';out.mkdir(exist_ok=False)
    E=4741091328+64*1024**2+128*1024**2;g=c.resource(E);g.check(force=True)
    c.write(c.ROOT/'MEMORY_ESTIMATE.json',dict(estimate=E,inventory=dict(previous_measured_private=4741091328,new_feature_label_payload=64*1024**2,graph_allocator=128*1024**2),CPU_estimate=2894262272+64*1024**2,policy='warning_first_remaining_v3',not_available_derived=True))
    c.verify(c.json_read(c.ROOT/'INPUT_SOURCE_LOCK.json')['files'],g.check)
    train=c.read_prepared('train');val=c.read_prepared('val');tiny=c.read_prepared('tiny');real8=c.subset(tiny,np.arange(8))
    math=math_checks(real8)
    resume_d=c.subset(train,hash_indices(train['x']['sample_id'],513,'train'));resume_v=c.subset(val,hash_indices(val['x']['sample_id'],12,'val'))
    recovery=resume(resume_d,resume_v,out,g)
    reviews=[];timing=[]
    for arm in ('G','H'):
        seed_everything(42,True);m=ActionUtilityHead().cuda();nonzero(m)
        score=infer(m,tiny['z'+arm]);folder=out/f'saved128_{arm}';t=time.perf_counter();save(folder,score,tiny,arm,'preflight128');reviews.append(saved(folder,tiny,arm,'preflight128'));save_time=time.perf_counter()-t
        # Actual changed-GT fixture, no score/mask/feature change. Input sample fixed first8.
        changed=copy.deepcopy(real8);changed['gt']+=100
        stats=c.load_npz(c.ROOT/'prepared/statistics8.npz')
        assert np.array_equal(transform(changed,stats,arm),real8['z'+arm])
        mask=mask_observed(real8['x']['past'],real8['x']['native']);mask2=mask_observed(changed['x']['past'],changed['x']['native'])
        assert np.array_equal(mask,real8['valid']) and np.array_equal(mask,mask2)
        before=infer(m,transform(real8,stats,arm));after=infer(m,transform(changed,stats,arm));assert np.array_equal(before,after) and np.array_equal(select(before,mask),select(after,mask2))
        # Deliberately corrupt a copied output; never touch the real output or old source.
        shutil.copytree(folder,out/f'bad_output_{arm}')
        with (out/f'bad_output_{arm}/output.npz').open('ab') as f:f.write(b'CORRUPT')
        try:read(out/f'bad_output_{arm}',tiny,arm,'preflight128')
        except ValueError:pass
        else:raise AssertionError('bad output accepted')
        # GPU training benchmark is throwaway, fixed4warm32 measured, no retained model.
        e=Engine(tiny,arm,stage='benchmark');steps=[]
        for i in range(36):
            torch.cuda.synchronize();t=time.perf_counter();e.step();torch.cuda.synchronize()
            if i>=4:steps.append(time.perf_counter()-t)
        t=time.perf_counter();infer(m,tiny['z'+arm]);score_time=time.perf_counter()-t
        timing.append(dict(arm=arm,step_seconds=steps,full128_score_seconds=score_time,full128_save_AND_independent_review_seconds=save_time))
    badfolder=out/'bad_prepared';badfolder.mkdir();shutil.copyfile(c.ROOT/'prepared/statistics8.npz',badfolder/'statistics8.npz')
    c.write(badfolder/'done.json',dict(files={'statistics8.npz':c.sha(badfolder/'statistics8.npz')}))
    with (badfolder/'statistics8.npz').open('ab') as f:f.write(b'BAD_STAT')
    try:c.verify_prepared(badfolder)
    except ValueError:pass
    else:raise AssertionError('corrupt stats accepted')
    # Measure distance construction and actual original960-guard work (not cached lookup timing).
    t=time.perf_counter();geometry8(real8['x']['native']);geo=time.perf_counter()-t
    t=time.perf_counter();mask_observed(real8['x']['past'],real8['x']['native']);guard=time.perf_counter()-t
    sourceio=c.json_read(c.ROOT/'INPUT_SOURCE_LOCK.json')['seconds'];savecost=max(t['full128_save_AND_independent_review_seconds'] for t in timing)
    compute=sum(3*256*max(t['step_seconds']) for t in timing)
    total=compute+30*savecost+1536*32*20*48/1e7+4*sourceio+180
    bench=dict(status='MEASURED_FULL_T_PLANNING',arms=timing,geometry8_8windows_seconds=geo,original960mask_8windows_seconds=guard,
        input_source_SHA_seconds=sourceio,T_conservative_seconds=total,T_budget_seconds=1800,T_admitted=total<=1800,
        includes_all30_savepoints_review=True,full_source_readback_allowance=4*sourceio,extra_IO_history_label_audit_seconds=180,
        fixture_optimizer_updates=136+72,not_raw_F16_end_to_end=True)
    c.write(c.ROOT/'BENCHMARK.json',bench)
    disk=shutil.disk_usage(c.ROOT);required=1610612736;c.write(c.ROOT/'DISK_ADMISSION.json',dict(free=disk.free,additional_inventory=required,factor=1.3,passed=disk.free>=required*1.3,no_delete=True));assert disk.free>=required*1.3
    c.verify(c.json_read(c.ROOT/'INPUT_SOURCE_LOCK.json')['files'],g.check);g.check(force=True)
    c.write(c.ROOT/'PREFLIGHT.json',dict(status='PASS_ACTUAL_PREFLIGHT' if total<=1800 else 'PASS_CHECKS_T_BUDGET_STOP',math=math,resume=recovery,saved128_both_arms_independent_BEFORE_science=reviews,
        changedGT_forward_features_masks_scores_choices_EXACT=True,all_inputs_opt_trainstats_only=True,stats_model_buffers=0,
        fixture_updates=dict(resume_each_arm68_total136=136,benchmark_36_each_total72=72),scientific_updates=0,seconds=time.perf_counter()-start))
    c.write(c.ROOT/'RESOURCE_PREFLIGHT.json',dict(estimate=g.estimate,observations=g.observations))
    print('PREFLIGHT PASS T planning',total,'seconds admitted',total<=1800,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import datetime,traceback
        c.write(c.ROOT/('FAILURE_PREFLIGHT_'+datetime.datetime.now().strftime('%H%M%S')+'.json'),dict(traceback=traceback.format_exc(),scientific_updates=0))
        raise
