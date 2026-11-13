"""Explicitly partial regression only. Does NOT admit formal training."""
import inspect,json,time
import numpy as np
import torch
from env import *
from editor import ActionEditor,loss,choose,numpy_loss_and_score_grad
from feature_builder import build,WindowMoments
def main():
    if (HERE/'PARTIAL_CHECK.json').exists():raise FileExistsError('Preserve previous partial evidence')
    guard=Guard(ESTIMATE);guard.check('startup');require_e0();seed_all(20261025)
    from data_contract import load as load_e0,ENC,SPLIT
    data=load_e0(save=False);source=e0.verify()
    with np.load(ROOT/SPLIT) as z:fit=z['grouped_train_index']
    raw_ix=fit[data['opt_row'][:256]]
    with np.load(ROOT/ENC) as z:past=z['past'][raw_ix];tokens=z['tokens'][raw_ix]
    pool=np.load(E0/'seed42/input_pools.npz');n=pool['native'][:256];k=pool['kin'][:256]
    X=[];eligible=[];target=[];moments=WindowMoments()
    for start in (0,128):
        with np.load(E0/f'seed42/chunk_{start:05d}_{start+128:05d}.npz') as z:
            for i in range(128):
                row=start+i;x=build(past[row],tokens[row],n[row],k[row],z['motion'][i]);X.append(x);eligible.append(z['eligible'][i]);target.append(z['target'][i]);moments.add(x,z['eligible'][i])
    X=np.asarray(X);eligible=np.asarray(eligible);target=np.asarray(target);mean,std=moments.finish()
    # Fixture stats only. NEVER save or use these as full4096 production statistics.
    assert tuple(inspect.signature(build).parameters)==('past','tokens','native','kin','action_motion')
    assert np.all(X[:,0,266:323]==0) and np.all(X[:,0,-1]==1) and np.all(X[:,1:,-1]==0)
    scores=torch.zeros(3,401,dtype=torch.float64,requires_grad=True);mask=torch.zeros(3,401,dtype=torch.bool);mask[:,0]=True;mask[1,1:5]=True;mask[2,1:]=True
    targets=torch.zeros_like(mask);targets[0,0]=True;targets[1,1:3]=True;targets[2,0]=True
    fixture=[]
    for arm in ARMS:
        score=scores.detach().clone().requires_grad_();value=loss(score,mask,targets,arm);value.backward()
        nv,ng=numpy_loss_and_score_grad(score.detach().numpy(),mask.numpy(),targets.numpy(),arm)
        assert abs(value.item()-nv)<1e-10 and np.max(abs(score.grad.numpy()-ng))<1e-12
        assert not score.grad[0].any() and not choose(score,mask).any()
        fixture.append(dict(arm=arm,scalar_error=abs(value.item()-nv),gradient_error=float(np.max(abs(score.grad.numpy()-ng)))))
    if not torch.cuda.is_available():raise RuntimeError('Actual GPU partial check unavailable; do not claim PASS')
    outputs=[]
    for arm in ARMS:
        seed_all(42);model=ActionEditor(mean,std).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,foreach=False)
        x=torch.from_numpy(X[:32].astype(np.float32)).cuda();m=torch.from_numpy(eligible[:32]).cuda();t=torch.from_numpy(target[:32]).cuda()
        initial=model(x);assert torch.equal(initial,torch.zeros_like(initial)) and not choose(initial,m).any()
        connected=set();first_front_zero=None
        for step in range(8):
            sl=slice(step*32,(step+1)*32);xb=torch.from_numpy(X[sl].astype(np.float32)).cuda();mb=torch.from_numpy(eligible[sl]).cuda();tb=torch.from_numpy(target[sl]).cuda()
            optimizer.zero_grad(set_to_none=True);ss=model(xb);value=loss(ss,mb,tb,arm);value.backward()
            assert torch.isfinite(ss).all() and torch.isfinite(value)
            for name,p in model.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all()
                if p.grad.abs().sum()>0:connected.add(name)
            if step==0:first_front_zero=all(not p.grad.any() for name,p in model.named_parameters() if not name.startswith('net.4'))
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        assert first_front_zero and connected==set(dict(model.named_parameters()))
        outputs.append(dict(arm=arm,actual_GPU_batches=8,windows=256,initial_allzero_scores_STOP=True,first_front_zero_expected=True,all_parameter_tensors_connected=list(sorted(connected))))
    guard.check();assert e0.verify()==source
    report=dict(status='PARTIAL_CODE_AND_GPU8_VERIFIED_NOT_FORMAL_PREFLIGHT',at=e0.now(),source_e0_digest=source,real_windows=256,fixture_stats_only=True,full4096_stats_NOT_CREATED=True,synthetic_loss_scoregradient=fixture,actual_GPU=outputs,resource=guard.observations,formal_training_started=False,missing_gates=['full4096 opt-only common stats and independent statistics','fixed double FD twoeps all activity faces','I/S513 real subset2epochs full optimizer/buffers/ALLRNG EXACT','actual completed badsplit/stat/checkpoint/output branch contracts','128 actual saved fullquality cluster motion and all current sources lock','4warm32 benchmark whole pipeline RAM disk estimate'],no_old_files_deleted=True)
    write_json(HERE/'PARTIAL_CHECK.json',report);print(json.dumps(dict(status=report['status'],actual_GPU=outputs)))
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import traceback
        write_json(HERE/('PARTIAL_FAILURE_'+str(time.time_ns())+'.json'),dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),at=e0.now()));raise
