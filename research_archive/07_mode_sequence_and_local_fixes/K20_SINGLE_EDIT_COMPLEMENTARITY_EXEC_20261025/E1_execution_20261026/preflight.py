"""Current-source full E1 numerical/resume/saved-output preflight; no formal training."""
import copy,math,time,shutil,traceback
import numpy as np
import torch
from env import *
from editor import ActionEditor,loss,choose,numpy_loss_and_score_grad
from data_pipeline import CACHE,OptData,contract,verify_prepared
from train_paired import initialize,train_epochs,read_completed
from evaluate_fixed import population,write_population,audit_population
from source_control import build_manifest,verify,snapshot_candidate
def exact(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        if a.dtype!=b.dtype or not torch.equal(a.cpu(),b.cpu()):raise AssertionError('Tensor EXACT '+path)
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise AssertionError('Array EXACT '+path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError('Keys '+path)
        for k in a:exact(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise AssertionError('Length '+path)
        for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'/'+str(i))
    elif a!=b:raise AssertionError('Scalar EXACT '+path)
def full_state(model,opt,g,history):return copy.deepcopy(dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),rng=capture_rng(g),history=history))
def parameter_fd():
    seed_all(20261025);m=ActionEditor(np.zeros(323),np.ones(323)).double()
    with torch.no_grad():
        for p in m.parameters():p.fill_(.01)
    x=torch.zeros(4,401,324,dtype=torch.float64);x[:,:,0]=-.5;x[:,0,0]=.5;x[:,0,323]=1;x[:,1,0]=2;x[:,2,0]=1
    valid=torch.ones(4,401,dtype=torch.bool);target=torch.zeros_like(valid);target[:,1:3]=True
    base=[p.detach().clone() for p in m.parameters()];generator=torch.Generator().manual_seed(20261025)
    directions=[]
    for _ in range(4):
        direction=[torch.randn(p.shape,generator=generator,dtype=torch.float64) for p in m.parameters()];norm=torch.sqrt(sum((d*d).sum() for d in direction));directions.append([d/norm for d in direction])
    reports=[]
    def face(s):return (s.masked_fill(~target,-torch.inf).argmax(1).tolist(),s.masked_fill(~(valid&~target),-torch.inf).argmax(1).tolist())
    for arm in ARMS:
        m.zero_grad(set_to_none=True);s=m(x);v=loss(s,valid,target,arm);v.backward();analytic=[p.grad.clone() for p in m.parameters()]
        nv,ng=numpy_loss_and_score_grad(s.detach().numpy(),valid.numpy(),target.numpy(),arm)
        if abs(v.item()-nv)>1e-10:raise ValueError('Independent NumPy loss scalar')
        m.zero_grad(set_to_none=True);m(x).backward(torch.from_numpy(ng));manual=[p.grad.clone() for p in m.parameters()]
        if max(float(abs(a-b).max()) for a,b in zip(analytic,manual))>1e-10:raise ValueError('Independent score-gradient all parameter chain')
        expected_face=face(s.detach());records=[]
        for di,direction in enumerate(directions):
            expected=sum(float((g*d).sum()) for g,d in zip(analytic,direction))
            for eps in (1e-4,1e-5):
                values=[];faces=[]
                for sign in (1,-1):
                    with torch.no_grad():
                        for p,b,d in zip(m.parameters(),base,direction):p.copy_(b+sign*eps*d)
                        score=m(x);values.append(float(loss(score,valid,target,arm)));faces.append(face(score))
                with torch.no_grad():
                    for p,b in zip(m.parameters(),base):p.copy_(b)
                if any(f!=expected_face for f in faces):raise ValueError('Registered FD active max face switched; no eps search allowed')
                numerical=(values[0]-values[1])/(2*eps);error=abs(numerical-expected)
                if error>1e-6+1e-3*abs(expected):raise ValueError('Registered parameter FD tolerance')
                records.append(dict(direction=di,eps=eps,error=error,reference=expected,numerical=numerical,max_faces_exact=True))
        reports.append(dict(arm=arm,scalar_error=abs(v.item()-nv),records=records))
    return reports
def resume_checks(folder,source,guard):
    reports=[];models={};last_state=None;last_contract=None
    data=OptData(42,513)
    for arm in ARMS:
        expected=contract(42,arm,source);expected.update(population_windows=513,epochs=2,steps=34,fixture='actual_first513_opt_resume')
        model,opt,g=initialize(42,data,'cuda');history=[];train_epochs(model,opt,g,data,2,history,arm,guard=guard);continuous=full_state(model,opt,g,history)
        model2,opt2,g2=initialize(42,data,'cuda');h=[];train_epochs(model2,opt2,g2,data,1,h,arm,guard=guard)
        ck=folder/f'{arm}_one_epoch_resume.pt';save_checkpoint(ck,model2,opt2,g2,h,expected)
        model3,opt3,g3=initialize(42,data,'cuda');h3=load_checkpoint(ck,model3,opt3,g3,expected);train_epochs(model3,opt3,g3,data,2,h3,arm,guard=guard)
        exact(continuous,full_state(model3,opt3,g3,h3));models[arm]=model3
        reports.append(dict(arm=arm,windows=513,epochs=2,steps=34,all_parameters_buffers_optimizer_TorchCPU_CUDA_loader_numpy_python_RNG_history='EXACT'))
        last_state=(model3,opt3,g3,h3);last_contract=expected
    model,opt,g,h=last_state
    fixture=folder/'completed_reader';fixture.mkdir()
    save_checkpoint(fixture/'checkpoint.pt',model,opt,g,h,last_contract)
    shutil.copyfile(CACHE/'stats_seed42.npz',fixture/'stats.npz');write_json(fixture/'split_ids.json',dict(FIT_opt=data.ids.tolist()));np.save(fixture/'output.npy',np.arange(5))
    done=dict(status='EXPLICIT_2EPOCH_FIXTURE_NOT_FORMAL',contract=last_contract,epoch_rows=2,steps=34,outputs={p.name:sha256(p) for p in fixture.iterdir()})
    write_json(fixture/'done.json',done);read_completed(fixture,last_contract)
    rejected=[]
    for kind in ('source','split','statistics','checkpoint','output'):
        bad=folder/('bad_'+kind);shutil.copytree(fixture,bad);bd=read_json(bad/'done.json')
        if kind in ('source','split','statistics'):bd['contract'][kind]='bad';write_json(bad/'done.json',bd)
        elif kind=='checkpoint':
            ckpt=torch.load(bad/'checkpoint.pt',map_location='cpu',weights_only=False);ckpt['contract']['source']='bad';torch.save(ckpt,bad/'checkpoint.pt');bd['outputs']['checkpoint.pt']=sha256(bad/'checkpoint.pt');write_json(bad/'done.json',bd)
        else:np.save(bad/'output.npy',np.ones(5))
        try:read_completed(bad,last_contract)
        except ValueError:rejected.append(kind)
        else:raise AssertionError('Actual completed reader accepted bad '+kind)
    bad={**last_contract,'statistics':'bad'}
    try:load_checkpoint(fixture/'checkpoint.pt',model,opt,g,bad)
    except ValueError:rejected.append('resume')
    else:raise AssertionError('Actual resume accepted bad contract')
    return reports,models,dict(valid_model_SHA_skip=True,bad_actual_reader_rejected=rejected)
def gpu_and_benchmark(source,guard):
    data=OptData(42);models={};timing={};connected_reports=[];paired_initial=None
    for arm in ARMS:
        model,opt,g=initialize(42,data,'cuda');models[arm]=model
        initial_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if paired_initial is None:paired_initial=initial_state
        else:exact(paired_initial,initial_state)
        x,valid,target=data.batch(np.arange(32),'cuda');initial=model(x)
        assert torch.equal(initial,torch.zeros_like(initial)) and not choose(initial,valid).any()
        connected=set();first_zero=None
        for step in range(8):
            x,v,t=data.batch(np.arange(step*32,(step+1)*32),'cuda');opt.zero_grad(set_to_none=True);s=model(x);obj=loss(s,v,t,arm);obj.backward()
            if not torch.isfinite(s).all() or not torch.isfinite(obj):raise FloatingPointError('GPU real8')
            for name,p in model.named_parameters():
                if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('GPU gradient finite')
                if p.grad.abs().sum()>0:connected.add(name)
            if step==0:first_zero=all(not p.grad.any() for name,p in model.named_parameters() if not name.startswith('net.4'))
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if not first_zero or connected!=set(dict(model.named_parameters())):raise ValueError('Phased all-parameter gradient connection')
        connected_reports.append(dict(arm=arm,actual8batch32=True,all_tensors_connected=True,initial_native_STOP_identity=True))
        # Disposable timing state is never a completed scientific model/checkpoint.
        times=[]
        for step in range(36):
            x,v,t=data.batch(np.arange(32),'cuda');torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True);obj=loss(model(x),v,t,arm);obj.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();torch.cuda.synchronize()
            if step>=4:times.append(time.perf_counter()-start)
        timing[arm]=float(np.mean(times));guard.check()
    exact(models['I'].feature_mean,models['S'].feature_mean);exact(models['I'].feature_std,models['S'].feature_std)
    return connected_reports,timing
def main():
    if (HERE/'source_manifest.json').exists():raise FileExistsError('Production source locked; preserve and use new repair version')
    guard=Guard(ESTIMATE);guard.check('startup');require_e0();ready=verify_prepared();manifest=build_manifest();folder=snapshot_candidate(manifest);start=time.perf_counter()
    try:
        fd=parameter_fd();resumes,models,contracts=resume_checks(folder,manifest['digest'],guard)
        gpu,timing=gpu_and_benchmark(manifest['digest'],guard)
        data=population(42,'opt',manifest['digest'],limit=128);t0=time.perf_counter();saved=write_population(42,'opt',data,models,manifest['digest'],folder/'saved_actual128',guard);audit=audit_population(folder/'saved_actual128',manifest['digest'],guard);pipeline=time.perf_counter()-t0
        guard.check();assert verify(manifest)==manifest['digest']
        # Includes hold/inner 3seed400 action construction+direct independent review and final full metrics; conservative explicit component.
        train_compute=3*1024*sum(timing.values());eval_seconds=max(pipeline/128*(4096+3825+1856)*3,180);new_action_audit_seconds=(3825+1856)*3*.04
        planning=1.25*(train_compute+eval_seconds+new_action_audit_seconds+120)
        if shutil.disk_usage(HERE).free<1.3*5*1024**3:raise OSError('E1 result/temp remaining disk allowance5GiB*1.3')
        out=dict(status='PASS',at=e0.now(),source_digest=manifest['digest'],source_count=len(manifest['files']),full_opt_stats_reports=ready['reports'],parameter_FD=fd,resume=resumes,actual_GPU8=gpu,completed_reader=contracts,saved_actual128_audit=audit,eval_weights_buffers_exact=saved['eval_weights_buffers_exact'],resource=guard.observations,benchmark=dict(FIT32_seconds=timing,actual128_fullsaved_pipeline_seconds=pipeline,train_compute_seconds=train_compute,evaluation_allowance_seconds=eval_seconds,new_hold_inner_allaction_audit_allowance_seconds=new_action_audit_seconds,planning_seconds=planning,ETA_seconds=[.7*planning,1.5*planning],no_production_guarantee=True),elapsed_seconds=time.perf_counter()-start,no_hold_inner_GT=True,no_formal_training=True)
        write_json(folder/'PREFLIGHT.json',out)
        # Lock exactly the tested source manifest; no subsequent scientific code changes.
        write_json(HERE/'source_manifest.json',manifest);write_json(HERE/'PREFLIGHT.json',out);print(json.dumps(dict(status='PASS',sources=len(manifest['files']),benchmark=out['benchmark'])),flush=True)
    except BaseException as exc:
        write_json(folder/'FAILURE.json',dict(type=type(exc).__name__,message=str(exc),at=e0.now(),traceback=traceback.format_exc(),resource=guard.observations));raise
if __name__=='__main__':main()
