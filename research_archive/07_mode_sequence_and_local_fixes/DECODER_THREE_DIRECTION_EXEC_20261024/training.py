"""Explicit one-package runner, never auto-chains directions or launches OS jobs.

--prepare only prepares new-owned inputs. --preflight tests actual FIT subsets.
--train requires a matching actual GPU preflight AND a separate package-specific
human authorization record. Nothing in importing or code unit tests trains.
"""
import argparse
import copy
import random
import shutil
import sys
import time
import os
from contextlib import contextmanager
from pathlib import Path
sys.dont_write_bytecode=True
from common import *
import torch
from models import ContextK20,objective
from f16.checkpoint import capture_rng,restore_rng
from f16.resource import Guard
sys.path.insert(0,str(ROOT))
from audit.verify import independent_metrics,compare

PACKAGES={'A':('A_cv_free','A_cv_basis'),'C1':('C1_hard','C1_awta'),
          'C2':('C2_isomlp','C2_setmix')}

@contextmanager
def exclusive(execution):
    """One actual worker per execution; no process polling or old task access."""
    import msvcrt
    path=execution/'worker.lock'
    with path.open('a+b') as lock:
        if path.stat().st_size==0:lock.write(b'0');lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError('SAME_EXECUTION_WORKER_ALREADY_ACTIVE')
        try:yield
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)

def save_checkpoint_atomic(blob,path):
    temp=path.with_name(path.name+'.writing')
    torch.save(blob,temp);os.replace(temp,path)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_model(arm,seed,execution,device='cuda'):
    st=np.load(execution/'statistics.npz');seed_all(seed);p=None
    if arm.startswith('A_'): p=np.load(execution/'projection.npy')
    model=ContextK20(st['mean'],st['std'],arm,p).to(device)
    initial=torch.load(OLD/f'initial_states/seed{seed}.pt',map_location='cpu',weights_only=True)
    missing,extra=model.load_state_dict(initial,strict=False)
    expected={'projection'} if arm.startswith('A_') else {'mix1.weight','mix1.bias','mix2.weight','mix2.bias'} if arm.startswith('C2_') else set()
    if set(missing)!=expected or extra: raise ValueError('INITIAL_STATE_CONTRACT_MISMATCH')
    return model

def optimizer(model):
    # Original AdamW defaults and clipping retained; do not change foreach recipe.
    return torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)

def contract(execution,arm,seed):
    return dict(arm=arm,seed=seed,source_digest=verify_sources(execution/'source_manifest.json'),
        protocol=sha256(execution/'protocol.json'),stats=sha256(execution/'statistics.npz'),
        IDs=sha256(execution/'split_ids.npz'),orders=sha256(OLD/f'batch_orders/seed{seed}.npz'),
        initial=sha256(OLD/f'initial_states/seed{seed}.pt'),
        projection=sha256(execution/'projection.npy') if arm.startswith('A_') else None)

def clone_state(m): return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def exact(a,b):
    if torch.is_tensor(a): return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

def batch_update(model,opt,d,ix,step):
    device=next(model.parameters()).device
    x=torch.from_numpy(np.ascontiguousarray(d['x'][ix])).to(device)
    y=torch.from_numpy(np.ascontiguousarray(d['gt'][ix].astype(np.float32))).to(device)
    pred=model(x);value,diag=objective(pred,y,model.arm,step)
    opt.zero_grad(set_to_none=True);value.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
    if not torch.isfinite(norm) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('NONFINITE_OR_MISSING_GRADIENT')
    opt.step()
    return dict(loss=float(value.detach()),task=float(diag['task'].detach()),div=float(diag['div'].detach()),
        preclip_grad_norm=float(norm),n=len(ix),tau=diag['tau'],
        weights=diag['weights'].mean(0).cpu().numpy(),entropy=float(diag['entropy'].mean().detach()),
        winner=diag['winner'].detach().cpu().numpy())

def predict(model,x):
    device=next(model.parameters()).device;result=np.empty((len(x),20,12,2),np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(x),256):
            n=min(256,len(x)-start);b=np.zeros((256,210),np.float32);b[:n]=x[start:start+n]
            result[start:start+n]=model(torch.from_numpy(b).to(device)).cpu().numpy()[:n]
    return result

def prepare(package,execution,screen=None):
    if execution.exists(): raise FileExistsError('New execution directory required')
    execution.mkdir(parents=True)
    spec=read_json(HERE/('direction_A_time_basis' if package=='A' else 'direction_C_set_mechanisms')/'protocol.json')
    if package=='A':
        if screen is None: raise ValueError('A requires completed registered screen')
        result=read_json(screen/'RESULT.json');name=result['selected_basis']
        if name is None or not result['decisions'][name]['passed']: raise ValueError('A_SCREEN_GATE_FAILED')
        if sha256(screen/'bases.npz')!=result['bases_SHA']: raise ValueError('BASIS_CHANGED')
        sys.path.insert(0,str(HERE/'direction_A_time_basis'))
        from basis import projector
        with np.load(screen/'bases.npz') as z:p=projector(z[name])
        np.save(execution/'projection.npy',p)
        spec.update(selected_basis=name,screen_RESULT_SHA=sha256(screen/'RESULT.json'))
    spec.update(package=package,arms=PACKAGES[package],status='PREPARED_NOT_PREFLIGHTED',epochs=8,steps_each=592)
    write_json(execution/'protocol.json',spec)
    for name in ('statistics.npz','split_ids.npz'):shutil.copyfile(OLD/name,execution/name)
    lock_sources(execution/'source_manifest.json')
    write_json(execution/'runtime_state.json',dict(status='PREPARED_NOT_STARTED',package=package))

def real_preflight(package,execution,estimate):
    from test_kernels import run_tests
    kernel_tests=run_tests()
    if not torch.cuda.is_available(): raise RuntimeError('Actual CUDA preflight required')
    guard=Guard(estimate);guard.check('startup');source=verify_sources(execution/'source_manifest.json')
    d=load_data();arms=PACKAGES[package];reports={};paired=[];timings={}
    expected_mean,expected_std=fit_stats(d['FIT']['x'])
    with np.load(execution/'statistics.npz') as z:
        # S1 stores float32 inference buffers AND the float64 accumulation proof.
        # Compare each against the same precision; do not change either scaler.
        np.testing.assert_allclose(z['mean_float64'],expected_mean,atol=1e-11,rtol=0)
        np.testing.assert_allclose(z['std_float64'],expected_std,atol=1e-11,rtol=0)
        np.testing.assert_array_equal(z['mean'],expected_mean.astype(np.float32))
        np.testing.assert_array_equal(z['std'],expected_std.astype(np.float32))
    for arm in arms:
        model=make_model(arm,42,execution);paired.append(clone_state(model));opt=optimizer(model)
        gradients={};generator=torch.Generator().manual_seed(42)
        for step in range(1,9):
            guard.check();batch_update(model,opt,d['FIT'],np.arange((step-1)*256,step*256),step)
            for name,param in model.named_parameters():
                gradients[name]=gradients.get(name,False) or bool(torch.isfinite(param.grad).all() and param.grad.norm()>0)
        # Exact softmax-key cancellation does not apply to these plain MLPs.
        if not all(gradients.values()): raise ValueError('LEGAL_PARAMETER_GRADIENT_DISCONNECTED: '+str([k for k,v in gradients.items() if not v]))
        # Full two-epoch 513 FIT +12 inner interruption fixture, original batch256.
        def fixture(interrupt):
            seed_all(42);m=make_model(arm,42,execution);o=optimizer(m);g=torch.Generator().manual_seed(42)
            history=[];step=0;assign=None
            for ep in range(2):
                order=torch.randperm(513,generator=g).numpy()
                for begin in range(0,513,256):
                    step+=1;rec=batch_update(m,o,d['FIT'],order[begin:begin+256],step)
                    history.append(rec);assign=rec['winner']
                    if interrupt and step==2:
                        blob=dict(model=clone_state(m),optimizer=copy.deepcopy(o.state_dict()),rng=capture_rng(g),
                            history=copy.deepcopy(history),step=step,assignment=assign.copy())
                        temp=execution/(arm+'_resume_fixture.pt');torch.save(blob,temp)
                        blob=torch.load(temp,map_location='cpu',weights_only=False)
                        m=make_model(arm,42,execution);o=optimizer(m)
                        m.load_state_dict(blob['model']);o.load_state_dict(blob['optimizer']);restore_rng(blob['rng'],g)
                        history=blob['history'];assign=blob['assignment'];step=blob['step']
                history.append(dict(inner_paths=predict(m,d['inner']['x'][:12])))
            return dict(model=clone_state(m),optimizer=o.state_dict(),rng=capture_rng(g),history=history,assignment=assign)
        if not exact(fixture(False),fixture(True)): raise ValueError('ALL_RNG_OPTIMIZER_RESUME_NOT_EXACT')
        pred=predict(model,d['inner']['x'][:128]);sample={k:v[:128] for k,v in d['inner'].items()}
        saved=execution/(arm+'_smoke_saved128.npz')
        np.savez_compressed(saved,paths=pred,gt=sample['gt'],native=sample['native'],groups=sample['groups'],
            sample_id=sample['ids'],track=sample['track'])
        rows,_=report_paths(pred,sample,execution/(arm+'_smoke_metrics.json'))
        with np.load(saved) as stored:
            assert np.array_equal(stored['paths'],pred) and np.array_equal(stored['sample_id'],sample['ids'])
            independent=independent_metrics(stored)
        audit=compare({g:{k:rows[g][k] for k in independent[g]} for g in independent},independent)
        if audit['status']!='PASS':raise ValueError(audit)
        # Measured throwaway steps, no benchmark checkpoint selection.
        start=time.perf_counter()
        for s in range(1,37):
            if s==5:torch.cuda.synchronize();start=time.perf_counter()
            batch_update(model,opt,d['FIT'],np.arange(256),s)
        torch.cuda.synchronize();timings[arm]=(time.perf_counter()-start)/32
        reports[arm]=dict(gradient_all_parameters=gradients,resume='EXACT_ALL_STATE',saved128_audit=audit)
        del model,opt;torch.cuda.empty_cache()
    params=[dict((k,v) for k,v in p.items() if k!='projection') for p in paired]
    assert exact(params[0],params[1])
    # Bind generated execution inputs as an explicit extension, not old hash edits.
    execution_inputs={n:sha256(execution/n) for n in ('statistics.npz','split_ids.npz','protocol.json')}
    if package=='A':execution_inputs['projection.npy']=sha256(execution/'projection.npy')
    status='PASS'
    profile_binding=None
    if package=='C2':
        # Do not silently substitute student Linear-only counts for full-model
        # MACs. A missing full frozen-encoder pipeline measurement is explicit.
        fullpath=execution/'FULL_PIPELINE_PROFILE.json'
        if not fullpath.exists():status='CORE_PASS_PENDING_FULL_PIPELINE_PROFILE'
        else:
            full=read_json(fullpath)
            if full.get('status')!='PASS' or full.get('source_digest')!=source:
                raise ValueError('FULL_PIPELINE_PROFILE_SOURCE_MISMATCH')
            for arm in arms:
                p=full['arms'][arm]
                if p['whole_parameters']!=365711 or p['full_model_MAC_per_window']<=0:
                    raise ValueError('FULL_MODEL_PARAMETER_MAC_PROFILE_INVALID')
                if not p.get('includes_frozen_encoder') or not p.get('counts_all_executed_matmul_convolution_ops'):
                    raise ValueError('INCOMPLETE_FULL_MODEL_MAC_PROFILE')
                for b in ('1','256'):
                    if p['latency'][b]['seconds_per_batch']<=0:raise ValueError('GPU_LATENCY_NOT_MEASURED')
            profile_binding=sha256(fullpath)
    write_json(execution/'PREFLIGHT.json',dict(status=status,scope='ACTUAL_GPU8_SUBSET2EPOCH_SAVED128',
        package=package,source_digest=source,execution_inputs=execution_inputs,reports=reports,
        kernel_tests=kernel_tests,full_pipeline_profile_SHA=profile_binding,
        benchmark_step_seconds=timings,estimate_bytes=estimate,resource_observations=guard.observations,
        full_FIT_twice_replayed=False,production_started=False))
    guard.check();assert verify_sources(execution/'source_manifest.json')==source

def train(package,execution,authorization,estimate):
    auth=read_json(authorization);pre=read_json(execution/'PREFLIGHT.json')
    if auth.get('package')!=package or not auth.get('direct_human_instruction'):
        raise ValueError('Separate explicit human authorization required for this single package')
    if pre.get('status')!='PASS' or pre['package']!=package or pre['source_digest']!=verify_sources(execution/'source_manifest.json'):
        raise ValueError('Current actual preflight missing')
    if estimate!=pre['estimate_bytes']:raise ValueError('MEMORY_ESTIMATE_CHANGED_REPREFLIGHT_REQUIRED')
    if package=='C2' and sha256(execution/'FULL_PIPELINE_PROFILE.json')!=pre['full_pipeline_profile_SHA']:
        raise ValueError('FULL_PIPELINE_PROFILE_CHANGED')
    for rel,digest in pre['execution_inputs'].items():
        if sha256(execution/rel)!=digest:raise ValueError('EXECUTION_INPUT_CHANGED')
    # No generic "PASS" stub can bypass current independent kernel unit suite.
    from test_kernels import run_tests
    run_tests()
    guard=Guard(estimate);guard.check('startup');data=load_data();allruns={}
    for arm in PACKAGES[package]:
        for seed in SEEDS:
            run=execution/f'{arm}_seed{seed}';run.mkdir(exist_ok=True)
            con=contract(execution,arm,seed)
            if (run/'done.json').exists():
                done=read_json(run/'done.json')
                if done['contract']!=con:raise ValueError('DONE_CONTRACT_MISMATCH')
                for name,digest in done['outputs'].items():
                    if sha256(run/name)!=digest:raise ValueError('DONE_OUTPUT_CHANGED')
                allruns[f'{arm}_{seed}']=done;continue
            model=make_model(arm,seed,execution);opt=optimizer(model);g=torch.Generator().manual_seed(seed)
            history=[];step=0;previous=None
            if (run/'resume.pt').exists():
                b=torch.load(run/'resume.pt',map_location='cpu',weights_only=False)
                if b['contract']!=con:raise ValueError('RESUME_CONTRACT_MISMATCH')
                model.load_state_dict(b['model']);opt.load_state_dict(b['optimizer']);restore_rng(b['rng'],g)
                history=b['history'];step=b['step'];previous=b['assignment']
                blob=b
            orders=np.load(OLD/f'batch_orders/seed{seed}.npz')['orders']
            for ep in range(len(history),8):
                model.train();assign=np.full(18909,-1,np.int16);recs=[]
                write_json(execution/'runtime_state.json',dict(status='RUNNING',stage='TRAIN',arm=arm,seed=seed,epoch=ep+1))
                for start in range(0,18909,256):
                    guard.check();ix=orders[ep,start:start+256];step+=1
                    rec=batch_update(model,opt,data['FIT'],ix,step);assign[ix]=rec.pop('winner');recs.append(rec)
                weights=sum(v['weights']*v['n'] for v in recs)/18909
                hist=dict(epoch=ep+1,steps=74,loss=sum(v['loss']*v['n'] for v in recs)/18909,
                    soft_weight_histogram=weights,entropy=sum(v['entropy']*v['n'] for v in recs)/18909,
                    hard_histogram=np.bincount(assign,minlength=20),hard_active=int(len(np.unique(assign))),
                    winner_switch_rate=None if previous is None else float((assign!=previous).mean()),batches=recs)
                history.append(hist);previous=assign
                blob=dict(model=clone_state(model),optimizer=copy.deepcopy(opt.state_dict()),rng=capture_rng(g),
                    history=history,step=step,assignment=assign,contract=con)
                save_checkpoint_atomic(blob,run/'resume.pt');write_json(run/'history.json',history)
            assert step==592 and len(history)==8
            save_checkpoint_atomic(blob,run/'checkpoint.pt');before=clone_state(model);jobs={}
            for split,d in data.items():
                guard.check();p=predict(model,d['x']);assert exact(before,clone_state(model))
                np.savez_compressed(run/f'predictions_{split}.npz',paths=p,gt=d['gt'],native=d['native'],
                    sample_id=d['ids'],track=d['track'],groups=d['groups'])
                rows,_=report_paths(p,d,run/f'metrics_{split}.json')
                old_ids,old=old_paths(seed,split)
                assert np.array_equal(old_ids,d['ids'])
                write_json(run/f'frozen_old_subsets_{split}.json',frozen_old_subsets(p,d,old))
                with np.load(run/f'predictions_{split}.npz') as z:ind=independent_metrics(z)
                audit=compare({g:{k:rows[g][k] for k in ind[g]} for g in ind},ind)
                write_json(run/f'audit_{split}.json',audit)
                if audit['status']!='PASS':raise ValueError('SAVED_AUDIT_FAILED')
                if arm=='C1_hard':
                    ids,old=old_paths(seed,split)
                    if not np.array_equal(ids,d['ids']) or not np.array_equal(old,p):
                        raise ValueError('CONTROL_REPRODUCTION_UNRESOLVED_NO_RETRAIN')
                jobs[split]=rows
            outputs={p.name:sha256(p) for p in run.iterdir() if p.is_file() and p.name not in ('done.json','resume.pt')}
            done=dict(status='SUCCESS',contract=con,outputs=outputs,metrics=jobs,eval_buffers_exact=True)
            write_json(run/'done.json',done);allruns[f'{arm}_{seed}']=done
            del model,opt;torch.cuda.empty_cache()
    from decisions import analyze
    decision=analyze(package,execution,data)
    guard.check()
    write_json(execution/'RESULT.json',dict(status='SUCCESS',runs=allruns,decision=decision,
        source_digest=verify_sources(execution/'source_manifest.json'),automatic_next=False))
    write_json(execution/'RESOURCE.json',dict(estimate=estimate,observations=guard.observations))
    write_json(execution/'runtime_state.json',dict(status='COMPLETED_REVIEW_REQUIRED',automatic_next=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('package',choices=PACKAGES);ap.add_argument('--execution',required=True,type=Path)
    group=ap.add_mutually_exclusive_group(required=True);group.add_argument('--prepare',action='store_true');group.add_argument('--preflight',action='store_true');group.add_argument('--train',action='store_true')
    ap.add_argument('--screen',type=Path);ap.add_argument('--authorization',type=Path);ap.add_argument('--estimated-peak-gib',type=float)
    a=ap.parse_args()
    a.execution=owned_output(a.execution,'direction_A_time_basis' if a.package=='A' else 'direction_C_set_mechanisms')
    if a.prepare:prepare(a.package,a.execution,a.screen)
    else:
        if a.estimated_peak_gib is None:ap.error('Explicit inventory estimate required')
        e=int(a.estimated_peak_gib*2**30)
        if a.preflight:real_preflight(a.package,a.execution,e)
        elif a.authorization is None:ap.error('Separate package authorization record required')
        else:
            with exclusive(a.execution):
                try:train(a.package,a.execution,a.authorization,e)
                except BaseException as error:
                    import traceback
                    write_json(a.execution/'FAILURE.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
                        preserve_partial_outputs=True,no_automatic_retry=True))
                    raise
