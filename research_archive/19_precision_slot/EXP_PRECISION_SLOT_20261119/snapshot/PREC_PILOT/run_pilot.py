"""Finite PREC-PILOT using the original training loop and frozen historical controls."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
FAST = WORK/'qoder_work/EXPLORE_FAST_EXEC_20261116'
OLD = WORK/'qoder_work/NEXT_PHASE_EXEC_20261114'
DIAG = WORK/'qoder_work/EXP_PREC_DIAG_EXEC_20261119'
sys.path.insert(0,str(FAST/'code'))
from explore_common import read, sha, verify, replace_json, now, train_contract, load_training, validate_result, save_npz
from run_explore import training_loop, evaluate, ResumeProbeStop

CONTROLS = ('B0','E3_self')
NEW = ('Prior','Prior_self')
ARM_PARENT = dict(zip(NEW,CONTROLS))
SEED, EPOCHS, STEPS = 42, 32, 340


class PauseRequested(Exception):
    pass


def current(phase, **values):
    replace_json(ROOT/'CURRENT.json',dict(at=now(),phase=phase,**values))


def load(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def control_folder(arm):
    return FAST/'runs'/f'{arm}_E32_s42'


def prepare():
    p = ROOT/'SOURCE_MANIFEST.json'
    if p.exists():
        verify(read(p)['files'])
        return
    frozen = read(FAST/'SOURCE_MANIFEST.json')['files']
    verify(frozen)
    inputs = dict(frozen)
    control_values = {}
    b0_order = None
    for arm in CONTROLS:
        folder = control_folder(arm)
        result = validate_result(folder)
        assert result['contract']==train_contract(arm,EPOCHS,SEED,21719),arm
        count = 295474 if arm=='B0' else 302477
        assert result['train']['parameters']==count and result['train']['steps_per_epoch']==STEPS
        assert result['train']['updates']==STEPS*EPOCHS and result['status']=='DONE'
        hashes = [read(folder/f'ckpt_epoch{epoch:03}.json')['order_hash'] for epoch in range(1,EPOCHS+1)]
        assert hashes[-1]==result['train']['order_hash'] and (b0_order is None or b0_order==hashes)
        b0_order = hashes
        inputs.update(result['files'])
        inputs[str(folder/'result.json')] = sha(folder/'result.json')
        control_values[arm] = dict(contract=result['contract'],order_hashes=hashes,parameters=count,
                                   dev_predictions_sha256=sha(folder/'dev_predictions.npz'),
                                   final_sha256=sha(folder/'ckpt_epoch032.pt'))
    for split in ('fit','dev'):
        path = OLD/f'C0/{split}.npz'
        inputs[str(path)] = sha(path)
    diag = read(DIAG/'DONE.json')
    verify(diag['files'])
    for path,digest in read(DIAG/'INPUT_MANIFEST.json')['files'].items():
        if Path(path).suffix in ('.py','.md','.json'):
            if path in inputs:
                assert inputs[path]==digest
            inputs[path]=digest
    for path in (DIAG/'DONE.json',DIAG/'INDEPENDENT_VERIFICATION.json',FAST/'SOURCE_MANIFEST.json'):
        inputs[str(path)]=sha(path)
    receipt = ROOT/'CONTROLS.json'
    replace_json(receipt,dict(at=now(),status='PASS',controls=control_values,batch=64,updates=10880,
                              each_epoch_order_equal=True,old_protocol='EXPLORE_FAST fixed 32-epoch final EMA'))
    code = [*ROOT.glob('*.py'),ROOT/'PROTOCOL.md',receipt,*sorted((ROOT/'standalone').glob('*.ps1'))]
    inputs.update({str(path):sha(path) for path in code})
    verify(inputs)
    replace_json(p,dict(at=now(),files=inputs))
    print('PILOT_SOURCE_AND_CONTROLS_PASS',len(inputs),flush=True)


def configure_trainer(tc, arm, allow_pause):
    from prior_decoder import PriorDecoder
    from set_refine_decoder import attach_set_refiner
    make_model = tc.make_model
    class PilotTrainer(tc.Trainer):
        def __init__(self,*args,**kwargs):
            assert kwargs['arm']==arm
            with_self = arm=='Prior_self'
            def factory(device,seed):
                model,cfg = make_model(device,seed)
                if with_self:
                    attach_set_refiner(model,'self').to(device)
                model.decoder = PriorDecoder(model.decoder,with_self)
                return model,cfg
            kwargs['arm'] = 'B0'
            tc.make_model = factory
            try:
                super().__init__(*args,**kwargs)
            finally:
                tc.make_model = make_model
            self.arm = arm

        def step(self):
            at_boundary = self.order is None or self.cursor==len(self.order)
            if allow_pause and at_boundary and self.epoch<self.epochs and (ROOT/'PAUSE.request').exists():
                raise PauseRequested(f'{self.arm} complete epochs {self.epoch}')
            return super().step()
    return SimpleNamespace(Trainer=PilotTrainer,save_checkpoint=tc.save_checkpoint)


def smoke(tc, rf, guard, fit):
    import torch
    from prior_decoder import CV_SLOT
    pre = ROOT/'preflight'/str(time.time_ns())
    pre.mkdir(parents=True)
    sample = {k:v[:128] for k,v in fit.items()}
    fit_batch = torch.as_tensor(fit['past'][:64],device='cuda:0')
    nv = torch.as_tensor(fit['num_valid'][:64],device='cuda:0')
    y = torch.as_tensor(fit['future'][:64],device='cuda:0')
    prediction_checks = {}
    for arm in NEW:
        proxy = configure_trainer(tc,arm,False)
        paired = tc.Trainer(sample,np.arange(128),epochs=2,device='cuda:0',seed=SEED,arm=ARM_PARENT[arm])
        new = proxy.Trainer(sample,np.arange(128),epochs=2,device='cuda:0',seed=SEED,arm=arm)
        tc.exact(tc.cpu_tree(paired.model.state_dict()),tc.cpu_tree(new.model.state_dict()),arm+'_INITIAL_STATE')
        with torch.no_grad():
            new.model.eval()
            paired.model.eval()
            x,z,_,tx = tc.prediction(new.model,fit_batch,nv,y)
            px,pz,_,_ = tc.prediction(paired.model,fit_batch,nv,y)
            torch.testing.assert_close(x[:,:,:CV_SLOT],px[:,:,:CV_SLOT],atol=1e-6,rtol=0)
            torch.testing.assert_close(z[:,:,:CV_SLOT],pz[:,:,:CV_SLOT],atol=2e-6,rtol=0)
            world = tc.world_paths(x,tx)
            d = fit_batch[:,0,-1]-fit_batch[:,0,-2]
            cv = fit_batch[:,0,-1,None]+torch.arange(1,13,device=d.device)[None,:,None]*d[:,None]
            cv_error = float((world[:,CV_SLOT]-cv).abs().max())
            assert cv_error<=1e-4
            new_path,new_score,_,_ = tc.prediction(new.model,fit_batch,nv,y.flip(0)+3)
            assert torch.equal(x,new_path) and torch.equal(z,new_score)
        new.step()
        key = 'decoder.proposal_delta.weight'
        assert key in new.connected
        prediction_checks[arm] = dict(initial_shared_weights='EXACT',initial_19_candidates='MATCH',
                                      initial_CV_world_max_error=cv_error,full_batch64_backward='PASS',
                                      future_perturbation_paths_and_logits='EXACT')
        del new,paired
        torch.cuda.empty_cache()
    proxy = configure_trainer(tc,'Prior_self',False)
    contract = dict(study='PREC_PILOT_TECHNICAL_ONLY',fit_rows=128,epochs=2,seed=SEED)
    full,_ = training_loop(proxy,rf,guard,sample,np.arange(128),'Prior_self',2,SEED,pre/'full',contract)
    del full
    try:
        training_loop(proxy,rf,guard,sample,np.arange(128),'Prior_self',2,SEED,pre/'resume',contract,stop_after=1)
    except ResumeProbeStop:
        pass
    else:
        raise AssertionError('Expected complete-epoch interruption')
    resumed,_ = training_loop(proxy,rf,guard,sample,np.arange(128),'Prior_self',2,SEED,pre/'resume',contract)
    del resumed
    x,y = (torch.load(pre/d/'ckpt_epoch002.pt',map_location='cpu',weights_only=False)
           for d in ('full','resume'))
    tc.exact(x,y,'PILOT_RESUME_STATE')
    replace_json(ROOT/'PREFLIGHT_PASS.json',dict(at=now(),status='PASS',checks=prediction_checks,
                 source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
                 exact_full_checkpoint_resume=True,technical_fixture_directory=str(pre),scope='FIT only'))
    print('REAL_CUDA_PREFLIGHT_PASS',flush=True)


def check_orders(directory, train):
    control = read(ROOT/'CONTROLS.json')['controls']['B0']['order_hashes']
    got = [read(directory/f'ckpt_epoch{epoch:03}.json') for epoch in range(1,EPOCHS+1)]
    assert [r['order_hash'] for r in got]==control
    assert [r['updates'] for r in got]==[STEPS*epoch for epoch in range(1,EPOCHS+1)]
    assert train['order_hash']==control[-1] and train['updates']==EPOCHS*STEPS


def train(tc,rf,guard,fit,dev,arm):
    directory = ROOT/'runs'/arm
    p = directory/'DONE.json'
    contract = dict(study='EXP_PREC_PILOT_20261119',arm=arm,epochs=EPOCHS,seed=SEED,
                    train_set='C0_full_fit',batch=64,final_weights='EMA_0.995',
                    source_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
                    reused_control=ARM_PARENT[arm],control_protocol=train_contract(ARM_PARENT[arm],EPOCHS,SEED,len(fit['ids'])))
    if p.exists():
        sealed=read(p)
        verify(sealed['files'])
        assert sealed['contract']==contract
        check_orders(directory,sealed['train'])
        return sealed
    current('TRAIN',arm=arm,epochs=EPOCHS,seed=SEED,progress_file=str(directory/'PROGRESS.json'))
    proxy=configure_trainer(tc,arm,True)
    t0=time.monotonic()
    model,info=training_loop(proxy,rf,guard,fit,np.arange(len(fit['ids'])),arm,EPOCHS,SEED,
                             directory,contract,progress=directory/'PROGRESS.json')
    check_orders(directory,info)
    assert info['parameters']==read(ROOT/'CONTROLS.json')['controls'][ARM_PARENT[arm]]['parameters']
    current('EVALUATE_DEV',arm=arm)
    metrics=evaluate(tc,guard,model.ema.module,dev,directory)
    del model
    files={str(f):sha(f) for f in directory.iterdir() if f.is_file() and f.name!='PROGRESS.json'}
    sealed = dict(status='DONE',arm=arm,contract=contract,train=info,dev=metrics,
                  wall_seconds_this_session=time.monotonic()-t0,files=files)
    replace_json(p,sealed)
    return sealed


def main(action):
    if action=='prepare':
        prepare()
        return
    manifest = read(ROOT/'SOURCE_MANIFEST.json')['files']
    verify(manifest)
    if (ROOT/'DONE.json').exists() and action=='run':
        verify(read(ROOT/'DONE.json')['files'])
        print('ALREADY_COMPLETE',flush=True)
        return
    tc,rf,guard = load_training()
    fit = load(OLD/'C0/fit.npz')
    if action=='preflight':
        guard.check(force=True)
        smoke(tc,rf,guard,fit)
        return
    preflight=read(ROOT/'PREFLIGHT_PASS.json')
    assert preflight['status']=='PASS' and preflight['source_manifest_sha256']==sha(ROOT/'SOURCE_MANIFEST.json')
    guard.check(force=True)
    dev = load(OLD/'C0/dev.npz')
    import torch
    started=time.monotonic()
    runs={}
    try:
        for arm in NEW:
            runs[arm]=train(tc,rf,guard,fit,dev,arm)
            torch.cuda.empty_cache()
    except PauseRequested as e:
        current('PAUSED',message=str(e))
        return
    current('ANALYZE')
    from analyze_pilot import analyze
    analyze(dev,runs)
    verify(manifest)
    names=['RESULT.json','REPORT.md','PREFLIGHT_PASS.json','SOURCE_MANIFEST.json','CONTROLS.json']
    files=[ROOT/name for name in names]+[ROOT/'runs'/a/'DONE.json' for a in NEW]+sorted((ROOT/'windows').glob('*.npz'))
    replace_json(ROOT/'DONE.json',dict(status='SUCCESS',new_runs=2,session_seconds=time.monotonic()-started,
                  files={str(p):sha(p) for p in files}))
    current('COMPLETE',new_runs=2)
    print('PREC_PILOT_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','preflight','run'))
    arguments=parser.parse_args()
    try:
        main(arguments.action)
    except Exception as error:
        resource_error = isinstance(error,MemoryError) or type(error).__name__=='OutOfMemoryError'
        if arguments.action=='run' and not resource_error:
            current('FAILED',message=str(error))
        (ROOT/f'{arguments.action}_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf-8')
        if resource_error:
            print(traceback.format_exc(),file=sys.stderr,flush=True)
            sys.exit(75)
        raise
