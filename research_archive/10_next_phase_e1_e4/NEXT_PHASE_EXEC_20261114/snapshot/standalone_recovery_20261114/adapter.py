"""Infrastructure-only continuation. Original sealed science and failures are immutable.

Changes: fix seal-name shadowing in completion bookkeeping, reuse validated final
weights, and persist resumable state at each epoch. Trainer.step is never modified.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import runpy
import sys
import time

REC=Path(__file__).resolve().parent
BASE=REC.parent
sys.path.insert(0,str(BASE))
import execution as original
from train_core import Trainer,save_checkpoint,exact,cpu_tree,rng_state
from common import read_json,write_json,sha,seal,verify,now,ROOT
import torch
import numpy as np
import resource_guard as rg

RUNROOT=BASE/'runs'
DATA_PROVIDER=original.data_read


def atomic(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,p)


def status(**values):atomic(REC/'CURRENT.json',dict(at=now(),**values))
original.status=status


def source_check():
    verify(read_json(BASE/'SOURCE_MANIFEST.json')['files'])
    if (REC/'RECOVERY_MANIFEST.json').exists():verify(read_json(REC/'RECOVERY_MANIFEST.json')['files'])


def finite_tree(value):
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():raise ValueError('NONFINITE_SAVED_STATE')
    elif isinstance(value,dict):
        for x in value.values():finite_tree(x)
    elif isinstance(value,(list,tuple)):
        for x in value:finite_tree(x)


def validate_final(spec,out):
    """CPU-only proof that the saved model completed every requested update."""
    state=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
    contract=read_json(out/'contract.json');exact(contract,state['contract'],'FINAL_CONTRACT')
    exact(spec,contract['spec'],'REQUESTED_SPEC')
    if contract['source']!=sha(BASE/'SOURCE_MANIFEST.json') or contract['fit']!=sha(BASE/'C0/fit.npz'):
        raise ValueError('FINAL_SOURCE_OR_SPLIT_CONTRACT')
    data=DATA_PROVIDER('fit');idx=original.indices_for(spec,data);steps=int(np.ceil(len(idx)/64));total=steps*spec['epochs']
    if state['epoch']!=spec['epochs'] or state['cursor']!=len(idx) or state['updates']!=total:raise ValueError('FINAL_INCOMPLETE_COUNTERS')
    if len(state['history'])!=total or len(state['batch_orders'])!=total:raise ValueError('FINAL_INCOMPLETE_HISTORY')
    ids=data['ids'][idx];expected=sorted(ids.tolist())
    for ep in range(spec['epochs']):
        rows=state['history'][ep*steps:(ep+1)*steps];batches=state['batch_orders'][ep*steps:(ep+1)*steps]
        if sorted([i for batch in batches for i in batch])!=expected:raise ValueError('MISSING_OR_DUPLICATED_EPOCH_IDS')
        if any(r['epoch']!=ep+1 or r['update']!=ep*steps+j+1 for j,r in enumerate(rows)):raise ValueError('BAD_UPDATE_HISTORY')
    finite_tree({k:state[k] for k in ('model','ema','optimizer','loss_buffers')})
    if spec['arm']!='B0':
        if not (out/'PAIRED_CONTROL_AUDIT.json').exists():
            cc=read_json(BASE/'E2E_V2_CONTRACT.json');reference=cc['reference_names'][spec['seed']-42]
            root=original.OLD/'B0_seed42' if reference=='OLD42' else BASE/'runs'/reference
            own_initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
            ref_initial=torch.load(root/'initial.pt',map_location='cpu',weights_only=False)
            for key,value in ref_initial['model'].items():exact(value,own_initial['model'][key],'RECOVER_SHARED_INIT/'+key)
            exact(ref_initial['rng'],own_initial['rng'],'RECOVER_INITIAL_RNG')
            ref_final=torch.load(root/'final.pt',map_location='cpu',weights_only=False)
            exact(ref_final['batch_orders'],state['batch_orders'],'RECOVER_BATCH_IDS')
            write_json(out/'PAIRED_CONTROL_AUDIT.json',dict(status='PASS',reference=reference,shared_init_all_tensors=True,
                global_initial_RNG=True,all_batch_IDs=True,epochs=spec['epochs'],updates=total,recovered_without_training=True))
            del own_initial,ref_initial,ref_final
    audit=dict(status='PASS_COMPLETED_SAVED_TRAINING',at=now(),spec=spec,epochs=state['epoch'],updates=total,
        completed_before_recovery=True,all_epoch_IDs_exact=True,all_state_finite=True,
        final_sha=sha(out/'final.pt'),initial_sha=sha(out/'initial.pt'),retrained=False)
    receipt=REC/'saved_model_audits'/f"{spec['name']}.json"
    if not receipt.exists():write_json(receipt,audit)
    del data,state;gc.collect()
    return audit


def finalize_existing(spec,out):
    source_check();audit=validate_final(spec,out)
    if not (out/'epoch_summary.json').exists():
        rows=[read_json(out/f'epoch{ep:02}.json') for ep in range(1,spec['epochs']+1)]
        write_json(out/'epoch_summary.json',rows)
    if not (out/'FINAL_LOCK.json').exists():
        write_json(out/'FINAL_LOCK.json',dict(at=now(),files=seal([out/'initial.pt',out/'final.pt',out/'contract.json']),
            updates=audit['updates'],seconds=sum(r['seconds'] for r in read_json(out/'epoch_summary.json')),
            completion_record_recovered=True,training_not_repeated=True,
            recovery_audit=sha(REC/'saved_model_audits'/f"{spec['name']}.json")))
    verify(read_json(out/'FINAL_LOCK.json')['files'])
    status(stage='SAVED_FINAL_RECOVERED_NO_RETRAIN',run=spec['name'],epoch=spec['epochs'],updates=audit['updates'])
    print('REUSED_COMPLETED_FINAL',spec['name'],audit['updates'],flush=True)


def train(spec,_fixture_stop_after_epoch=None):
    source_check();torch.set_num_threads(1)
    out=RUNROOT/spec['name']
    if (out/'final.pt').exists():return finalize_existing(spec,out)
    if spec['arm']!='B0':
        contract_seal=read_json(BASE/'E2E_V2_SEAL.json');verify(contract_seal['files'])
        if read_json(BASE/'PREFLIGHT_METHODS.json')['status']!='PASS':raise ValueError('METHOD_PREFLIGHT_REQUIRED')
    out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();guard=original.guard();data=DATA_PROVIDER('fit');idx=original.indices_for(spec,data)
    t=Trainer(data,idx,epochs=spec['epochs'],seed=spec['seed'],order_seed=spec['order'],arm=spec['arm'])
    contract=dict(spec=spec,source=sha(BASE/'SOURCE_MANIFEST.json'),fit=sha(BASE/'C0/fit.npz'),
                  final_selection='FIXED_FINAL_EMA_0.995',batch=64,clip=5)
    if spec['arm']!='B0':contract['E2E_V2']=sha(BASE/'E2E_V2_CONTRACT.json')
    if (out/'contract.json').exists():exact(read_json(out/'contract.json'),contract,'RESUME_CONTRACT')
    else:write_json(out/'contract.json',contract)
    if not (out/'initial.pt').exists():save_checkpoint(out/'initial.pt',t.snapshot(contract))
    # Verify the actual immutable starting point before selecting any saved epoch.
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
    exact(contract,initial['contract']);exact(initial['model'],cpu_tree(t.model.state_dict()),'FRESH_INITIAL_MATCH')
    reference_root=None
    if spec['arm']!='B0':
        cc=read_json(BASE/'E2E_V2_CONTRACT.json');reference=cc['reference_names'][spec['seed']-42]
        reference_root=original.OLD/'B0_seed42' if reference=='OLD42' else BASE/'runs'/reference
        shared=torch.load(reference_root/'initial.pt',map_location='cpu',weights_only=False)
        for key,value in shared['model'].items():exact(value,t.model.state_dict()[key],'FORMAL_SHARED_INIT/'+key)
        exact(shared['rng'],rng_state(t.generator),'FORMAL_ALL_INITIAL_RNG');del shared
    elif spec['seed']==42:
        shared=torch.load(original.OLD/'B0_seed42/initial.pt',map_location='cpu',weights_only=False)
        exact(shared['model'],t.model.state_dict(),'OLD_B0_INITIAL');del shared
    candidates=sorted((out/'resume').glob('epoch*.pt'))
    if candidates:
        resume_path=candidates[-1]
        state=torch.load(resume_path,map_location='cpu',weights_only=False)
        exact(state['contract'],contract);finite_tree({k:state[k] for k in ('model','ema','optimizer')})
        receipt_path=resume_path.with_suffix('.json')
        if not receipt_path.exists():write_json(receipt_path,dict(at=now(),files=seal([resume_path]),recovered_atomic_checkpoint_receipt=True))
        verify(read_json(receipt_path)['files']);t.load(state,contract);del state
    else:
        resume_path=out/'initial.pt';t.load(initial,contract)
    del initial
    if t.cursor not in (0,len(idx)):raise ValueError('ONLY_FULL_EPOCH_RECOVERY_SUPPORTED')
    first_epoch=t.epoch+1
    if spec['split']=='fit85' and spec['epochs']==64:
        final_ema=cpu_tree(t.ema.module.state_dict())
        for completed_epoch in range(1,t.epoch+1):
            cp=out/'resume'/f'epoch{completed_epoch:02}.pt';epochfile=out/f'epoch{completed_epoch:02}.pt'
            if not epochfile.exists():os.link(cp,epochfile)
            target=out/f'calib_epoch{completed_epoch:02}.json'
            if not target.exists():
                prior=torch.load(cp,map_location='cpu',weights_only=False);t.ema.module.load_state_dict(prior['ema'],strict=True);del prior
                cal=original.subset(data,np.load(BASE/'E1/calib_split.npz')['calib_index'])
                paths,logits=original.infer(t.ema.module,cal,guard=guard)
                summary,_=original.ex.summarize(dict(paths=paths,logits=logits),cal);write_json(target,summary)
                del cal,paths,logits
        t.ema.module.load_state_dict(final_ema,strict=True);del final_ema
    status(stage='TRAIN_RESUMED' if candidates else 'TRAIN',run=spec['name'],epoch=t.epoch,epochs=spec['epochs'],
           updates=t.updates,checkpoint=str(resume_path))
    for ep in range(first_epoch,spec['epochs']+1):
        started=time.perf_counter();losses=[]
        for _ in range(t.steps_per_epoch):
            row=t.step();losses.append(row['loss']);guard.check()
        row=dict(epoch=ep,mean_train_loss=float(np.mean(losses)),updates=t.updates,seconds=time.perf_counter()-started)
        if (out/f'epoch{ep:02}.json').exists():
            # A crash after writing metadata but before its checkpoint may replay only this epoch.
            prior=read_json(out/f'epoch{ep:02}.json')
            if prior['mean_train_loss']!=row['mean_train_loss'] or prior['updates']!=row['updates']:raise ValueError('REPLAYED_EPOCH_NOT_EXACT')
        else:write_json(out/f'epoch{ep:02}.json',row)
        snapshot=t.snapshot(contract)
        checkpoint=out/'resume'/f'epoch{ep:02}.pt';checkpoint.parent.mkdir(exist_ok=True)
        save_checkpoint(checkpoint,snapshot)
        write_json(checkpoint.with_suffix('.json'),dict(at=now(),files=seal([checkpoint]),epoch=ep,updates=t.updates))
        if spec['split']=='fit85' and spec['epochs']==64:
            # Preserve the originally specified per-epoch calibration checkpoints.
            epochfile=out/f'epoch{ep:02}.pt'
            if not epochfile.exists():os.link(checkpoint,epochfile)
            cal=original.subset(data,np.load(BASE/'E1/calib_split.npz')['calib_index'])
            paths,logits=original.infer(t.ema.module,cal,guard=guard)
            summary,_=original.ex.summarize(dict(paths=paths,logits=logits),cal)
            target=out/f'calib_epoch{ep:02}.json'
            if not target.exists():write_json(target,summary)
            del cal,paths,logits
        status(stage='TRAIN',run=spec['name'],epoch=ep,epochs=spec['epochs'],updates=t.updates,epoch_seconds=row['seconds'],
               durable_checkpoint=str(checkpoint))
        print('EPOCH_CHECKPOINTED',spec['name'],row,flush=True)
        if _fixture_stop_after_epoch==ep:raise InterruptedError('INTENTIONAL_PREFLIGHT_INTERRUPTION')
    save_checkpoint(out/'final.pt',t.snapshot(contract))
    write_json(out/'epoch_summary.json',[read_json(out/f'epoch{ep:02}.json') for ep in range(1,spec['epochs']+1)])
    if reference_root is not None:
        other=torch.load(reference_root/'final.pt',map_location='cpu',weights_only=False)
        exact(other['batch_orders'],t.batch_orders,'FORMAL_ALL_EPOCH_BATCH_IDS')
        write_json(out/'PAIRED_CONTROL_AUDIT.json',dict(status='PASS',reference=reference,shared_init_all_tensors=True,
            global_initial_RNG=True,all_batch_IDs=True,epochs=spec['epochs'],updates=t.updates));del other
    write_json(out/'FINAL_LOCK.json',dict(at=now(),files=seal([out/'initial.pt',out/'final.pt',out/'contract.json']),
        updates=t.updates,seconds=time.perf_counter()-start,resource=rg.snapshot(),per_epoch_full_state_saved=True))
    source_check()


def dispatch(script,args):
    if script=='execution.py':
        mode,spec_text=args;spec=json.loads(spec_text)
        if mode=='train':train(spec)
        elif mode=='eval':original.evaluate(spec,spec['split']!='fit85')
        else:raise ValueError('UNKNOWN_MODE')
    else:
        sys.argv=[str(BASE/script),*args];runpy.run_path(str(BASE/script),run_name='__main__')


if __name__=='__main__':dispatch(sys.argv[1],sys.argv[2:])
