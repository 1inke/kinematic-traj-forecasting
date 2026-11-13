"""Freeze exact dependencies and causal context cache. No DSF scientific updates."""
import time,shutil
from run_common import *

def main():
    t=time.perf_counter();torch.set_num_threads(1);rg=rg_module();guard=rg.Guard(3*1024**3);guard.check(mode='startup',force=True)
    check_files(read(SETUP/'FINAL_INTERFACE_SEAL.json')['files']);old=read(STAGE_A/'FINAL_SEAL.json');check_files(old['files']);check_files(old['sources'])
    if read(STAGE_A/'WORKER_EXIT.json')['status']!='SUCCESS':raise ValueError('STAGE_A_NOT_COMPLETE')
    adapter,evidence=load_current('cuda');before=cpu(adapter.model.state_dict());out=RUN/'cache';out.mkdir(exist_ok=False)
    reports={}
    for split in ('fit','dev'):
        progress('PREPARE_CAUSAL_CONTEXTS',split=split)
        with np.load(BASE/'C0'/f'{split}.npz',allow_pickle=False) as f:ids=f['ids'].copy();past=f['past'].copy();valid=f['num_valid'].copy()
        if len(ids)!=(21719 if split=='fit' else 4753) or len(np.unique(ids))!=len(ids):raise ValueError('POPULATION_ID_COUNT')
        values={k:[] for k in ('context','past_loc','center','origin','matrix','native','native_logits')};max_identity=0.
        for start in range(0,len(ids),32):
            ix=slice(start,start+32);p=torch.as_tensor(past[ix],device='cuda');n=torch.as_tensor(valid[ix],device='cuda');b=adapter.encode(p,n)
            with torch.no_grad():
                y,l=decode_logits(adapter,torch.zeros(len(p),20,64,device='cuda'),b)
            if not torch.equal(y,b['native']):raise ValueError('CONTEXT_CACHE_ZERO_NATIVE_NOT_BITWISE')
            if not torch.equal(l,adapter.model.last_mode_logits[:,0]):raise ValueError('LOGIT_REPRODUCTION')
            center,origin,matrix=b['transform']
            current=dict(context=b['context'],past_loc=b['past_loc'],center=center,origin=origin,matrix=matrix,native=y,native_logits=l)
            for k,v in current.items():values[k].append(v.detach().cpu().numpy())
            guard.check()
        a={k:np.concatenate(v) for k,v in values.items()};a.update(ids=ids,raw_past=past,num_valid=valid)
        # Original stored compute64 vs this fixed-compute32 can differ numerically.
        with np.load(BASE/'populations'/split/'predictions.npz',allow_pickle=False) as f:
            if not np.array_equal(f['ids'],ids):raise ValueError('OLD_REFERENCE_ID')
            old_error=float(np.max(abs(f['paths']-a['native'])))
            if old_error>2e-5:raise ValueError('OLD_NATIVE_PATH_CACHE_TOLERANCE '+str(old_error))
        np.savez_compressed(out/f'{split}.npz',**a)
        with np.load(out/f'{split}.npz',allow_pickle=False) as f:
            for k,v in a.items():
                if not np.array_equal(f[k],v):raise ValueError('CONTEXT_READBACK')
        reports[split]=dict(N=len(ids),fixed_compute=32,zero_control_bitwise=True,old_stored_compute64_max_path_error=old_error,bytes_payload=sum(x.nbytes for x in a.values()),files=seal([out/f'{split}.npz']),no_future_or_group_access=True)
    exact(before,cpu(adapter.model.state_dict()),'FROZEN_ENCODER_CONTEXT_PREP')
    write(RUN/'CACHE_RECEIPT.json',dict(status='PASS_CAUSAL_FROZEN_CONTEXT_CACHE_READBACK_NO_DSF_UPDATES',splits=reports,generator=evidence,elapsed_seconds=time.perf_counter()-t))
    files=dict(old['sources']);files.update(old['files']);files.update(read(SETUP/'FINAL_INTERFACE_SEAL.json')['files'])
    files.update(read(BASE/'source_manifest.json')['files']);files.update(read(BASE/'B0_FINAL_LOCK.json')['files'])
    for x in reports.values():files.update(x['files'])
    files.update(seal([BASE/'C0/fit.npz',BASE/'C0/dev.npz',RUN/'CACHE_RECEIPT.json',STAGE_A/'CALIBRATION.json',ROOT/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py']))
    files.update(seal([p for p in RUN.iterdir() if p.suffix in ('.py','.md','.ps1','.json') and p.name not in ('progress.json',)]))
    write(RUN/'PRE_SOURCE.json',dict(status='PRE_REGISTERED_SOURCE_BEFORE_GPU_PREFLIGHT',files=files,generator_checkpoint_sha=evidence['checkpoint_sha256']))
    guard.check(force=True);write(RUN/'PREP_RESOURCE.json',dict(policy='warning_first_v3',estimate=3*1024**3,observations=guard.observations,snapshot=rg.snapshot(),disk_free=shutil.disk_usage(RUN).free))
    print('CONTEXT_PREPARATION_PASS',time.perf_counter()-t,flush=True)

if __name__=='__main__':main()
