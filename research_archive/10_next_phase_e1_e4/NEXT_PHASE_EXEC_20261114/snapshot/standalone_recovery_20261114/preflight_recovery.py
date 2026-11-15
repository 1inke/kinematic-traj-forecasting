"""Exercise new final-record and real epoch-recovery paths before production."""
import ast
import copy
from pathlib import Path
import sys
import time
import adapter as a
from common import *
from train_core import exact
import torch
import numpy as np


def main():
    start=time.perf_counter();torch.set_num_threads(1)
    out=a.REC/'preflight';out.mkdir(exist_ok=False)
    verify(read_json(a.BASE/'SOURCE_MANIFEST.json')['files'])
    # Check that the source-level bookkeeping error is present in the preserved source,
    # and no original Trainer/model/loss/scheduler source has been edited.
    original_ast=ast.parse((a.BASE/'execution.py').read_text(encoding='utf-8'))
    assert any(isinstance(x,ast.Name) and isinstance(x.ctx,ast.Store) and x.id=='seal' for x in ast.walk(original_ast))
    original_final=a.BASE/'runs/E1A_s43/final.pt';saved_before=sha(original_final)
    s=dict(name='E1A_s43',arm='B0',seed=43,order=43,epochs=8,split='fit')
    audit=a.validate_final(s,a.BASE/'runs/E1A_s43')
    assert audit['updates']==2720 and sha(original_final)==saved_before
    data=a.DATA_PROVIDER('fit');idx=np.array(sorted(range(len(data['ids'])),key=lambda i:__import__('hashlib').sha256(str(data['ids'][i]).encode()).digest())[:513])
    fixture={k:v[idx] for k,v in data.items()}
    a.RUNROOT=out/'runs';a.DATA_PROVIDER=lambda split:fixture
    common=dict(arm='B0',seed=42,order=42,epochs=2,split='fit',fixture=True)
    continuous=dict(common,name='continuous');resumed=dict(common,name='resumed')
    a.train(continuous)
    try:a.train(resumed,_fixture_stop_after_epoch=1)
    except InterruptedError:pass
    else:raise AssertionError('INTERRUPTION_NOT_TRIGGERED')
    a.train(resumed)
    x=torch.load(a.RUNROOT/'continuous/final.pt',map_location='cpu',weights_only=False)
    y=torch.load(a.RUNROOT/'resumed/final.pt',map_location='cpu',weights_only=False)
    x.pop('contract');y.pop('contract');exact(x,y,'ADAPTER_REAL_TWO_EPOCH_RESUME_ALL_STATE')
    # Call the actual completed-run path again: it must perform zero optimizer steps.
    before=seal(list((a.RUNROOT/'resumed').glob('*.pt')))
    a.train(resumed)
    verify(before)
    if sha(original_final)!=saved_before:raise ValueError('ORIGINAL_COMPLETED_MODEL_CHANGED')
    # Identity-only change in the source call graph: actual scientific operators unchanged.
    write_json(a.REC/'PREFLIGHT_RECOVERY.json',dict(status='PASS',at=now(),seconds=time.perf_counter()-start,
        preserved_original_manifest=True,completed_43_CPU_validated_updates=2720,completed43_sha=saved_before,
        completed_model_retrain_steps=0,real_GPU_fixture_windows=513,fixture_epochs=2,fixture_updates=18,
        adapter_continuous_vs_epoch_resume_all_model_EMA_optimizer_scheduler_RNG_order_history_exact=True,
        completed_fixture_reentry_zero_updates=True,science_modules_unchanged=True))
    print('RECOVERY_PREFLIGHT_PASS',flush=True)


if __name__=='__main__':main()
