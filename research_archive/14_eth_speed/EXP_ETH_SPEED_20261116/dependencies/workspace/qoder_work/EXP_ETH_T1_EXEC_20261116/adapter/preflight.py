"""FIT-only real CUDA resume proof, plus preserved source and B0 receipt audit."""
import sys
from pathlib import Path
from types import SimpleNamespace
from eth_exec_common import *


def main():
    ec,original=imports();ec.verify_source()
    tc,rf,guard=ec.load_training()
    original.preflight(SimpleNamespace(explore_dir=str(EXPLORE),out=str(ROOT)))
    cfg=read(ec.PARALLEL/'PARALLEL_CONFIG.json')
    data=rf.load_split(cfg['data_dir'],'eth','train');idx,_=rf.scene_tail_split(data['scene'])
    assert len(idx)==28791
    old,ck,state=original.b0_checkpoint(ec,cfg,rf,idx)
    verify(read(old/'DONE.json')['files'])
    c=dict(study='EXP_ETH_T1_PREFLIGHT',epochs=2,seed=42,arm='T1',n_train=513,train_index_sha=rf.index_sha(idx[:513]))
    import torch
    continuous=ROOT/'preflight/CONTINUOUS';resumed=ROOT/'preflight/RESUMED'
    if continuous.exists() or resumed.exists():raise FileExistsError('Preserve prior preflight artifacts; inspect before retry.')
    trainer,info=__import__('run_explore').training_loop(tc,rf,guard,data,idx[:513],'T1',2,42,continuous,c)
    continuous_state=torch.load(continuous/'ckpt_epoch002.pt',map_location='cpu',weights_only=False)
    del trainer
    import gc;gc.collect();torch.cuda.empty_cache()
    rx=__import__('run_explore')
    try:rx.training_loop(tc,rf,guard,data,idx[:513],'T1',2,42,resumed,c,stop_after=1)
    except rx.ResumeProbeStop:pass
    else:raise AssertionError('Expected an interruption after epoch one')
    trainer,reinfo=rx.training_loop(tc,rf,guard,data,idx[:513],'T1',2,42,resumed,c)
    recovered=torch.load(resumed/'ckpt_epoch002.pt',map_location='cpu',weights_only=False)
    tc.exact(continuous_state,recovered)
    assert reinfo['updates']==18 and reinfo['parameters']==295474
    import numpy as np
    sample={key:(value[:4].copy() if isinstance(value,np.ndarray) and len(value)==len(data['ids']) else value) for key,value in data.items()}
    p0,l0=tc.infer(trainer.ema.module,sample,guard=guard)
    sample['future']=sample['future']+100
    p1,l1=tc.infer(trainer.ema.module,sample,guard=guard)
    assert np.array_equal(p0,p1) and np.array_equal(l0,l1)
    artifact=dict(at=now(),status='PASS',real_CUDA=True,fit_only=True,test_model_inference=False,
        exact_resume='ALL_STATE_EXACT',updates=18,parameters=295474,future_label_inference_invariance=True,
        full_train_windows=len(idx),steps_per_epoch=450,formal_updates=28800,
        B0_checkpoint=str(ck),B0_checkpoint_sha256=sha(ck),B0_contract=state['contract'],original_B0_DONE_verified=True)
    replace_json(ROOT/'preflight/REAL_RESUME_PASS.json',artifact)
    print('PASS: real CUDA T1 exact resume, B0 identity and source hashes, FIT-only future-label invariance.')


if __name__=='__main__':main()
