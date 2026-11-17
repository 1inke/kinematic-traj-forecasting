"""FIT-only real CUDA wiring, complete-state resume and lightweight metric checks."""
import ast
import numpy as np
from explore_common import *
from run_explore import training_loop,ResumeProbeStop,evaluate


def main():
    for p in (ROOT/'code').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    verify(read(PARALLEL/'recovery_20261116/SOURCE_MANIFEST.json')['files'])
    tc,rf,guard=load_training()
    import torch
    with np.load(EXEC/'C0/fit.npz',allow_pickle=False) as z:data={k:z[k][:513] for k in z.files}
    indices=np.arange(513);checks=[]
    pre=ROOT/'preflight'/str(time.time_ns());pre.mkdir(parents=True,exist_ok=False)
    contract=dict(study='EXPLORATION_PREFLIGHT_FIT_ONLY',epochs=2,seed=42,n_train=513)
    full,info=training_loop(tc,rf,guard,data,indices,'B0',2,42,pre/'full',dict(contract,arm='B0'))
    expected=torch.load(pre/'full/ckpt_epoch002.pt',map_location='cpu',weights_only=False)
    del full
    try:
        training_loop(tc,rf,guard,data,indices,'B0',2,42,pre/'split',dict(contract,arm='B0'),stop_after=1)
    except ResumeProbeStop:pass
    else:raise AssertionError('PREFLIGHT_DID_NOT_INTERRUPT')
    resumed,_=training_loop(tc,rf,guard,data,indices,'B0',2,42,pre/'split',dict(contract,arm='B0'))
    actual=torch.load(pre/'split/ckpt_epoch002.pt',map_location='cpu',weights_only=False)
    tc.exact(expected,actual,'EXPLORATION_RESUME_ALL_STATE')
    checks.append('B0_513_FIT_WINDOWS_18_UPDATES_INTERRUPTED_AND_CONTINUOUS_ALL_STATE_EXACT')
    evaluated={k:v[:128] for k,v in data.items()}
    metric=evaluate(tc,guard,resumed.ema.module,evaluated,pre/'split')
    assert metric['overall']['n']==128 and all(np.isfinite(v) for stats in metric['motion_allK20'].values() for v in stats.values())
    checks.append('SAVED_PREDICTIONS_AND_LIGHT_MOTION_METRICS_FINITE')
    del resumed,expected,actual
    torch.cuda.empty_cache()
    parameters={}
    for arm in ('T1','E3_self'):
        trainer,details=training_loop(tc,rf,guard,data,indices,arm,2,42,pre/arm,dict(contract,arm=arm))
        parameters[arm]=details['parameters']
        metrics=evaluate(tc,guard,trainer.ema.module,evaluated,pre/arm)
        assert np.isfinite(metrics['overall']['minADE'])
        checks.append(arm+'_REAL_CUDA_TWO_EPOCHS_AND_EVALUATION_PASS')
        del trainer;torch.cuda.empty_cache()
    replace_json(ROOT/'PREFLIGHT_PASS.json',dict(at=now(),status='PASS',scope='FIT_ONLY_REAL_CUDA',
                 checks=checks,parameters=parameters,folder=str(pre),no_official_test_metrics_used=True,
                 unchanged_training_sources_verified=True))
    print(json.dumps(read(ROOT/'PREFLIGHT_PASS.json'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

