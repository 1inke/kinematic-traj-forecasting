"""Six fresh256-update models. No held labels; no automatic formal execution."""
import datetime
import time
import numpy as np
import contracts_discrete as c
import torch
from engine_discrete import Engine,infer
from evaluate_discrete import save
from preflight_checks import trees_equal

def main():
    start=time.perf_counter();pf=c.json_read(c.ROOT/'PREFLIGHT.json');bench=c.json_read(c.ROOT/'BENCHMARK.json')
    assert pf['status']=='PASS_ACTUAL_PREFLIGHT' and bench['T_admitted'] and bench['T_conservative_seconds']<=1800
    manifest=c.json_read(c.ROOT/'source_manifest.json');c.verify(manifest['files'])
    d=c.read_prepared('tiny');g=c.resource(c.json_read(c.ROOT/'MEMORY_ESTIMATE.json')['estimate']);g.check(force=True)
    root=c.ROOT/'tiny';root.mkdir(exist_ok=False);paired={};orders={}
    for seed in (42,43,44):
        for arm in ('G','H'):
            out=root/f'{arm}_{seed}';out.mkdir(exist_ok=False);e=Engine(d,arm,seed)
            if seed not in paired:paired[seed]=e.initial
            else:assert trees_equal(paired[seed],e.initial)
            run_manifest=dict(contract=e.contract,stats=c.binding(d),initial={k:c.digest(v.cpu().numpy()) for k,v in e.initial.items()},seed=seed,arm=arm,updates=256,epochs=64)
            c.write(out/'run_manifest.json',run_manifest)
            for point in (0,64,128,192,256):
                while e.updates<point:g.check();e.step()
                c.atomic_checkpoint(out/f'checkpoint_{point:03d}.pt',e.snapshot())
                score=infer(e.model,d['z'+arm],g.check);before={k:v.detach().clone() for k,v in e.model.state_dict().items()}
                save(out/f'point_{point:03d}',score,d,arm,'tiny',extra=dict(seed=seed,update=point,primary=point==256))
                assert trees_equal(before,e.model.state_dict())
                print('TINY',arm,seed,'update',point,'SAVED',flush=True)
            if seed not in orders:orders[seed]=e.batch_order
            else:assert orders[seed]==e.batch_order
            assert e.updates==256 and e.epoch==64 and len(e.history)==64
            c.write(out/'history.json',e.history)
            c.npz_new(out/'preupdate_trace.npz',score=np.stack([r['score'] for r in e.trace]),batch_indices=np.array(e.batch_order),loss=np.array([r['loss'] for r in e.trace]),grad_norm=np.array([r['unclipped_grad_norm'] for r in e.trace]))
            c.write(out/'done.json',dict(status='ALL256_AND_FIVE_SAVEPOINTS_COMPLETE_REVIEW_REQUIRED',contract=e.contract,files={str(p.relative_to(out)):c.sha(p) for p in out.rglob('*') if p.is_file()}))
            del e;torch.cuda.empty_cache()
    c.verify(manifest['files'],g.check);g.check(force=True)
    c.write(c.ROOT/'RESOURCE_TINY.json',dict(estimate=g.estimate,observations=g.observations))
    c.write(c.ROOT/'TINY_COMPUTATION.json',dict(status='ALL_SIX_TINY_SAVED_REVIEW_REQUIRED',new_models=6,updates=1536,formal_models=0,paired_initial_batch_order_EXACT=True,seconds=time.perf_counter()-start))
    print('TINY ALL6 COMPLETE saved review required',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        c.write(c.ROOT/('FAILURE_TINY_'+datetime.datetime.now().strftime('%H%M%S')+'.json'),dict(traceback=traceback.format_exc()))
        raise
