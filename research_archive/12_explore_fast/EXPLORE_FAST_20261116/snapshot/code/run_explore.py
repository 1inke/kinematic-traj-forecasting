"""Existing Trainer and five-fold resume loop; short full-C0-fit exploration only."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import sys
import time
import numpy as np
from explore_common import *


class ResumeProbeStop(RuntimeError):
    pass


def training_loop(tc, rf, guard, data, indices, arm, epochs, seed, directory,
                  contract, progress=None, stop_after=None):
    """Only adapt arm selection/metadata; optimizer, batches, loss and RNG stay original."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for pt in sorted(directory.glob('ckpt_epoch*.pt')):
        side = pt.with_suffix('.json')
        if side.exists():
            assert sha(pt) == read(side)['sha256'], 'CHECKPOINT_SHA_MISMATCH'
        else:
            pt.rename(pt.with_name(pt.name + '.uncommitted.' + str(time.time_ns())))
    for pending in directory.glob('ckpt_epoch*.pt.new'):
        pending.rename(pending.with_name(pending.name + '.uncommitted.' + str(time.time_ns())))
    last = [time.perf_counter()]

    class SelectedTrainer(tc.Trainer):
        def __init__(self, *args, **kwargs):
            kwargs['arm'] = arm
            super().__init__(*args, **kwargs)
            count = sum(p.numel() for p in self.model.parameters())
            if arm == 'B0' and count != 295474:
                raise ValueError('WRONG_B0_MODEL')
            # Original loop asserts total parameters; adapters may add their own parameters.
            rf.B0_PARAMETERS = count

        def step(self):
            guard.check()
            return super().step()

    def save(path, state):
        tc.save_checkpoint(path, state)
        elapsed = time.perf_counter() - last[0]
        side = dict(at=now(), file=str(path), sha256=sha(path), epoch=state['epoch'],
                    updates=state['updates'], order_hash=state['fivefold_order_hash'],
                    epoch_wall_seconds=elapsed)
        replace_json(Path(path).with_suffix('.json'), side)
        if progress:
            status(progress, dict(status='TRAIN', arm=arm, seed=seed, epochs=epochs,
                                  **{k:v for k,v in side.items() if k != 'at'}))
        last[0] = time.perf_counter()
        if stop_after == state['epoch']:
            raise ResumeProbeStop('PREFLIGHT_AFTER_COMPLETE_EPOCH')

    proxy = SimpleNamespace(Trainer=SelectedTrainer, save_checkpoint=save)
    # The legacy loop hashes integer IDs only. Integer row indices change metadata, not inputs.
    train_data = dict(data, ids=np.arange(len(data['ids']), dtype=np.int64))
    return rf.train_one(proxy, train_data, indices, epochs, seed, 'cuda:0', directory, contract)


def evaluate(tc, guard, model, data, directory):
    from fivefold_metrics import per_window, aggregate
    paths, logits = tc.infer(model, data, guard=guard)
    paths, logits = paths.astype(np.float32), logits.astype(np.float32)
    window = per_window(paths, logits, data['future'], data['past'][:, 0])
    metrics = aggregate(window, data['group'])
    past = data['past'][:, 0].astype(np.float64)
    points = np.concatenate((np.broadcast_to(past[:, None, -1:, :], (len(past),20,1,2)),
                             paths.astype(np.float64)), axis=2)
    velocity = np.diff(points, axis=2) / .4
    acceleration = np.diff(velocity, axis=2) / .4
    jerk = np.diff(acceleration, axis=2) / .4
    junction = np.linalg.norm(velocity[:, :, 0] - ((past[:, -1]-past[:, -2])/.4)[:, None], axis=-1)/.4
    def stats(values):
        return dict(mean=float(values.mean()), p95=float(np.percentile(values,95)))
    metrics['motion_allK20'] = dict(speed=stats(np.linalg.norm(velocity,axis=-1)),
        acceleration=stats(np.linalg.norm(acceleration,axis=-1)),
        jerk=stats(np.linalg.norm(jerk,axis=-1)), junction_acceleration=stats(junction))
    save_npz(directory / 'dev_predictions.npz',paths=paths,logits=logits,ids=data['ids'],group=data['group'])
    save_npz(directory / 'dev_window_metrics.npz',ids=data['ids'],group=data['group'],**window)
    return metrics


def paired_details(directory, reference):
    if not (reference / 'result.json').exists():
        raise ValueError('PAIRED_B0_REQUIRED_BEFORE_METHOD')
    validate_result(reference)
    with np.load(directory/'dev_window_metrics.npz',allow_pickle=False) as m, np.load(reference/'dev_window_metrics.npz',allow_pickle=False) as b:
        assert np.array_equal(m['ids'],b['ids']) and np.array_equal(m['group'],b['group'])
        groups = {}
        for name,mask in [('overall',np.ones(len(m['ids']),bool)),('stable',m['group']==0),
                          ('transition',m['group']==1),('change',m['group']==2)]:
            if not mask.any():
                continue
            oldgood=b['coverage'][mask];newgood=m['coverage'][mask]
            lost=int((oldgood & ~newgood).sum());rescue=int((~oldgood & newgood).sum())
            groups[name]=dict(n=int(mask.sum()),reference_good=int(oldgood.sum()),lost=lost,rescued=rescue,
                lost_fraction_of_reference_good=lost/int(oldgood.sum()) if oldgood.any() else None,
                coverage_diff_pp=100*float(newgood.mean()-oldgood.mean()))
    return groups


def run(arm, epochs, seed):
    verify_source()
    if epochs not in (16,32) or arm not in ('B0','T1','E3_self') or seed not in (42,43):
        raise ValueError('OUTSIDE_FINITE_EXPLORATION_PLAN')
    directory=ROOT/'runs'/f'{arm}_E{epochs}_s{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    progress=ROOT/'standalone/slots'/f'{arm}_E{epochs}_s{seed}.json'
    tc,rf,guard=load_training()
    with np.load(EXEC/'C0/fit.npz',allow_pickle=False) as z:fit={k:z[k] for k in z.files}
    contract=train_contract(arm,epochs,seed,len(fit['ids']))
    if (directory/'result.json').exists():
        result=validate_result(directory)
        if result['contract']!=contract:raise ValueError('RUN_CONTRACT_CHANGED')
        return
    guard.check(force=True)
    t0=time.perf_counter()
    trainer,info=training_loop(tc,rf,guard,fit,np.arange(len(fit['ids'])),arm,epochs,seed,directory,contract,progress)
    status(progress,dict(status='EVALUATE_DEV',arm=arm,epochs=epochs,seed=seed,epoch=epochs))
    with np.load(EXEC/'C0/dev.npz',allow_pickle=False) as z:dev={k:z[k] for k in z.files}
    metrics=evaluate(tc,guard,trainer.ema.module,dev,directory)
    paired=paired_details(directory,ROOT/'runs'/f'B0_E{epochs}_s{seed}') if arm!='B0' else None
    result=dict(at=now(),status='DONE',arm=arm,epochs=epochs,seed=seed,train_set='fit',
                primary='change_coverage',contract=contract,train=info,dev=metrics,paired=paired,
                wall_seconds=time.perf_counter()-t0,interpretation='EXPLORATORY_NOT_METHOD_SUCCESS')
    files=[p for p in directory.iterdir() if p.is_file() and p.suffix in ('.npz','.pt','.json')]
    result['files']={str(p):sha(p) for p in files}
    replace_json(directory/'result.json',result)
    status(progress,dict(status='DONE',arm=arm,epochs=epochs,seed=seed,epoch=epochs,
                         result=str(directory/'result.json')))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True);ap.add_argument('--epochs',type=int,required=True)
    ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();run(a.arm,a.epochs,a.seed)
