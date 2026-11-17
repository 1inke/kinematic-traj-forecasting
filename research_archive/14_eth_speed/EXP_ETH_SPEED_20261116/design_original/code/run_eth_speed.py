"""CLI phases: prepare (read-only audit), diag (zero training), aug (one run).

No phase automatically starts another phase that trains a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from speed_io import config, frozen, imports, monitor, now, read, sha, verify, write
from eth_speed_math import (fit_profile, scales, speed_bins, transform_past, inverse_paths,
                            stratify, diagnostic_decision, gain, observed_speed)


class PauseRequested(Exception):
    pass


def index_sha(indices):
    return hashlib.sha256(np.asarray(indices, np.int64).tobytes()).hexdigest()


def check_pause(root, deadline):
    if (Path(root)/'PAUSE.flag').exists() or time.monotonic() >= deadline:
        raise PauseRequested('Requested pause or 2.5h session limit reached')


def save_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    # Recomputed incomplete case outputs can replace only files in this new root.
    from uuid import uuid4
    import os
    temp = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    with temp.open('xb') as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def save_torch(path, state):
    # A shutdown during serialization leaves an ignored unique temp, never a
    # truncated canonical checkpoint. A committed epoch also needs its sidecar.
    import os
    import torch
    from uuid import uuid4
    path = Path(path); temp = path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    with temp.open('xb') as f:
        torch.save(state, f); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def load_population(c, fold):
    fd, fm = imports(c)
    train = fd.load_split(c['data_dir'], fold, 'train')
    idx, val_idx = fd.scene_tail_split(train['scene'])
    b = c['baselines'][fold]; receipt = read(b['receipt'])
    contract = receipt['contract']
    if index_sha(idx) != contract['train_index_sha'] or contract['epochs'] != 64 or contract['seed'] != 42:
        raise ValueError('BASELINE_TRAINING_PROTOCOL_MISMATCH')
    profile = fit_profile(train['past'][idx])
    return fd, fm, train, idx, val_idx, profile


def model_from_checkpoint(c, fold, checkpoint=None):
    fd, fm, tc, bf = imports(c, training=True)
    import torch
    b = c['baselines'][fold]
    path = checkpoint or b['checkpoint']
    if checkpoint is None and sha(path) != b['sha256']:
        raise ValueError('CHECKPOINT_SHA_MISMATCH')
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['epoch'] != 64 or state['updates'] != 64*math.ceil(read(b['receipt'])['contract']['n_train']/64):
        raise ValueError('CHECKPOINT_NOT_FIXED_FINAL_64')
    model, cfg = bf.make_model(c['device'], 42)
    model.load_state_dict(state['ema'], strict=True)
    if sum(p.numel() for p in model.parameters()) != 295474:
        raise ValueError('MODEL_PARAMETER_COUNT')
    return tc, model


def metric_case(fm, paths, logits, data, profile, directory, extra=None):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    # Save first; all displayed metrics are recomputed from the saved float32 paths.
    paths = np.asarray(paths, np.float32); logits = np.asarray(logits, np.float32)
    if not np.isfinite(paths).all() or not np.isfinite(logits).all():
        raise ValueError('NONFINITE_PREDICTION')
    save_npz(directory/'predictions.npz', paths=paths, logits=logits, ids=data['ids'],
             group=data['group'], scene=data['scene'])
    w = fm.per_window(paths, logits, data['future'], data['past'][:, 0])
    save_npz(directory/'window_metrics.npz', ids=data['ids'], group=data['group'],
             observed_speed=observed_speed(data['past']), speed_bin=speed_bins(data['past'], profile), **w)
    summary = stratify(w, data['past'], data['future'], profile, data['group'])
    write(directory/'summary.json', dict(summary=summary, extra=extra or {},
          files={str(directory/f): sha(directory/f) for f in ('predictions.npz', 'window_metrics.npz')}))
    return summary


def prepare(c, root, contract_sha):
    from audit_eth_speed import run_audit
    path = Path(root)/'DIAG'/'PREPROCESSING_AUDIT.json'
    if path.exists():
        result = read(path)
        if result['input_contract_sha256'] != contract_sha or not result['pipeline_consistent']:
            raise ValueError('STALE_OR_FAILED_AUDIT')
        return result
    return run_audit(c, root, contract_sha)


def diag(c, root, contract_sha, deadline):
    audit = prepare(c, root, contract_sha)
    root = Path(root); directory = root/'DIAG'
    if (directory/'DONE.json').exists():
        verify(read(directory/'DONE.json')['files'])
        print('DIAG already complete; verified; no inference repeated.', flush=True)
        return
    # A pause applies before model construction or any inference.
    check_pause(root, deadline)
    results = {}
    for fold in ('eth', 'hotel'):
        fd, fm, train, idx, val_idx, profile = load_population(c, fold)
        test = fd.load_split(c['data_dir'], fold, 'test')
        del train
        tc, model = model_from_checkpoint(c, fold)
        # The model inference API receives only observed positions and valid-agent counts.
        s = scales(test['past'], profile)
        definitions = [('identity', np.ones_like(s), 'motion'), ('high_motion', s, 'motion'),
                       ('high_scene', s, 'scene'), ('reverse_motion', 1/s, 'motion'),
                       ('fixed075_motion', np.full_like(s, .75), 'motion')]
        cases = {}; low_ok = False; replay_ok = False
        for name, scale, kind in definitions:
            check_pause(root, deadline)
            monitor(root, phase='DIAG', fold=fold, case=name, training=False)
            cd = directory/fold/name; result_file = cd/'summary.json'
            if result_file.exists():
                rec = read(result_file); verify(rec['files']); cases[name] = rec['summary']
                continue
            p = transform_past(test['past'], test['num_valid'], scale, kind)
            past_only = dict(past=p, num_valid=test['num_valid'], ids=test['ids'])
            paths, logits = tc.infer(model, past_only, batch=64)
            paths = inverse_paths(paths, test['past'], scale)
            cases[name] = metric_case(fm, paths, logits, test, profile, cd,
                                       dict(scale_min=float(scale.min()), scale_max=float(scale.max()),
                                            transformed_windows=int((scale != 1).sum()), kind=kind))
        original = np.load(directory/fold/'identity/predictions.npz')
        # Independent identity replay protects against stale predictions/model mismatch.
        # No full additional TEST pass: first 8 observations are enough for a wiring check.
        probe = dict(past=test['past'][:8], num_valid=test['num_valid'][:8], ids=test['ids'][:8])
        paths_probe, logits_probe = tc.infer(model, probe, batch=64)
        replay_ok = bool(np.allclose(paths_probe, original['paths'][:8], atol=1e-4, rtol=1e-5)
                         and np.allclose(logits_probe, original['logits'][:8], atol=1e-4, rtol=1e-5))
        if fold == 'eth' and c.get('eth_saved_B0'):
            with np.load(c['eth_saved_B0']) as saved:
                replay_ok = replay_ok and bool(np.array_equal(saved['ids'], test['ids'])
                    and np.array_equal(saved['group'], test['group'])
                    and np.allclose(original['paths'], saved['paths'], atol=1e-4, rtol=1e-5)
                    and np.allclose(original['logits'], saved['logits'], atol=1e-4, rtol=1e-5))
        primary = np.load(directory/fold/'high_motion/predictions.npz')
        unchanged = s == 1
        low_ok = bool(np.allclose(primary['paths'][unchanged], original['paths'][unchanged], atol=1e-4, rtol=1e-5))
        if not replay_ok or not low_ok:
            raise ValueError('IDENTITY_CONTROL_FAILED')
        saved = c.get('eth_saved_T1') if fold == 'eth' else None
        if saved:
            with np.load(saved) as z:
                if not np.array_equal(z['ids'], test['ids']) or not np.array_equal(z['group'], test['group']):
                    raise ValueError('SAVED_T1_POPULATION_MISMATCH')
                cases['T1_saved_descriptive_only'] = metric_case(fm, z['paths'], z['logits'], test, profile,
                     directory/fold/'T1_saved_descriptive_only', dict(training=False, inference=False, source=saved))
        results[fold] = dict(profile=profile, train_n=len(idx), train_index_sha256=index_sha(idx),
                             cases=cases, low_speed_identity_passed=low_ok, baseline_replay_passed=replay_ok,
                             test_speed_summary=fit_profile(test['past']))
        save_npz(directory/fold/'scale_inputs.npz', ids=test['ids'], observed_speed=observed_speed(test['past']),
                 scale=s, training_index=idx, validation_index=val_idx)
        original.close(); primary.close()
        del original, primary, model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    decision = diagnostic_decision(results, audit)
    write(directory/'RESULTS.json', dict(input_contract_sha256=contract_sha, folds=results))
    write(directory/'DECISION.json', dict(input_contract_sha256=contract_sha, **decision))
    from report_eth_speed import render
    render(directory)
    files = {str(p): sha(p) for p in directory.rglob('*') if p.is_file() and p.name != 'DONE.json'}
    write(directory/'DONE.json', dict(at=now(), input_contract_sha256=contract_sha, files=files))
    monitor(root, phase='DIAG_COMPLETE', training=False, decision=decision['status'])
    print(json.dumps(decision, ensure_ascii=False), flush=True)


def aug(c, root, contract_sha, deadline):
    root = Path(root); decision = read(root/'DIAG/DECISION.json')
    completed = read(root/'DIAG/DONE.json'); verify(completed['files'])
    if decision['input_contract_sha256'] != contract_sha or decision['status'] != 'ELIGIBLE_EXPLORATORY':
        raise ValueError('DIAG_DOES_NOT_SUPPORT_THIS_TRAINING_PLAN')
    rd = root/'SPEED_AUG/B0_wide_eth_E64_s42'; rd.mkdir(parents=True, exist_ok=True)
    if (rd/'DONE.json').exists():
        verify(read(rd/'DONE.json')['files']); print('AUG already complete; verified.', flush=True); return
    check_pause(root, deadline)
    fd, fm, tc, bf = imports(c, training=True)
    from speed_aug_adapter import trainer_type
    fd, fm, data, idx, val_idx, profile = load_population(c, 'eth')
    contract = dict(study='EXP_SPEED_AUG_20261116', input_contract_sha256=contract_sha,
                    diagnostic_decision_sha256=sha(root/'DIAG/DECISION.json'), arm='B0_SPEED_WIDE',
                    epochs=64, seed=42, batch=64, train_index_sha256=index_sha(idx),
                    augmentation=dict(speed_uniform=[.5, 2.], probability=.5, reflection=True),
                    fixed_final_ema=.995)
    trainer = trainer_type(tc, bf)(data, idx, batch=64, epochs=64, device=c['device'], seed=42, arm='B0')
    if sum(p.numel() for p in trainer.model.parameters()) != 295474:
        raise ValueError('MODEL_PARAMETER_COUNT')
    import torch
    # Ignore an epoch whose pt was committed just before shutdown but whose
    # sidecar was not; resume the previous committed epoch and recompute it.
    checkpoints = sorted(p for p in rd.glob('ckpt_epoch*.pt') if p.with_suffix('.json').exists())
    order_hash = hashlib.sha256(b'FIVEFOLD_ORDER').hexdigest()
    if checkpoints:
        sidecar = read(checkpoints[-1].with_suffix('.json'))
        if sha(checkpoints[-1]) != sidecar['sha256']:
            raise ValueError('RESUME_CHECKPOINT_CORRUPT')
        state = torch.load(checkpoints[-1], map_location='cpu', weights_only=False)
        trainer.load(state, contract); order_hash = state['fivefold_order_hash']
    training_start = time.monotonic(); last_epoch_end = training_start
    while trainer.updates < 64*trainer.steps_per_epoch:
        trainer.step()
        batch_ids = np.asarray(trainer.batch_orders.pop(), np.int64)
        order_hash = hashlib.sha256(bytes.fromhex(order_hash)+batch_ids.tobytes()).hexdigest()
        if trainer.cursor == len(trainer.order):
            target = rd/f'ckpt_epoch{trainer.epoch:03d}.pt'
            state = trainer.snapshot(contract); state['fivefold_order_hash'] = order_hash
            save_torch(target, state)
            write(target.with_suffix('.json'), dict(epoch=trainer.epoch, updates=trainer.updates,
                  sha256=sha(target), order_hash=order_hash, at=now(),
                  epoch_wall_seconds=time.monotonic()-last_epoch_end,
                  session_elapsed_seconds=time.monotonic()-training_start))
            last_epoch_end = time.monotonic()
            # Only obsolete checkpoints created by THIS run; keep latest two states.
            for old in sorted(rd.glob('ckpt_epoch*.pt'))[:-2]:
                old.unlink()
            monitor(root, phase='TRAIN_SPEED_AUG', epoch=trainer.epoch, epochs=64,
                    updates=trainer.updates, total_updates=64*trainer.steps_per_epoch)
            check_pause(root, deadline)
    expected = read(c['baselines']['eth']['receipt'])['train']['order_hash']
    if order_hash != expected:
        raise ValueError('BASELINE_BATCH_ORDER_MISMATCH')
    # Same val and test population, no inference scaling for the primary comparison.
    val = fd.subset(data, val_idx); test = fd.load_split(c['data_dir'], 'eth', 'test')
    del data
    outputs = {}
    for label, population in (('val', val), ('test', test)):
        check_pause(root, deadline)
        paths, logits = tc.infer(trainer.ema.module, dict(past=population['past'],
             num_valid=population['num_valid'], ids=population['ids']), batch=64)
        outputs[label] = metric_case(fm, paths, logits, population, profile, rd/label)
    base = read(root/'DIAG/RESULTS.json')['folds']['eth']['cases']['identity']
    a = gain(base['overall']['minADE'], outputs['test']['overall']['minADE'])
    f = gain(base['overall']['minFDE'], outputs['test']['overall']['minFDE'])
    verdict = 'PROMISING_SINGLE_SEED' if a is not None and f is not None and a >= .02 and f >= .02 else 'NO_CLEAR_JOINT_GAIN'
    write(rd/'RESULTS.json', dict(at=now(), contract=contract, order_hash=order_hash,
          comparisons=dict(ADE_gain=a, FDE_gain=f, verdict=verdict), outputs=outputs, baseline_test=base,
          baseline_promoted=False, causal_conclusion='NOT_ESTABLISHED'))
    files = {str(p): sha(p) for p in rd.rglob('*') if p.is_file() and p.name != 'DONE.json'}
    write(rd/'DONE.json', dict(at=now(), input_contract_sha256=contract_sha, files=files))
    monitor(root, phase='AUG_COMPLETE', verdict=verdict)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'diag', 'aug'))
    parser.add_argument('--config'); parser.add_argument('--out')
    args = parser.parse_args(); c = config(args.config); root = Path(args.out or c['output_dir'])
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 9000
    # Scheduled and terminal entry points use the same cross-study Windows mutex.
    mutex = None
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        api.CreateMutexW.restype = wintypes.HANDLE
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        mutex = api.CreateMutexW(None, False, 'Local\\TrajectoryFivefoldB0Standalone20261115')
        if not mutex or ctypes.get_last_error() == 183:
            raise RuntimeError('ANOTHER_EXPERIMENT_CONTROLLER_IS_ACTIVE')
    try:
        monitor(root, phase='VERIFY_INPUTS', requested=args.phase)
        digest = frozen(c, root)
        if args.phase == 'prepare':
            prepare(c, root, digest)
        elif args.phase == 'diag':
            diag(c, root, digest, deadline)
        else:
            aug(c, root, digest, deadline)
        write(root/'LAST_EXIT.json', dict(at=now(), phase=args.phase, status='SUCCESS', exit_code=0))
    except PauseRequested as exc:
        monitor(root, phase='PAUSED', reason=str(exc), safe_to_shutdown=True)
        write(root/'LAST_EXIT.json', dict(at=now(), phase=args.phase, status='PAUSED', exit_code=0))
    except BaseException as exc:
        monitor(root, phase='FAILED', error=str(exc))
        write(root/'LAST_EXIT.json', dict(at=now(), phase=args.phase, status='FAILED', exit_code=1,
                                         error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if mutex:
            api.CloseHandle(mutex)


if __name__ == '__main__':
    main()
