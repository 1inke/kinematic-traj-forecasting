"""Execution-only safeguards around the unchanged supplied five-fold orchestration.

Trainer.step, model/loss/data split/budget rules and metric formulas are unchanged.
"""
from __future__ import annotations
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run_fivefold as rf
from fivefold_common import FOLDS, BUDGETS, now, read_json, sha256

CONTEXT = {}
STOP_AFTER_EPOCH = None   # Only the isolated real-resume preflight sets this hook.


class PreflightInterrupt(RuntimeError):
    pass


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def status(**values):
    atomic(ROOT / 'standalone/CURRENT.json', {'at': now(), **values})


def stable(value):
    if isinstance(value, dict):
        return {key: stable(item) for key, item in value.items() if key not in ('at', 'train_seconds_this_session')}
    if isinstance(value, (list, tuple)):
        return [stable(item) for item in value]
    return value


def verify_files(files):
    for name, expected in files.items():
        if sha256(name) != expected:
            raise ValueError('SEALED_FILE_CHANGED: ' + str(name))


def validate_done(directory):
    receipt = read_json(directory / 'DONE.json')
    if not receipt.get('files'):
        raise ValueError('DONE_HAS_NO_OUTPUT_PROOF')
    verify_files(receipt['files'])
    metrics = read_json(directory / 'metrics.json')
    manifest = ROOT / 'SOURCE_MANIFEST.json'
    if manifest.exists() and metrics['contract'].get('study_source') != sha256(manifest):
        raise ValueError('DONE_SOURCE_MISMATCH')


def save_once(path, value):
    path = Path(path)
    if path.name == 'DONE.json':
        files = [p for p in path.parent.iterdir() if p.is_file() and p.suffix in ('.pt', '.npz', '.json') and p.name != 'DONE.json']
        value = dict(value, files={str(p.resolve()): sha256(p) for p in files})
    if path.exists():
        if stable(read_json(path)) != stable(value):
            raise ValueError('EXISTING_RESULT_DIFFERS: ' + str(path))
        return
    atomic(path, value)


def quarantine(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('RECOVERY_PATH_OUTSIDE_STUDY')
    target = path.parent / 'uncommitted_recovery_files'
    target.mkdir(exist_ok=True)
    path.rename(target / (path.name + '.' + str(time.time_ns())))


class NumpyProxy:
    def __getattr__(self, name):
        return getattr(np, name)

    def savez_compressed(self, path, **arrays):
        path = Path(path)
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('OUTPUT_OUTSIDE_STUDY')
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != set(arrays) or any(not np.array_equal(saved[key], value) for key, value in arrays.items()):
                    raise ValueError('SAVED_EVALUATION_ARRAYS_DIFFER')
            return
        temp = path.with_suffix('.npz.pending')
        with temp.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)


def install(tc, output, audit_path=None, data_dir=None):
    import torch
    torch.set_num_threads(1)  # Matches NEXT_PHASE execution.py train/eval.
    if getattr(tc, '_fivefold_runtime_installed', False):
        CONTEXT.update(output=Path(output))
        return
    tc._fivefold_runtime_installed = True
    CONTEXT.update(output=Path(output), epoch_started=time.perf_counter())
    import resource_guard as rg
    estimate = read_json(Path(tc.__file__).parent / 'PREFLIGHT_E1.json')['estimated_peak_bytes']
    guard = rg.Guard(estimate)
    CONTEXT['guard'] = guard
    base_trainer, base_save = tc.Trainer, tc.save_checkpoint
    base_train_one, base_eval = rf.train_one, rf.evaluate
    base_contract, base_decide, base_final = rf.base_contract, rf.phase_decide, rf.phase_final

    class AuditedTrainer(base_trainer):
        def __init__(self, *args, **kwargs):
            guard.check(force=True)
            super().__init__(*args, **kwargs)
            if sum(p.numel() for p in self.model.parameters()) != rf.B0_PARAMETERS:
                raise ValueError('WRONG_B0_PARAMETER_COUNT_BEFORE_TRAINING')

        def step(self):
            guard.check()
            return super().step()

    def checked_save(path, state):
        path = Path(path)
        if path.with_suffix('.pt.new').exists():
            quarantine(path.with_suffix('.pt.new'))
        base_save(path, state)
        elapsed = time.perf_counter() - CONTEXT['epoch_started']
        receipt = dict(at=now(), file=str(path.resolve()), sha256=sha256(path), epoch=state['epoch'],
                       updates=state['updates'], order_hash=state['fivefold_order_hash'], epoch_wall_seconds=elapsed)
        atomic(path.with_suffix('.json'), receipt)
        CONTEXT['epoch_started'] = time.perf_counter()
        spec = state['contract']
        status(stage='TRAIN', run=path.parent.name, phase=spec.get('phase'), fold=spec.get('fold'),
               seed=spec.get('seed'), epoch=state['epoch'], epochs=spec['epochs'], updates=state['updates'],
               epoch_wall_seconds=elapsed, durable_checkpoint=str(path.resolve()))
        if STOP_AFTER_EPOCH is not None and state['epoch'] == STOP_AFTER_EPOCH:
            raise PreflightInterrupt('SIMULATED_STOP_AFTER_DURABLE_EPOCH')

    def checked_train(*args, **kwargs):
        directory = Path(args[6] if len(args) > 6 else kwargs['run_dir'])
        for path in sorted(directory.glob('ckpt_epoch*.pt')):
            receipt_path = path.with_suffix('.json')
            if not receipt_path.exists():
                quarantine(path)
            elif sha256(path) != read_json(receipt_path)['sha256']:
                raise ValueError('CHECKPOINT_SHA_MISMATCH')
        CONTEXT['epoch_started'] = time.perf_counter()
        trainer, info = base_train_one(*args, **kwargs)
        save_once(directory / 'TRAIN_COMPLETE.json', dict(at=now(), contract=args[7] if len(args) > 7 else kwargs['contract'], train=info))
        return trainer, info

    def checked_evaluate(tc, model, data, directory, split):
        status(stage='EVALUATE', run=Path(directory).name, split=split, windows=len(data['ids']))
        guard.check(force=True)
        result = base_eval(tc, model, data, directory, split)
        # Re-read the published prediction bytes and independently audit ADE/FDE.
        with np.load(Path(directory) / f'{split}_predictions.npz', allow_pickle=False) as saved:
            if not np.array_equal(saved['ids'], data['ids']):
                raise ValueError('EVALUATION_ID_MISMATCH')
            distance = np.sqrt(np.sum((saved['paths'].astype(np.float64) - data['future'][:, None].astype(np.float64)) ** 2, axis=-1))
            ade, fde = distance.mean(-1).min(1), distance[:, :, -1].min(1)
        with np.load(Path(directory) / f'{split}_window_metrics.npz', allow_pickle=False) as saved:
            if np.max(np.abs(ade - saved['minADE'])) > 2e-12 or np.max(np.abs(fde - saved['minFDE'])) > 2e-12:
                raise ValueError('SAVED_FLOAT32_PREDICTION_METRICS_MISMATCH')
        save_once(Path(directory) / 'EVALUATION_AUDIT.json', dict(id_order_exact=True, saved_prediction_ADE_FDE_rechecked=True, tolerance=2e-12))
        return result

    def checked_contract(*args, **kwargs):
        contract = base_contract(*args, **kwargs)
        manifest = ROOT / 'SOURCE_MANIFEST.json'
        contract.update(study_source=sha256(manifest) if manifest.exists() else 'ISOLATED_PREFLIGHT',
                        audit_sha256=sha256(audit_path) if audit_path else 'REAL_RESUME_FIXTURE')
        return contract

    def checked_decide(args):
        directory = Path(args.out)
        for fold in FOLDS:
            for epoch in BUDGETS:
                validate_done(directory / 'runs' / rf.run_name('budget', fold, epoch, 42))
        decision = base_decide(args)
        rows = {fold: {epoch: read_json(directory / 'runs' / rf.run_name('budget', fold, epoch, 42) / 'metrics.json')['val']['overall']
                       for epoch in BUDGETS} for fold in FOLDS}
        expected = rf.decide_budget(rows)
        if any(decision[key] != value for key, value in expected.items()) or decision['test_accessed'] is not False:
            raise ValueError('BUDGET_DECISION_DOES_NOT_MATCH_FROZEN_RULE')
        save_once(directory / 'DECISION_SEAL.json', dict(
            decision_sha256=sha256(directory / 'BUDGET_DECISION.json'),
            budget_done_sha256={rf.run_name('budget', f, e, 42): sha256(directory / 'runs' / rf.run_name('budget', f, e, 42) / 'DONE.json')
                                for f in FOLDS for e in BUDGETS}, no_model_test_metrics_used=True,
            F0_fixed_integrity_audit_opened_test_files=True))
        return decision

    def checked_final(args, *remaining):
        checked_decide(args)
        return base_final(args, *remaining)

    tc.Trainer, tc.save_checkpoint = AuditedTrainer, checked_save
    rf.train_one, rf.evaluate, rf.base_contract = checked_train, checked_evaluate, checked_contract
    rf.phase_decide, rf.phase_final = checked_decide, checked_final
    rf.write_json_once, rf.np = save_once, NumpyProxy()
    if audit_path and data_dir:
        audit = rf.load_audit(audit_path)
        original_load = rf.load_split

        def checked_load(directory, fold, split):
            if split == 'test' and not (Path(output) / 'DECISION_SEAL.json').exists():
                raise ValueError('TEST_READ_BEFORE_BUDGET_SEAL')
            from fivefold_common import fold_files
            for key, path in fold_files(directory, fold).items():
                if key.endswith('_' + split) and sha256(path) != audit['by_fold'][fold]['sha256'][key]:
                    raise ValueError('DATA_CHANGED_SINCE_F0')
            return original_load(directory, fold, split)
        rf.load_split = checked_load


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'summarize':
        import argparse
        import summarize_fivefold as sf
        p = argparse.ArgumentParser(); p.add_argument('mode'); p.add_argument('--out', required=True); p.add_argument('--exp156', required=True)
        args = p.parse_args()
        output = Path(args.out)
        verify_files(read_json(ROOT / 'SOURCE_MANIFEST.json')['files'])
        for directory in (output / 'runs').iterdir():
            validate_done(directory)
        decision, rows = sf.collect(output)
        summary = sf.summarize(decision, rows, read_json(args.exp156))
        save_once(output / 'SUMMARY.json', summary)
        text = sf.markdown(summary)
        text += '\nEXP-156/133 use single seeds and different checkpoint-selection protocols. These are scale references only; differences cannot isolate component contributions or training-protocol effects. Literature protocols remain unverified.\n'
        target = output / 'SUMMARY.md'
        if target.exists() and target.read_text(encoding='utf-8') != text:
            raise ValueError('EXISTING_SUMMARY_DIFFERS')
        target.write_text(text, encoding='utf-8')
        print(text)
        return
    def option(name):
        return sys.argv[sys.argv.index(name) + 1]
    output = Path(option('--out')).resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError('OUTPUT_MUST_BE_INSIDE_NEW_STUDY')
    smoke = '--smoke' in sys.argv
    manifest = ROOT / 'SOURCE_MANIFEST.json'
    if not smoke:
        verify_files(read_json(manifest)['files'])
    tc, _ = rf.import_exec(option('--exec-dir'))
    install(tc, output, option('--audit'), option('--data-dir'))
    for directory in (output / 'runs').glob('*'):
        if (directory / 'DONE.json').exists():
            validate_done(directory)
    rf.main()


if __name__ == '__main__':
    main()
