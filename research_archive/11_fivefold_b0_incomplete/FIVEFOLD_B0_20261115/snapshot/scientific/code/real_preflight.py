"""Real CUDA recovery and prediction-output checks, isolated from formal runs."""
import argparse
import gc
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import runtime as rt
import run_fivefold as rf
from fivefold_common import now, sha256
from fivefold_data import load_split, scene_tail_split, subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exec-dir', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--audit', required=True)
    args = parser.parse_args()
    import torch
    started = time.perf_counter()
    root = rt.ROOT / 'preflight/SMOKE_RESUME'
    tc, sources = rf.import_exec(args.exec_dir)
    rt.install(tc, root)
    audit = rf.load_audit(args.audit)
    data = load_split(args.data_dir, 'hotel', 'train')
    tr, va = scene_tail_split(data['scene'])
    tr, va = tr[:513], va[:256]
    contract = rf.base_contract(SimpleNamespace(device='cuda:0'), sources, audit,
                                'REAL_RESUME', 'hotel', 2, 42, tr)
    reference, interrupted = root / 'reference', root / 'interrupted'
    if reference.exists() or interrupted.exists():
        raise RuntimeError('PREFLIGHT_ALREADY_EXISTS_USE_RECORDED_PROOF')
    trainer, info = rf.train_one(tc, data, tr, 2, 42, 'cuda:0', reference, contract)
    assert info['updates'] == 18 and info['parameters'] == 295474
    reference_state = torch.load(reference / 'ckpt_epoch002.pt', map_location='cpu', weights_only=False)
    del trainer
    gc.collect(); torch.cuda.empty_cache()
    rt.STOP_AFTER_EPOCH = 1
    try:
        rf.train_one(tc, data, tr, 2, 42, 'cuda:0', interrupted, contract)
    except rt.PreflightInterrupt:
        pass
    else:
        raise AssertionError('INTERRUPTION_HOOK_NOT_REACHED')
    finally:
        rt.STOP_AFTER_EPOCH = None
    first = torch.load(interrupted / 'ckpt_epoch001.pt', map_location='cpu', weights_only=False)
    assert first['updates'] == 9
    gc.collect(); torch.cuda.empty_cache()
    trainer, resumed_info = rf.train_one(tc, data, tr, 2, 42, 'cuda:0', interrupted, contract)
    resumed_state = torch.load(interrupted / 'ckpt_epoch002.pt', map_location='cpu', weights_only=False)
    tc.exact(reference_state, resumed_state, 'CONTINUOUS_VS_DURABLE_EPOCH_RESUME_ALL_STATE')
    val = subset(data, va)
    paths, logits = tc.infer(trainer.ema.module, val)
    changed = dict(val, future=val['future'] + np.float32(100))
    paths_changed, logits_changed = tc.infer(trainer.ema.module, changed)
    assert np.array_equal(paths, paths_changed) and np.array_equal(logits, logits_changed)
    assert paths.shape == (256, 20, 12, 2) and logits.shape == (256, 20)
    result = rf.evaluate(tc, trainer.ema.module, val, interrupted, 'val')
    prediction_sha = sha256(interrupted / 'val_predictions.npz')
    assert result == rf.evaluate(tc, trainer.ema.module, val, interrupted, 'val')
    assert prediction_sha == sha256(interrupted / 'val_predictions.npz')
    before = sha256(interrupted / 'ckpt_epoch002.pt')
    del trainer
    gc.collect(); torch.cuda.empty_cache()
    trainer, reused_info = rf.train_one(tc, data, tr, 2, 42, 'cuda:0', interrupted, contract)
    assert before == sha256(interrupted / 'ckpt_epoch002.pt') and reused_info['updates'] == 18
    rt.atomic(rt.ROOT / 'preflight/REAL_RESUME.json', dict(
        at=now(), status='PASS', seconds=time.perf_counter()-started, device=torch.cuda.get_device_name(0),
        train_windows=513, epochs=2, updates=18, interrupted_after_updates=9, parameters=295474,
        all_checkpoint_state_exact=True, model_EMA_optimizer_scheduler_loss_RNG_order_history_exact=True,
        prediction_ignores_future_exact=True, published_float32_metrics_rechecked=True,
        repeat_evaluation_reuses_identical_bytes=True, finished_training_reentry_no_updates=True,
        order_hash=resumed_info['order_hash'], checkpoint_sha256=before,
        resource_last=rt.CONTEXT['guard'].last))
    print('REAL_CUDA_RECOVERY_PREFLIGHT_PASS', flush=True)


if __name__ == '__main__':
    main()
