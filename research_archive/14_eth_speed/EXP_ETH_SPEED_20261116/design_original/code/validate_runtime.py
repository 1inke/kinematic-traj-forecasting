"""CPU/synthetic full-engine preflight; never reads benchmark trajectories.

Exercises actual B0 + wide augmentation, exact mid-epoch resume and label-free
inference. Run with --device cuda:0 later to validate the execution device.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from speed_io import config, imports, now, sha, write
from speed_aug_adapter import trainer_type


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(); c = config(); fd, fm, tc, bf = imports(c, training=True)
    import torch
    start = time.monotonic(); rng = np.random.default_rng(4); n = 65
    past = np.zeros((n, 9, 8, 2), np.float32)
    past[:, :3] = rng.normal(size=(n, 3, 1, 2)) + np.cumsum(rng.normal(.03, .08, (n, 3, 8, 2)), axis=2)
    future = past[:, 0, -1, None]+np.arange(1, 13)[None, :, None]*(past[:, 0, -1, None]-past[:, 0, -2, None])
    data = dict(past=past, future=future.astype(np.float32), ids=np.arange(n), num_valid=np.full(n, 3))
    Trainer = trainer_type(tc, bf); index = np.arange(n); contract = dict(kind='SYNTHETIC_ONLY', epochs=2)
    full = Trainer(data, index, batch=64, epochs=2, device=args.device, seed=42)
    initial = tc.cpu_tree(full.model.state_dict())
    for _ in range(3): full.step()
    state = full.snapshot(contract)
    full.step(); expected = full.snapshot(contract)
    resumed = Trainer(data, index, batch=64, epochs=2, device=args.device, seed=42)
    tc.exact(initial, tc.cpu_tree(resumed.model.state_dict()), 'PAIRED_INIT')
    resumed.load(state, contract); resumed.step()
    tc.exact(expected, resumed.snapshot(contract), 'EXACT_SYNTHETIC_RESUME')
    # The underlying standard B0 initialization must also remain the same.
    plain = tc.Trainer(data, index, batch=64, epochs=2, device=args.device, seed=42)
    tc.exact(initial, tc.cpu_tree(plain.model.state_dict()), 'UNCHANGED_B0_INIT')
    before = tc.cpu_tree(resumed.ema.module.state_dict())
    probe = dict(past=past[:4], num_valid=data['num_valid'][:4], ids=np.arange(4), future=future[:4])
    paths, logits = tc.infer(resumed.ema.module, probe)
    probe['future'] = np.full_like(probe['future'], 999999)
    changed_paths, changed_logits = tc.infer(resumed.ema.module, probe)
    np.testing.assert_array_equal(paths, changed_paths); np.testing.assert_array_equal(logits, changed_logits)
    tc.exact(before, tc.cpu_tree(resumed.ema.module.state_dict()), 'NO_EVAL_STATE_MUTATION')
    assert paths.shape == (4, 20, 12, 2) and logits.shape == (4, 20)
    write(args.out, dict(status='PASS', at=now(), device=args.device, elapsed_seconds=time.monotonic()-start,
          benchmark_data_read=False, formal_experiment_started=False, synthetic_n=n,
          synthetic_updates=4, exact_mid_epoch_resume=True, baseline_initialization_identical=True,
          inference_future_label_invariant=True, parameters=sum(p.numel() for p in resumed.model.parameters()),
          code_sha256={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}))
    print('PASS: synthetic full B0 engine, exact resume, paired init and future-label invariance.', flush=True)


if __name__ == '__main__':
    main()
