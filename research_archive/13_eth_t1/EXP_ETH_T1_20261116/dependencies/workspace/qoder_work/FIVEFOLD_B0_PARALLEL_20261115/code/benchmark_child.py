"""Isolated train-only benchmark; no test set or formal run is touched."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent / 'FIVEFOLD_B0_EXEC_20261115'
sys.path.insert(0, str(OLD / 'code'))
import run_fivefold as rf
from fivefold_common import read_json
from fivefold_data import load_split, scene_tail_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label', required=True)
    p.add_argument('--gate', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    import torch
    torch.set_num_threads(1)
    config = read_json(OLD / 'STUDY_CONFIG.json')
    tc, sources = rf.import_exec(config['exec_dir'])
    import resource_guard as rg
    guard = rg.Guard(read_json(Path(config['exec_dir']) / 'PREFLIGHT_E1.json')['estimated_peak_bytes'])
    guard.check(force=True)
    data = load_split(config['data_dir'], 'hotel', 'train')
    tr, _ = scene_tail_split(data['scene'])
    tr = tr[:4096]  # 64 updates per epoch; 2 epochs = 128 measured updates.
    trainer = tc.Trainer(data, tr, batch=64, epochs=2, device='cuda:0', seed=42, arm='B0')
    # Warm kernels and allocator; restore all state/RNG before timing actual work.
    initial = trainer.snapshot({'benchmark': True})
    trainer.step(); trainer.batch_orders.pop()
    trainer.load(initial, {'benchmark': True})
    torch.cuda.synchronize()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'READY').write_text(str(time.time()), encoding='ascii')
    deadline = time.monotonic() + 120
    while not Path(args.gate).exists():
        if time.monotonic() > deadline:
            raise TimeoutError('BENCHMARK_GATE_TIMEOUT')
        time.sleep(.05)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(128):
        guard.check()
        trainer.step(); trainer.batch_orders.pop()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    state = trainer.snapshot({'benchmark': True})
    tc.save_checkpoint(output / 'final.pt', state)
    result = dict(label=args.label, updates=trainer.updates, seconds=elapsed,
                  seconds_per_update=elapsed/128, parameters=sum(p.numel() for p in trainer.model.parameters()),
                  cuda_peak_allocated=torch.cuda.max_memory_allocated(), cuda_peak_reserved=torch.cuda.max_memory_reserved(),
                  resource=guard.check(force=True), sources=sources)
    (output / 'RESULT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
