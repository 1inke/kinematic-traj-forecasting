"""Compare identical serial vs two-process CUDA work; verify all final state exactly."""
import json
from pathlib import Path
import subprocess
import sys
import time

from job_control import ChildJob

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent / 'FIVEFOLD_B0_EXEC_20261115'
sys.path.insert(0, str(OLD / 'code'))
from fivefold_common import now, read_json
import runtime as rt


def wave(labels, parent, config, job):
    gate = parent / ('GATE_' + '_'.join(labels))
    active = []
    for label in labels:
        output = parent / label
        log = (parent / (label + '.log')).open('xb')
        process = job.popen([config['python'], '-B', str(ROOT / 'code/benchmark_child.py'),
                             '--label', label, '--gate', str(gate), '--output', str(output)],
                            stdout=log, stderr=subprocess.STDOUT, cwd=config['workspace'])
        active.append((process, log, output))
    deadline = time.monotonic() + 120
    while not all((output / 'READY').exists() for _, _, output in active):
        if any(process.poll() is not None for process, _, _ in active) or time.monotonic() > deadline:
            raise RuntimeError('BENCHMARK_INIT_FAILURE')
        time.sleep(.1)
    start = time.perf_counter()
    gate.write_text(now(), encoding='ascii')
    for process, log, output in active:
        code = process.wait(); log.close()
        if code:
            raise RuntimeError('BENCHMARK_CHILD_FAILURE ' + str(output))
    return time.perf_counter() - start


def main():
    config = read_json(OLD / 'STUDY_CONFIG.json')
    parent = ROOT / 'preflight/BENCHMARK'
    parent.mkdir(exist_ok=False)
    job = ChildJob()
    try:
        one = wave(['serial1'], parent, config, job)
        two = wave(['serial2'], parent, config, job)
        parallel = wave(['parallel1', 'parallel2'], parent, config, job)
    finally:
        job.close()
    # GPU workers have exited: CPU-only comparison of all serialized tensor/RNG/order states.
    import torch
    sys.path.insert(0, config['exec_dir'])
    import train_core as tc
    reference = torch.load(parent / 'serial1/final.pt', map_location='cpu', weights_only=False)
    for label in ('serial2', 'parallel1', 'parallel2'):
        state = torch.load(parent / label / 'final.pt', map_location='cpu', weights_only=False)
        tc.exact(reference, state, 'SERIAL_VS_PARALLEL_' + label)
    results = {label: read_json(parent / label / 'RESULT.json') for label in ('serial1', 'serial2', 'parallel1', 'parallel2')}
    serial_compute = sum(results[k]['seconds'] for k in ('serial1','serial2'))
    parallel_compute = max(results[k]['seconds'] for k in ('parallel1','parallel2'))
    report = dict(at=now(), status='PASS', exact_all_state_serial_parallel=True, updates_each=128,
                  serial_wave_seconds=one+two, parallel_wave_seconds=parallel,
                  end_to_end_speedup=(one+two)/parallel, compute_speedup=serial_compute/parallel_compute,
                  max_concurrency=2, results=results,
                  planning_note='Short same-data benchmark; full epochs may differ under thermal/load changes.')
    rt.atomic(ROOT / 'preflight/BENCHMARK_PASS.json', report)
    print(json.dumps({k:v for k,v in report.items() if k!='results'}), flush=True)


if __name__ == '__main__':
    main()
