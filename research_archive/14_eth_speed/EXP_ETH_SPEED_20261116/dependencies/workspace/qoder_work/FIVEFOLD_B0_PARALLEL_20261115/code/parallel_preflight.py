"""Real parallel smoke, unchanged completed outputs, and forced-owner termination checks."""
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil
from job_control import ChildJob

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent / 'FIVEFOLD_B0_EXEC_20261115'
sys.path.insert(0, str(OLD / 'code'))
import runtime as rt
import run_fivefold as rf
from fivefold_common import now, read_json, sha256


def containment(config):
    directory = ROOT / 'preflight/CONTAINMENT'
    directory.mkdir(exist_ok=False)
    with (directory / 'owner.log').open('xb') as log:
        owner = subprocess.Popen([config['python'], '-B', str(ROOT / 'code/containment_owner.py'),
                                  str(directory), config['python']], stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        deadline = time.monotonic() + 30
        try:
            while not (directory / 'actual_child_pid').exists():
                if owner.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('CONTAINMENT_OWNER_FAILED')
                time.sleep(.1)
            actual_pid = int((directory / 'actual_child_pid').read_text())
            identity = psutil.Process(actual_pid).create_time()
            owner.kill(); owner.wait(timeout=10)
            deadline = time.monotonic() + 10
            while psutil.pid_exists(actual_pid) and psutil.Process(actual_pid).create_time() == identity:
                if time.monotonic() > deadline:
                    raise RuntimeError('FORCED_OWNER_EXIT_LEFT_ORPHAN')
                time.sleep(.1)
        finally:
            if owner.poll() is None:
                owner.kill(); owner.wait()
    return dict(forced_owner_exit_killed_actual_venv_child=True, actual_child_pid=actual_pid,
                owner_exit_code=owner.returncode, child_creation_time=identity)


def main():
    config = read_json(OLD / 'STUDY_CONFIG.json')
    result = containment(config)
    # Existing validated ENVIRONMENT is created before dispatch: no concurrent first-writer race.
    output = OLD / 'preflight/PARALLEL_SMOKE'
    if output.exists():
        raise RuntimeError('PARALLEL_SMOKE_ALREADY_EXISTS')
    output.mkdir()
    rt.atomic(output / 'ENVIRONMENT.json', read_json(Path(config['output']) / 'ENVIRONMENT.json'))
    child_job = ChildJob()
    first = None
    try:
        for repeat in (1, 2):
            active = []
            for fold in ('hotel', 'univ'):
                log = (ROOT / f'logs/parallel_smoke_{fold}_{repeat}.log').open('xb')
                command = [config['python'], '-B', str(ROOT / 'code/parallel_child.py'), 'smoke_' + fold,
                           'budget', '--smoke', '--folds', fold, '--out', str(output),
                           '--exec-dir', config['exec_dir'], '--data-dir', config['data_dir'], '--audit', config['audit']]
                active.append((child_job.popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=config['workspace']), log))
            for process, log in active:
                code = process.wait(); log.close()
                if code:
                    raise RuntimeError('PARALLEL_REAL_SMOKE_FAILURE')
            files = {}
            for fold in ('hotel','univ'):
                directory = output / 'runs' / rf.run_name('smoke', fold, 1, 42)
                rt.validate_done(directory)
                m = read_json(directory / 'metrics.json')
                assert m['train']['updates'] == 8 and m['train']['parameters'] == 295474
                files.update({str(p): sha256(p) for p in directory.iterdir() if p.is_file()})
            if first is not None:
                assert files == first, 'PARALLEL_DONE_SKIP_CHANGED_SAVED_BYTES'
            first = files
    finally:
        child_job.close()
    rt.atomic(ROOT / 'preflight/PARALLEL_SMOKE_PASS.json', dict(at=now(), status='PASS',
              simultaneous_fold_workers=2, per_worker_updates=8, saved_predictions_audited=True,
              repeated_parallel_launch_DONE_skip_exact=True, per_worker_status_no_shared_temp=True,
              files=first, containment=result))
    print('REAL_PARALLEL_SMOKE_AND_CONTAINMENT_PASS', flush=True)


if __name__ == '__main__':
    main()
