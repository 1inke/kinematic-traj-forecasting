"""Finite Windows-scheduler experiment worker. No GPT/app/API dependency."""
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'code'))
import runtime as rt
from fivefold_common import now, read_json, sha256

MUTEX = None
ATTEMPT = None


def claim():
    global MUTEX
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    api.CreateMutexW.restype = ctypes.c_void_p
    MUTEX = api.CreateMutexW(None, False, 'Local\\TrajectoryFivefoldB0Standalone20261115')
    if not MUTEX:
        raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error() == 183:
        raise RuntimeError('ANOTHER_FIVEFOLD_WORKER_EXISTS')


def run(label, arguments, config):
    import resource_guard as rg
    estimate = read_json(Path(config['exec_dir']) / 'PREFLIGHT_E1.json')['estimated_peak_bytes']
    admission = rg.decide(rg.snapshot(), estimate, 'startup')
    rt.atomic(ATTEMPT / (label + '_admission.json'), dict(at=now(), **admission))
    if admission['status'] != 'PASS':
        raise MemoryError('CURRENT_STARTUP_RESOURCE_SHORTAGE')
    command = [config['python'], '-B', str(ROOT / 'code/runtime.py'), *arguments]
    with (ATTEMPT / (label + '.stdout.log')).open('xb') as stdout, (ATTEMPT / (label + '.stderr.log')).open('xb') as stderr:
        child = subprocess.Popen(command, cwd=config['workspace'], stdout=stdout, stderr=stderr,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        rt.atomic(HERE / 'CHILD_STATE.json', dict(at=now(), label=label, pid=child.pid,
                  state='RUNNING', command=command, attempt=str(ATTEMPT)))
        exit_code = child.wait()
    receipt = dict(at=now(), pid=child.pid, exit_code=exit_code, command=command,
                   logs={str(p): sha256(p) for p in ATTEMPT.glob(label + '.*.log')})
    rt.atomic(ATTEMPT / (label + '_receipt.json'), receipt)
    rt.atomic(HERE / 'CHILD_STATE.json', dict(receipt, label=label,
              state='SUCCESS' if exit_code == 0 else 'FAILURE', attempt=str(ATTEMPT)))
    if exit_code:
        raise RuntimeError(f'CHILD_FAILURE {label} exit={exit_code}; logs={ATTEMPT}')


def main():
    global ATTEMPT
    claim()
    config = read_json(ROOT / 'STUDY_CONFIG.json')
    rt.verify_files(read_json(ROOT / 'SOURCE_MANIFEST.json')['files'])
    if read_json(ROOT / 'preflight/PREFLIGHT_PASS.json')['status'] != 'PASS':
        raise RuntimeError('REAL_PREFLIGHT_REQUIRED')
    if (HERE / 'RESULT.json').exists() and read_json(HERE / 'RESULT.json').get('status') == 'SUCCESS':
        rt.verify_files(read_json(ROOT / 'FINAL_OUTPUT_SEAL.json')['files'])
        return
    ATTEMPT = HERE / 'attempts' / str(time.time_ns())
    ATTEMPT.mkdir(parents=True, exist_ok=False)
    for name in ('WORKER_EXIT.json', 'RESULT.json', 'FAILURE.json'):
        old = HERE / name
        if old.exists():
            shutil.copy2(old, ATTEMPT / ('prior_' + name))
            old.unlink()
    sys.path.insert(0, config['exec_dir'])
    wake = ctypes.windll.kernel32.SetThreadExecutionState
    wake.argtypes = [ctypes.c_uint]; wake.restype = ctypes.c_uint
    if not wake(0x80000001):
        raise OSError('CANNOT_INHIBIT_AUTOMATIC_SLEEP')
    started = time.perf_counter()
    rt.atomic(HERE / 'ACTIVE_WORKER.json', dict(at=now(), pid=os.getpid(), attempt=str(ATTEMPT),
              dependency_on_Codex_API_or_app=False, wake_scope='Active worker lifetime'))
    rt.status(stage='STARTING', pid=os.getpid(), attempt=str(ATTEMPT))
    try:
        output = config['output']
        run('experiment', ['all', '--out', output, '--exec-dir', config['exec_dir'],
                          '--data-dir', config['data_dir'], '--audit', config['audit'], '--device', 'cuda:0'], config)
        run('summary', ['summarize', '--out', output, '--exp156', config['exp156']], config)
        directories = list((Path(output) / 'runs').iterdir())
        if len(directories) != 25:
            raise ValueError('EXPECTED_10_BUDGET_PLUS_15_FINAL_RUNS')
        for directory in directories:
            rt.validate_done(directory)
        rt.verify_files(read_json(ROOT / 'SOURCE_MANIFEST.json')['files'])
        files = [p for p in Path(output).rglob('*') if p.is_file() and not p.name.endswith(('.tmp', '.pending'))]
        rt.save_once(ROOT / 'FINAL_OUTPUT_SEAL.json', dict(at=now(), files={str(p): sha256(p) for p in files}))
        result = dict(at=now(), status='SUCCESS', completed_runs=25,
                      seconds_this_session=time.perf_counter()-started, source_sha256=sha256(ROOT / 'SOURCE_MANIFEST.json'))
        rt.atomic(HERE / 'RESULT.json', result)
        rt.status(**result)
        rt.atomic(HERE / 'WORKER_EXIT.json', dict(at=now(), status='SUCCESS', exit_code=0, attempt=str(ATTEMPT)))
    except BaseException:
        failure = dict(at=now(), status='FAILURE', traceback=traceback.format_exc(), attempt=str(ATTEMPT))
        rt.atomic(HERE / 'FAILURE.json', failure)
        rt.status(**failure)
        rt.atomic(HERE / 'WORKER_EXIT.json', dict(at=now(), status='FAILURE', exit_code=1, attempt=str(ATTEMPT)))
        raise
    finally:
        wake(0x80000000)


if __name__ == '__main__':
    main()
