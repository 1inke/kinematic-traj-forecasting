"""Two-process finite scientific schedule, externally owned by Windows Task Scheduler."""
import ctypes
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT.parent / 'FIVEFOLD_B0_EXEC_20261115'
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(OLD / 'code'))
import runtime as rt
import run_fivefold as rf
from fivefold_common import now, read_json, sha256, FOLDS, FINAL_SEEDS
from job_control import ChildJob

MUTEX = None
ATTEMPT = None
ACTIVE = []
CHILD_JOB = None
CONFIG = None


def claim():
    global MUTEX
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    api.CreateMutexW.restype = ctypes.c_void_p
    # Same mutex as the serial worker: both launchers cannot own this study.
    MUTEX = api.CreateMutexW(None, False, 'Local\\TrajectoryFivefoldB0Standalone20261115')
    if not MUTEX:
        raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error() == 183:
        raise RuntimeError('ANOTHER_FIVEFOLD_CONTROLLER_EXISTS')


def publish(stage, **extra):
    active = []
    for item in ACTIVE:
        path = HERE / 'slots' / item['label'] / 'CURRENT.json'
        row = dict(label=item['label'], pid=item['process'].pid, started_at=item['started_at'])
        if path.exists():
            row['progress'] = read_json(path)
        active.append(row)
    done = sorted(p.parent.name for p in (Path(CONFIG['output']) / 'runs').glob('*/DONE.json'))
    value = dict(at=now(), status='RUNNING', stage=stage, controller='PARALLEL', concurrency_limit=2,
                 active=active, completed_runs=done, completed_count=len(done), **extra)
    for base in (HERE, OLD / 'standalone'):
        rt.atomic(base / 'CURRENT.json', value)
        rt.atomic(base / 'CHILD_STATE.json', dict(at=now(), state='RUNNING', active=active, attempt=str(ATTEMPT)))


def required_runs(spec):
    output = Path(CONFIG['output']) / 'runs'
    if spec['kind'] == 'budget':
        return [output / rf.run_name('budget', spec['fold'], e, 42) for e in (64, 128)]
    if spec['kind'] == 'final':
        E = read_json(Path(CONFIG['output']) / 'BUDGET_DECISION.json')['E_star']
        return [output / rf.run_name('final', spec['fold'], E, spec['seed'])]
    return []


def arguments(spec):
    common = ['--out', CONFIG['output'], '--exec-dir', CONFIG['exec_dir'], '--data-dir', CONFIG['data_dir'],
              '--audit', CONFIG['audit'], '--device', 'cuda:0']
    if spec['kind'] == 'summary':
        return ['summarize', '--out', CONFIG['output'], '--exp156', CONFIG['exp156']]
    if spec['kind'] == 'decide':
        return ['decide', *common]
    args = [spec['kind'], *common, '--folds', spec['fold']]
    if spec['kind'] == 'final':
        args += ['--seeds', str(spec['seed'])]
    return args


def pool(specs, stage, limit=2):
    import resource_guard as rg
    queue = list(specs)
    estimate = read_json(Path(CONFIG['exec_dir']) / 'PREFLIGHT_E1.json')['estimated_peak_bytes']
    while queue or ACTIVE:
        for item in list(ACTIVE):
            code = item['process'].poll()
            if code is None:
                continue
            item['stdout'].close(); item['stderr'].close()
            ACTIVE.remove(item)
            if code == 0:
                for directory in required_runs(item['spec']):
                    rt.validate_done(directory)
            receipt = dict(at=now(), label=item['label'], pid=item['process'].pid, exit_code=code,
                           command=item['command'], parallel_source_sha256=sha256(ROOT / 'SOURCE_MANIFEST.json'),
                           logs={str(p): sha256(p) for p in item['folder'].glob('*.log')},
                           done_sha256={str(p / 'DONE.json'): sha256(p / 'DONE.json') for p in required_runs(item['spec'])} if code == 0 else {})
            rt.atomic(item['folder'] / 'RECEIPT.json', receipt)
            if code != 0:
                raise RuntimeError('CHILD_FAILURE ' + item['label'] + ' exit=' + str(code))
        while queue and len(ACTIVE) < limit:
            spec = queue[0]
            runs = required_runs(spec)
            if runs and all((p / 'DONE.json').exists() for p in runs):
                for directory in runs:
                    rt.validate_done(directory)
                rt.save_once(ATTEMPT / 'skipped' / (spec['label'] + '.json'),
                             dict(at=now(), reason='ALL_DONE_SHA_VERIFIED', runs=[str(p) for p in runs]))
                queue.pop(0)
                continue
            admission = rg.decide(rg.snapshot(), estimate, 'startup')
            if admission['status'] != 'PASS':
                if not ACTIVE:
                    raise MemoryError('CURRENT_RESOURCE_SHORTAGE_NO_ACTIVE_JOB ' + str(admission))
                break  # Keep one current job; retry admission on actual resource changes.
            queue.pop(0)
            folder = ATTEMPT / spec['label']
            folder.mkdir(parents=True, exist_ok=False)
            rt.atomic(folder / 'ADMISSION.json', dict(at=now(), **admission))
            command = [CONFIG['python'], '-B', str(ROOT / 'code/parallel_child.py'), spec['label'], *arguments(spec)]
            stdout, stderr = (folder / 'stdout.log').open('xb'), (folder / 'stderr.log').open('xb')
            try:
                process = CHILD_JOB.popen(command, cwd=CONFIG['workspace'], stdout=stdout, stderr=stderr)
            except BaseException:
                stdout.close(); stderr.close()
                raise
            item = dict(label=spec['label'], process=process, spec=spec, folder=folder, command=command,
                        stdout=stdout, stderr=stderr, started_at=now())
            ACTIVE.append(item)
        publish(stage, queued=[s['label'] for s in queue])
        if queue or ACTIVE:
            time.sleep(2)


def finish():
    output = Path(CONFIG['output'])
    runs = list((output / 'runs').iterdir())
    if len(runs) != 25:
        raise ValueError('EXPECTED_25_COMPLETED_RUNS')
    for directory in runs:
        rt.validate_done(directory)
    rt.verify_files(read_json(ROOT / 'SOURCE_MANIFEST.json')['files'])
    paths = [p for p in output.rglob('*') if p.is_file() and not p.name.endswith(('.tmp', '.pending'))]
    rt.save_once(OLD / 'FINAL_OUTPUT_SEAL.json', dict(at=now(), files={str(p): sha256(p) for p in paths},
                 parallel_execution_source_sha256=sha256(ROOT / 'SOURCE_MANIFEST.json')))
    result = dict(at=now(), status='SUCCESS', completed_runs=25, controller='PARALLEL',
                  source_sha256=sha256(ROOT / 'SOURCE_MANIFEST.json'), output_seal_sha256=sha256(OLD / 'FINAL_OUTPUT_SEAL.json'))
    for base in (HERE, OLD / 'standalone'):
        rt.atomic(base / 'RESULT.json', result)
        rt.atomic(base / 'CURRENT.json', result)
        rt.atomic(base / 'WORKER_EXIT.json', dict(at=now(), status='SUCCESS', exit_code=0, attempt=str(ATTEMPT)))


def main():
    global ATTEMPT, CHILD_JOB, CONFIG
    claim()
    CONFIG = read_json(ROOT / 'PARALLEL_CONFIG.json')
    sys.path.insert(0, CONFIG['exec_dir'])
    rt.verify_files(read_json(ROOT / 'SOURCE_MANIFEST.json')['files'])
    if read_json(ROOT / 'preflight/PREFLIGHT_PASS.json')['status'] != 'PASS':
        raise RuntimeError('PARALLEL_REAL_PREFLIGHT_REQUIRED')
    if (HERE / 'RESULT.json').exists() and read_json(HERE / 'RESULT.json').get('status') == 'SUCCESS':
        rt.verify_files(read_json(OLD / 'FINAL_OUTPUT_SEAL.json')['files'])
        return
    ATTEMPT = HERE / 'attempts' / str(time.time_ns())
    ATTEMPT.mkdir(parents=True, exist_ok=False)
    for base in (HERE, OLD / 'standalone'):
        for name in ('WORKER_EXIT.json', 'RESULT.json', 'FAILURE.json', 'ACTIVE_WORKER.json'):
            path = base / name
            if path.exists():
                archive = ATTEMPT / ('prior_' + ('old_' if base != HERE else '') + name)
                shutil.copy2(path, archive)
                if name != 'ACTIVE_WORKER.json':
                    path.unlink()
    wake = ctypes.windll.kernel32.SetThreadExecutionState
    wake.argtypes = [ctypes.c_uint]; wake.restype = ctypes.c_uint
    if not wake(0x80000001):
        raise OSError('CANNOT_INHIBIT_AUTOMATIC_SLEEP')
    CHILD_JOB = ChildJob()
    try:
        for base in (HERE, OLD / 'standalone'):
            rt.atomic(base / 'ACTIVE_WORKER.json', dict(at=now(), pid=os.getpid(), controller='PARALLEL',
                      attempt=str(ATTEMPT), dependency_on_Codex_API_or_app=False, children_kill_on_owner_exit=True))
        budget = [dict(label='budget_' + f, kind='budget', fold=f) for f in FOLDS]
        pool(budget, 'PARALLEL_BUDGET')
        pool([dict(label='decide', kind='decide')], 'BUDGET_DECISION', limit=1)
        final = [dict(label=f'final_{f}_s{s}', kind='final', fold=f, seed=s) for f in FOLDS for s in FINAL_SEEDS]
        pool(final, 'PARALLEL_FINAL')
        pool([dict(label='summary', kind='summary')], 'SUMMARY', limit=1)
        finish()
    except BaseException:
        CHILD_JOB.terminate()
        stopped = []
        for item in ACTIVE:
            code = item['process'].wait(timeout=30)
            item['stdout'].close(); item['stderr'].close()
            stopped.append(dict(label=item['label'], pid=item['process'].pid, exit_code=code,
                                reason='STOPPED_AFTER_CONTROLLER_FAILURE'))
        failure = dict(at=now(), status='FAILURE', controller='PARALLEL', traceback=traceback.format_exc(),
                       stopped_children=stopped, attempt=str(ATTEMPT))
        for base in (HERE, OLD / 'standalone'):
            rt.atomic(base / 'FAILURE.json', failure)
            rt.atomic(base / 'CURRENT.json', failure)
            rt.atomic(base / 'WORKER_EXIT.json', dict(at=now(), status='FAILURE', exit_code=1, attempt=str(ATTEMPT)))
        raise
    finally:
        CHILD_JOB.close()
        wake(0x80000000)


if __name__ == '__main__':
    main()
