"""Finite short-budget exploration, service owned, at most two independent children."""
import ctypes
import os
from pathlib import Path
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from explore_common import *
sys.path.insert(0,str(PARALLEL/'code'))
from job_control import ChildJob
from summarize_explore import calibrate,summarize

ACTIVE=[]
ATTEMPT=None
JOB=None
MUTEX=None
STARTED=None


class TimeBudgetReached(RuntimeError):pass


def publish(stage,queued=None):
    rows=[]
    for item in ACTIVE:
        label=item['label'];p=ROOT/'standalone/slots'/f'{label}.json'
        rows.append(dict(label=label,pid=item['process'].pid,started_at=item['started_at'],
                         progress=read(p) if p.exists() else None))
    status(ROOT/'standalone/CURRENT.json',dict(status='RUNNING',stage=stage,controller='EXPLORATION',
           concurrency_limit=2,active=rows,queued=queued or [],
           completed_runs=[p.parent.name for p in sorted((ROOT/'runs').glob('*/result.json'))],
           automatic_full_confirmation=False,attempt=str(ATTEMPT)))


def pool(specs,stage,limit=2):
    queue=list(specs)
    sys.path.insert(0,str(EXEC));import resource_guard as rg
    estimate=read(EXEC/'PREFLIGHT_E1.json')['estimated_peak_bytes']
    while queue or ACTIVE:
        if time.monotonic()-STARTED>9000:
            raise TimeBudgetReached('2.5_HOUR_EXPLORATION_SESSION_LIMIT; COMPLETE_EPOCHS_PRESERVED')
        for item in list(ACTIVE):
            code=item['process'].poll()
            if code is None:continue
            item['stdout'].close();item['stderr'].close();ACTIVE.remove(item)
            replace_json(item['folder']/'EXIT.json',dict(at=now(),exit_code=code,command=item['command']))
            if code!=0:raise RuntimeError('EXPLORATION_CHILD_FAILED '+item['label'])
            if item['label']!='reference':validate_result(ROOT/'runs'/item['label'])
        while queue and len(ACTIVE)<limit:
            spec=queue[0];label=spec['label']
            if label=='reference' and (ROOT/'BUDGET_REFERENCE.json').exists():
                verify({str(Path(row['checkpoint'])):row['checkpoint_sha256'] for row in read(ROOT/'BUDGET_REFERENCE.json')['rows']})
                queue.pop(0);continue
            if label!='reference' and (ROOT/'runs'/label/'result.json').exists():
                validate_result(ROOT/'runs'/label);queue.pop(0);continue
            admission=rg.decide(rg.snapshot(),estimate,'startup')
            if admission['status']!='PASS':
                if not ACTIVE:raise MemoryError('EXPLORATION_STARTUP_RESOURCE_SHORTAGE')
                break
            queue.pop(0);folder=ATTEMPT/label;folder.mkdir(parents=True,exist_ok=False)
            command=[read(ROOT/'CONFIG.json')['python'],'-X','utf8','-B',str(ROOT/'code'/spec['script']),*spec.get('args',[])]
            stdout=(folder/'stdout.log').open('xb');stderr=(folder/'stderr.log').open('xb')
            process=JOB.popen(command,cwd=str(WORKSPACE),stdout=stdout,stderr=stderr)
            ACTIVE.append(dict(label=label,process=process,stdout=stdout,stderr=stderr,folder=folder,
                               command=command,started_at=now()))
        publish(stage,[s['label'] for s in queue])
        if queue or ACTIVE:time.sleep(2)


def specs(epochs,arms):
    return [dict(label=f'{arm}_E{epochs}_s{seed}',script='run_explore.py',
                 args=['--arm',arm,'--epochs',str(epochs),'--seed',str(seed)]) for arm,seed in arms]


def main():
    global MUTEX,ATTEMPT,JOB,STARTED
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_wchar_p];api.CreateMutexW.restype=ctypes.c_void_p
    MUTEX=api.CreateMutexW(None,False,'Local\\TrajectoryFivefoldB0Standalone20261115')
    if not MUTEX or ctypes.get_last_error()==183:raise RuntimeError('ANOTHER_STUDY_CONTROLLER_EXISTS')
    verify_source()
    if read(ROOT/'PREFLIGHT_PASS.json')['status']!='PASS':raise ValueError('REAL_PREFLIGHT_REQUIRED')
    if (ROOT/'standalone/RESULT.json').exists() and read(ROOT/'standalone/RESULT.json').get('status')=='SUCCESS':return
    ATTEMPT=ROOT/'standalone/attempts'/str(time.time_ns());ATTEMPT.mkdir(parents=True,exist_ok=False)
    for name in ('FAILURE.json','WORKER_EXIT.json','RESULT.json','ACTIVE_WORKER.json'):
        p=ROOT/'standalone'/name
        if p.exists():
            import shutil
            shutil.copy2(p,ATTEMPT/('prior_'+name))
            if name!='ACTIVE_WORKER.json':p.unlink()
    replace_json(ROOT/'standalone/ACTIVE_WORKER.json',dict(at=now(),pid=os.getpid(),attempt=str(ATTEMPT),
                 controller='EXPLORATION',concurrency_limit=2,dependency_on_Codex_API_or_app=False))
    wake=api.SetThreadExecutionState;wake.argtypes=[ctypes.c_uint];wake.restype=ctypes.c_uint
    if not wake(0x80000001):raise OSError('CANNOT_HOLD_SYSTEM_AWAKE')
    JOB=ChildJob();STARTED=time.monotonic()
    try:
        pool([dict(label='reference',script='reference_eval.py')],'REUSE_EXISTING_BUDGET_MODELS',limit=1)
        chosen=None;decisions=[]
        for epochs in (16,32):
            pool(specs(epochs,[('B0',42),('B0',43)]),f'CALIBRATE_E{epochs}_B0')
            pool(specs(epochs,[('T1',42),('E3_self',42)]),f'CALIBRATE_E{epochs}_SIGNAL_CONTROLS')
            decision=calibrate(epochs);decisions.append(decision);summarize()
            if decision['usable_as_initial_screen']:
                chosen=epochs;break
        result=dict(at=now(),status='SUCCESS',stage='EXPLORATION_CALIBRATION_COMPLETE',
                    chosen_initial_exploration_budget=chosen,decisions=decisions,
                    if_no_budget_passes='Short-budget screening not calibrated; no automatic 64/128 training.',
                    automatic_full_confirmation=False,session_wall_seconds=time.monotonic()-STARTED)
        verify_source();summarize();replace_json(ROOT/'EXPLORATION_BUDGET.json',result)
        replace_json(ROOT/'standalone/RESULT.json',result)
        status(ROOT/'standalone/CURRENT.json',dict(status='SUCCESS',stage=result['stage'],active=[],
               chosen_initial_exploration_budget=chosen,automatic_full_confirmation=False))
        replace_json(ROOT/'standalone/WORKER_EXIT.json',dict(at=now(),status='SUCCESS',exit_code=0))
    except BaseException as exc:
        JOB.terminate()
        for item in ACTIVE:
            item['process'].wait(timeout=30);item['stdout'].close();item['stderr'].close()
        paused=isinstance(exc,TimeBudgetReached)
        record=dict(at=now(),status='PAUSED_TIME_BUDGET' if paused else 'FAILURE',traceback=traceback.format_exc(),
                    automatic_resume=False,complete_epoch_checkpoints_preserved=True)
        replace_json(ROOT/'standalone'/('TIME_BUDGET_PAUSE.json' if paused else 'FAILURE.json'),record)
        status(ROOT/'standalone/CURRENT.json',{k:v for k,v in record.items() if k!='at'})
        replace_json(ROOT/'standalone/WORKER_EXIT.json',dict(at=now(),status=record['status'],exit_code=0 if paused else 1))
        if not paused:raise
    finally:
        JOB.close();wake(0x80000000)


if __name__=='__main__':main()

