"""Service-owned finite ETH-T1 seed42 experiment, up to two child processes."""
from pathlib import Path
import os
import sys
import time
import ctypes
import traceback
import shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'adapter'))
from eth_exec_common import *
sys.path.insert(0,str(WORKSPACE/'qoder_work/FIVEFOLD_B0_PARALLEL_20261115/code'))
from job_control import ChildJob

ACTIVE=[]
STARTED=0
JOB=None
ATTEMPT=None


class TimeBudgetReached(RuntimeError):pass


def publish(stage,queue):
    rows=[]
    for item in ACTIVE:
        p=ROOT/'PROGRESS.json' if item['label']=='train' else ROOT/'standalone/slots'/f"{item['label']}.json"
        rows.append(dict(label=item['label'],pid=item['process'].pid,started_at=item['started_at'],progress=read(p) if p.exists() else None))
    status(ROOT/'standalone/CURRENT.json',dict(status='RUNNING',stage=stage,controller='ETH_T1',concurrency_limit=2,
        active=rows,queued=list(queue),new_training_runs_planned=1,epochs=64,seed=42,automatic_seed43=False,COMBO_started=False,attempt=str(ATTEMPT)))


def pool(labels,stage,limit=2):
    queue=list(labels)
    sys.path.insert(0,str(WORKSPACE/'qoder_work/NEXT_PHASE_EXEC_20261114'));import resource_guard as rg
    estimate=read(WORKSPACE/'qoder_work/NEXT_PHASE_EXEC_20261114/PREFLIGHT_E1.json')['estimated_peak_bytes']
    while queue or ACTIVE:
        if time.monotonic()-STARTED>9000:raise TimeBudgetReached('2.5_HOUR_SESSION_LIMIT; COMPLETE_EPOCHS_PRESERVED')
        for item in list(ACTIVE):
            code=item['process'].poll()
            if code is None:continue
            item['stdout'].close();item['stderr'].close();ACTIVE.remove(item)
            replace_json(item['folder']/'EXIT.json',dict(at=now(),exit_code=code,command=item['command']))
            if code!=0:raise RuntimeError('ETH_T1_CHILD_FAILED '+item['label'])
        while queue and len(ACTIVE)<limit:
            admission=rg.decide(rg.snapshot(),estimate,'startup')
            if admission['status']!='PASS':
                if not ACTIVE:raise MemoryError('ETH_T1_RESOURCE_SHORTAGE')
                break
            label=queue.pop(0);folder=ATTEMPT/label;folder.mkdir(parents=True,exist_ok=False)
            stdout=(folder/'stdout.log').open('xb');stderr=(folder/'stderr.log').open('xb')
            command=[CONFIG['python'],'-X','utf8','-B',str(ROOT/'adapter/child.py'),label]
            child=JOB.popen(command,cwd=str(WORKSPACE),stdout=stdout,stderr=stderr)
            ACTIVE.append(dict(label=label,process=child,stdout=stdout,stderr=stderr,folder=folder,command=command,started_at=now()))
        publish(stage,queue)
        if queue or ACTIVE:time.sleep(2)


def main():
    global JOB,STARTED,ATTEMPT
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_wchar_p];api.CreateMutexW.restype=ctypes.c_void_p
    mutex=api.CreateMutexW(None,False,'Local\\TrajectoryFivefoldB0Standalone20261115')
    if not mutex or ctypes.get_last_error()==183:raise RuntimeError('ANOTHER_STUDY_CONTROLLER_EXISTS')
    verify_source()
    if read(ROOT/'PREFLIGHT_PASS.json')['status']!='PASS':raise ValueError('REAL_PREFLIGHT_REQUIRED')
    if (ROOT/'standalone/RESULT.json').exists() and read(ROOT/'standalone/RESULT.json').get('status')=='SUCCESS':
        verify(read(ROOT/'FINAL_OUTPUT_SEAL.json')['files']);return
    ATTEMPT=ROOT/'standalone/attempts'/str(time.time_ns());ATTEMPT.mkdir(parents=True,exist_ok=False)
    for name in ('FAILURE.json','WORKER_EXIT.json','RESULT.json','ACTIVE_WORKER.json','TIME_BUDGET_PAUSE.json'):
        p=ROOT/'standalone'/name
        if p.exists():
            shutil.copyfile(p,ATTEMPT/('prior_'+name))
            if name!='ACTIVE_WORKER.json':p.unlink()
    replace_json(ROOT/'standalone/ACTIVE_WORKER.json',dict(at=now(),pid=os.getpid(),attempt=str(ATTEMPT),dependency_on_Codex_API_or_app=False))
    wake=api.SetThreadExecutionState;wake.argtypes=[ctypes.c_uint];wake.restype=ctypes.c_uint
    if not wake(0x80000001):raise OSError('CANNOT_HOLD_SYSTEM_AWAKE')
    JOB=ChildJob();STARTED=time.monotonic()
    try:
        pool(['train','reference'],'TRAIN_T1_AND_EVALUATE_REUSED_B0',limit=2)
        pool(['evaluate'],'EVALUATE_T1',limit=1)
        pool(['analyze'],'ANALYZE_PAIRED_ETH',limit=1)
        verify_source();validate_seal(run_dir('B0'),'EVALUATION_SEAL.json');validate_seal(run_dir('T1'),'EVALUATION_SEAL.json')
        files=[p for p in (ROOT/'runs').rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.pt')]
        files += [ROOT/n for n in ('ETH_T1_RESULT.json','ETH_T1_RESULT.md','INTERPRETATION.json')]
        replace_json(ROOT/'FINAL_OUTPUT_SEAL.json',dict(at=now(),source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),files={str(p):sha(p) for p in files}))
        verdict=read(ROOT/'ETH_T1_RESULT.json')['verdict']
        result=dict(at=now(),status='SUCCESS',stage='ETH_T1_SEED42_COMPLETE',verdict=verdict,new_training_runs=1,epochs=64,seed=42,
            automatic_seed43=False,COMBO_started=False,baseline_upgraded=False,session_wall_seconds=time.monotonic()-STARTED)
        replace_json(ROOT/'standalone/RESULT.json',result)
        status(ROOT/'standalone/CURRENT.json',dict(status='SUCCESS',stage=result['stage'],active=[],verdict=verdict,automatic_seed43=False,COMBO_started=False))
        replace_json(ROOT/'standalone/WORKER_EXIT.json',dict(at=now(),status='SUCCESS',exit_code=0))
    except BaseException as exc:
        JOB.terminate()
        for item in ACTIVE:
            item['process'].wait(timeout=30);item['stdout'].close();item['stderr'].close()
        timed=isinstance(exc,TimeBudgetReached)
        record=dict(at=now(),status='PAUSED_TIME_BUDGET' if timed else 'FAILURE',traceback=traceback.format_exc(),automatic_resume=False,complete_epoch_checkpoints_preserved=True)
        replace_json(ROOT/'standalone'/('TIME_BUDGET_PAUSE.json' if timed else 'FAILURE.json'),record)
        status(ROOT/'standalone/CURRENT.json',{k:v for k,v in record.items() if k!='at'})
        replace_json(ROOT/'standalone/WORKER_EXIT.json',dict(at=now(),status=record['status'],exit_code=0 if timed else 1))
        if not timed:raise
    finally:
        JOB.close();wake(0x80000000)


if __name__=='__main__':main()
