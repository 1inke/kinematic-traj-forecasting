"""Human-requested stop after seed45. No science source or optimizer is modified.

Temporarily suspend only the waiting coordinator, let its already-running training
child finish, record its real exit and immutable outputs, then stop the OS task.
This closes the race in which seed46 could otherwise start immediately afterward.
"""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil

HERE=Path(__file__).resolve().parent
REC=HERE.parent;BASE=REC.parent
sys.path.insert(0,str(BASE))
from common import read_json,write_json,verify,sha,seal,now

TASK='Trajectory-NextPhase-E1E4-Standalone-20261114'
TARGET='E1C_E64_s45'
NEXT='E1C_E64_s46'


def atomic(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)


def suspend_verified(parent,expected_command):
    command=' '.join(parent.cmdline())
    if expected_command not in command:raise ValueError('WRONG_COORDINATOR_COMMAND')
    parent.suspend()
    if parent.status()!=psutil.STATUS_STOPPED:raise ValueError('COORDINATOR_NOT_SUSPENDED')


def wait_exit_code(pid):
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.OpenProcess.argtypes=[ctypes.c_uint,ctypes.c_int,ctypes.c_uint];api.OpenProcess.restype=ctypes.c_void_p
    api.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint];api.WaitForSingleObject.restype=ctypes.c_uint
    api.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint)];api.GetExitCodeProcess.restype=ctypes.c_int
    handle=api.OpenProcess(0x00100000|0x1000,False,pid)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    return api,handle


def main():
    if (HERE/'CONTROLLER_START.json').exists():raise ValueError('DO_NOT_DUPLICATE_PAUSE_CONTROLLER')
    verify(read_json(REC/'RECOVERY_MANIFEST.json')['files']);verify(read_json(BASE/'SOURCE_MANIFEST.json')['files'])
    active=read_json(REC/'ACTIVE_WORKER.json');stage=read_json(REC/'CHILD_STATE.json')
    if stage['state']!='RUNNING' or stage['label']!=TARGET+'_train':raise ValueError('SEED45_NOT_THE_CURRENT_CHILD')
    if (BASE/'runs'/NEXT).exists():raise ValueError('SEED46_ALREADY_HAS_ARTIFACTS')
    parent=psutil.Process(active['pid']);child=psutil.Process(stage['pid'])
    expected=str(REC/'worker_standalone.py')
    if TARGET not in ' '.join(child.cmdline()) or 'adapter.py' not in ' '.join(child.cmdline()):raise ValueError('WRONG_TRAINING_CHILD')
    if child.ppid()!=parent.pid:raise ValueError('CHILD_PARENT_ID_MISMATCH')
    api,handle=wait_exit_code(child.pid)
    write_json(HERE/'CONTROLLER_START.json',dict(at=now(),pid=os.getpid(),human_instruction='seed45跑完先暂停，我有事情要关闭电脑',
        parent_pid=parent.pid,parent_creation=parent.create_time(),child_pid=child.pid,child_creation=child.create_time(),
        current=read_json(REC/'CURRENT.json'),task=TASK,source_sha=sha(__file__)))
    suspended=False;stopped=False
    try:
        suspend_verified(parent,expected);suspended=True
        write_json(HERE/'GATE_ARMED.json',dict(at=now(),status='ARMED',parent_suspended_only=True,child_training_continues=True,
            seed46_blocked=True,target=TARGET))
        atomic(HERE/'CURRENT.json',dict(at=now(),status='WAITING_FOR_SEED45_COMPLETION',target=TARGET))
        while api.WaitForSingleObject(handle,10000)==258:
            # No periodic model polling or app automation; wait on the exact child handle.
            pass
        code=ctypes.c_uint()
        if not api.GetExitCodeProcess(handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
        if code.value!=0:raise RuntimeError('SEED45_CHILD_EXIT_'+str(code.value))
        run=BASE/'runs'/TARGET
        lock=read_json(run/'FINAL_LOCK.json');verify(lock['files'])
        # Independent CPU-only saved-state audit, no model forward or optimizer step.
        sys.path.insert(0,str(REC));import adapter
        spec=dict(name=TARGET,arm='B0',seed=45,order=45,epochs=64,split='fit')
        audit=adapter.validate_final(spec,run)
        if audit['updates']!=21760:raise ValueError('SEED45_NOT_FULL_64_EPOCHS')
        if (BASE/'runs'/NEXT).exists():raise ValueError('NEXT_SEED_WAS_STARTED')
        attempt=Path(active['attempt']);label=stage['label']
        stdout=attempt/'logs'/f'{label}.stdout.log';stderr=attempt/'logs'/f'{label}.stderr.log'
        record=dict(at=now(),command=stage['command'],pid=child.pid,exit_code=0,logs=seal([stdout,stderr]),
            outputs=seal([run/'FINAL_LOCK.json',*[Path(p) for p in lock['files']]]),
            recorded_by='human_requested_pause_controller',real_exit_obtained_from_Windows_process_handle=True)
        receipt=attempt/'receipts'/f'{label}.json'
        if receipt.exists():raise ValueError('EXISTING_CHILD_RECEIPT_DO_NOT_OVERWRITE')
        write_json(receipt,record)
        # Only the waiting coordinator remains in the task. No training child is killed.
        result=subprocess.run(['powershell.exe','-NoProfile','-Command',
            "Stop-ScheduledTask -TaskName '"+TASK+"'; (Get-ScheduledTask -TaskName '"+TASK+"').State.ToString()"],
            capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW,check=True)
        for _ in range(20):
            if not parent.is_running():break
            time.sleep(.25)
        if parent.is_running():raise RuntimeError('COORDINATOR_DID_NOT_EXIT')
        stopped=True;suspended=False
        if (BASE/'runs'/NEXT).exists():raise ValueError('SEED46_UNEXPECTED_ARTIFACTS')
        receipt_files=seal([run/'FINAL_LOCK.json',run/'final.pt',run/'initial.pt',run/'contract.json',receipt])
        done=dict(status='PAUSED_BY_USER',at=now(),reason='seed45完整64轮结束后暂停，用户需关闭电脑',
            completed_run=TARGET,completed_epochs=64,updates=21760,seed46_started=False,
            worker_stopped=True,child_exit_code=0,task_scheduler_output=result.stdout.strip(),
            files=receipt_files,resume_entry=str(REC/'START.cmd'),
            next_stage='E1C_E64_s46_train',safe_to_shutdown=True,experiment_complete=False)
        write_json(HERE/'PAUSE_COMPLETE.json',done)
        atomic(REC/'CHILD_STATE.json',dict(at=now(),label=label,pid=child.pid,state='SUCCESS',exit_code=0,attempt=str(attempt)))
        atomic(REC/'CURRENT.json',done);atomic(REC/'WORKER_EXIT.json',dict(**done,exit_code=0))
        atomic(HERE/'CURRENT.json',done)
    except BaseException:
        error=dict(status='PAUSE_CONTROLLER_FAILURE',at=now(),traceback=traceback.format_exc())
        write_json(HERE/'FAILURE.json',error);atomic(HERE/'CURRENT.json',error)
        if suspended and parent.is_running():parent.resume()
        raise
    finally:
        api.CloseHandle.argtypes=[ctypes.c_void_p];api.CloseHandle(handle)


if __name__=='__main__':main()
