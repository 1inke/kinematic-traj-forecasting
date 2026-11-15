"""Windows Task Scheduler owns this worker; no Codex process or API is used."""
import ast
import ctypes
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REC=Path(__file__).resolve().parent;BASE=REC.parent
sys.path.insert(0,str(BASE))
import worker as oldworker
from common import ROOT,read_json,write_json,verify,sha,seal,now
import resource_guard as rg

ATT=REC/'attempts'/str(time.time_ns())
ATT.mkdir(parents=True,exist_ok=False)
MUTEX=None


def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)


def claim():
    global MUTEX
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_wchar_p];api.CreateMutexW.restype=ctypes.c_void_p
    MUTEX=api.CreateMutexW(None,False,'Local\\TrajectoryNextPhaseStandalone20261114')
    if not MUTEX:raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error()==183:raise RuntimeError('ANOTHER_STANDALONE_WORKER_EXISTS')


def write_once(path,value):
    path=Path(path)
    if path.exists():
        old=read_json(path)
        def stable(v):return {k:x for k,x in v.items() if k!='at'}
        if stable(old)!=stable(value):raise ValueError('EXISTING_RECEIPT_DIFFERS '+str(path))
        if 'files' in old:verify(old['files'])
        return
    write_json(path,value)


def outputs(script,args):
    files=[]
    if script=='make_calib_split.py':files=[BASE/'E1/calib_split.npz']
    elif script=='execution.py':
        mode,s=args[0],json.loads(args[1]);root=BASE/'runs'/s['name']
        marker=root/('FINAL_LOCK.json' if mode=='train' else 'EVALUATION_DONE.json');files=[marker]
        verify(read_json(marker)['files']);files.extend(Path(p) for p in read_json(marker)['files'])
    elif script=='analysis_phase.py':
        files=[BASE/'E1/E1A_NULL.json'] if args[0]=='e1a' else [BASE/'RESULT_ANALYSIS.json',BASE/'RESULT_ANALYSIS.md',*list((BASE/'analysis').glob('*.json'))]
    elif script=='plot_calibration.py':files=list((BASE/'E1/figures').iterdir())
    elif script=='respline_saved.py':files=list(Path(args[args.index('--outdir')+1]).glob('*.npz'))
    elif script=='diagnostics_phase.py':
        if args[0]=='zero':files=[BASE/'E2_zero'/f's{args[1]}_{args[2]}'/'DIAGNOSTIC_RESULT.json']
        else:files=[BASE/'runs'/args[1]/args[2]/'GEOMETRY_DIAGNOSTICS.json']
    elif script=='cost_methods.py':files=[BASE/'cost'/f'{args[0]}.json']
    elif script=='preflight_phase.py':files=[BASE/'PREFLIGHT_METHODS.json']
    return seal(files)


def cached(label,script,args):
    for receipt in sorted((REC/'attempts').glob(f'*/receipts/{label}.json'),reverse=True):
        r=read_json(receipt)
        if r['exit_code']==0:
            verify(r['outputs']);verify(r['logs']);return r
    # Only these two finished before the old bookkeeping crash. Their artifacts
    # are independently validated and sealed in RECOVERY_MANIFEST before launch.
    if label in ('numpy_parts','make_calib_split'):
        path=BASE/'receipts'/f'{label}.json'
        if path.exists():
            r=read_json(path)
            if r['exit_code']==0:
                verify(read_json(REC/'RECOVERY_MANIFEST.json')['files'])
                return dict(source='original_success_receipt',outputs=outputs(script,args))
    return None


def run(label,script,args):
    prior=cached(label,script,args)
    if prior:
        write_json(ATT/'skipped'/f'{label}.json',dict(at=now(),reason='VERIFIED_COMPLETED_STAGE',evidence=prior));return
    estimate=read_json(BASE/'PREFLIGHT_E1.json')['estimated_peak_bytes']
    admission=rg.decide(rg.snapshot(),estimate,'startup');write_json(ATT/'admission'/f'{label}.json',dict(at=now(),**admission))
    if admission['status']!='PASS':raise MemoryError('CURRENT_STARTUP_RESOURCE_SHORTAGE')
    log=ATT/'logs';log.mkdir(exist_ok=True)
    command=[sys.executable,'-B',str(REC/'adapter.py'),script,*args]
    with (log/f'{label}.stdout.log').open('xb') as stdout,(log/f'{label}.stderr.log').open('xb') as stderr:
        child=subprocess.Popen(command,stdout=stdout,stderr=stderr,cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW)
        atomic(REC/'CHILD_STATE.json',dict(at=now(),label=label,pid=child.pid,command=command,state='RUNNING',attempt=str(ATT)))
        exitcode=child.wait()
    record=dict(at=now(),command=command,pid=child.pid,exit_code=exitcode,
        logs=seal([log/f'{label}.stdout.log',log/f'{label}.stderr.log']),outputs=outputs(script,args) if exitcode==0 else {})
    write_json(ATT/'receipts'/f'{label}.json',record)
    atomic(REC/'CHILD_STATE.json',dict(at=now(),label=label,pid=child.pid,state='SUCCESS' if exitcode==0 else 'FAILURE',exit_code=exitcode,attempt=str(ATT)))
    if exitcode:raise RuntimeError(f'CHILD_FAILURE {label} exit={exitcode}; logs={log}')


def train(spec):run(spec['name']+'_train','execution.py',['train',json.dumps(spec)])
def evaluate(spec):run(spec['name']+'_eval','execution.py',['eval',json.dumps(spec)])


def budget():
    p=BASE/'E1/BUDGET_DECISION.json'
    if p.exists():return read_json(p)['E_star']
    return oldworker.choose_budget()


def contract(E,refs):
    if (BASE/'E2E_V2_SEAL.json').exists():
        verify(read_json(BASE/'E2E_V2_SEAL.json')['files'])
        prior=read_json(BASE/'E2E_V2_CONTRACT.json')
        if prior['E_star']!=E or prior['reference_names']!=refs:raise ValueError('SEALED_CONTRACT_MISMATCH')
        return
    oldworker.seal_contract(E,refs)


def main():
    claim()
    verify(read_json(REC/'RECOVERY_MANIFEST.json')['files'])
    if (REC/'RESULT.json').exists() and read_json(REC/'RESULT.json').get('status')=='SUCCESS':return
    # Process-scoped sleep inhibition only. No power-plan, display, OS or login setting is changed.
    wake_api=ctypes.windll.kernel32.SetThreadExecutionState
    wake_api.argtypes=[ctypes.c_uint];wake_api.restype=ctypes.c_uint
    if not wake_api(0x80000001):raise OSError('Cannot hold system awake for active experiment')
    atomic(REC/'ACTIVE_WORKER.json',dict(at=now(),pid=os.getpid(),attempt=str(ATT),standalone_scheduler_owned=True,
        dependency_on_Codex_API_or_app=False,awake_scope='worker lifetime; display may sleep'))
    try:
        # Reuse the sealed original finite scientific stage sequence byte-for-byte,
        # adapting only lifecycle record destinations, singleton lock and run wrappers.
        source=inspect.getsource(oldworker.main)
        lockline="fd=os.open(HERE/'worker.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)"
        if source.count(lockline)!=1:raise ValueError('UNEXPECTED_WORKER_LOCK_SOURCE')
        source=source.replace(lockline,'pass  # process-owned named mutex already held')
        for name in ('WORKER_START.json','RESULT.json','WORKER_EXIT.json','FAILURE.json'):
            source=source.replace("HERE/'"+name+"'","ATT/'"+name+"'")
        source=source.replace("HERE/'CURRENT.json'","REC/'CURRENT.json'")
        original_filter="and 'design_original' not in p.parts"
        source=source.replace(original_filter,original_filter+" and not (p.parent==REC and p.name in ('RESULT.json','WORKER_EXIT.json','ACTIVE_WORKER.json'))")
        ns=dict(oldworker.__dict__,REC=REC,ATT=ATT,run=run,train=train,evaluate=evaluate,choose_budget=budget,
            seal_contract=contract,atomic=atomic,write_json=write_once)
        exec(compile(ast.parse(source),str(REC/'worker_standalone.py')+':original_finite_stage_sequence','exec'),ns)
        ns['main']()
        result=read_json(ATT/'RESULT.json');atomic(REC/'RESULT.json',result);atomic(REC/'WORKER_EXIT.json',read_json(ATT/'WORKER_EXIT.json'))
    except BaseException:
        error=dict(status='FAILURE',at=now(),pid=os.getpid(),attempt=str(ATT),traceback=traceback.format_exc())
        if not (ATT/'FAILURE.json').exists():write_json(ATT/'FAILURE.json',error)
        atomic(REC/'WORKER_EXIT.json',dict(**error,exit_code=1));atomic(REC/'CURRENT.json',error);raise
    finally:wake_api(0x80000000)


if __name__=='__main__':main()
