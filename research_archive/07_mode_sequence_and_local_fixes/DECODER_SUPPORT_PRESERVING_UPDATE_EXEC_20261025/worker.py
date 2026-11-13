"""Sole independent G0 worker; later formal training requires its own actual gates."""
import sys,os,msvcrt,traceback,json
from pathlib import Path
from datetime import datetime,timedelta,timezone
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent

def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,path)

def main():
    lock=(HERE/'worker.lock').open('a+b')
    if (HERE/'worker_state.json').exists():raise FileExistsError('G0 worker already existed; no duplicate')
    if not (HERE/'worker.lock').stat().st_size:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    start=now();state=dict(status='RUNNING',pid=os.getpid(),parent_pid=os.getppid(),started_at=start)
    write(HERE/'worker_state.json',state);write(HERE/'runtime_state.json',dict(status='RUNNING',stage='G0_IMPORT_INITIAL_SOURCE',started_at=start))
    try:
        from g0_one_step import main as produce
        from task_adapter import read_json,verify_manifest,write_json
        from review_g0 import audit
        from prelaunch import verify_extension
        assert read_json(HERE/'PREFLIGHT.json')['status']=='PASS'
        source=verify_manifest();extension=verify_extension();result=produce();write_json(HERE/'runtime_state.json',dict(status='RUNNING',stage='G0_INDEPENDENT_SAVED_AUDIT',at=now()))
        review=audit();assert review['status']=='PASS_INDEPENDENT_SAVED_G0';assert verify_manifest()==source and verify_extension()==extension
        write(HERE/'RESULT.json',dict(status='SUCCESS_G0_EXECUTION_REVIEW_REQUIRED',decision=result['status'],source_digest=source,G0_packets=48,formal_training=0,G1='READY_FOR_OWN_PREFLIGHT' if result['status']=='G0_PASS_ALLOW_G1' else 'NOT_TRIGGERED_G0_GATE',G2='G0_SAVED_AUDIT_PASS'))
        write(HERE/'exit_status.json',dict(status='SUCCESS',at=now()));state.update(status='SUCCESS',finished_at=now());write(HERE/'worker_state.json',state)
        write(HERE/'runtime_state.json',dict(status='COMPLETED_REVIEW_REQUIRED',stage='G0_COMPLETE',active_worker=False,started_at=start,finished_at=state['finished_at']))
    except BaseException as e:
        write(HERE/'FAILURE.json',dict(type=type(e).__name__,message=str(e),at=now(),traceback=traceback.format_exc(),preserve_all=True));state.update(status='FAILURE',finished_at=now());write(HERE/'worker_state.json',state);write(HERE/'exit_status.json',dict(status='FAILURE',at=now()));raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()

if __name__=='__main__':main()
