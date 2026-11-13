import os,sys,json,traceback,msvcrt
from pathlib import Path
from datetime import datetime
H=Path(__file__).resolve().parent
def write(name,z):
    p=H/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(z,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
def main():
    lock=(H/'worker.lock').open('a+b');lock.seek(0)
    if not (H/'worker.lock').stat().st_size:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    started=datetime.now().astimezone().isoformat();base=dict(pid=os.getpid(),parent=os.getppid(),started_at=started)
    write('worker_state.json',dict(status='RUNNING',stage='AUDIT_RECOVERY_IMPORT',**base))
    with (H/'worker_stdout.log').open('a',encoding='utf-8',buffering=1) as out,(H/'worker_stderr.log').open('a',encoding='utf-8',buffering=1) as err:
        sys.stdout=out;sys.stderr=err
        try:
            from recovery import main as run
            run();finished=datetime.now().astimezone().isoformat()
            write('exit_status.json',dict(status='SUCCESS',finished_at=finished,**base));write('worker_state.json',dict(status='SUCCESS',finished_at=finished,**base));write('runtime_state.json',dict(status='COMPLETED_REVIEW_READY',no_retraining=True,first_progress_query_done=False))
        except BaseException:
            traceback.print_exc();finished=datetime.now().astimezone().isoformat()
            if 'common' in sys.modules:
                try:sys.modules['common'].failure('STOP_AUDIT_RECOVERY_WORKER')
                except BaseException:traceback.print_exc()
            write('exit_status.json',dict(status='FAILURE',finished_at=finished,**base));write('worker_state.json',dict(status='FAILURE',finished_at=finished,**base));raise
if __name__=='__main__':main()
