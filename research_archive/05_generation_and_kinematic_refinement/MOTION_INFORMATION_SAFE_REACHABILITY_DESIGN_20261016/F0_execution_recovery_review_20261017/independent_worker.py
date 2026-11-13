import sys,os,msvcrt,traceback
from pathlib import Path
from datetime import datetime
H=Path(__file__).resolve().parent
if __name__=='__main__':
    lock=(H/'worker.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    sys.stdout=(H/'worker_stdout.log').open('a',encoding='utf-8',buffering=1);sys.stderr=(H/'worker_stderr.log').open('a',encoding='utf-8',buffering=1)
    import common as c
    base=dict(pid=os.getpid(),parent=os.getppid(),started_at=datetime.now().astimezone().isoformat());c.write('worker_state.json',dict(status='RUNNING',**base))
    try:
        import run
        run.main();status='SUCCESS'
    except Exception:
        traceback.print_exc();c.failure('STOP_F0_WORKER');status='FAILURE'
    final=dict(status=status,finished_at=datetime.now().astimezone().isoformat(),**base);c.write('worker_state.json',final);c.write('exit_status.json',final)
    sys.exit(0 if status=='SUCCESS' else 1)
