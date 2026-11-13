"""Exclusive OS worker; never depends on the conversation process lifetime."""
import os,sys
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='2'
sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parent
log=open(HERE/'worker_stdout.log','a',encoding='utf-8',buffering=1);err=open(HERE/'worker_stderr.log','a',encoding='utf-8',buffering=1);sys.stdout=log;sys.stderr=err
import msvcrt,traceback,time
from core import write_json,stamp
lock=open(HERE/'worker.lock','a+b');lock.seek(0)
if lock.read(1)==b'':lock.write(b'0');lock.flush()
lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
started=stamp();write_json(HERE/'worker_state.json',dict(status='RUNNING',pid=os.getpid(),parent=os.getppid(),started_at=started,stage='IMPORT_S1'))
try:
    import run_s1
    run_s1.main()
    write_json(HERE/'exit_status.json',dict(status='SUCCESS',at=stamp()))
    write_json(HERE/'worker_state.json',dict(status='SUCCESS',pid=os.getpid(),parent=os.getppid(),started_at=started,finished_at=stamp()))
except BaseException:
    failure=dict(status='FAILURE',at=stamp(),traceback=traceback.format_exc());write_json(HERE/'FAILURE.json',failure);write_json(HERE/'exit_status.json',failure);write_json(HERE/'worker_state.json',dict(**failure,pid=os.getpid(),started_at=started));traceback.print_exc();raise
finally:
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close();log.flush();err.flush()
