"""Single byte-locked Windows OS worker, independent of this conversation."""
import os,sys,msvcrt,traceback
from pathlib import Path
from datetime import datetime,timezone,timedelta
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[2]/'src'))
from f16.io import write_json
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
sys.stdout=open(HERE/'worker_stdout.log','a',encoding='utf-8',buffering=1);sys.stderr=open(HERE/'worker_stderr.log','a',encoding='utf-8',buffering=1)
lock=open(HERE/'worker.lock','a+b');lock.seek(0)
if not lock.read(1):lock.write(b'0');lock.flush()
lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
started=now();write_json(HERE/'worker_state.json',dict(status='RUNNING',pid=os.getpid(),parent=os.getppid(),started_at=started,stage='IMPORT_T1'))
try:
    import run_t1
    run_t1.main()
    write_json(HERE/'exit_status.json',dict(status='SUCCESS',at=now()))
    write_json(HERE/'worker_state.json',dict(status='SUCCESS',pid=os.getpid(),parent=os.getppid(),started_at=started,finished_at=now()))
except BaseException:
    failure=dict(status='FAILURE',at=now(),traceback=traceback.format_exc());write_json(HERE/'FAILURE.json',failure);write_json(HERE/'exit_status.json',failure);write_json(HERE/'worker_state.json',dict(**failure,pid=os.getpid(),started_at=started));traceback.print_exc();raise
finally:
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
