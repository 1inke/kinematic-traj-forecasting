from adapter import *
from datetime import datetime,timedelta,timezone
import os,msvcrt,traceback
def stamp():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def verify_recovery():
    m=read(CHILD/'source_manifest.json');assert digest(m['files'])==m['digest']
    for v in m['files'].values():assert sha(v['path'])==v['sha256'],v['path']
    assert verify_manifest()==m['inherited_parent_digest'];return m['digest']
def main():
    lock=(CHILD/'worker.lock').open('a+b')
    if (CHILD/'worker_state.json').exists():raise FileExistsError('No duplicate recovery worker')
    if (CHILD/'worker.lock').stat().st_size==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    pre=read(CHILD/'PREFLIGHT.json');assert pre['status']=='PASS_ACTUAL_FAILED_POPULATION_AUDITOR_REGRESSION';source=verify_recovery()
    start=stamp();state=dict(status='RUNNING',pid=os.getpid(),parent_pid=os.getppid(),started_at=start)
    write(CHILD/'worker_state.json',state)
    control=dict(status='RUNNING',active_worker=True,stage='INDEPENDENT_SAVED_AUDIT_ONLY',started_at=start,
        query_at=(datetime.fromisoformat(start)+timedelta(seconds=90)).isoformat(),analysis_deadline=(datetime.fromisoformat(start)+timedelta(seconds=180)).isoformat(),first_progress_query_count=0)
    write(CHILD/'runtime_state.json',control)
    try:
        r=audit();assert r['status']=='PASS_INDEPENDENT_SAVED_Q0';assert verify_recovery()==source
        write(CHILD/'RESULT.json',dict(status='SUCCESS_AUDIT_RECOVERY',audit=r['status'],source_digest=source,Q0_reproduced=False,parent_worker_status='FAILURE_PRESERVED',
             parent_scientific_outputs='SUCCESS_SAVED_ATTRIBUTION_REVIEW_REQUIRED',Q1a='STOP_MAP_ADMISSION_UNRESOLVED',Q1_probe='NOT_EXECUTED',Q2='NOT_TRIGGERED',formal_training=0))
        write(CHILD/'exit_status.json',dict(status='SUCCESS',at=stamp()));state.update(status='SUCCESS',finished_at=stamp());write(CHILD/'worker_state.json',state)
        control.update(status='COMPLETED_REVIEW_REQUIRED',active_worker=False,stage='Q0_AUDIT_COMPLETE_MAP_STOP',finished_at=state['finished_at']);write(CHILD/'runtime_state.json',control)
    except BaseException as e:
        write(CHILD/'FAILURE.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),at=stamp(),preserve_all=True))
        state.update(status='FAILURE',finished_at=stamp());write(CHILD/'worker_state.json',state);write(CHILD/'exit_status.json',dict(status='FAILURE',at=stamp()));raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
if __name__=='__main__':main()
