"""One independent no-trigger saved-analysis worker; real map gate stop is final."""
from common import *
from datetime import datetime,timedelta,timezone
import os,time,traceback,msvcrt
def stamp():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def main():
    lock=(HERE/'worker.lock').open('a+b')
    if (HERE/'worker_state.json').exists():raise FileExistsError('Existing actual worker state: no duplicate')
    if (HERE/'worker.lock').stat().st_size==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    started=stamp();pre=read(HERE/'PREFLIGHT.json');assert pre['status']=='PASS_ACTUAL_SAVED_CPU8X128';assert verify_manifest()==pre['source_digest']
    state=dict(status='RUNNING',pid=os.getpid(),parent_pid=os.getppid(),started_at=started)
    write(HERE/'worker_state.json',state)
    control=dict(status='RUNNING',active_worker=True,started_at=started,stage='Q0_SAVED_ATTRIBUTION',first_progress_query_done=False,first_progress_query_count=0,
        query_at=(datetime.fromisoformat(started)+timedelta(seconds=pre['planning_seconds']/2)).isoformat(),
        analysis_deadline=(datetime.fromisoformat(started)+timedelta(seconds=pre['planning_seconds'])).isoformat())
    write(HERE/'runtime_state.json',control)
    try:
        from saved_attribution import produce
        q0=produce();control.update(stage='Q0_INDEPENDENT_SAVED_REVIEW');write(HERE/'runtime_state.json',control)
        from audit_saved import audit
        review=audit();inventory=read(HERE/'q1_map_inventory/MAP_INVENTORY.json')
        assert not inventory['all_required_recordings_qualified']
        # Current registry conclusively lacks required material. Do not implement
        # untriggered map probes/training or silently call this a negative AUC.
        write(HERE/'RESULT.json',dict(status='SUCCESS_Q0_WITH_MAP_ADMISSION_STOP',Q0=q0['status'],Q0_audit=review['status'],
            Q1a=inventory['status'],Q1='NOT_EXECUTED_MAP_ADMISSION_FAILED',Q2='NOT_TRIGGERED_MAP_ADMISSION_FAILED',
            formal_experiments=0,source_digest=verify_manifest(),no_old_model_forward=True,no_training=True,needs_human='qualified static maps, independent controls and provenance for all6 recordings; no automatic external download'))
        write(HERE/'exit_status.json',dict(status='SUCCESS',at=stamp()));state.update(status='SUCCESS',finished_at=stamp());write(HERE/'worker_state.json',state)
        control.update(status='COMPLETED_REVIEW_REQUIRED_MAP_STOP',active_worker=False,stage='Q0_COMPLETE_Q1A_MAP_STOP',finished_at=state['finished_at']);write(HERE/'runtime_state.json',control)
    except BaseException as error:
        write(HERE/'FAILURE.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),at=stamp(),preserve_all=True))
        state.update(status='FAILURE',finished_at=stamp());write(HERE/'worker_state.json',state);write(HERE/'exit_status.json',dict(status='FAILURE',at=stamp()));raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
if __name__=='__main__':main()
