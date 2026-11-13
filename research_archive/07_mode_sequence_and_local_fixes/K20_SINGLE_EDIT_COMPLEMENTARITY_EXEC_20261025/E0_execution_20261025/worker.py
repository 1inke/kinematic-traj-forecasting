"""One sole independent E0 CPU worker; no training and no automatic E1 bypass."""
import os,sys,msvcrt,traceback,time
sys.dont_write_bytecode=True
from common import *
def main():
    lock=(HERE/'worker.lock').open('a+b')
    if (HERE/'worker_state.json').exists():raise FileExistsError('Current worker already existed; duplicate forbidden')
    if not (HERE/'worker.lock').stat().st_size:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    start=now();state=dict(status='RUNNING',pid=os.getpid(),parent_pid=os.getppid(),started_at=start)
    write_json(HERE/'worker_state.json',state);stage('E0_INITIAL_SOURCE_INPUT_SHA',started_at=start)
    guard=Guard(ESTIMATE)
    try:
        guard.check('startup');source=verify()
        from run_e0 import main as produce
        from review_saved import audit_seed
        data,gates,source2=produce(guard);assert source==source2
        audits=[audit_seed(seed,data,source,guard) for seed in SEEDS]
        write_json(HERE/'ARITHMETIC_AUDIT.json',dict(status='PASS_ALL_4915200_ACTIONS_AND_ORACLE',records=audits,no_model_replay=True))
        stage('E0_FINAL_SOURCE_INPUT_OUTPUT_SHA')
        assert verify()==source
        outputs={}
        for seed in SEEDS:
            done=validate_done(HERE/f'seed{seed}'/'done.json',source)
            for file,ss in done['outputs'].items():outputs[f'seed{seed}/{file}']=ss
        write_json(HERE/'PRESERVATION.json',dict(status='PASS',source_digest=source,all_inherited_and_inputs_unchanged=True,outputs=outputs,original_designs_and_failures_preserved=True))
        last=guard.check();write_json(HERE/'RESOURCE.json',dict(status='PASS',estimate=ESTIMATE,factor=1.3,observations=guard.observations,last=last,production_lifetime_peak_private=last['snapshot']['peak_private'],production_lifetime_peak_RSS=last['snapshot']['peak_rss']))
        decision='E0_PASS_ALLOW_E1_OWN_PREFLIGHT' if all(g['pass_all'] for g in gates) else 'E0_STOP'
        write_json(HERE/'DECISION.json',dict(status=decision,gates=gates,E1='OWN_ACTUAL_GPU_PREFLIGHT_REQUIRED' if decision.startswith('E0_PASS') else 'NOT_TRIGGERED',no_threshold_or_pool_changes=True))
        write_json(HERE/'RESULT.json',dict(status='PASS_EXECUTION_E0_REVIEW_REQUIRED',scientific_decision=decision,source_digest=source,windows=4096,seeds=list(SEEDS),nonSTOP_actions=4915200,formal_training=0,all_saved_audit='PASS',finished_at=now()))
        state.update(status='SUCCESS',finished_at=now());write_json(HERE/'exit_status.json',dict(status='SUCCESS',at=now()));write_json(HERE/'worker_state.json',state)
        write_json(HERE/'runtime_state.json',dict(status='COMPLETED_REVIEW_REQUIRED',stage='E0_COMPLETE',active_worker=False,started_at=start,finished_at=state['finished_at'],scientific_decision=decision))
    except BaseException as e:
        write_json(HERE/'FAILURE.json',dict(type=type(e).__name__,message=str(e),at=now(),traceback=traceback.format_exc(),preserve_all=True,resource_observations=guard.observations))
        state.update(status='FAILURE',finished_at=now());write_json(HERE/'worker_state.json',state);write_json(HERE/'exit_status.json',dict(status='FAILURE',at=now()));raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
if __name__=='__main__':main()
