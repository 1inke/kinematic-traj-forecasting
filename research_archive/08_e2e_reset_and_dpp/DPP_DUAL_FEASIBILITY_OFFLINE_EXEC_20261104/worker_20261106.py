"""Sole independent Stage A worker. Real preflight must already have passed."""
import os,time,traceback
from common import *
from threadpoolctl import threadpool_limits

def main():
    if not authorize(True,'independent_stage_A_worker'):return
    with (RUN/'worker_20261106.lock').open('x') as f:f.write(str(os.getpid()))
    started=time.perf_counter()
    write(RUN/'WORKER_START.json',dict(at=now(),pid=os.getpid(),parent_pid=os.getppid(),stage='VERIFY_REAL_PREFLIGHT',training=False))
    rg=guard_module();guard=rg.Guard(2*1024**3)
    try:
        source=read(RUN/'source_manifest.json');verify(source['files'])
        pre=read(RUN/'PREFLIGHT_REAL.json')
        if pre['status']!='PASS_ACTUAL_SAVED128_KIN_DPP_QUALITY_READBACK' or pre['source_manifest']!=sha(RUN/'source_manifest.json'):
            raise ValueError('REAL_PREFLIGHT_CONTRACT')
        guard.check(force=True)
        with threadpool_limits(limits=1):
            import evaluate,review
            write(RUN/'WORKER_EVALUATE_STARTED.json',dict(at=now(),pid=os.getpid()))
            print('STAGE_A_EVALUATE',now(),flush=True);evaluate.main(True)
            write(RUN/'WORKER_REVIEW_STARTED.json',dict(at=now(),pid=os.getpid()))
            print('STAGE_A_INDEPENDENT_REVIEW',now(),flush=True);review.main(True)
        final=read(RUN/'FINAL_SEAL.json');verify(final['files']);verify(final['sources'])
        guard.check(force=True)
        write(RUN/'WORKER_RESOURCE.json',dict(policy='warning_first_remaining_v3',estimate_bytes=2*1024**3,observations=guard.observations,snapshot=rg.snapshot()))
        write(RUN/'WORKER_SUCCESS.json',dict(status='SUCCESS',at=now(),pid=os.getpid(),elapsed_seconds=time.perf_counter()-started,no_training=True,files=seal([RUN/'FINAL_SEAL.json',RUN/'WORKER_RESOURCE.json'])))
        write(RUN/'WORKER_EXIT.json',dict(status='SUCCESS',at=now(),pid=os.getpid(),exit_code=0))
        print('STAGE_A_ALL_SAVED_REVIEW_SUCCESS',now(),flush=True)
    except BaseException as e:
        write(RUN/'WORKER_FAILURE.json',dict(status='FAILURE',at=now(),pid=os.getpid(),error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started))
        write(RUN/'WORKER_EXIT.json',dict(status='FAILURE',at=now(),pid=os.getpid(),exit_code=1))
        raise

if __name__=='__main__':main()
