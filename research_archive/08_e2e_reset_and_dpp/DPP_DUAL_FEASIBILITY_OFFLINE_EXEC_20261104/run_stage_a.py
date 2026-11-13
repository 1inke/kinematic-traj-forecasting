"""Manual-only entry. No task scheduler, no background process, no auto-start."""
import argparse
from common import *
def main(execute=False):
    if not authorize(execute,'manual_stage_A_pipeline'):return
    import prepare,preflight,evaluate,review
    from threadpoolctl import threadpool_limits
    write(RUN/'MANUAL_PIPELINE_STARTED.json',dict(at=now(),mode='CPU_SAVED_DATA_ONLY',training=False))
    with threadpool_limits(limits=1):
        prepare.main(True)
        preflight.main(True)
        evaluate.main(True)
        review.main(True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');main(p.parse_args().execute)

