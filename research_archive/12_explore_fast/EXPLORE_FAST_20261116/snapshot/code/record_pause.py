from explore_common import *

if __name__=='__main__':
    verify_source()
    checkpoints={};completed=[]
    for directory in (ROOT/'runs').glob('*'):
        if not directory.is_dir():continue
        if (directory/'result.json').exists():
            validate_result(directory);completed.append(directory.name)
        pts=[p for p in sorted(directory.glob('ckpt_epoch*.pt')) if p.with_suffix('.json').exists()]
        if pts:
            side=read(pts[-1].with_suffix('.json'));assert sha(pts[-1])==side['sha256']
            checkpoints[directory.name]=side
    event=ROOT/'standalone/pause_events'/str(time.time_ns())
    event.mkdir(parents=True,exist_ok=False)
    record=dict(at=now(),status='PAUSED_BY_USER',automatic_resume=False,active=[],
                durable_checkpoints=checkpoints,completed_runs=completed)
    replace_json(event/'PAUSE_COMPLETE.json',record)
    replace_json(ROOT/'standalone/CURRENT.json',record)
    replace_json(ROOT/'standalone/WORKER_EXIT.json',record)
    print('PAUSED; exploration complete epochs verified.')

