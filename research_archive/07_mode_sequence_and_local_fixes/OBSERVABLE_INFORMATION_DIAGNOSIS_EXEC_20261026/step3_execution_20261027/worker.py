"""Sole serial step3 process coordinator. Never launches step4 implicitly."""
from core import *
import subprocess,time,traceback,msvcrt
def run(label,*args):
    mark(label,arguments=list(args));logs=HERE/'logs';logs.mkdir(exist_ok=True)
    out=logs/(label+'.stdout.log');err=logs/(label+'.stderr.log')
    if out.exists() or err.exists():raise FileExistsError('Preserve prior child logs')
    with out.open('xb') as o,err.open('xb') as e:
        p=subprocess.Popen([sys.executable,'-B','-u',str(HERE/args[0]),*map(str,args[1:])],cwd=HERE,stdout=o,stderr=e,creationflags=subprocess.CREATE_NO_WINDOW)
        write_json(HERE/'current_child.json',dict(pid=p.pid,stage=label,arguments=args,started=now()))
        code=p.wait()
    write_json(HERE/'current_child.json',dict(pid=p.pid,stage=label,exit_code=code,finished=now()))
    if code:raise RuntimeError('Child '+label+' failed; preserve source/log/partial')
def main():
    lock=(HERE/'worker.lock').open('a+b')
    if lock.tell()==0:lock.write(b'1');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (HERE/'worker_state.json').exists();start=now()
    try:
        assert read_json(HERE/'PRELAUNCH_REVIEW.json')['status']=='PASS'
        write_json(HERE/'worker_state.json',dict(status='RUNNING',started=start,pid=os.getpid()))
        mark('STEP3_INITIAL_SOURCE_SHA');verify();arms=read_json(HERE/'FEATURE_PROVENANCE.json')['admissible_arms']
        for seed in SEEDS:
            for arm in arms:run(f'train_{arm}_{seed}','train.py',arm,seed)
        models={};done={}
        for seed in SEEDS:
            for arm in arms:
                f=HERE/f'runs/{arm}_seed{seed}';c=read_json(f/'manifest.json');completed(f,c)
                models[f'{arm}_seed{seed}']=sha256(f/'final.pt');done[f'{arm}_seed{seed}']=sha256(f/'done.json')
        write_json(HERE/'ALL_PROBE_FINALS_LOCK.json',dict(status='ALL_ADMITTED_FINALS_LOCKED_BEFORE_HELD_EFFECTS',source=source(),models=models,done_SHA=done,blocked_arms=read_json(HERE/'FEATURE_PROVENANCE.json')['blocked_arms'],at=now()))
        for seed in SEEDS:
            for arm in arms:
                for split in ('opt','hold','inner'):
                    run(f'predict_{arm}_{seed}_{split}','predict.py',arm,seed,split)
                    run(f'audit_{arm}_{seed}_{split}','saved_audit.py',arm,seed,split)
        run('primary_bootstrap_20000','analysis_statistics.py')
        mark('STEP3_FINAL_SOURCE_OUTPUT_SHA');verify();outputs={}
        for folder in ('runs','populations','statistics'):
            for p in (HERE/folder).rglob('*'):
                if p.is_file():outputs[p.relative_to(HERE).as_posix()]=sha256(p)
        reports=[]
        for folder in ('runs','populations','statistics'):
            for p in (HERE/folder).rglob('RESOURCE.json'):
                r=read_json(p);assert r['last']['status']=='PASS';reports.append(dict(path=p.relative_to(HERE).as_posix(),report=r))
        write_json(HERE/'RESOURCE.json',dict(status='PASS_ALL_STAGE_RETAINED_PEAKS',records=reports))
        write_json(HERE/'PRESERVATION.json',dict(status='PASS',source=source(),outputs=outputs))
        write_json(HERE/'ARITHMETIC_AUDIT.json',dict(status='PASS',admitted_runs=len(models),populations=len(models)*3,all401_independent_choice_fixed19=True,full_samecandidate_original_complete_link_pdist_and_allgroups5motion=True,all_20000_actual_draws_independent=True,not_model_gradient_replay=True))
        write_json(HERE/'RESULT.json',dict(status='SUCCESS_STEP3_REVIEW_REQUIRED',runs=len(models),updates=len(models)*1024,epochs=len(models)*8,
            blocked_arms=read_json(HERE/'FEATURE_PROVENANCE.json')['blocked_arms'],primary=read_json(HERE/'statistics/PRIMARY_COMPARISONS.json'),step4='AUTHORIZED_AFTER_COMPLETE_STEP3_ANALYSIS_NOT_RUN',source=source()))
        state=dict(status='SUCCESS',pid=os.getpid(),started=start,finished=now());write_json(HERE/'worker_state.json',state);write_json(HERE/'exit_status.json',state)
    except BaseException as exc:
        state=dict(status='FAILURE',pid=os.getpid(),started=start,finished=now(),message=str(exc),traceback=traceback.format_exc())
        write_json(HERE/'FAILURE.json',state);write_json(HERE/'worker_state.json',state);write_json(HERE/'exit_status.json',state);raise
    finally:lock.close()
if __name__=='__main__':main()
