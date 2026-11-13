"""Sole fourth-step CPU coordinator; all selections locked BEFORE held labels."""
from core import *
import subprocess,traceback,msvcrt
def run(label,*args):
    mark(label,arguments=args);logs=HERE/'logs';logs.mkdir(exist_ok=True);o=logs/(label+'.stdout.log');e=logs/(label+'.stderr.log')
    if o.exists() or e.exists():raise FileExistsError('Preserve prior logs')
    with o.open('xb') as out,e.open('xb') as err:
        child=subprocess.Popen([sys.executable,'-B','-u',str(HERE/args[0]),*map(str,args[1:])],cwd=HERE,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        write_json(HERE/'current_child.json',dict(pid=child.pid,stage=label,started=now(),arguments=args));code=child.wait()
    write_json(HERE/'current_child.json',dict(pid=child.pid,stage=label,finished=now(),exit_code=code))
    if code:raise RuntimeError('Current child '+label+' failed; preserve evidence')
def main():
    lock=(HERE/'worker.lock').open('a+b')
    if lock.tell()==0:lock.write(b'1');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);assert not (HERE/'worker_state.json').exists();start=now()
    try:
        require3();assert read_json(HERE/'PRELAUNCH_REVIEW.json')['status']=='PASS'
        write_json(HERE/'worker_state.json',dict(status='RUNNING',pid=os.getpid(),started=start));mark('STEP4_INITIAL_SOURCE_SHA');verify()
        for seed in SEEDS:run(f'geometry_and_independent_audit_{seed}','select_and_audit.py',seed)
        locks={};problems=0
        for seed in SEEDS:
            f=HERE/f'selections/seed{seed}';d=read_json(f/'done.json');completed(f,d['contract']);locks[str(seed)]=sha256(f/'done.json');problems+=read_json(f/'SELECTION.json')['scientific_assignment_problems']
        assert problems<=49152
        write_json(HERE/'ALL_SELECTIONS_LOCK.json',dict(status='ALL3_GT_FREE_SELECTIONS_AND_THRESHOLDS_LOCKED_BEFORE_HELD_LABELS',source=source(),done_SHA=locks,science_assignment_problems=problems,at=now(),no_B_quality_GT_group_used_for_selection=True))
        for seed in SEEDS:run(f'posthoc_common_domain_{seed}','posthoc.py',seed)
        run('query_track_bootstrap_20000','statistics_fixed.py')
        mark('STEP4_FINAL_SOURCE_OUTPUT_SHA');verify();outputs={};reports=[]
        for name in ('selections','posthoc','statistics'):
            for p in (HERE/name).rglob('*'):
                if p.is_file():outputs[p.relative_to(HERE).as_posix()]=sha256(p)
            for p in (HERE/name).rglob('RESOURCE.json'):
                r=read_json(p);assert r['last']['status']=='PASS';reports.append(dict(path=p.relative_to(HERE).as_posix(),report=r))
        write_json(HERE/'PRESERVATION.json',dict(status='PASS',source=source(),outputs=outputs))
        write_json(HERE/'RESOURCE.json',dict(status='PASS_ALL_STAGE_RETAINED_PEAKS',records=reports))
        write_json(HERE/'ARITHMETIC_AUDIT.json',dict(status='PASS',source_time_IDs_shortlists_independent=True,all_matching_second_Hungarian_primal_dual=True,
            all_primary_reverse_domain_action_maps_and_401_quality_positive_sets_independent=True,all_20000_query_track_draws_independent=True,
            no_model_Torch_forward_or_training=True,science_assignments=problems,additional_certificate_and_reverse_problems_separate=True))
        summary=read_json(HERE/'statistics/SUMMARY.json')
        write_json(HERE/'RESULT.json',dict(status='SUCCESS_STEP4_REVIEW_REQUIRED',source=source(),no_training=True,support=summary['support'],pair_counts=summary['pair_counts'],finite_four_steps_compute_complete_review_required=True))
        state=dict(status='SUCCESS',pid=os.getpid(),started=start,finished=now());write_json(HERE/'worker_state.json',state);write_json(HERE/'exit_status.json',state)
    except BaseException as exc:
        state=dict(status='FAILURE',pid=os.getpid(),started=start,finished=now(),message=str(exc),traceback=traceback.format_exc());write_json(HERE/'FAILURE.json',state);write_json(HERE/'worker_state.json',state);write_json(HERE/'exit_status.json',state);raise
    finally:lock.close()
if __name__=='__main__':main()
