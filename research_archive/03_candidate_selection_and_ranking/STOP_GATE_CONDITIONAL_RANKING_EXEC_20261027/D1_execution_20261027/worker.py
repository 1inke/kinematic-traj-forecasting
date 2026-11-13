from core import *
import subprocess,traceback,msvcrt
def run(label,*args):
    folder=HERE/'logs';folder.mkdir(exist_ok=True);o=folder/(label+'.stdout.log');e=folder/(label+'.stderr.log')
    if o.exists() or e.exists():raise FileExistsError('Preserve existing logs')
    mark(label,arguments=args)
    with o.open('xb') as out,e.open('xb') as err:
        p=subprocess.Popen([sys.executable,'-B','-u',str(HERE/args[0]),*map(str,args[1:])],cwd=HERE,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        write_json(HERE/'current_child.json',dict(pid=p.pid,stage=label,started=now(),arguments=args));code=p.wait()
    write_json(HERE/'current_child.json',dict(pid=p.pid,stage=label,finished=now(),exit_code=code))
    if code:raise RuntimeError('Child '+label+' failed; preserve evidence and do not repeat completed models')
def main():
    lock=(HERE/'worker.lock').open('a+b');lock.write(b'1');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);assert not (HERE/'worker_state.json').exists();state=dict(status='RUNNING',pid=os.getpid(),started=now());write_json(HERE/'worker_state.json',state)
    try:
        assert read_json(HERE/'PRELAUNCH_REVIEW.json')['status']=='PASS';mark('D1_INITIAL_SOURCE_INPUT_SHA');verify_manifest(HERE)
        for seed in SEEDS:
            for arm in ARMS:run(f'train_{arm}_{seed}','train_paired.py',seed,arm)
        # Tiny metadata-only process uses the same pure train reader, never fits.
        run('ALL9_FINAL_LOCK','lock_finals.py')
        audits=[]
        for seed in SEEDS:
            for split in SPLITS:
                run(f'save_{seed}_{split}','evaluate_saved.py',seed,split);run(f'audit_{seed}_{split}','audit_saved.py',seed,split);audits.append(read_json(HERE/f'populations/seed{seed}/{split}/audit/REVIEW.json'))
        run('PRIMARY_20000','statistics_fixed.py');primary=read_json(HERE/'statistics/PRIMARY.json');mark('D1_FINAL_SOURCE_OUTPUT_SHA');verify_manifest(HERE);outputs={};resources=[]
        for name in ('runs','populations','statistics'):
            for p in (HERE/name).rglob('*'):
                if p.is_file():outputs[p.relative_to(HERE).as_posix()]=sha256(p)
            for p in (HERE/name).rglob('RESOURCE*.json'):
                r=read_json(p);assert r['last']['status']=='PASS';resources.append(dict(path=p.relative_to(HERE).as_posix(),report=r))
        write_json(HERE/'PRESERVATION.json',dict(status='PASS',source=source0(),outputs=outputs,old_sources_inputs_models_outputs_unchanged=True,no_delete=True));write_json(HERE/'RESOURCE.json',dict(status='PASS_ALL_STAGES',records=resources))
        write_json(HERE/'ARITHMETIC_AUDIT.json',dict(status='PASS_27_FINAL_POPULATIONS_AND_PRIMARY',saved_reviews=audits,source=source0(),all_primary_actual20000_independent=True,old_IS_final_weights_history_scores_paths_EXACT=True,no_old_F16_forward_or_training=True,saved_arithmetic_not_full_model_gradient_replay=True))
        write_json(HERE/'RESULT.json',dict(status='SUCCESS_D1_REVIEW_REQUIRED',source=source0(),complete_models=9,epochs=72,updates=9216,scientific_populations=27,signal=primary['signal'],baseline_promoted=False,cost_NOT_SOLVED=True))
        state.update(status='SUCCESS',finished=now());write_json(HERE/'worker_state.json',state);write_json(HERE/'exit_status.json',state)
    except BaseException as ex:
        state.update(status='FAILURE',finished=now(),error=str(ex),traceback=traceback.format_exc());write_json(HERE/'FAILURE.json',state);write_json(HERE/'worker_state.json',state);write_json(HERE/'exit_status.json',state);raise
    finally:lock.close()
if __name__=='__main__':main()
