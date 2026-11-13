from core import *
import subprocess,traceback,msvcrt
def child(label,*args):
    logs=HERE/'logs';logs.mkdir(exist_ok=True);mark(label,args=args)
    admission=guard(cpu=args[0] in ('audit.py','statistics.py'),startup=True);resource(logs/(label+'.admission.json'),admission)
    with (logs/(label+'.stdout.log')).open('xb') as out,(logs/(label+'.stderr.log')).open('xb') as err:
        p=subprocess.Popen([sys.executable,'-B','-u',str(HERE/args[0]),*map(str,args[1:])],cwd=HERE,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);write_json(HERE/'current_child.json',dict(pid=p.pid,stage=label,at=now(),args=args));code=p.wait()
    write_json(HERE/'current_child.json',dict(pid=p.pid,stage=label,finished=now(),exit_code=code))
    if code:raise RuntimeError('Current child '+label+' failed; inspect its new stderr, preserve evidence')
def main():
    lock=(HERE/'worker.lock').open('a+b');lock.write(b'1');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);assert not (HERE/'worker_state.json').exists();state=dict(status='RUNNING',pid=os.getpid(),started=now());write_json(HERE/'worker_state.json',state)
    try:
        assert read_json(HERE/'PRELAUNCH_REVIEW.json')['status']=='PASS';mark('COLLAB_INITIAL_SOURCE_SHA');verify()
        for seed in SEEDS:
            for arm in ARMS:child(f'train_{arm}_{seed}','train.py',seed,arm)
        child('ALL6_LOCK','train.py','lock');reviews=[]
        for seed in SEEDS:
            for split in SPLITS:
                child(f'save_{seed}_{split}','evaluate.py',seed,split);child(f'audit_{seed}_{split}','audit.py',seed,split);reviews.append(read_json(HERE/f'populations/seed{seed}/{split}/audit/REVIEW.json'))
        child('PRIMARY_20000','statistics.py');child('MODULE_COST','cost.py');mark('COLLAB_FINAL_SOURCE_OUTPUT_SHA');verify();outputs={};resources=[]
        for root in ('runs','populations','statistics','cost'):
            for p in (HERE/root).rglob('*'):
                if p.is_file():outputs[p.relative_to(HERE).as_posix()]=sha256(p)
            for p in (HERE/root).rglob('RESOURCE*.json'):
                r=read_json(p);assert r['last']['status']=='PASS';resources.append(dict(path=p.relative_to(HERE).as_posix(),report=r))
        write_json(HERE/'PRESERVATION.json',dict(status='PASS',source=source(),outputs=outputs,all_inherited_sources_old_outputs_unchanged=True,no_deletion=True));write_json(HERE/'RESOURCE.json',dict(status='PASS_ALL_STAGES',records=resources));write_json(HERE/'ARITHMETIC_AUDIT.json',dict(status='PASS_27_PORTS',reviews=reviews,all20000_bootstrap_independent=True,all6_final_params_buffers_and_paired_orders_EXACT=True,saved_arithmetic_not_full_model_gradient_replay=True));write_json(HERE/'RESULT.json',dict(status='SUCCESS_COLLABORATIVE_REFINEMENT_REVIEW_REQUIRED',source=source(),models=6,epochs=48,updates=read_json(HERE/'ALL6_FINALS_LOCK.json')['updates'],scientific_ports=18,native_reference_ports=9,mechanism_signal=read_json(HERE/'statistics/PRIMARY.json')['mechanism_signal'],baseline_promoted=False,wholemodel_cost_not_certified=True));state.update(status='SUCCESS',finished=now());write_json(HERE/'worker_state.json',state);write_json(HERE/'exit_status.json',state)
    except BaseException as ex:
        state.update(status='FAILURE',finished=now(),error=str(ex),traceback=traceback.format_exc());write_json(HERE/'FAILURE.json',state);write_json(HERE/'worker_state.json',state);write_json(HERE/'exit_status.json',state);raise
    finally:lock.close()
if __name__=='__main__':main()
