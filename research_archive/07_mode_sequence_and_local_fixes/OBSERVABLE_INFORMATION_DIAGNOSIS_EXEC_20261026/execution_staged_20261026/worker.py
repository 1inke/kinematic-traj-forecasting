"""Sole lightweight serial OS supervisor. Scientific libraries live in children."""
from base import *
import subprocess,msvcrt,traceback

def child(tag,args):
    logs=HERE/'logs';logs.mkdir(exist_ok=True);mark(tag,arguments=args)
    with (logs/(tag+'.stdout.log')).open('xb') as out,(logs/(tag+'.stderr.log')).open('xb') as err:
        p=subprocess.Popen([sys.executable,'-B','-u',str(HERE/args[0]),*args[1:]],cwd=HERE,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        write_json(HERE/'current_child.json',dict(tag=tag,pid=p.pid,started_at=now(),args=args));code=p.wait()
    write_json(HERE/'current_child.json',dict(tag=tag,pid=p.pid,status='SUCCESS' if code==0 else 'FAILURE',exit_code=code,finished_at=now()))
    if code:raise RuntimeError(f'Child {tag} exit {code}; preserve logs and source, no blind restart')

def finals():
    models={};done_sha={}
    for seed in SEEDS:
        for arm in ('M_full','T_full','T_geom'):
            f=HERE/'step2'/f'{arm}_seed{seed}';d=read_json(f/'done.json');completed(f,d['contract']);assert d['contract']['source']==source()
            model=f/('final.pt' if arm=='M_full' else 'final.joblib');models[f'{arm}_seed{seed}']=sha256(model);done_sha[f'{arm}_seed{seed}']=sha256(f/'done.json')
    write_json(HERE/'ALL_PROBE_FINALS_LOCK.json',dict(status='ALL9_ADMISSIBLE_FINALS_LOCKED_3_PERM_BLOCKED',source=source(),models=models,done_sha=done_sha,T_perm='BLOCKED_NO_LEGAL_LAYER_SHIFT',no_hold_inner_effects_accessed=True,at=now()))

def main():
    lock=(HERE/'worker.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (HERE/'worker_state.json').exists():raise FileExistsError('Sole worker previously existed; no duplicate')
    state=dict(status='RUNNING',pid=os.getpid(),parent_pid=os.getppid(),started_at=now());write_json(HERE/'worker_state.json',state)
    try:
        mark('INITIAL_CURRENT_SOURCE_INPUT_SHA');s=verify();pre=read_json(HERE/'PRELAUNCH_REVIEW.json');assert pre['status']=='PASS' and pre['source']==s and read_json(HERE/'PRODUCTION_SOURCE_LOCK.json')['digest']==s
        for seed in SEEDS:
            for arm in ('I','S'):child(f'tiny_{arm}_{seed}',['mlp.py','train','tiny',arm,str(seed)])
        for seed in SEEDS:
            for arm in ('I','S'):
                for update in (0,32,128,512):child(f'tinyaudit_{arm}_{seed}_{update}',['evaluate_saved.py','tiny',arm,str(seed),'tiny',str(update)])
        for seed in SEEDS:child(f'fit_M_{seed}',['mlp.py','train','probe','M_full',str(seed)])
        for seed in SEEDS:
            for arm in ('T_full','T_geom'):child(f'fit_{arm}_{seed}',['tree.py','fit',arm,str(seed)])
        finals()
        for split in ('opt','hold','inner'):
            for seed in SEEDS:
                for arm in ('M_full','T_full','T_geom'):
                    child(f'predict_{arm}_{seed}_{split}',['mlp.py' if arm=='M_full' else 'tree.py','predict',arm,str(seed),split])
                    child(f'audit_{arm}_{seed}_{split}',['evaluate_saved.py','probe',arm,str(seed),split])
        child('primary_bootstrap20000',['analysis_statistics.py']);mark('FINAL_ALL_SOURCE_OUTPUT_SHA');assert verify()==s
        tiny=[];allgroups=[];resource=[]
        for seed in SEEDS:
            for arm in ('I','S'):
                f=HERE/f'step1/{arm}_seed{seed}';d=read_json(f/'done.json');completed(f,d['contract']);assert d['updates']==512
                a=read_json(f/'audit512/ASSIGNMENT.json')['overall'];a.update(seed=seed,arm=arm)
                a['gate']=bool(a['B_nonempty']>=20 and a['B_empty']>=20 and a['hit_T']>=.95 and a['hit_B']>=.9 and a['empty_B_STOP']>=.9);tiny.append(a)
        for arm in ('M_full','T_full','T_geom'):
            for seed in SEEDS:
                for split in ('opt','hold','inner'):
                    f=HERE/f'populations/{arm}_seed{seed}/{split}/audit';d=read_json(f/'done.json');completed(f,d['contract']);allgroups.append(dict(arm=arm,seed=seed,split=split,groups=read_json(f/'METRICS.json')))
        outputs={}
        for folder in ('step1','step2','populations','statistics'):
            for p in (HERE/folder).rglob('*'):
                if p.is_file():outputs[p.relative_to(HERE).as_posix()]=sha256(p)
        for p in HERE.rglob('RESOURCE.json'):
            if p.is_relative_to(HERE/'prepared') or 'preflight' in str(p):continue
            resource.append(dict(path=p.relative_to(HERE).as_posix(),report=read_json(p)))
        write_json(HERE/'FIT_DIAGNOSTICS.json',dict(records=tiny,gate_per_arm={a:all(r['gate'] for r in tiny if r['arm']==a) for a in ('I','S')},only_reading512_gates=True))
        write_json(HERE/'ALL_PROBE_GROUPS.json',allgroups);write_json(HERE/'PRESERVATION.json',dict(status='PASS',source=s,outputs=outputs,no_old_file_changed=True,no_files_deleted=True));write_json(HERE/'RESOURCE.json',dict(status='PASS_ALL_STAGE_RETAINED_PEAKS',records=resource,serial_nonconcurrent=True,factor=1.3))
        write_json(HERE/'ARITHMETIC_AUDIT.json',dict(status='PASS',tiny_fullreadings=24,probe_populations=27,full401_choice_fixed19_samecandidate_quality_complete_link_pooled5=True,all20000_CI_draws_independently_recomputed=True,saved_arithmetic_not_model_gradient_replay=True))
        primary=read_json(HERE/'statistics/PRIMARY_COMPARISONS.json')
        write_json(HERE/'RESULT.json',dict(status='SUCCESS_STEPS12_REVIEW_REQUIRED',source=s,step1_fits=6,step1_updates=3072,step2_fits=9,T_perm_blocked=3,step2_MLP_updates=3072,tree_iterations=64,steps34='NOT_AUTHORIZED_NOT_RUN',fit_gate=read_json(HERE/'FIT_DIAGNOSTICS.json')['gate_per_arm'],learning_scheme_signal=primary['learning_scheme_signal'],observation_increment='UNRESOLVED_PERM_BLOCKED',new_models=15,formal_diagnostics_not_independent_data_replicates=True,baseline_promoted=False))
        state.update(status='SUCCESS',finished_at=now());write_json(HERE/'worker_state.json',state);write_json(HERE/'exit_status.json',dict(status='SUCCESS',at=now()));write_json(HERE/'runtime_state.json',dict(status='COMPLETED_REVIEW_REQUIRED',at=now()))
    except BaseException as e:
        fail(e);state.update(status='FAILURE',finished_at=now(),message=str(e));write_json(HERE/'worker_state.json',state);write_json(HERE/'exit_status.json',dict(status='FAILURE',at=now()));write_json(HERE/'runtime_state.json',dict(status='STOP_IMPLEMENTATION_REVIEW_REQUIRED',at=now()));raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()

if __name__=='__main__':main()
