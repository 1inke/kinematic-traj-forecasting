"""Default PLAN only. Training, resume, held evaluation require later consent."""
import argparse, time, traceback
from pathlib import Path
from joint_common import *

def prepare_execution(execution,preflight):
    package=verify_package();verify(package['dependencies']);p=read(preflight)
    if p.get('status')!='PASS_REAL_JOINT_GPU_RESUME_SAVED_PREFLIGHT' or p['script_manifest_sha']!=sha(HERE/'SCRIPT_MANIFEST.json'):
        raise ValueError('CURRENT_REAL_PREFLIGHT_REQUIRED')
    verify(p['source_files']);verify(p['dependency_files'])
    execution=Path(execution).resolve()
    if not execution.is_relative_to(HERE/'executions'):raise ValueError('NEW_EXECUTION_NAMESPACE_REQUIRED')
    if not execution.exists():
        execution.mkdir(parents=True,exist_ok=False)
        files=dict(package['files']);files.update(package['dependencies']);files[str(Path(preflight).resolve())]=sha(preflight)
        write(execution/'SOURCE_LOCK.json',dict(at=now(),files=files,preflight_path=str(Path(preflight).resolve()),preflight_sha=sha(preflight)))
    lock=read(execution/'SOURCE_LOCK.json');verify(lock['files'])
    return execution,p,lock

def checkpoint(folder,trainer,tag):
    file=folder/(tag+'.pt');save_checkpoint(file,trainer.snapshot())
    write(folder/(tag+'.receipt.json'),dict(path=str(file),sha=sha(file),updates=trainer.updates,
        epoch=trainer.epoch,complete_cursor=trainer.cursor==len(trainer.indices)))

def train(execution,preflight,family='all',execute=False):
    authorize('train',execute)
    root,pre,lock=prepare_execution(execution,preflight)
    import torch
    from joint_train import Trainer
    from joint_models import parameter_inventory
    # OS lock is released on exit, but its file is NEVER deleted.
    import msvcrt
    with (root/'worker.lock').open('a+b') as handle:
        if handle.tell()==0:handle.write(b'1');handle.flush()
        handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        try:
            torch.set_num_threads(1);guard=resource_module().Guard(pre['GPU_E']);guard.check(force=True)
            data=cache('fit');target,_=labels('fit',data['ids']);source=sha(root/'SOURCE_LOCK.json')
            started=time.perf_counter();write(root/f'START_{time.time_ns()}.json',dict(at=now(),pid=os.getpid(),family=family))
            paired={}
            for f in protocol()['families']:
                if family!='all' and f!=family:continue
                for seed in (42,43,44):
                    for arm in protocol()['arms']:
                        name=f'{f}_{arm}_seed{seed}';folder=root/'runs'/name
                        if (folder/'done.json').exists():
                            done=read(folder/'done.json');verify(done['files'])
                            if done['source_sha']!=source or done['updates']!=13580:raise ValueError('INCOMPATIBLE_COMPLETED_RUN')
                            continue # NEVER retrain a completed compatible model.
                        trainer=Trainer(f,arm,seed,data,target,source)
                        if not folder.exists():
                            folder.mkdir(parents=True,exist_ok=False)
                            save_checkpoint(folder/'initial.pt',trainer.snapshot())
                            write(folder/'initial_binding.json',dict(path=str(folder/'initial.pt'),sha=sha(folder/'initial.pt'),source_sha=source))
                            write(folder/'contract.json',trainer.contract)
                            write(folder/'parameters.json',parameter_inventory(trainer.model))
                        else:
                            exact(read(folder/'contract.json'),trainer.contract)
                            receipts=list(folder.glob('*.receipt.json'))
                            if receipts:
                                latest=max((read(p) for p in receipts),key=lambda d:d['updates'])
                                trainer.restore(trusted_checkpoint(latest['path'],latest['sha']))
                            else:
                                # Initial-only retry consumes NO training noise beyond the saved initial RNG.
                                binding=read(folder/'initial_binding.json')
                                if binding['source_sha']!=source:raise ValueError('INITIAL_SOURCE_BINDING')
                                trainer.restore(trusted_checkpoint(binding['path'],binding['sha']))
                        try:
                            while trainer.updates<13580:
                                row=trainer.step();guard.check()
                                if trainer.cursor==len(trainer.indices):
                                    trainer.check_frozen();checkpoint(folder,trainer,f'epoch{trainer.epoch:02}')
                                    print(name,row,flush=True)
                        except MemoryError:
                            checkpoint(folder,trainer,f'resource_stop_u{trainer.updates}_{time.time_ns()}')
                            write(root/f'RESOURCE_STOP_{time.time_ns()}.json',dict(at=now(),observations=guard.observations))
                            raise
                        trainer.check_frozen();save_checkpoint(folder/'final.pt',trainer.snapshot())
                        write(folder/'history.json',trainer.history);write(folder/'epoch_orders.json',trainer.orders)
                        binding=read(folder/'initial_binding.json')
                        initial=trusted_checkpoint(binding['path'],binding['sha'])['model']
                        key=(f,seed)
                        if key in paired:exact(paired[key][0],initial);exact(paired[key][1],trainer.orders)
                        else:paired[key]=(initial,cpu(trainer.orders))
                        write(folder/'done.json',dict(status='PASS_FINAL20_TRAINING_NOT_EVALUATION',source_sha=source,
                            epochs=20,updates=13580,files=seal(folder.iterdir())))
                        del trainer
            # Recheck pairing even when some runs were SHA-reused after interruption.
            complete=all((root/'runs'/n/'done.json').is_file() for n in run_names())
            if complete:
                finals={}
                for f in ('existing','cvae'):
                    for seed in (42,43,44):
                        states=[];orders=[]
                        for arm in protocol()['arms']:
                            n=f'{f}_{arm}_seed{seed}';p=root/'runs'/n;d=read(p/'done.json');verify(d['files'])
                            states.append(trusted_checkpoint(p/'initial.pt',d['files'][str((p/'initial.pt').resolve())])['model'])
                            orders.append(read(p/'epoch_orders.json'));finals[n]=d
                        exact(states[0],states[1]);exact(orders[0],orders[1])
                if not (root/'ALL12_FINALS_LOCK.json').exists():
                    write(root/'ALL12_FINALS_LOCK.json',dict(at=now(),status='ALL12_FINALS_SEALED_BEFORE_DEV_GT',runs=finals))
            write(root/f'TRAIN_EXIT_{time.time_ns()}.json',dict(at=now(),all12_complete=complete,
                elapsed=time.perf_counter()-started,resource=guard.observations,
                next='evaluate explicitly' if complete else 'other family training pending; dev locked'))
        except BaseException as e:
            write(root/f'FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(e),traceback=traceback.format_exc(),retain_all=True))
            raise
        finally:
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)

def evaluate_all(execution,execute=False):
    authorize('evaluate',execute);root=Path(execution).resolve()
    if not root.is_relative_to(HERE/'executions'):raise ValueError('EXECUTION_NAMESPACE')
    lock=read(root/'SOURCE_LOCK.json');verify(lock['files']);finals=read(root/'ALL12_FINALS_LOCK.json')
    if set(finals['runs'])!=set(run_names()):raise ValueError('ALL12_FINAlS_REQUIRED')
    for d in finals['runs'].values():verify(d['files'])
    import torch
    from joint_train import predict
    from joint_models import make
    from joint_evaluate import evaluate
    from independent_math import reconstructor
    from joint_statistics import compare,protection
    pre=read(lock['preflight_path']);guard=resource_module().Guard(pre['GPU_E']);guard.check(force=True)
    outputs=[]
    # One exclusive OS lock for evaluation too; never overlap with another launcher.
    import msvcrt
    with (root/'worker.lock').open('r+b') as handle:
        msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        try:
            for split in ('fit','dev'):
                data=cache(split)
                for name in ['native']+run_names():
                    p=root/'populations'/name/split
                    if (p/'done.json').exists():verify(read(p/'done.json')['files']);outputs.append(p/'done.json');continue
                    if p.exists():raise RuntimeError('PARTIAL_POPULATION_REQUIRES_NEW_RECOVERY_NAMESPACE_NO_OVERWRITE '+str(p))
                    decode=None
                    if name=='native':paths=data['native'].astype(np.float64);control=np.empty((len(paths),0))
                    else:
                        f,rest=name.split('_',1);seed=int(name.rsplit('seed',1)[1])
                        d=finals['runs'][name];file=root/'runs'/name/'final.pt'
                        state=trusted_checkpoint(file,d['files'][str(file.resolve())]);model=make(f,seed,'cuda')
                        model.load_state_dict(state['model'],strict=True);paths,control=predict(model,data,'cuda',guard)
                        decode=reconstructor(model,f);del model,state
                    evaluate(p,paths,control,data,split,execution=root,guard=guard,reconstruct=decode);outputs.append(p/'done.json')
            compare(root);protection(root);verify(lock['files']);guard.check(force=True)
            write(root/'RESOURCE.json',dict(status='RECORDED_WARNING_FIRST_V3',estimate=pre['GPU_E'],observations=guard.observations))
            write(root/'RESULT.json',dict(status='JOINT_TWO_FAMILIES_SAVED_SCIENCE_COMPLETE_COST_NOT_MEASURED',at=now(),
                new_runs=12,epochs=240,updates=162960,populations=26,no_automatic_baseline_upgrade=True,
                raw_end_to_end_cost='NOT_MEASURED_NOT_DEPLOYMENT_PASS',scene_joint='NOT_TRIGGERED'))
            files=dict(lock['files'])
            for p in outputs:files.update(read(p)['files']);files[str(p)]=sha(p)
            for d in finals['runs'].values():files.update(d['files'])
            files.update(seal([root/n for n in ('ALL12_FINALS_LOCK.json','PRIMARY_COMPARISON.json','BOOTSTRAP_DRAWS.npz','PROTECTION_GATES.json','RESOURCE.json','RESULT.json')]))
            write(root/'FINAL_SCIENCE_SEAL.json',dict(status='SAVED_SCIENCE_SOURCE_AND_OUTPUT_SHA_NOT_RAW_COST',files=files))
        finally:
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)

def cli(family_default='all'):
    p=argparse.ArgumentParser();p.add_argument('action',choices=('plan','train','evaluate'),nargs='?',default='plan')
    p.add_argument('--family',choices=('all','existing','cvae'),default=family_default)
    p.add_argument('--execution');p.add_argument('--preflight');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.action=='plan':print(json.dumps(dict(control=read(HERE/'control.json'),protocol=protocol()),ensure_ascii=False,indent=2));return
    if not a.execution:p.error('--execution is required')
    if a.action=='train':
        if not a.preflight:p.error('--preflight is required')
        train(a.execution,a.preflight,a.family,a.execute)
    else:evaluate_all(a.execution,a.execute)

if __name__=='__main__':cli()
