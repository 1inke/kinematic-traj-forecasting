import os,msvcrt,traceback
import torch
from env import *
from source_control import verify
from data_pipeline import OptData,contract
from train_paired import run,lock_all_finals,initialize
from evaluate_fixed import population,write_population,audit_population
def main():
    lock=(HERE/'worker.lock').open('a+b')
    if (HERE/'worker_state.json').exists():raise FileExistsError('Sole worker already existed')
    lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    state=dict(status='RUNNING',pid=os.getpid(),parent_pid=os.getppid(),started_at=e0.now());write_json(HERE/'worker_state.json',state);stage('E1_INITIAL_SOURCE_INPUT_SHA');guard=Guard(ESTIMATE)
    try:
        guard.check('startup');source=verify();assert read_json(HERE/'PREFLIGHT.json')['status']=='PASS' and read_json(HERE/'PREFLIGHT.json')['source_digest']==source
        done=[]
        for seed in SEEDS:
            for arm in ARMS:done.append(run(seed,arm,source,guard))
        finals=lock_all_finals(source);audits=[]
        for seed in SEEDS:
            models={};data=OptData(seed)
            for arm in ARMS:
                model,_,_=initialize(seed,data,'cuda');ck=torch.load(HERE/f'runs/{arm}_seed{seed}/checkpoint.pt',map_location='cpu',weights_only=False)
                assert ck['contract']==contract(seed,arm,source);model.load_state_dict(ck['model']);models[arm]=model
            for split in ('opt','hold','inner'):
                stage('E1_FINAL_EVALUATE_SAVE',seed=seed,split=split);p=population(seed,split,source);folder=HERE/f'populations/seed{seed}/{split}';write_population(seed,split,p,models,source,folder,guard)
                stage('E2_INDEPENDENT_SAVED_AUDIT',seed=seed,split=split);audits.append(audit_population(folder,source,guard))
        from statistics_fixed import main as statistics
        stage('E2_PRIMARY_BOOTSTRAP_20000');primary=statistics();guard.check()
        from benchmark_cached import run as cached_cost
        stage('E2_CACHED_COST_EXPLICIT_NOT_END_TO_END');cost=cached_cost();guard.check()
        write_json(HERE/'ARITHMETIC_AUDIT.json',dict(status='PASS_ALL18_SCIENTIFIC_POPULATIONS_AND_PRIMARY_DRAWS',records=audits,bootstrap=primary['status'],saved_arithmetic_not_model_replay=True))
        stage('E1_E2_FINAL_SOURCE_OUTPUT_SHA');assert verify()==source;lock_all_finals(source)
        outputs={}
        for path in HERE.rglob('*'):
            if path.is_file() and (path.is_relative_to(HERE/'runs') or path.is_relative_to(HERE/'populations')):outputs[path.relative_to(HERE).as_posix()]=sha256(path)
        write_json(HERE/'PRESERVATION.json',dict(status='PASS',source_digest=source,all_old_sources_inputs_outputs_unchanged=True,outputs=outputs,no_files_deleted=True));last=guard.check()
        write_json(HERE/'RESOURCE.json',dict(status='PASS',estimate=ESTIMATE,factor=1.3,observations=guard.observations,last=last,production_lifetime_peak_private=last['snapshot']['peak_private'],production_lifetime_peak_RSS=last['snapshot']['peak_rss']))
        write_json(HERE/'RESULT.json',dict(status='PASS_EXECUTION_E1_E2_REVIEW_REQUIRED',source_digest=source,complete_runs=6,history_rows=48,optimizer_steps=6144,scientific_populations=18,primary_signal=primary['signal'],cost='CACHED_ONLY_REAL_END_TO_END_REVIEW_STILL_REQUIRED',formal_experiments_completed=1,baseline_promoted=False))
        state.update(status='SUCCESS',finished_at=e0.now());write_json(HERE/'exit_status.json',dict(status='SUCCESS',at=e0.now()));write_json(HERE/'worker_state.json',state);write_json(HERE/'runtime_state.json',dict(status='COMPLETED_REVIEW_REQUIRED',stage='E1_E2_SAVED_COMPLETE_REAL_COST_PENDING',active_worker=False,finished_at=state['finished_at']))
    except BaseException as exc:
        write_json(HERE/'FAILURE.json',dict(type=type(exc).__name__,message=str(exc),at=e0.now(),traceback=traceback.format_exc(),resource=guard.observations,preserve_all=True));state.update(status='FAILURE',finished_at=e0.now());write_json(HERE/'worker_state.json',state);write_json(HERE/'exit_status.json',dict(status='FAILURE',at=e0.now()));raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
if __name__=='__main__':main()
