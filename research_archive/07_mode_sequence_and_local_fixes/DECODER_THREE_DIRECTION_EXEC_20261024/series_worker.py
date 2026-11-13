"""Finite serial series owned by one independent, byte-locked OS worker.

No automated retuning or blind restart. Implementation failures preserve partial
outputs and wake the app's reviewer; scientific negative gates stop that branch
but do not invalidate the other independently authorized directions.
"""
import os,sys,time,subprocess,traceback,hashlib
from datetime import datetime,timezone,timedelta
from pathlib import Path
sys.dont_write_bytecode=True
from common import HERE,ROOT,read_json,write_json,sha256,verify_sources

def stamp():return datetime.now(timezone(timedelta(hours=8))).isoformat()

def stage(label,**kw):
    old=read_json(HERE/'series_state.json') if (HERE/'series_state.json').exists() else {}
    old.update(stage=label,at=stamp(),**kw);write_json(HERE/'series_state.json',old)

def call(label,args):
    stage(label,status='RUNNING',active_worker=True)
    logs=HERE/'series_logs';logs.mkdir(exist_ok=True)
    with (logs/(label+'.stdout.log')).open('xb') as stdout,(logs/(label+'.stderr.log')).open('xb') as stderr:
        proc=subprocess.Popen([sys.executable,'-B','-u',*map(str,args)],cwd=str(ROOT),stdout=stdout,stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW)
        stage(label,active_child_pid=proc.pid,child_args=list(map(str,args)))
        code=proc.wait()
    write_json(logs/(label+'.exit.json'),dict(exit_code=code,finished_at=stamp()))
    if code:raise RuntimeError(f'{label} failed exit={code}; preserve logs and source, no blind retry')

def authorize(package):
    path=HERE/(package+'_EXECUTION_AUTHORIZATION.json')
    if not path.exists():
        write_json(path,dict(package=package,direct_human_instruction='自行跑这三个方向的实验，有问题自己解决。',
            recorded_at=stamp(),scope='2026-10-24 finite three-direction design; independent serial packages only; A conditional gate unchanged'))
    return path

def package_run(package,execution,screen=None):
    args=[HERE/'training.py',package,'--execution',execution,'--prepare']
    if screen is not None:args+=['--screen',screen]
    call(package+'_PREPARE',args)
    if package=='C2':
        call('C2_FULL_PIPELINE_PROFILE',[HERE/'profile_full_pipeline.py','--execution',execution,'--estimated-peak-gib','6'])
    call(package+'_ACTUAL_PREFLIGHT',[HERE/'training.py',package,'--execution',execution,'--preflight','--estimated-peak-gib','6'])
    pre=read_json(execution/'PREFLIGHT.json')
    if pre['status']!='PASS':raise ValueError((package,'ACTUAL_PREFLIGHT_NOT_PASS',pre['status']))
    call(package+'_FORMAL_TRAIN_SAVE_AUDIT',[HERE/'training.py',package,'--execution',execution,
        '--train','--authorization',authorize(package),'--estimated-peak-gib','6'])
    result=read_json(execution/'RESULT.json')
    if result['status']!='SUCCESS':raise ValueError((package,'EXECUTION_NOT_SUCCESS'))
    for run in result['runs'].values():
        folder=execution/f"{run['contract']['arm']}_seed{run['contract']['seed']}"
        for rel,expected in run['outputs'].items():
            if sha256(folder/rel)!=expected:raise ValueError('COMPLETED_OUTPUT_SHA_CHANGED '+str(folder/rel))
    return result['decision']

def main():
    import msvcrt
    start=stamp();timer=time.perf_counter();lock=(HERE/'series_worker.lock').open('a+b')
    if (HERE/'worker_state.json').exists():raise FileExistsError('Prior worker state exists: no duplicate launch')
    if (HERE/'series_worker.lock').stat().st_size==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    worker=dict(status='RUNNING',pid=os.getpid(),parent_pid=os.getppid(),started_at=start)
    write_json(HERE/'worker_state.json',worker)
    try:
        source=verify_sources(HERE/'series_source_manifest.json')
        pre=read_json(HERE/'PRELAUNCH.json');assert pre['status'].startswith('PASS_ACTUAL')
        assert read_json(HERE/'prelaunch_encoder/ENCODER_SMOKE.json')['status']=='PASS'
        spec=read_json(HERE/'series_spec.json')
        query=datetime.fromisoformat(start)+timedelta(minutes=spec['initial_planning_minutes']/2)
        deadline=datetime.fromisoformat(start)+timedelta(minutes=spec['initial_planning_minutes'])
        stage('INITIAL_CURRENT_SOURCE_SHA',status='RUNNING',active_worker=True,started_at=start,
            query_at=query.isoformat(),initial_completion_at=deadline.isoformat(),completed_stages=[],
            first_progress_query_count=0,source_digest=source)
        A=HERE/'direction_A_time_basis/screen_execution_20261024'
        B=HERE/'direction_B_scene_prior/probe_execution_20261024'
        C=HERE/'direction_C_set_mechanisms'
        call('A_SCREEN',[HERE/'direction_A_time_basis/screen.py','--output',A,'--estimated-peak-gib','3'])
        ar=read_json(A/'RESULT.json');stage('A_SCREEN_COMPLETE',completed_stages=['A_SCREEN'])
        call('B_PROBE',[HERE/'direction_B_scene_prior/probe.py','--records',HERE/'direction_B_scene_prior/records_20261024.json',
            '--output',B,'--estimated-peak-gib','3'])
        br=read_json(B/'RESULT.json');stage('B_PROBE_COMPLETE',completed_stages=['A_SCREEN','B_PROBE'])
        train_results={}
        if ar['selected_basis'] is not None:
            train_results['A']=package_run('A',HERE/'direction_A_time_basis/train_execution_20261024',A)
        else:train_results['A']=dict(status='NOT_TRIGGERED_SCREEN_GATE',conclusion=ar['conclusion'])
        train_results['C1']=package_run('C1',C/'C1_execution_20261024')
        train_results['C2']=package_run('C2',C/'C2_execution_20261024')
        stage('FINAL_SOURCE_OUTPUT_SHA',active_child_pid=None)
        assert verify_sources(HERE/'series_source_manifest.json')==source
        summary=dict(status='SUCCESS_EXECUTION_REVIEW_REQUIRED',finished_at=stamp(),elapsed_seconds=time.perf_counter()-timer,
            A_screen=ar['decisions'],A_selected_basis=ar['selected_basis'],B_decision=br['decisions'],
            B_conclusion=br['conclusion'],training_decisions=train_results,source_digest=source,
            formal_experiments_completed=2+int(ar['selected_basis'] is not None),
            no_baseline_reset=True,no_further_variants_or_automatic_training=True)
        write_json(HERE/'SERIES_RESULT.json',summary)
        write_json(HERE/'exit_status.json',dict(status='SUCCESS',at=stamp()))
        worker.update(status='SUCCESS',finished_at=stamp());write_json(HERE/'worker_state.json',worker)
        stage('COMPUTATION_COMPLETE_REVIEW_REQUIRED',status='COMPLETED',active_worker=False,
            completed_stages=['A_SCREEN','B_PROBE','A_TRAIN_CONDITIONAL','C1','C2'],
            elapsed_seconds=time.perf_counter()-timer)
    except BaseException as error:
        failure=dict(status='FAILURE',type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),at=stamp(),
            stage=read_json(HERE/'series_state.json') if (HERE/'series_state.json').exists() else None,
            preserve_all_sources_models_logs_partial_outputs=True)
        write_json(HERE/'FAILURE.json',failure);write_json(HERE/'exit_status.json',dict(status='FAILURE',at=stamp()))
        worker.update(status='FAILURE',finished_at=stamp());write_json(HERE/'worker_state.json',worker)
        stage('IMPLEMENTATION_OR_RUNTIME_FAILURE_REVIEW_REQUIRED',status='FAILURE',active_worker=False)
        raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()

if __name__=='__main__':
    from contextlib import redirect_stdout,redirect_stderr
    with (HERE/'worker_stdout.log').open('x',encoding='utf-8') as out,(HERE/'worker_stderr.log').open('x',encoding='utf-8') as err:
        with redirect_stdout(out),redirect_stderr(err):main()
