"""Sole serial OS coordinator: four admitted fresh pilots, then saved review."""
import os
import sys
import json
import subprocess
import traceback
from ud_common import *


def state(**items):
    value=dict(at=now(),**items);temp=RUN/'WORKER_STATE.json.new'
    with temp.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    os.replace(temp,RUN/'WORKER_STATE.json')


def control(**items):
    data=read_json(RUN/'runtime_control.json');data.update(items);data['at']=now()
    temp=RUN/'runtime_control.json.new'
    with temp.open('w',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
    os.replace(temp,RUN/'runtime_control.json')


def main():
    if (RUN/'WORKER_START.json').exists():raise RuntimeError('NO_DUPLICATE_WORKER')
    with (RUN/'worker.lock').open('x') as f:f.write(str(os.getpid()))
    start=now();write_json(RUN/'WORKER_START.json',dict(at=start,pid=os.getpid(),parent=os.getppid(),command=sys.argv))
    state(status='RUNNING',stage='UD_INITIAL_SOURCE_SHA')
    manifest=read_json(RUN/'source_manifest.json');verify(manifest['files'])
    pre=read_json(RUN/'PRELAUNCH_REVIEW.json')
    if pre['status']!='PASS_AUTHORIZED_ALL4_PILOTS' or pre['planning_seconds']>1800:raise ValueError('UD_NOT_ADMITTED')
    import importlib.util
    path=ROOT/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py'
    spec=importlib.util.spec_from_file_location('worker_guard_v3',path);rg=importlib.util.module_from_spec(spec);spec.loader.exec_module(rg)
    estimates=read_json(RUN/'MEMORY_ESTIMATE.json');logs=RUN/'logs';logs.mkdir(exist_ok=False);admissions=[]
    jobs=[(arm,'train_arm.py',['--arm',arm],estimates['GPU_E']) for arm in ARMS]+[('FINAL_REVIEW','compare_ud.py',[],estimates['CPU_E'])]
    for name,program,args,E in jobs:
        guard=rg.Guard(E);guard.check(mode='startup',force=True);admissions.extend(guard.observations)
        command=[sys.executable,'-B',str(RUN/program)]+args
        with (logs/(name+'_stdout.log')).open('x',encoding='utf-8') as out,(logs/(name+'_stderr.log')).open('x',encoding='utf-8') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            state(status='RUNNING',stage=name,child_pid=child.pid,child_command=command)
            code=child.wait()
        if code!=0:raise RuntimeError(name+' failed '+str(code)+'; preserve sources/logs/checkpoints, no automatic repeat')
    if read_json(RUN/'RESULT.json')['status']!='SUCCESS_AUTHORIZED_ALL4_UD_PILOTS_REVIEWED':raise ValueError('INCOMPLETE_RESULT')
    verify(manifest['files']);write_json(RUN/'RESOURCE_ADMISSION.json',dict(status='PASS_V3_PREPOPEN',observations=admissions))
    control(status='ANALYZED_FOUR_PILOTS_NEXT_CONDITIONAL_REVIEW',active_worker=False,next_query=None)
    state(status='SUCCESS',stage='UD_FOUR_PILOTS_AND_FULL_SAVED_REVIEW_COMPLETE')
    write_json(RUN/'WORKER_EXIT.json',dict(status='SUCCESS',exit_code=0,started_at=start,finished_at=now()))
    write_json(RUN/'FINAL_SEAL.json',dict(status='PASS',files=seal([RUN/n for n in ('RESULT.json','PRESERVATION.json','RESOURCE_ADMISSION.json','WORKER_EXIT.json','completion_review/REVIEW.json')])) )


if __name__=='__main__':
    try:main()
    except BaseException:
        failure=dict(status='FAILURE',at=now(),traceback=traceback.format_exc())
        state(**failure)
        if not (RUN/'WORKER_EXIT.json').exists():write_json(RUN/'WORKER_EXIT.json',failure)
        control(status='STOP_FAILURE_PRESERVE_ALL',active_worker=False,next_query=None)
        raise
