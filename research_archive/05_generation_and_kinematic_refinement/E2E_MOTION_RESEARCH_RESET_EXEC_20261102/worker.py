"""Sole on-demand coordinator; independently survives the conversation."""
import datetime
import importlib.util
import os
import subprocess
import sys
import traceback
from pathlib import Path
from common import HERE, ROOT, now, read_json, write_json, verify


def state(**x):
    import json
    temp=HERE/'WORKER_STATE.json.new'
    value=dict(at=now());value.update(x)
    with temp.open('w',encoding='utf-8') as f:json.dump(value,f,indent=2)
    os.replace(temp,HERE/'WORKER_STATE.json')


def main():
    if (HERE/'WORKER_START.json').exists():raise RuntimeError('NO_DUPLICATE_WORKER')
    with (HERE/'worker.lock').open('x') as f:f.write(str(os.getpid()))
    started=now()
    write_json(HERE/'WORKER_START.json',dict(at=started,pid=os.getpid(),parent=os.getppid(),command=sys.argv))
    state(status='RUNNING',stage='B0_COORDINATOR_INITIAL_SOURCE_SHA')
    verify(read_json(HERE/'source_manifest.json')['files'])
    pre=read_json(HERE/'PRELAUNCH_REVIEW.json')
    if pre['status']!='PASS_C0_B0_ALLOW_SOLE_WORKER' or pre['planning_seconds']>1800:
        raise ValueError('B0_PRELAUNCH_NOT_ADMITTED')
    spec=importlib.util.spec_from_file_location('e2e_rg',ROOT/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py')
    rg=importlib.util.module_from_spec(spec);spec.loader.exec_module(rg)
    E=read_json(HERE/'MEMORY_ESTIMATE.json')
    logs=HERE/'logs';logs.mkdir(exist_ok=False)
    reads=[]
    for name,program,estimate in [('SCIENCE','train_pilot.py',E['GPU_E']),('COMPLETION','completion_review.py',E['CPU_E'])]:
        guard=rg.Guard(estimate);guard.check(mode='startup',force=True);reads.extend(guard.observations)
        with (logs/(name+'_stdout.log')).open('x',encoding='utf-8') as out,(logs/(name+'_stderr.log')).open('x',encoding='utf-8') as err:
            child=subprocess.Popen([sys.executable,'-B',str(HERE/program)],cwd=ROOT,
                stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            state(status='RUNNING',stage=name,child_pid=child.pid,child_command=[sys.executable,'-B',str(HERE/program)])
            code=child.wait()
        if code!=0:raise RuntimeError(name+' failed '+str(code)+'; preserve all current source/logs/partial outputs')
    if read_json(HERE/'RESULT.json')['status']!='SUCCESS_C0_B0_COMPLETE_REVIEWED_PILOT':raise ValueError('RESULT_NOT_COMPLETE')
    write_json(HERE/'RESOURCE_ADMISSION.json',dict(status='PASS_V3_PREPOPEN',observations=reads))
    state(status='SUCCESS',stage='B0_SCIENCE_AND_FULL_SAVED_AUDIT_COMPLETE')
    write_json(HERE/'WORKER_EXIT.json',dict(status='SUCCESS',exit_code=0,finished_at=now(),started_at=started))


if __name__=='__main__':
    try:main()
    except BaseException:
        failure=dict(status='FAILURE',at=now(),traceback=traceback.format_exc())
        state(**failure)
        if not (HERE/'WORKER_EXIT.json').exists():write_json(HERE/'WORKER_EXIT.json',failure)
        raise
