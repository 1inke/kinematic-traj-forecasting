"""Exactly one startup identity check after the sole worker creates its record."""
import json
import subprocess
import time
from datetime import datetime, timedelta
from common import *


if __name__=='__main__':
    if (HERE/'startup_check.json').exists():raise ValueError('STARTUP_ALREADY_PERFORMED')
    for _ in range(30):
        if (HERE/'WORKER_START.json').exists():break
        time.sleep(1)
    result=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(HERE/'check_once.ps1')],capture_output=True,text=True,encoding='utf-8',check=True)
    check=json.loads(result.stdout)
    if check['XML_trigger_count']!=0:raise ValueError('TASK_TRIGGER_MISMATCH')
    if check['worker']['status']!='RUNNING' or check['exit'] is not None:raise ValueError('STARTUP_NOT_RUNNING')
    if not check['process_chain'] or check['process_chain'][0]['ProcessId']!=check['actual_pid']:
        raise ValueError('ACTUAL_PID_NOT_VERIFIED')
    check['status']='PASS_ONE_STARTUP_ACTUAL_WORKER_CLI_CREATE_PARENT_TASK0'
    write_json(HERE/'startup_check.json',check)
    control=read_json(HERE/'runtime_control.json')
    start=read_json(HERE/'WORKER_START.json')['at'];dt=datetime.fromisoformat(start)
    control.update(status='STARTED_C0_COMPLETE_B0_RUNNING',startup_count=1,
        startup_at=check['checked_at'],actual_python=check['actual_pid'],worker_start=start,
        query_at=(dt+timedelta(seconds=control['planning_seconds']/2)).isoformat(),
        initial_completion=(dt+timedelta(seconds=control['planning_seconds'])).isoformat())
    # Control metadata only, not a scientific source file.
    temp=HERE/'runtime_control.json.new'
    with temp.open('w',encoding='utf-8') as f:json.dump(control,f,ensure_ascii=False,indent=2)
    import os
    os.replace(temp,HERE/'runtime_control.json')
    print(json.dumps(check,ensure_ascii=True),flush=True)
    print('QUERY_AT',control['query_at'],'COMPLETION',control['initial_completion'],flush=True)
