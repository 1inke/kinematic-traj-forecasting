"""One actual startup identity receipt; no repeated process polling."""
import json
import os
import subprocess
import time
from datetime import datetime,timedelta
from ud_common import *


if __name__=='__main__':
    if (RUN/'startup_check.json').exists():raise ValueError('STARTUP_ALREADY_PERFORMED')
    for _ in range(30):
        if (RUN/'WORKER_START.json').exists():break
        time.sleep(1)
    result=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(RUN/'check_once.ps1')],capture_output=True,text=True,encoding='utf-8',check=True)
    check=json.loads(result.stdout)
    if check['XML_trigger_count']!=0 or check['worker']['status']!='RUNNING' or check['exit'] is not None:raise ValueError('STARTUP_NOT_RUNNING')
    if not check['process_chain'] or check['process_chain'][0]['ProcessId']!=check['actual_pid']:raise ValueError('ACTUAL_PID_NOT_VERIFIED')
    check['status']='PASS_ONE_STARTUP_ACTUAL_WORKER_CLI_CREATE_PARENT_TASK0'
    write_json(RUN/'startup_check.json',check)
    c=read_json(RUN/'runtime_control.json');start=read_json(RUN/'WORKER_START.json')['at'];dt=datetime.fromisoformat(start)
    c.update(status='STARTED_FOUR_AUTHORIZED_PILOTS',active_worker=True,startup_count=1,startup_at=check['checked_at'],actual_python=check['actual_pid'],
        worker_start=start,query_at=(dt+timedelta(seconds=c['planning_seconds']/2)).isoformat(),
        initial_completion=(dt+timedelta(seconds=c['planning_seconds'])).isoformat())
    tmp=RUN/'runtime_control.json.new'
    with tmp.open('w',encoding='utf-8') as f:json.dump(c,f,ensure_ascii=False,indent=2)
    os.replace(tmp,RUN/'runtime_control.json')
    print(json.dumps(check,ensure_ascii=True),flush=True)
    print('QUERY_AT',c['query_at'],'COMPLETION',c['initial_completion'],flush=True)
