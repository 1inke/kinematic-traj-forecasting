"""Sole OS worker, serial GPU science then CPU saved audit; no auto formal."""
import datetime
import os
import subprocess
import sys
import traceback
import contracts_discrete as c

def now():return datetime.datetime.now().astimezone().isoformat()
def state(v):
    import json
    p=c.ROOT/'WORKER_STATE.json';tmp=p.with_name(p.name+'.new')
    with tmp.open('w',encoding='utf-8') as f:json.dump(c.legacy.clean(dict(at=now(),**v)),f)
    os.replace(tmp,p)

def main():
    if (c.ROOT/'WORKER_START.json').exists():raise RuntimeError('worker start exists; refuse duplicate')
    c.write(c.ROOT/'WORKER_START.json',dict(pid=os.getpid(),parent=os.getppid(),started_at=now(),command=sys.argv))
    pf=c.json_read(c.ROOT/'PREFLIGHT.json');assert pf['status']=='PASS_ACTUAL_PREFLIGHT'
    source=c.json_read(c.ROOT/'source_manifest.json');c.verify(source['files'])
    logs=c.ROOT/'logs';logs.mkdir(exist_ok=False);est=c.json_read(c.ROOT/'MEMORY_ESTIMATE.json');readings=[]
    for name,program,E in [('TINY','run_tiny_discrete.py',est['estimate']),('REVIEW','audit_tiny_discrete.py',est['CPU_estimate'])]:
        g=c.resource(E);g.check(mode='startup',force=True);readings.extend(g.observations)
        with (logs/f'{name}_stdout.log').open('x',encoding='utf-8') as out,(logs/f'{name}_stderr.log').open('x',encoding='utf-8') as err:
            child=subprocess.Popen([sys.executable,'-B',str(c.ROOT/program)],cwd=c.PROJECT,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            state(dict(status='RUNNING',stage=name,child_redirector_pid=child.pid,child_command=[sys.executable,'-B',str(c.ROOT/program)]))
            code=child.wait()
        if code:raise RuntimeError(f'{name} child exited {code}; preserve logs, no automatic restart')
    decision=c.json_read(c.ROOT/'TINY_DECISION.json')['status']
    c.write(c.ROOT/'RESOURCE_ADMISSION.json',dict(policy='warning_first_v3_prePopen',observations=readings))
    state(dict(status='SUCCESS',stage='COMPLETE_TINY_SAVED_AUDIT',decision=decision))
    c.write(c.ROOT/'WORKER_EXIT.json',dict(status='SUCCESS',exit_code=0,finished_at=now(),decision=decision))

if __name__=='__main__':
    try:main()
    except BaseException:
        state(dict(status='FAILURE',stage='PRESERVE_NO_AUTO_RETRY',traceback=traceback.format_exc()))
        if not (c.ROOT/'WORKER_EXIT.json').exists():c.write(c.ROOT/'WORKER_EXIT.json',dict(status='FAILURE',finished_at=now(),traceback=traceback.format_exc()))
        raise
