"""Lightweight sole coordinator; serial GPU science then CPU saved review."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,os,subprocess,sys,traceback
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
def now():return datetime.datetime.now().astimezone().isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2)
def state(x):
    tmp=HERE/'WORKER_STATE.json.new'
    with tmp.open('w',encoding='utf-8') as f:json.dump(dict(at=now(),**x),f)
    os.replace(tmp,HERE/'WORKER_STATE.json')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    if (HERE/'WORKER_START.json').exists():raise RuntimeError('No duplicate worker')
    with (HERE/'worker.lock').open('x') as f:f.write(str(os.getpid()))
    write(HERE/'WORKER_START.json',dict(pid=os.getpid(),parent=os.getppid(),started_at=now(),command=sys.argv))
    state(dict(status='RUNNING',stage='Q_COORDINATOR_INITIAL_SOURCE_SHA'))
    assert read(HERE/'PREFLIGHT.json')['status']=='PASS_ACTUAL_COMPLETE_PREFLIGHT'
    assert read(HERE/'BENCHMARK.json')['conservative_full_seconds']<=1800
    for p,s in read(HERE/'source_manifest.json')['files'].items():
        if sha(p)!=s:raise ValueError('LOCKED_SOURCE_CHANGED '+p)
    spec=importlib.util.spec_from_file_location('sole_Q_guard',PROJECT/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py');rg=importlib.util.module_from_spec(spec);spec.loader.exec_module(rg)
    E=read(HERE/'MEMORY_ESTIMATE.json');logs=HERE/'logs';logs.mkdir(exist_ok=False);readings=[]
    for name,program,estimate in [('SCIENCE','train_stage.py',E['GPU_E']),('REVIEW','completion_review.py',E['CPU_E'])]:
        g=rg.Guard(estimate);g.check(mode='startup',force=True);readings.extend(g.observations)
        with (logs/f'{name}_stdout.log').open('x',encoding='utf-8') as out,(logs/f'{name}_stderr.log').open('x',encoding='utf-8') as err:
            child=subprocess.Popen([sys.executable,'-B',str(HERE/program)],cwd=PROJECT,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            state(dict(status='RUNNING',stage=name,child_redirector_pid=child.pid,child_command=[sys.executable,'-B',str(HERE/program)]));code=child.wait()
        if code:raise RuntimeError(name+' exited '+str(code)+'; retain all code/logs/partial outputs')
    dec=read(HERE/'DECISION.json');write(HERE/'RESOURCE_ADMISSION.json',dict(status='PASS',policy='warning_first_v3_prePopen',observations=readings))
    state(dict(status='SUCCESS',stage='Q_SCIENCE_AND_CPU_AUDIT_COMPLETE',decision=dec['status']))
    write(HERE/'WORKER_EXIT.json',dict(status='SUCCESS',exit_code=0,finished_at=now(),decision=dec['status']))
if __name__=='__main__':
    try:main()
    except BaseException:
        x=dict(status='FAILURE',at=now(),traceback=traceback.format_exc());state(x)
        if not (HERE/'WORKER_EXIT.json').exists():write(HERE/'WORKER_EXIT.json',x)
        raise
