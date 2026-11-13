"""One persistent owned child; no real-problem wall limit; synthetic-only watchdog regression. No unrelated kills."""
import sys,subprocess,json,queue,threading,time,os
from pathlib import Path
H=Path(__file__).resolve().parent
ENVROOT=H.parent/'F0_execution_authorized_20261017'
class Solver:
    def __init__(self,guard,folder):
        self.guard=guard;self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True);self.restarts=0;self.process=None;self.start()
    def start(self):
        self.q=queue.Queue();self.err=(self.folder/f'service_{self.restarts}_stderr.log').open('a',encoding='utf-8')
        self.process=subprocess.Popen([str(ENVROOT/'.solver_env/Scripts/python.exe'),'-B','-u',str(H/'solver_service.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW,env=dict(os.environ,PYTHONNOUSERSITE='1'))
        self.guard(self.process.pid)
        def reader(p,q):
            for line in p.stdout:q.put(line)
            q.put(None)
        threading.Thread(target=reader,args=(self.process,self.q),daemon=True).start()
        line=self.q.get(timeout=120);assert line is not None,'SOLVER_SERVICE_EXIT'
        self.ready=json.loads(line);assert self.ready['status']=='READY';self.guard(self.process.pid)
    def close(self):
        if self.process is not None:
            if self.process.poll() is None:self.process.terminate()
            self.process.wait(timeout=10);self.process.stdin.close();self.process.stdout.close();self.err.close();self.process=None
    def solve(self,native,gt,slot,accel,jerk,request_id,fixture_sleep=0):
        if self.process is None:self.start()
        self.guard(self.process.pid)
        req=dict(native=native.tolist(),gt=gt.tolist(),slot=slot,accel_budget=1.01*accel,jerk_budget=1.01*jerk,request_id=request_id,fixture_sleep=fixture_sleep)
        payload=json.dumps(req,separators=(',',':'));start=time.perf_counter();self.process.stdin.write(payload+'\n');self.process.stdin.flush()
        try:
            if fixture_sleep:line=self.q.get(timeout=max(0.,2.-(time.perf_counter()-start)))
            else:
                while True:
                    try:line=self.q.get(timeout=30.);break
                    except queue.Empty:
                        assert self.process.poll() is None,'SOLVER_SERVICE_EXIT';self.guard(self.process.pid)
        except queue.Empty:
            result=dict(status='WATCHDOG_TIMEOUT',primal=None,request_id=request_id,wall_seconds=time.perf_counter()-start)
            self.guard(self.process.pid);self.close();self.restarts+=1;return result
        assert line is not None,'SOLVER_SERVICE_EXIT';result=json.loads(line);assert result['request_id']==request_id
        result['wall_seconds']=time.perf_counter()-start;self.guard(self.process.pid)
        if fixture_sleep and result['wall_seconds']>2:
            result['returned_after_deadline']=True;result['late_primal_not_accepted']=True;result['primal']=None
        return result
