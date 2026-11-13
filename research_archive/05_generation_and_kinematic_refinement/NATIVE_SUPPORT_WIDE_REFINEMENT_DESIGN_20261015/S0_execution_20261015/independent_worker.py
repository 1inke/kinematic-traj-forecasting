import os,sys,json,traceback,msvcrt
from pathlib import Path
from datetime import datetime
H=Path(__file__).resolve().parent

def write(p,z):
    path=H/p;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(z,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)

def main():
    lock=(H/'worker.lock').open('a+b');lock.seek(0)
    if not (H/'worker.lock').stat().st_size:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    started=datetime.now().astimezone().isoformat();write('worker_state.json',dict(status='RUNNING',pid=os.getpid(),parent=os.getppid(),started_at=started,stage='INITIAL_S0_IMPORT'))
    with (H/'worker_stdout.log').open('a',encoding='utf-8',buffering=1) as out,(H/'worker_stderr.log').open('a',encoding='utf-8',buffering=1) as err:
        sys.stdout=out;sys.stderr=err
        try:
            from run_s0 import main as run
            run()
            import common as c
            result=c.read(H/'RESULT.json');d=result['decision'];lines=['# S0 native-support safe opportunity results','',f"Decision: {d['status']}",'','Privileged GT-direction witness under registered conservative map, not global safe-space optimum, deployable output or teacher. No training.','', '|split/group|n|native good|safe4 good|safe20 good|','|---|---:|---:|---:|---:|']
            for split,r in result['records'].items():
                for g in c.GROUPS:
                    a,b,z=[r['metrics'][p][g] for p in c.PORTS];lines.append(f"|{split}/{g}|{a['n']}|{a['good']}|{b['good']}|{z['good']}|")
            lines+=['',f"FITchange integer gate {d['integer_lhs']} >= {d['integer_rhs']}: {d['coverage_pass']}",'','S1 only after own actual preflight if S0 passes. Failure stops this map without radius/margin/width/LR sweeps. Native-good zero lost and full literal motion audit checked. Inline further analysis pending; no causal or generalization claim.']
            (H/'RESULT_ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');c.write('RESULT_ANALYSIS.json',dict(status='SAVED_REVIEW_READY',decision=d,full_result='RESULT.json',independent_audit='ARITHMETIC_AUDIT.json',not_external_review=True))
            finished=datetime.now().astimezone().isoformat();write('exit_status.json',dict(status='SUCCESS',started_at=started,finished_at=finished));write('worker_state.json',dict(status='SUCCESS',pid=os.getpid(),parent=os.getppid(),started_at=started,finished_at=finished));write('runtime_state.json',dict(status='COMPLETED_REVIEW_READY',decision=d['status'],S1_started=False,new_formal_count=0,report='RESULT_ANALYSIS.md'))
        except BaseException:
            traceback.print_exc()
            if 'common' in sys.modules:
                try:sys.modules['common'].failure('STOP_S0_WORKER')
                except Exception:traceback.print_exc()
            finished=datetime.now().astimezone().isoformat();write('exit_status.json',dict(status='FAILURE',started_at=started,finished_at=finished));write('worker_state.json',dict(status='FAILURE',pid=os.getpid(),parent=os.getppid(),started_at=started,finished_at=finished));raise

if __name__=='__main__':main()
