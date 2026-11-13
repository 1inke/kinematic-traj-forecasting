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
    started=datetime.now().astimezone().isoformat();write('worker_state.json',dict(status='RUNNING',pid=os.getpid(),parent=os.getppid(),started_at=started,stage='INITIAL_A0_IMPORT'))
    with (H/'worker_stdout.log').open('a',encoding='utf-8',buffering=1) as out,(H/'worker_stderr.log').open('a',encoding='utf-8',buffering=1) as err:
        sys.stdout=out;sys.stderr=err
        try:
            from run_a0 import main as run
            run()
            import common as c
            result=c.read(H/'RESULT.json');d=result['decision'];lines=['# A0 analytic envelope results','',f"Decision: {d['status']}",'','All20 and bottom4 are relaxed oracle bounds, not safe/model optima. No training or new baseline.','', '|split/group|n|native good|bottom4 UB good|all20 UB good|','|---|---:|---:|---:|---:|']
            for split,r in result['records'].items():
                for g in c.GROUPS:
                    a,b,z=[r['metrics'][p][g] for p in c.PORTS];lines.append(f"|{split}/{g}|{a['n']}|{a['good']}|{b['good']}|{z['good']}|")
            lines+=['',f"FIT-change gains: bottom4 {d['bottom4_gain_pp']:.10g}pp, all20 {d['all20_gain_pp']:.10g}pp. Integer comparisons control the gate, not rounded display.",'','A1 only if fixed-bottom4 FIT bound reaches5pp, after its own actual preflight. B/V never automatically bypass A0/A1. Samecandidate ADE<=.2 and FDE<=.4; untouched16 exact. Existing failed branches and old sources/models/results preserved.']
            (H/'RESULT_ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
            finished=datetime.now().astimezone().isoformat();write('exit_status.json',dict(status='SUCCESS',started_at=started,finished_at=finished));write('worker_state.json',dict(status='SUCCESS',pid=os.getpid(),parent=os.getppid(),started_at=started,finished_at=finished));write('runtime_state.json',dict(status='COMPLETED_REVIEW_READY',decision=d['status'],A1_started=False,B_started=False,new_formal_count=0,report='RESULT_ANALYSIS.md'))
        except BaseException:
            traceback.print_exc()
            if 'common' in sys.modules:
                try:sys.modules['common'].failure('STOP_A0_WORKER')
                except Exception:traceback.print_exc()
            finished=datetime.now().astimezone().isoformat();write('exit_status.json',dict(status='FAILURE',started_at=started,finished_at=finished));write('worker_state.json',dict(status='FAILURE',pid=os.getpid(),parent=os.getppid(),started_at=started,finished_at=finished));raise
if __name__=='__main__':main()
