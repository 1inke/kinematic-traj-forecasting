import os,sys,json,traceback,msvcrt
from pathlib import Path
from datetime import datetime
H=Path(__file__).resolve().parent
def write(name,z):
    p=H/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(z,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
def main():
    lock=(H/'worker.lock').open('a+b');lock.seek(0)
    if not (H/'worker.lock').stat().st_size:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);started=datetime.now().astimezone().isoformat();write('worker_state.json',dict(status='RUNNING',pid=os.getpid(),parent=os.getppid(),started_at=started,stage='INITIAL_U0_IMPORT'))
    with (H/'worker_stdout.log').open('a',encoding='utf-8',buffering=1) as out,(H/'worker_stderr.log').open('a',encoding='utf-8',buffering=1) as err:
        sys.stdout=out;sys.stderr=err
        try:
            from run import main as run
            run()
            import common as c
            result=c.read(H/'RESULT.json');lines=['# U0 learned unprojected all20 temporal refinement','',str(result['decision']),'','No oracle or safe ray; new head trained with GT supervised loss only. Physical failures are reported rather than fallback.','', '|job change group|n|nativegood|newgood|rescue|lost|motionPASS|','|---|---:|---:|---:|---:|---:|---|']
            for key,row in result['jobs'].items():
                r=c.read(row['metrics'])['change'];lines.append(f"|{key}|{r['n']}|{r['native_good']}|{r['good']}|{r['rescue']}|{r['lost']}|{r['motion_pass']}|")
            lines+=['','Full evidence in RESULT/BOOTSTRAP/ARITHMETIC_AUDIT; manual evidence-driven relationship analysis pending. Accuracy potential is not safe baseline success. No automatic next loss/protection/scorer training.']
            (H/'RESULT_ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');finished=datetime.now().astimezone().isoformat();write('exit_status.json',dict(status='SUCCESS',started_at=started,finished_at=finished));write('worker_state.json',dict(status='SUCCESS',pid=os.getpid(),parent=os.getppid(),started_at=started,finished_at=finished));write('runtime_state.json',dict(status='COMPLETED_REVIEW_READY',decision=result['decision'],new_formal_experiments=1))
        except BaseException:
            traceback.print_exc()
            if 'common' in sys.modules:
                try:sys.modules['common'].failure('STOP_U0_WORKER')
                except Exception:traceback.print_exc()
            finished=datetime.now().astimezone().isoformat();write('exit_status.json',dict(status='FAILURE',started_at=started,finished_at=finished));write('worker_state.json',dict(status='FAILURE',pid=os.getpid(),parent=os.getppid(),started_at=started,finished_at=finished));raise
if __name__=='__main__':main()
