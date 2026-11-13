import os,sys,json,traceback,msvcrt
from pathlib import Path
from datetime import datetime
H=Path(__file__).resolve().parent
def write(name,z):
    p=H/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(z,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
def main():
    lock=(H/'worker.lock').open('a+b');lock.seek(0)
    if not (H/'worker.lock').stat().st_size:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    start=datetime.now().astimezone().isoformat();base=dict(pid=os.getpid(),parent=os.getppid(),started_at=start)
    write('worker_state.json',dict(status='RUNNING',stage='INITIAL_U1_CPU_IMPORT',**base))
    with (H/'worker_stdout.log').open('a',encoding='utf-8',buffering=1) as out,(H/'worker_stderr.log').open('a',encoding='utf-8',buffering=1) as err:
        sys.stdout=out;sys.stderr=err
        try:
            from run_audit import main as run
            run();import common as c
            primary=c.read(H/'PRIMARY_COMPARISONS.json');lines=['# U1 direction/retention audit: computational results ready','','Manual evidence/counterevidence synthesis pending. No training or model replay.','','|primary outer-FIT metric|gap seedmean|valid draws|family4 CI|','|---|---:|---:|---|']
            for r in primary['rows']:lines.append(f"|{r['metric']}|{r['estimate']}|{r['valid_draws']}|{r['CI_family4']}|")
            lines+=['','All24jobs/allgroups/thinlayers in jobs/*/*/*_summary.json. Full conditional branch review must consider actual fixedslot changes andall20 counterevidence. NEXT_BRANCH flags are not unique causes.']
            (H/'RESULT_ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');write('RESULT_ANALYSIS.json',dict(status='COMPUTATION_COMPLETE_MANUAL_SYNTHESIS_PENDING',no_training=True,primary='PRIMARY_COMPARISONS.json',alljob_summaries='JOBS.json',branches='NEXT_BRANCH.json'))
            finish=datetime.now().astimezone().isoformat();write('exit_status.json',dict(status='SUCCESS',finished_at=finish,**base));write('worker_state.json',dict(status='SUCCESS',finished_at=finish,**base))
        except BaseException:
            traceback.print_exc();finish=datetime.now().astimezone().isoformat()
            if 'common' in sys.modules:
                try:sys.modules['common'].failure('STOP_U1_WORKER')
                except BaseException:traceback.print_exc()
            write('exit_status.json',dict(status='FAILURE',finished_at=finish,**base));write('worker_state.json',dict(status='FAILURE',finished_at=finish,**base));raise
if __name__=='__main__':main()
