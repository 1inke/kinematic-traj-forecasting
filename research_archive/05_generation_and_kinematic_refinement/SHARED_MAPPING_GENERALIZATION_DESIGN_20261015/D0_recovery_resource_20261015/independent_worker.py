import os,sys,json,time,traceback,msvcrt
from pathlib import Path
from datetime import datetime
H=Path(__file__).resolve().parent
def write(p,z):
    t=(H/p).with_suffix('.tmp');t.write_text(json.dumps(z,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(t,H/p)
def main():
    lock=(H/'worker.lock').open('a+b');lock.seek(0)
    if (H/'worker.lock').stat().st_size==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    started=datetime.now().astimezone().isoformat();write('worker_state.json',dict(status='RUNNING',pid=os.getpid(),parent=os.getppid(),started_at=started,stage='INITIAL_D0_IMPORT'))
    with (H/'worker_stdout.log').open('a',encoding='utf-8',buffering=1) as out,(H/'worker_stderr.log').open('a',encoding='utf-8',buffering=1) as err:
        sys.stdout=out;sys.stderr=err
        try:
            from run_d0 import main as run
            run()
            from core import read
            result=read(H/'RESULT.json');decision=result['decision'];audit=read(H/'ARITHMETIC_AUDIT.json')
            lines=['# D0 finite-update diagnosis','',f"Decision: {decision['status']}",'','|seed|ordinary source dL|projected source dL|ordinary probe dJ|projected probe dJ|direction packets /8|ordinary lost|projected lost|gate|','|---|---:|---:|---:|---:|---:|---:|---:|---|']
            for r in decision['perseed']:lines.append(f"|{r['seed']}|{r['ordinary_source_delta_L']:.10g}|{r['projected_source_delta_L']:.10g}|{r['ordinary_probe_delta_J']:.10g}|{r['projected_probe_delta_J']:.10g}|{r['ordinary_joint_direction_packets']}|{r['ordinary_lost_mean']:.6g}|{r['projected_lost_mean']:.6g}|{r['ordinary_interference_gate'] and r['projection_gate']}|")
            lines+=['','All 3 seeds and all8 packets included. Same complete candidate ADE<=.2 AND FDE<=.4. Source and probe are previously seen FIT, track-disjoint but context may overlap. Temporary fresh-zero-moment AdamW is an intervention at final checkpoints, not original training resume or independent generalization.','',f"Independent saved audit: {audit['status']}, jobs={audit['jobs']}, maxerror={audit['max_error']}. Old scientific sources and parent checkpoints SHA preserved. NoUpdate exact. All groups and step0/1/4 snapshots stored. No new CI.",'','ARS inline DA2 (same assistant, not external): local dot products cannot establish finite-step causality or generalization; gates use actual source loss/probe J and retained-good lost. Fixed bottom4/.20m position envelope and saturated final head may limit intervention.','', 'ARS inline DA3 (same assistant, not external): D0 failure stops U1. D0 positive would only permit U1 own preflight/budget; no baseline/goal completion or revival of failed T1. Original major and five-motion1% remain.']
            (H/'RESULT_ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
            finished=datetime.now().astimezone().isoformat();write('exit_status.json',dict(status='SUCCESS',started_at=started,finished_at=finished));write('worker_state.json',dict(status='SUCCESS',pid=os.getpid(),parent=os.getppid(),started_at=started,finished_at=finished));write('runtime_state.json',dict(status='COMPLETED_REVIEW_READY',decision=decision['status'],U1_started=False,new_formal_count=0,report='RESULT_ANALYSIS.md'))
        except BaseException:
            traceback.print_exc();finished=datetime.now().astimezone().isoformat();write('exit_status.json',dict(status='FAILURE',started_at=started,finished_at=finished));write('worker_state.json',dict(status='FAILURE',pid=os.getpid(),parent=os.getppid(),started_at=started,finished_at=finished));raise
if __name__=='__main__':main()
