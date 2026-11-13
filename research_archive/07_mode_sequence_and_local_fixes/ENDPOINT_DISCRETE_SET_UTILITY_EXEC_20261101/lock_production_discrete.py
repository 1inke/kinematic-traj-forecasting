"""One new production source lock AFTER complete real preflight and own admission."""
import contracts_discrete as c

def main():
    pf=c.json_read(c.ROOT/'PREFLIGHT.json');bench=c.json_read(c.ROOT/'BENCHMARK.json');assert pf['status']=='PASS_ACTUAL_PREFLIGHT' and bench['T_admitted']
    files=dict(c.json_read(c.ROOT/'INPUT_SOURCE_LOCK.json')['files'])
    c.verify(files)
    for p in c.ROOT.rglob('*'):
        if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts and 'logs' not in p.parts and 'tiny' not in p.parts and p.name not in ('runtime_control.json','WORKER_START.json','WORKER_STATE.json','WORKER_EXIT.json','startup_check.json','halfway_check.json','completion_check.json'):
            key=str(p.resolve())
            if key in files and files[key]!=c.sha(p):raise ValueError('source conflict')
            files[key]=c.sha(p)
    c.write(c.ROOT/'PRELAUNCH_REVIEW.json',dict(status='PASS_D0_GPU_RESUME_SAVED_BEFORE_T_ADMISSION',preflight=c.sha(c.ROOT/'PREFLIGHT.json'),budget=bench['T_conservative_seconds'],
        D0=c.sha(c.ROOT/'D0_RESULT.json'),DA1=dict(verdict='PASS_WITH_LIMITATIONS',issue='new representation/objective/policy vs oldRS is not causal one-factor',strongest_counterargument='max-score noisy action utility may still sacrifice old success'),
        resource_v3_no_fixed6_no_delete=True))
    files[str((c.ROOT/'PRELAUNCH_REVIEW.json').resolve())]=c.sha(c.ROOT/'PRELAUNCH_REVIEW.json')
    c.write(c.ROOT/'source_manifest.json',dict(status='PRODUCTION_SOURCE_LOCK',files=files,count=len(files),not_all_historical_files=True,no_digest_overwrite=True))
    print('PRODUCTION SOURCE LOCK',len(files),flush=True)
if __name__=='__main__':main()
