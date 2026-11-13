import common as c
import analytic_envelope as e
from preflight import job_arrays
import numpy as np,time,gc
def main():
    t=time.perf_counter();pre=c.read(c.H/'PREFLIGHT.json');assert pre['status']=='PASS' and pre['planning_minutes']<=30
    files=c.read(c.H/'source_manifest.json')['files'];assert c.digest(files)==pre['source_digest'];c.stage('A0_INITIAL_SOURCE_SHA');c.verify(files);data,splits=c.load_data();records={}
    for name,ix in splits.items():
        c.stage('A0_ANALYTIC_CONSTRUCT_SAVE',split=name,completed_jobs=len(records),total=4);output=c.H/'jobs'/f'{name}.npz';mp=output.with_suffix('.json');dp=output.with_suffix('.done.json')
        if dp.exists():c.valid_done(c.read(dp),pre['source_digest']);record=c.read(mp)
        else:
            assert not output.exists() and not mp.exists(),'PRESERVE_PARTIAL_OUTPUTS_STOP';z=job_arrays(data,ix);met=e.metrics(z,z['groups'],z['track']);output.parent.mkdir(exist_ok=True);np.savez_compressed(output,**z);back=c.npz(output);assert all(np.array_equal(v,back[k]) for k,v in z.items())
            record=dict(split=name,n=len(ix),source_digest=pre['source_digest'],output=str(output),metrics=met,relaxed_oracle_not_deployable=True);c.write(mp,record);record['files']={str(output):c.sha(output),str(mp):c.sha(mp)};c.write(dp,dict(status='PASS',source_digest=pre['source_digest'],files=record['files']));del z,back,met
        if 'files' not in record:record['files']=c.read(dp)['files']
        records[name]=record;c.write('JOBS.json',records);gc.collect()
    decision=e.gate(records);c.write('DECISION.json',decision)
    from audit_saved import main as audit
    audit();c.stage('A0_FINAL_SOURCE_SHA',completed_jobs=4);c.verify(files);allout={p:h for r in records.values() for p,h in r['files'].items()};c.verify(allout)
    c.write('PRESERVATION.json',dict(status='PASS',source_count=len(files),inherited_all_previous_digests=pre['inherited_sources'],source_digest=c.digest(files),unchanged_initial_final=True));c.write('RESOURCE_USAGE.json',dict(status='PASS',**c.RESOURCE))
    c.write('RESULT.json',dict(status='PASS_DIAGNOSTIC_ONLY_ANALYTIC_ENVELOPE',decision=decision,records=records,scientific_jobs=4,populations=12,groups=48,source_count=len(files),source_digest=c.digest(files),outputs_SHA=allout,elapsed_seconds=time.perf_counter()-t,resource=c.RESOURCE,new_formal_count=0,neural_forward=False,optimizer=False,A1_started=False,B_started=False));c.stage('SUCCESS',decision=decision['status']);print('A0_DECISION',decision,flush=True)
if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_A0');raise
