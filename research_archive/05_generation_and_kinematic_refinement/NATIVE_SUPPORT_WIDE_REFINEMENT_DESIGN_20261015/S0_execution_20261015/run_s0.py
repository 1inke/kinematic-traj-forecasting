import common as c
import metrics as m
from preflight import job_arrays
import numpy as np,time,gc

def main():
    t=time.perf_counter();pre=c.read(c.H/'PREFLIGHT.json');assert pre['status']=='PASS' and pre['planning_minutes']<=30
    files=c.read(c.H/'source_manifest.json')['files'];assert c.digest(files)==pre['source_digest'];c.stage('S0_INITIAL_SOURCE_SHA');c.verify(files);data,splits=c.load_data();records={}
    for name,ix in splits.items():
        c.stage('S0_CONSTRUCT_SAVE',split=name,completed_jobs=len(records),total=4);output=c.H/'jobs'/f'{name}.npz';mp=output.with_suffix('.json');dp=output.with_suffix('.done.json')
        if dp.exists():c.valid_done(c.read(dp),pre['source_digest']);record=c.read(mp);record['files']=c.read(dp)['files']
        else:
            assert not output.exists() and not mp.exists(),'PRESERVE_PARTIAL_STOP';z=job_arrays(data,ix);met=m.metrics(z,z['groups'],z['track']);cert=m.certificate_summary(z,z['groups']);output.parent.mkdir(exist_ok=True);np.savez_compressed(output,**z);back=c.npz(output);assert all(np.array_equal(v,back[k]) for k,v in z.items())
            record=dict(split=name,n=len(ix),source_digest=pre['source_digest'],output=str(output),metrics=met,certificate=cert,GT_privileged_witness_not_teacher=True);c.write(mp,record);record['files']={str(output):c.sha(output),str(mp):c.sha(mp)};c.write(dp,dict(status='PASS',source_digest=pre['source_digest'],files=record['files']));del z,back,met,cert
        records[name]=record;c.write('JOBS.json',records);gc.collect()
    from audit_saved import main as audit
    audit();decision=m.gate(records);c.write('DECISION.json',decision);c.stage('S0_FINAL_SOURCE_SHA',completed_jobs=4);c.verify(files);outputs={p:h for r in records.values() for p,h in r['files'].items()};c.verify(outputs)
    c.write('PRESERVATION.json',dict(status='PASS',source_count=len(files),inherited_A0_digests=1823,source_digest=c.digest(files),initial_final_unchanged=True));c.write('RESOURCE.json',dict(status='PASS',**c.RESOURCE))
    c.write('RESULT.json',dict(status='PASS_DIAGNOSTIC_ONLY_NATIVE_SUPPORT_SAFE_OPPORTUNITY',decision=decision,records=records,scientific_jobs=4,ports=12,groups=48,outputs_SHA=outputs,source_digest=c.digest(files),source_count=len(files),elapsed_seconds=time.perf_counter()-t,new_formal_count=0,S1_started=False,no_model_optimizer_solver=True));c.stage('SUCCESS',decision=decision['status'])

if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_S0');raise
