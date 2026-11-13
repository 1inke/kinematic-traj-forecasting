"""Read archive headers only, estimate remaining saved-audit peak, one OS snapshot."""
import json,zipfile,hashlib,math
from pathlib import Path
import numpy as np
import memory_policy as p
H=Path(__file__).resolve().parent;P=H.parent;GiB=2**30
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def payload(path):
    total=0
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):continue
            with archive.open(name) as f:
                version=np.lib.format.read_magic(f)
                if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(f)
                elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(f)
                else:raise ValueError(('UNSUPPORTED_ARRAY_HEADER',version,path,name))
            assert not dtype.hasobject;total+=math.prod(shape)*dtype.itemsize
    return total
def tests():
    e=1000.;s=dict(physical_available_bytes=1040.,commit_available_bytes=910.,private_bytes=300.,RSS_bytes=200.)
    assert p.decide(s,e)['status']=='PASS'
    assert p.decide(dict(s,physical_available_bytes=1039.),e)['status']=='STOP_RESOURCE'
    assert p.decide(dict(s,commit_available_bytes=909.),e)['status']=='STOP_RESOURCE'
    assert p.decide(dict(s,physical_available_bytes=1300.,commit_available_bytes=1300.),e,mode='startup')['status']=='PASS'
    assert p.decide(dict(s,physical_available_bytes=1299.,commit_available_bytes=1300.),e,mode='startup')['status']=='STOP_RESOURCE'
    assert p.decide(dict(s,private_bytes=1301.),e)['status']=='STOP_RESOURCE'
    assert p.decide(dict(s,private_bytes=1000.,RSS_bytes=1000.,physical_available_bytes=0.,commit_available_bytes=0.),e)['status']=='PASS'
    bad=False
    try:p.decide(s,0.)
    except AssertionError:bad=True
    assert bad
    return dict(status='PASS',equality_boundary_accepted=True,one_byte_below_rejected=True,physical_commit_independent=True,allocated_memory_not_counted_twice=True,estimate_overrun_rejected=True,invalid_estimate_rejected=True,startup_uses_full_requirement=True)
def main():
    sources=[P/'common.py',P/'RESOURCE_FAILURE.json',P/'PREFLIGHT.json',P/'prelaunch_review/REVIEW.json',P/'JOBS.json'];before={str(f):sha(f) for f in sources};failure=read(P/'RESOURCE_FAILURE.json');review=read(P/'prelaunch_review/REVIEW.json');jobs=read(P/'JOBS.json');sizes=[]
    for key,row in jobs.items():sizes.append(dict(job=key,resident_payload_bytes=payload(Path(row['output']))+payload(Path(row['aux']))))
    largest=max(x['resident_payload_bytes'] for x in sizes);baseline=review['resource']['peak_private_GiB']*GiB;observed=max(failure['lifecycle']['peak_private_GiB'],failure['lifecycle']['peak_RSS_GiB'])*GiB
    static=baseline+4*largest;estimated=int(math.ceil(max(observed,static)))
    estimate=dict(status='ESTIMATE_NOT_NEW_AUDIT_MEASUREMENT',estimated_total_peak_bytes=estimated,estimated_total_peak_GiB=estimated/GiB,historical_full_worker_peak_bytes=observed,measured_prelaunch_baseline_bytes=baseline,largest_saved_job_payload_bytes=largest,static_component_estimate_bytes=static,temporary_payload_multiplier=4,explanation='Conservative max of already observed full worker peak including audited FIT jobs and measured baseline+4 resident job payloads (read/decompress, reconstruction, metrics/aux, comparison temporaries). Not an arbitrary floor; current allocated RSS/private deducted only during runtime.',job_header_payloads=sizes,source_SHA=before,scope='saved-output audit/finalization, no six-run retraining')
    p.write('MEMORY_ESTIMATE.json',estimate);fixtures=tests();snap=p.snapshot();startup=p.decide(snap,estimated,mode='startup');runtime=p.decide(snap,estimated)
    oldsnap=dict(physical_available_bytes=failure['physical_available_GiB']*GiB,commit_available_bytes=failure['commit_available_GiB']*GiB,private_bytes=failure['private_GiB']*GiB,RSS_bytes=failure['RSS_GiB']*GiB)
    old_replay=p.decide(oldsnap,estimated);assert all(sha(Path(f))==v for f,v in before.items())
    # The adapter is actually wired, including the four AST arithmetic globals.
    import common as adapter
    assert adapter.guard is p.guard and adapter.legacy.guard is p.guard and adapter.legacy.ns['guard'] is p.guard and adapter.legacy.H==H
    assert all(sha(Path(f))==v for f,v in before.items())
    result=dict(status='PASS_POLICY_CONFIGURATION',fixtures=fixtures,estimate_GiB=estimated/GiB,full_startup_threshold_GiB=1.3*estimated/GiB,one_current_snapshot=snap,startup_assessment=startup,runtime_assessment=runtime,historical_failure_replayed_with_new_rule=old_replay,old_source_and_evidence_SHA_unchanged=True,new_adapter_all_guard_calls_wired=True,audit_or_worker_started=False,training_restarted=False,scientific_thresholds_changed=False)
    p.write('POLICY_VALIDATION.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
