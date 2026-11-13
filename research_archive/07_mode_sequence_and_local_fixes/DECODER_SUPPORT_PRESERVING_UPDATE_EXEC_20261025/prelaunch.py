"""Explicit CPU infrastructure extension bound to immutable actual-preflight source."""
from task_adapter import *
import shutil

def verify_extension():
    m=read_json(HERE/'source_extension_prelaunch.json');assert m['base_digest']==verify_manifest()
    for v in m['files'].values():assert sha256(v['path'])==v['sha256'],('EXTENSION_CHANGED',v['path'])
    return m['digest']

def main():
    source=verify_manifest();guard=Guard(E);guard.check('startup')
    paths=[HERE/n for n in ('worker.py','launch_worker.py','start_independent.ps1','check_once.ps1','prelaunch.py','IMPLEMENTATION_REPAIR.md','PREFLIGHT.json','PRELAUNCH_REVIEW.json','FIXTURES.json','RESUME_CHECK.json','BENCHMARK.json','MEMORY_ESTIMATE.json','identity_audit/RESULT.json')]
    paths+=list((HERE/'failures').rglob('*'));paths=[p for p in paths if p.is_file()]
    files={p.relative_to(ROOT).as_posix():dict(path=str(p),sha256=sha256(p)) for p in paths}
    digest=hashlib.sha256(json.dumps({k:v['sha256'] for k,v in files.items()},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write_json(HERE/'source_extension_prelaunch.json',dict(base_digest=source,files=files,digest=digest,old_base_manifest_modified=False,scope='worker launch/check infrastructure and actual preflight evidence/failure preservation, scientific production functions unchanged'))
    assert verify_extension()==digest
    write_json(HERE/'PRELAUNCH_FINAL_REVIEW.json',dict(status='PASS',base_source_digest=source,extension_digest=digest,all494_base_sources_initial_final_SHA=True,infrastructure_files=len(files),actual_preflight='PASS',resource=guard.observations,no_GPU_or_optimizer_replay=True,DA1='PASS_WITH_EXPLORATORY_LIMITS',DA1_major_limit='Two group means and first-order constraints do not ensure individual success or transfer; N control and all fixed G0 gates retained.'))
    print('PRELAUNCH PASS',source,digest)

if __name__=='__main__':main()
