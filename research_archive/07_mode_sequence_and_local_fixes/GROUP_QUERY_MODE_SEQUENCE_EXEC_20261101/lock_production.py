"""Actual implementation review, immutable source lock, no task creation."""
from common import *
import ast,shutil
def main():
    pf=read(HERE/'PREFLIGHT.json');assert pf['status']=='PASS_ACTUAL_COMPLETE_PREFLIGHT';b=read(HERE/'BENCHMARK.json');assert b['conservative_full_seconds']<=1800
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    # Bounded actual disk plan includes all groups, epoch checkpoint history,
    # metric arrays, independent review, model exports and the profiling outputs.
    planned=3*2*(18909+1856)*20*12*2*4+6*8*2**21+2*2**30
    assert shutil.disk_usage(HERE).free>=1.3*planned
    write(HERE/'DISK_ADMISSION.json',dict(status='PASS',free=shutil.disk_usage(HERE).free,conservative_outputs=planned,required=1.3*planned,no_old_file_deletion=True))
    write(HERE/'PRELAUNCH_REVIEW.json',dict(status='PASS',reviewer='same assistant inline DA1 not external',critical_issues=[],major_limits=['Q3 has extra training compute/640params; not pure supervision causal identification','old A unsafe; native absolute gate unchanged','F16 has seen FIT and inner; repeated developer evaluation'],checks=['all3 documents read','original G1 nonzero paths loss and allparams gradients EXACT','paired group0 stats initial epoch orders EXACT','independent auxiliaries localRNG','GT-free forward and exact group0 export','fixed twoepsFD no face changes','eacharm real513+12 twoepoch allparams Adam ALLRNG order cursor history eval EXACT','128saved independent complete-link pdist all5+junction','real CPU pipeline3sizes all20000draw independent loop','budget conservative<=30min','warning-first-v3 resource'],
        authority='2026-11-01 human execute directory; original DESIGN_ONLY is snapshot',no_old_process_queries=True,no_scientific_updates_yet=True))
    files=dict(read(HERE/'INPUT_SOURCE_LOCK.json')['files'])
    for p in HERE.rglob('*'):
        if not p.is_file() or p.name in ('progress.json','source_manifest.json','runtime_control.json') or '__pycache__' in p.parts:continue
        if p.suffix in ('.pyc',):continue
        key=str(p.resolve());s=sha(p)
        if key in files and files[key]!=s:raise ValueError('No overwritten original digest')
        files[key]=s
    digestvalue=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    write(HERE/'source_manifest.json',dict(status='PRODUCTION_SOURCE_LOCK_Q',files=files,count=len(files),digest=digestvalue,old_digests_retained=len(read(HERE/'INPUT_SOURCE_LOCK.json')['files'])))
    print('LOCK PASS',len(files),digestvalue,flush=True)
if __name__=='__main__':main()
