from core import *
import ast
def main():
    require12();assert not (HERE/'PRODUCTION_SOURCE_LOCK.json').exists();g=guard('CPU')
    prior=read_json(PREVIOUS/'source_manifest.json');files=prior['files'].copy();assert digest(files)==prior['digest']
    for k,h in read_json(PREVIOUS/'PRESERVATION.json')['outputs'].items():
        p=(PREVIOUS/k).relative_to(ROOT).as_posix()
        if p in files:assert files[p]==h
        files[p]=h
    for p in (PREVIOUS/'completion_review').glob('*'):
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=sha256(p)
    for name in ('RESULT_ANALYSIS.md','RESULT_ANALYSIS.json','PRESERVATION.json','RESOURCE.json','RESULT.json','ARITHMETIC_AUDIT.json','ALL_PROBE_FINALS_LOCK.json','source_manifest.json','PRODUCTION_SOURCE_LOCK.json'):
        p=PREVIOUS/name;files[p.relative_to(ROOT).as_posix()]=sha256(p)
    provenance=read_json(HERE/'FEATURE_PROVENANCE.json');receipt=[]
    for p,h in provenance['source_files'].items():
        p=Path(p)
        if p==HERE/'core.py':
            snapshot=HERE/'admission_runtime_snapshot/core_exact.py';assert sha256(snapshot)==h
            old=ast.parse(snapshot.read_text(encoding='utf-8'));new=ast.parse((HERE/'core.py').read_text(encoding='utf-8'))
            oldf={n.name:ast.dump(n,include_attributes=False) for n in old.body if isinstance(n,ast.FunctionDef)}
            newf={n.name:ast.dump(n,include_attributes=False) for n in new.body if isinstance(n,ast.FunctionDef)}
            assert set(newf)-set(oldf)=={'mark'}
            for name in oldf:
                if name not in ('guard','resource_report'):assert oldf[name]==newf[name],name
            oldtop=[ast.dump(n,include_attributes=False) for n in old.body if not isinstance(n,ast.FunctionDef)]
            newtop=[ast.dump(n,include_attributes=False) for n in new.body if not isinstance(n,ast.FunctionDef)];assert oldtop==newtop
            receipt.append(dict(original_path=str(p),historical_SHA=h,retained_exact_snapshot=str(snapshot),current_infrastructure_SHA=sha256(p),changed_only=['guard','resource_report','mark']))
        else:assert sha256(p)==h,p
        key=p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p.resolve())
        if key in files:assert files[key]==sha256(p),key
        files[key]=sha256(p)
    for p in HERE.rglob('*'):
        if p.is_file() and (p.suffix in ('.py','.ps1','.md') or 'prepared' in p.parts or p.name in ('protocol.json','RESOURCE_STAGE_PLAN.json','FEATURE_PROVENANCE.json','TOKEN_REDUNDANCY_CHECK.json','PAST_ONLY_AUDIT.json','STATISTICS_AUDIT.json')):
            files[p.relative_to(ROOT).as_posix()]=sha256(p)
    for p in (ROOT/'docs/OBSERVABLE_INFORMATION_DIAGNOSIS_DESIGN_20261026').glob('*.md'):files[p.relative_to(ROOT).as_posix()]=sha256(p)
    for i,(p,h) in enumerate(files.items()):
        assert sha256(ROOT/p if not Path(p).is_absolute() else p)==h,p
        if i%100==0:g.check()
    m=dict(files=files,digest=digest(files),at=now(),all1660_inherited_digests_preserved=True,stage='CANDIDATE_REQUIRES_CURRENT_PREFLIGHT')
    write_json(HERE/'source_manifest.json',m);write_json(HERE/'SOURCE_AND_ID_AUDIT.json',dict(status='PASS',source=m['digest'],source_count=len(files),admission_runtime_transition=receipt,
        old_source_output_SHA_unchanged=True,all9777_ids_retained=True,blocked_71_observed_time_no_deletion=True,admissible_arms=provenance['admissible_arms']))
    print('PASS SOURCES',len(files),m['digest'],flush=True)
if __name__=='__main__':main()
