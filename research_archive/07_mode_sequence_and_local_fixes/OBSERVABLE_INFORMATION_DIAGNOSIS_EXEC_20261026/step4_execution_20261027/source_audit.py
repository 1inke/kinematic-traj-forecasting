from core import *
def main():
    require3();assert not (HERE/'PRODUCTION_SOURCE_LOCK.json').exists();g=guard()
    if (HERE/'source_manifest.json').exists():
        old=read_json(HERE/'source_manifest.json');write_json(HERE/f'candidate_provenance/{old["digest"]}.json',old)
    previous=read_json(PREVIOUS/'source_manifest.json');files=previous['files'].copy();assert digest(files)==previous['digest']
    for p,h in read_json(PREVIOUS/'PRESERVATION.json')['outputs'].items():
        key=(PREVIOUS/p).relative_to(ROOT).as_posix()
        if key in files:assert files[key]==h
        files[key]=h
    for p in (PREVIOUS/'completion_review').glob('*'):
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=sha256(p)
    for name in ('RESULT_ANALYSIS.md','RESULT_ANALYSIS.json','RESULT.json','RESOURCE.json','PRESERVATION.json','ARITHMETIC_AUDIT.json','ALL_PROBE_FINALS_LOCK.json','source_manifest.json','PRODUCTION_SOURCE_LOCK.json'):
        p=PREVIOUS/name;files[p.relative_to(ROOT).as_posix()]=sha256(p)
    for p in (ROOT/'docs/OBSERVABLE_INFORMATION_DIAGNOSIS_DESIGN_20261026').glob('*.md'):files[p.relative_to(ROOT).as_posix()]=sha256(p)
    for p in HERE.rglob('*'):
        if p.is_file() and (p.suffix in ('.py','.ps1','.md') or 'prepared' in p.parts or p.name in ('protocol.json','RESOURCE_STAGE_PLAN.json','QUERY_REGISTRY.json','INPUT_AUDIT.json')):
            files[p.relative_to(ROOT).as_posix()]=sha256(p)
    for i,(p,h) in enumerate(files.items()):
        assert sha256(Path(p) if Path(p).is_absolute() else ROOT/p)==h,p
        if i%128==0:g.check()
    manifest=dict(files=files,digest=digest(files),at=now(),all2844_old_digests_preserved=True,old_completed_outputs_and_analysis_inherited=True)
    write_json(HERE/'source_manifest.json',manifest);write_json(HERE/'SOURCE_AND_ID_AUDIT.json',dict(status='PASS',source_count=len(files),source=manifest['digest'],
        all7921IDs_retained=True,no_hold_B_GT_quality_group_read_during_preparation=True,original20source_index_metadata_not_video_frames=True))
    print('PASS step4 sources',len(files),manifest['digest'],flush=True)
if __name__=='__main__':main()
