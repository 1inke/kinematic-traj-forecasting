from base import *
def main():
    if (HERE/'PRODUCTION_SOURCE_LOCK.json').exists():raise RuntimeError('Production science is locked; no rebaseline')
    for oldname in ('source_manifest.json','SOURCE_AND_ID_AUDIT.json'):
        if (HERE/oldname).exists():
            provenance=HERE/'candidate_provenance';provenance.mkdir(exist_ok=True);write_json(provenance/(str(time.time_ns())+'_'+oldname),read_json(HERE/oldname))
    g=guard('source_prepare');m=read_json(OLD/'source_manifest.json');assert digest(m['files'])==m['digest'];files=m['files'].copy();n=0
    p=read_json(OLD/'PRESERVATION.json')
    for k,h in p['outputs'].items():
        key=(OLD/k).relative_to(ROOT).as_posix()
        if key in files and files[key]!=h:raise ValueError('Inherited lock conflict')
        files[key]=h
    ext=read_json(OLD/'source_extension_prelaunch.json');assert ext['base_source_digest']==m['digest'];files.update(ext['extension_files'])
    ce=read_json(OLD/'COMPLETION_EXTENSION.json')
    for k,h in ce['files'].items():files[(OLD/k).relative_to(ROOT).as_posix()]=h
    for k in ('source_manifest.json','source_extension_prelaunch.json','COMPLETION_EXTENSION.json','PRESERVATION.json','ALL6_FINALS_LOCK.json','RESULT_ANALYSIS.json'):files[(OLD/k).relative_to(ROOT).as_posix()]=sha256(OLD/k)
    for k,h in files.items():
        path=ROOT/k
        if not path.exists():raise FileNotFoundError('Missing immutable reused source; do not retrain: '+k)
        if sha256(path)!=h:raise ValueError('Inherited source/output changed '+k)
        n+=1
        if n%100==0:g.check()
    for p in DESIGN.glob('*.md'):files[p.relative_to(ROOT).as_posix()]=sha256(p)
    for p in HERE.iterdir():
        if p.suffix in ('.py','.ps1','.md') or p.name in ('protocol.json','RESOURCE_STAGE_PLAN.json'):files[p.relative_to(ROOT).as_posix()]=sha256(p)
    for p in (HERE/'prepared').rglob('*'):
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=sha256(p)
    for name in ('TINY_IDS.json','INPUT_COLLISIONS.json'):
        files[(HERE/name).relative_to(ROOT).as_posix()]=sha256(HERE/name)
    out=dict(files=files,digest=digest(files),inherited_E1_source_count=len(m['files']),inherited_E1_output_count=len(read_json(OLD/'PRESERVATION.json')['outputs']),at=now(),not_overwriting_old_digests=True,stage='IMPLEMENTATION_CANDIDATE_NOT_YET_PRODUCTION_LOCKED')
    write_json(HERE/'source_manifest.json',out)
    write_json(HERE/'SOURCE_AND_ID_AUDIT.json',dict(status='PASS_ALL_INHERITED_SHA',source_count=len(files),source=out['digest'],old700=True,old837outputs=True,old_prelaunch_extension=True,old_completion_extension=True,no_generator_forward=True,resource=g.observations))
    print('PASS SOURCE',len(files),out['digest'],flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:fail(e);raise
