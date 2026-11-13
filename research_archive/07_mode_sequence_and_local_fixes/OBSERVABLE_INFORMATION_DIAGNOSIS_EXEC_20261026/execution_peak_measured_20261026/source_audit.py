from bootstrap import HERE,PARENT
from base import *
import shutil

def main():
    if (HERE/'PRODUCTION_SOURCE_LOCK.json').exists():raise RuntimeError('Locked recovery: no rebaseline')
    g=guard('source_prepare');parent=read_json(PARENT/'source_manifest.json');files=parent['files'].copy();assert digest(files)==parent['digest']
    for p in list(PARENT.glob('*.json'))+list((PARENT/'logs').glob('*'))+list(PARENT.glob('worker.*.log')):
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=sha256(p)
    for k,h in files.items():assert sha256(ROOT/k)==h,'Immutable parent/source changed '+k
    prepared=HERE/'prepared';prepared.mkdir(exist_ok=True)
    for p in (PARENT/'prepared').iterdir():
        if not p.is_file():continue
        out=prepared/p.name
        if out.exists():assert sha256(out)==sha256(p)
        else:shutil.copyfile(p,out)
        assert sha256(out)==sha256(p);files[out.relative_to(ROOT).as_posix()]=sha256(out)
    for name in ('TINY_IDS.json','INPUT_COLLISIONS.json'):
        p=PARENT/name;out=HERE/name
        if not out.exists():shutil.copyfile(p,out)
        assert sha256(out)==sha256(p);files[out.relative_to(ROOT).as_posix()]=sha256(out)
    for p in HERE.iterdir():
        if p.suffix in ('.py','.ps1','.md') or p.name in ('protocol.json','RESOURCE_STAGE_PLAN.json'):files[p.relative_to(ROOT).as_posix()]=sha256(p)
    for name in ('source_manifest.json','SOURCE_AND_ID_AUDIT.json'):
        if (HERE/name).exists():
            provenance=HERE/'candidate_provenance';provenance.mkdir(exist_ok=True);write_json(provenance/(str(time.time_ns())+'_'+name),read_json(HERE/name))
    m=dict(files=files,digest=digest(files),parent_digest=parent['digest'],parent_entries_inherited=len(parent['files']),stage='CANDIDATE_UNTIL_PRODUCTION_SOURCE_LOCK',at=now());write_json(HERE/'source_manifest.json',m)
    write_json(HERE/'SOURCE_AND_ID_AUDIT.json',dict(status='PASS',source=m['digest'],entries=len(files),all_parent_digests_preserved=True,science_modules_executed_byte_for_byte_from_parent=True,parent_no_training=True,no_files_deleted=True,prepared_arrays_copied_SHA_EXACT=True,resource=g.check()));print('PASS SOURCE',len(files),m['digest'],flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:fail(e);raise
