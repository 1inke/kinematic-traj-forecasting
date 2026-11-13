"""Lossless NPZ dedup archive. No deletion is implemented in this program.

The PowerShell cleanup removes source containers only after this program has
preserved and independently replay-hashed every byte. All original arrays,
ZIP framing, timestamps and source SHA-256 remain recoverable.
"""
from pathlib import Path
import argparse, base64, hashlib, io, json, os, shutil, sqlite3, sys, tempfile, time, zipfile
import review

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
STORE=HERE/'npz_archive'
MAN=STORE/'manifests'
PACK=STORE/'payloads.bin'
DB=STORE/'index.sqlite'
BLOCK=4*1024**2

def bounded(rel):
    p=(ROOT/rel).resolve()
    if not p.is_relative_to(ROOT.resolve()) or p==ROOT.resolve():raise ValueError('OUTSIDE_ROOT')
    return p

def digest(p):return review.sha_file(p)

def atomic_json(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    with open(tmp,'w',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def connection():
    STORE.mkdir(exist_ok=True);MAN.mkdir(exist_ok=True)
    c=sqlite3.connect(DB);c.execute('PRAGMA synchronous=FULL')
    c.execute('CREATE TABLE IF NOT EXISTS chunks(sha TEXT PRIMARY KEY,offset INTEGER NOT NULL,size INTEGER NOT NULL)')
    c.execute('CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY,manifest TEXT NOT NULL,sha TEXT NOT NULL,size INTEGER NOT NULL,state TEXT NOT NULL)')
    c.commit();return c

def manifest_path(rel):return MAN/(hashlib.sha256(rel.encode('utf-8')).hexdigest()+'.json')

def pieces(man,c):
    with open(PACK,'rb') as f:
        for part in man['parts']:
            if 'literal' in part:
                yield base64.b64decode(part['literal'],validate=True)
            else:
                row=c.execute('SELECT offset,size FROM chunks WHERE sha=?',(part['sha256'],)).fetchone()
                assert row is not None and row[1]==part['length']
                f.seek(row[0]);remain=row[1];h=hashlib.sha256()
                while remain:
                    b=f.read(min(BLOCK,remain));assert b;remain-=len(b);h.update(b);yield b
                assert h.hexdigest()==part['sha256'],'CORRUPT_ARCHIVE_CHUNK'

def verify_manifest(mp,c,compare_source=False):
    man=json.loads(mp.read_text(encoding='utf-8'));h=hashlib.sha256();n=0
    source=open(bounded(man['path']),'rb') if compare_source else None
    try:
        for b in pieces(man,c):
            if source is not None:assert source.read(len(b))==b,'SOURCE_BYTE_MISMATCH'
            h.update(b);n+=len(b)
        if source is not None:assert not source.read(1)
    finally:
        if source is not None:source.close()
    assert n==man['bytes'] and h.hexdigest()==man['sha256'],'RECONSTRUCTION_SHA_MISMATCH'
    return man

def stage_one(rec,c,pack):
    p=bounded(rec['path']);st=p.stat()
    assert st.st_size==rec['bytes'] and st.st_mtime_ns==rec['mtime_ns'],'SOURCE_CHANGED'
    assert st.st_nlink==1,'EXISTING_HARDLINK_RETAIN'
    assert not (st.st_file_attributes & 1024),'REPARSE_FILE_RETAIN'
    mp=manifest_path(rec['path']);before=pack.seek(0,2);parts=[];source_hash=hashlib.sha256();count=0
    expected={(x['offset'],x['length']):x for x in rec['members']}
    with open(p,'rb') as f:
        for typ,offset,size,name in review.segments(p):
            f.seek(offset)
            if typ=='literal':
                data=f.read(size);assert len(data)==size;source_hash.update(data);count+=size
                parts.append(dict(literal=base64.b64encode(data).decode('ascii')));continue
            sha=expected[(offset,size)]['sha256'];row=c.execute('SELECT offset,size FROM chunks WHERE sha=?',(sha,)).fetchone()
            isnew=row is None
            if isnew:where=pack.seek(0,2)
            else:assert row[1]==size
            remain=size;ch=hashlib.sha256()
            while remain:
                data=f.read(min(remain,BLOCK));assert data;remain-=len(data);count+=len(data);source_hash.update(data);ch.update(data)
                if isnew:pack.write(data)
            assert ch.hexdigest()==sha,'PAYLOAD_CHANGED_SINCE_PLAN'
            if isnew:c.execute('INSERT INTO chunks VALUES(?,?,?)',(sha,where,size))
            parts.append(dict(sha256=sha,length=size,name=name))
    assert count==st.st_size and p.stat().st_mtime_ns==st.st_mtime_ns
    pack.flush();os.fsync(pack.fileno());c.commit()
    man=dict(format='EXACT_NPZ_ZIP_PAYLOAD_CAS_V1',path=rec['path'],bytes=count,sha256=source_hash.hexdigest(),mtime_ns=st.st_mtime_ns,
             attributes=st.st_file_attributes,scope=rec['scope'],report=rec['report'],reason=rec['reason'],parts=parts,
             archive_store='payloads.bin + index.sqlite',unique_data_discarded=False)
    atomic_json(mp,man)
    # Independent reader uses the persisted manifest and SQLite mapping, checks
    # every chunk hash, compares every reconstructed byte against the source,
    # and checks the complete original NPZ SHA-256.
    verify_manifest(mp,c,compare_source=True)
    c.execute('INSERT OR REPLACE INTO files VALUES(?,?,?,?,?)',(rec['path'],str(mp.relative_to(HERE)),man['sha256'],count,'VERIFIED_SOURCE_PRESENT'));c.commit()
    return dict(path=rec['path'],absolute_path=str(p),bytes=count,sha256=man['sha256'],manifest=str(mp),manifest_sha256=digest(mp),new_payload_bytes=pack.seek(0,2)-before)

def stage(limit_gib):
    plan=json.loads((HERE/'ARCHIVE_PLAN.json').read_text(encoding='utf-8'))
    assert digest(HERE/'APPROVED_SCOPES.json')==plan['approved_scopes_sha']
    assert shutil.disk_usage(ROOT).free>(limit_gib*1.2+.5)*1024**3,'INSUFFICIENT_STAGING_SPACE'
    approval=json.loads((HERE/'APPROVED_SCOPES.json').read_text(encoding='utf-8'))
    for scope in approval['scopes']:
        assert digest(bounded(scope['report']))==scope['report_sha256'],'RESULT_ANALYSIS_CHANGED'
    c=connection();records=[];total=0
    batchnum=1+len(list(HERE.glob('DELETE_BATCH_*.json')))
    with open(PACK,'ab+') as pack:
        for rec in plan['candidates']:
            row=c.execute('SELECT state FROM files WHERE path=?',(rec['path'],)).fetchone()
            if row is not None:continue
            if records and total+rec['bytes']>limit_gib*1024**3:break
            records.append(stage_one(rec,c,pack));total+=rec['bytes']
            if len(records)%15==0:print(json.dumps(dict(stage_files=len(records),source_GiB=total/1024**3)),flush=True)
    bp=HERE/f'DELETE_BATCH_{batchnum:03d}.json'
    atomic_json(bp,dict(root=str(ROOT),format='VERIFIED_LOSSLESS_NPZ_CLEANUP_V1',verified=True,files=records))
    print(json.dumps(dict(batch=str(bp),files=len(records),source_GiB=total/1024**3,new_payload_GiB=sum(r['new_payload_bytes'] for r in records)/1024**3),ensure_ascii=False))
    c.close()

def reconcile():
    c=connection();missing=0;present=0;total=0
    for rel,mp,sha,size,state in list(c.execute('SELECT * FROM files')):
        p=bounded(rel)
        if p.exists():
            assert p.stat().st_size==size;present+=1
        else:
            # Only transition when independently reverified from persisted data.
            if state!='ARCHIVED_SOURCE_REMOVED':verify_manifest(HERE/mp,c)
            c.execute('UPDATE files SET state=? WHERE path=?',('ARCHIVED_SOURCE_REMOVED',rel));missing+=1;total+=size
    c.commit();print(json.dumps(dict(removed_files=missing,present_files=present,removed_logical_GiB=total/1024**3)));c.close()

def verify_all():
    c=connection();records=list(c.execute('SELECT path,manifest,sha,size,state FROM files'));total=0
    for i,(rel,mp,sha,size,state) in enumerate(records):
        man=verify_manifest(HERE/mp,c);assert man['sha256']==sha and man['bytes']==size;total+=size
        if i%100==0:print(json.dumps(dict(verified=i,total=len(records))),flush=True)
    plan_count=len(json.loads((HERE/'ARCHIVE_PLAN.json').read_text(encoding='utf-8'))['candidates'])
    removed=sum(r[4]=='ARCHIVED_SOURCE_REMOVED' for r in records)
    result=dict(status='PASS',files=len(records),plan_files=plan_count,archive_plan_complete=len(records)==plan_count,
                source_removal_complete=removed==plan_count,source_files_removed=removed,
                original_bytes=total,archive_reconstruction_all_SHA256_exact=True,
                unique_scientific_data_discarded=False,pack_bytes=PACK.stat().st_size,checked_at=time.time())
    atomic_json(HERE/'FINAL_ARCHIVE_VERIFICATION.json',result);print(json.dumps(result));c.close()

def restore(paths,all_files,destination):
    c=connection();records=list(c.execute('SELECT path,manifest,sha,size,state FROM files'))
    matches=[r for r in records if all_files or any(r[0].replace('\\','/').lower().startswith(s.replace('\\','/').lower()) for s in paths)]
    assert matches,'No matching archived paths'
    base=Path(destination).resolve() if destination else ROOT
    for rel,mp,sha,size,state in matches:
        target=(base/rel).resolve();assert target.is_relative_to(base) and target!=base
        if target.exists():
            assert digest(target)==sha,'Existing different output; refusing overwrite'
            continue
        target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.name+'.restoring')
        assert not tmp.exists(),'Existing partial restoration retained'
        man=json.loads((HERE/mp).read_text(encoding='utf-8'));h=hashlib.sha256();n=0
        with open(tmp,'xb') as f:
            for b in pieces(man,c):f.write(b);h.update(b);n+=len(b)
            f.flush();os.fsync(f.fileno())
        assert n==size and h.hexdigest()==sha
        os.replace(tmp,target);os.utime(target,ns=(man['mtime_ns'],man['mtime_ns']))
        print(json.dumps(dict(restored=str(target),sha256=sha),ensure_ascii=False),flush=True)
    c.close()

def main():
    ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest='command',required=True)
    s=sp.add_parser('stage');s.add_argument('--batch-gib',type=float,default=2)
    sp.add_parser('reconcile');sp.add_parser('verify')
    r=sp.add_parser('restore');r.add_argument('--path',action='append',default=[]);r.add_argument('--all',action='store_true');r.add_argument('--destination')
    a=ap.parse_args()
    if a.command=='stage':stage(a.batch_gib)
    elif a.command=='reconcile':reconcile()
    elif a.command=='verify':verify_all()
    else:restore(a.path,a.all,a.destination)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
