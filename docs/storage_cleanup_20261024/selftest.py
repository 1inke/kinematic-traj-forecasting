from pathlib import Path
import hashlib, json, os, sys, zipfile
import archive as a
import review

def main():
    test=a.HERE/'selftest';test.mkdir(exist_ok=True)
    a.STORE=test/'store';a.MAN=a.STORE/'manifests';a.PACK=a.STORE/'payloads.bin';a.DB=a.STORE/'index.sqlite'
    c=a.connection();records=[];raw=(bytes(range(256))*1000)
    with open(a.PACK,'ab+') as pack:
        for i in range(2):
            p=test/f'fixture{i}.npz'
            with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr('renamed_shared'+str(i)+'.npy',raw)
                z.writestr('different.npy',raw[:10000]+bytes([i]))
            st=p.stat();members=[]
            with open(p,'rb') as f:
                for typ,off,n,name in review.segments(p):
                    if typ=='payload':f.seek(off);members.append(dict(offset=off,length=n,name=name,sha256=hashlib.sha256(f.read(n)).hexdigest()))
            rec=dict(path=str(p.relative_to(a.ROOT)),bytes=st.st_size,mtime_ns=st.st_mtime_ns,scope='selftest',report='selftest',reason='synthetic reconstruction/dedup test',members=members)
            row=a.stage_one(rec,c,pack);records.append(row)
    assert c.execute('SELECT count(*) FROM chunks').fetchone()[0]==3,'Shared compressed data must dedup even when member filenames differ'
    a.restore([str(test.relative_to(a.ROOT))],False,str(test/'restored'))
    for row in records:
        p=test/'restored'/row['path'];assert a.digest(p)==row['sha256']
    mp=Path(records[0]['manifest']);bad=json.loads(mp.read_text(encoding='utf-8'));bad['sha256']='0'*64
    badp=test/'bad_manifest.json';badp.write_text(json.dumps(bad),encoding='utf-8')
    refused=False
    try:a.verify_manifest(badp,c)
    except AssertionError:refused=True
    assert refused,'Corrupt manifest must be refused'
    refused=False
    try:a.bounded('../outside_workspace.npz')
    except ValueError:refused=True
    assert refused,'Outside root must be refused'
    result=dict(status='PASS',exact_full_ZIP_restore=True,renamed_member_dedup=True,corrupt_manifest_refused=True,outside_root_refused=True,synthetic_only=True)
    (test/'RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result));c.close()

if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
