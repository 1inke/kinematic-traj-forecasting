"""Read-only experiment/report inventory and exact ZIP-payload deduplication estimate."""
from pathlib import Path
import collections, hashlib, json, os, re, struct, sys, tempfile, time, zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
INV = Path(tempfile.gettempdir()) / 'codex_project_storage_inventory_20261024.json'
GIB = 1024**3

def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''): h.update(b)
    return h.hexdigest()

def inventory():
    rows = json.loads(INV.read_text(encoding='utf-8'))['files']
    roots, reports, sources = collections.defaultdict(list), collections.defaultdict(list), []
    for row in rows:
        rel = row[0]; p = ROOT / rel
        if rel.lower().endswith('.npz') and p.exists(): roots[rel.split('\\')[0]].append(row)
        if p.suffix.lower() == '.py' and 'site-packages' not in rel and '\\.git\\' not in rel:
            sources.append(rel)
        if p.suffix.lower() != '.md' or not p.exists(): continue
        name = p.name.lower()
        if not any(s in name for s in ('result_analysis', 'final_report', 'report.md', 'series_report', 'decision', '分析')): continue
        raw = p.read_bytes(); txt = raw.decode('utf-8-sig', errors='replace').replace('\\n', '\n')
        # Keep explicit outcomes with context; an implementation PASS is not a scientific PASS.
        lines = [s.strip() for s in txt.splitlines() if s.strip()]
        outcome = [s[:600] for s in lines if re.search(r'NOT_SUPPORTED|FALSIFIED|SCIENTIFIC_NEGATIVE|GATE_FAIL|major\s*FAIL|MAJOR_FAIL|STOP_[A-Z_]+|Decision:|verdict|科学.*(FAIL|失败|停止)|不启动|不再继续|停止.*(分支|方案|主线)|不能替换', s, re.I)]
        rec = dict(path=rel, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), read_full=True,
                   opening=' '.join([s for s in lines if not s.startswith('#')][:2])[:700], outcome=outcome[:12])
        reports[rel.split('\\')[0]].append(rec)
    reviews = []
    for name, files in sorted(roots.items(), key=lambda kv: -sum(r[1] for r in kv[1])):
        rr = sorted(reports[name], key=lambda r: (r['path'].count('\\'), 'result_analysis' not in r['path'].lower()))
        reviews.append(dict(root=name, npz_bytes=sum(r[1] for r in files), npz_count=len(files),
                            reports=rr, default_action='KEEP_UNLESS_EXPLICITLY_APPROVED', files=files))
    result = dict(root=str(ROOT), time=time.time(), roots=reviews, source_files=sources,
                  note='Read full report text; extracted outcomes are discovery aids, not automatic deletion authorization.')
    (HERE/'REPORT_REVIEW.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(roots=len(reviews),reports=sum(len(r['reports']) for r in reviews),npz_GiB=sum(r['npz_bytes'] for r in reviews)/GIB),ensure_ascii=False))

def segments(path):
    """Partition entire ZIP into preserved literal framing and compressed member payloads."""
    size = path.stat().st_size
    with zipfile.ZipFile(path) as z, open(path,'rb') as f:
        entries = sorted(z.infolist(), key=lambda i:i.header_offset)
        end = 0
        for zi in entries:
            f.seek(zi.header_offset); header=f.read(30)
            assert len(header)==30 and header[:4]==b'PK\x03\x04'
            fields=struct.unpack('<4s5H3I2H',header)
            start=zi.header_offset+30+fields[-2]+fields[-1]
            assert end<=start<=start+zi.compress_size<=size
            yield ('literal',end,start-end,None)
            yield ('payload',start,zi.compress_size,zi.filename)
            end=start+zi.compress_size
        yield ('literal',end,size-end,None)

def estimate():
    approval=json.loads((HERE/'APPROVED_SCOPES.json').read_text(encoding='utf-8'))
    review=json.loads((HERE/'REPORT_REVIEW.json').read_text(encoding='utf-8'))
    scope=approval['scopes']; candidates=[]; kept=[]
    protected=approval['protected_roots']; outcomes={}
    for row in review['roots']:
        for r in row['files']:
            rel,size=r[:2]; parts=rel.split('\\'); low=rel.lower(); p=ROOT/rel
            hits=[s for s in scope if low.startswith(s['prefix'].lower())]
            why=None
            if parts[0] in protected: why='CURRENT_INPUT_OR_RECENT_ROOT'
            elif not hits: why='NO_REVIEWED_RETIRED_SCOPE'
            elif size<8*1024**2: why='SMALL_EVIDENCE_RETAINED'
            elif any(k in parts[-1].lower() for k in ('split','statistics','stats','labels','metadata','indices','manifest','bootstrap','draws','normaliz')): why='REPRODUCTION_INPUT_OR_STATISTICAL_EVIDENCE'
            elif any(k in low for k in ('\\data\\','\\prepared\\','\\cache\\','\\latent_cache','\\features\\','\\failures\\','\\preflight','\\smoke','\\resume_smoke')):why='INPUT_OR_PREFLIGHT_PRESERVED'
            elif not p.exists() or p.stat().st_size!=size:why='STALE_INVENTORY'
            if why: kept.append(dict(path=rel,bytes=size,reason=why));continue
            candidates.append(dict(path=rel,bytes=size,scope=hits[0]['prefix'],report=hits[0]['report'],reason=hits[0]['reason']))
    unique={}; perroot=collections.defaultdict(lambda:dict(original=0,new_payload=0,count=0)); records=[]
    for i,rec in enumerate(candidates):
        p=ROOT/rec['path'];root=rec['path'].split('\\')[0]; ps=perroot[root];ps['original']+=rec['bytes'];ps['count']+=1
        st=p.stat(); member_records=[]; framing=0
        with open(p,'rb') as f:
            for typ,offset,n,name in segments(p):
                if typ=='literal':framing+=n;continue
                f.seek(offset);h=hashlib.sha256();remain=n
                while remain:
                    b=f.read(min(remain,4*1024**2));assert b;h.update(b);remain-=len(b)
                key=h.hexdigest()
                if key not in unique:unique[key]=n;ps['new_payload']+=n
                else:assert unique[key]==n
                member_records.append(dict(name=name,offset=offset,length=n,sha256=key))
        assert p.stat().st_size==st.st_size and p.stat().st_mtime_ns==st.st_mtime_ns
        rec.update(mtime_ns=st.st_mtime_ns,framing_bytes=framing,members=member_records)
        records.append(rec)
        if i%50==0:print(json.dumps(dict(progress=i,total=len(candidates),unique_payload_GiB=sum(unique.values())/GIB)),flush=True)
    plan=dict(root=str(ROOT),approved_scopes_sha=sha_file(HERE/'APPROVED_SCOPES.json'),candidates=records,kept=kept,
              original_bytes=sum(r['bytes'] for r in records),unique_compressed_payload_bytes=sum(unique.values()),perroot=perroot,
              lossless=True,unique_scientific_arrays_discarded=False,weights_deleted=False)
    (HERE/'ARCHIVE_PLAN.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(records),original_GiB=plan['original_bytes']/GIB,unique_payload_GiB=sum(unique.values())/GIB,
                         gross_duplicate_GiB=(plan['original_bytes']-sum(unique.values()))/GIB,perroot=perroot),ensure_ascii=False))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    {'inventory':inventory,'estimate':estimate}[sys.argv[1]]()
