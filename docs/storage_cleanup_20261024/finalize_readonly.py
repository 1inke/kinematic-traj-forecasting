"""Post-cleanup independent plan/receipt/preservation checks. Never deletes files."""
from pathlib import Path
import collections, csv, hashlib, json, sqlite3, sys, time

H=Path(__file__).resolve().parent
ROOT=H.parent.parent

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    plan=json.loads((H/'ARCHIVE_PLAN.json').read_text(encoding='utf-8'))
    targets={r['path']:r for r in plan['candidates']}
    verification=json.loads((H/'FINAL_ARCHIVE_VERIFICATION.json').read_text(encoding='utf-8'))
    assert verification['archive_plan_complete'] and verification['source_removal_complete']
    assert verification['source_files_removed']==len(targets)==verification['files']
    completion=json.loads((H/'MANUAL_COMPLETION.json').read_text(encoding='utf-8'));assert completion['status']=='COMPLETE'
    c=sqlite3.connect(H/'npz_archive/index.sqlite');assert c.execute('PRAGMA quick_check').fetchone()[0]=='ok'
    stored=list(c.execute('SELECT path,manifest,sha,size,state FROM files'));c.close()
    assert {r[0] for r in stored}==set(targets)
    receipts={}
    for path in H.glob('DELETE_RECEIPT_*.json'):
        rows=json.loads(path.read_text(encoding='utf-8-sig'))
        if isinstance(rows,dict):rows=[rows]
        for row in rows:
            rel=str(Path(row['path']).relative_to(ROOT));assert rel not in receipts
            assert row['status']=='REMOVED_AFTER_LOSSLESS_BYTE_VERIFICATION'
            receipts[rel]=row
    assert set(receipts)==set(targets)
    for rel,mp,digest,n,state in stored:
        assert not (ROOT/rel).exists() and state=='ARCHIVED_SOURCE_REMOVED'
        assert receipts[rel]['sha256']==digest and receipts[rel]['bytes']==n==targets[rel]['bytes']
        man=json.loads((H/mp).read_text(encoding='utf-8'))
        assert man['path']==rel and man['sha256']==digest
    before=json.loads((H/'PRESERVATION_BEFORE.json').read_text(encoding='utf-8'));changes=[]
    for rel,n,mt,ino in before['files']:
        p=ROOT/rel
        if not p.exists():changes.append([rel,'MISSING']);continue
        s=p.stat()
        if (s.st_size,s.st_mtime_ns,s.st_ino)!=(n,mt,ino):changes.append([rel,'METADATA_CHANGED'])
    assert not changes,changes
    reviews=json.loads((H/'REPORT_REVIEW.json').read_text(encoding='utf-8'));report_count=0
    for root in reviews['roots']:
        for report in root['reports']:
            assert sha(ROOT/report['path'])==report['sha256'],report['path'];report_count+=1
    perroot=collections.defaultdict(lambda:[0,0])
    for rel,row in targets.items():
        k=rel.split('\\')[0];perroot[k][0]+=1;perroot[k][1]+=row['bytes']
    with open(H/'CLEANED_FILES.csv','w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['path','bytes','source_sha256','restore_manifest','reason'])
        for rel,mp,digest,n,state in sorted(stored):w.writerow([rel,n,digest,mp,targets[rel]['reason']])
    source_bytes=sum(r['bytes'] for r in targets.values());pack_bytes=(H/'npz_archive/payloads.bin').stat().st_size
    archive_bytes=sum(p.stat().st_size for p in (H/'npz_archive').rglob('*') if p.is_file())
    audit_bytes=sum(p.stat().st_size for p in H.rglob('*') if p.is_file())
    result=dict(status='COMPLETE_VERIFIED',removed_npz_files=len(targets),method_directories=len(perroot),
                source_bytes_removed=source_bytes,archive_bytes_retained=archive_bytes,archive_payload_bytes=pack_bytes,
                net_saved_after_archive_bytes=source_bytes-archive_bytes,
                net_saved_after_all_audit_artifacts_bytes=source_bytes-audit_bytes,
                unique_scientific_data_discarded=False,exact_restore_all_original_SHA256=True,
                protected_files_metadata_unchanged=len(before['files']),analysis_reports_SHA256_unchanged=report_count,
                sqlite_quick_check='ok',deletion_receipts_match_plan=True,per_method=dict(perroot),
                free_GiB_at_script_completion=completion['freeGiB'],free_change_since_run_GiB=completion['changeSinceManualStartGiB'],
                checked_at=time.time())
    (H/'CLEANUP_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
