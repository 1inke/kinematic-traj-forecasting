"""Capture actual checks/environment, without claiming scientific S3 success."""
import os,sys,subprocess,hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from f16.io import sha256,read_json,write_json,verify_manifest
from f16.provenance import environment_record
from audit.security_scan import scan
import importlib.metadata

def main():
    out=Path(__file__).resolve().parent;tree=ET.parse(out/'pytest.xml');suite=tree.getroot().find('testsuite');errors,n=scan()
    if errors or int(suite.attrib['failures']) or int(suite.attrib['errors']):raise RuntimeError('Tests/security failed')
    verify_manifest(read_json(ROOT/'config/data_manifest.v1.json')['files'],ROOT)
    overlap=read_json(ROOT/'TRACK_OVERLAP_AUDIT_20261020/OVERLAP_REPORT.json')
    if overlap['script_sha256']!=sha256(ROOT/'TRACK_OVERLAP_AUDIT_20261020/check_overlap.py'):raise RuntimeError('Overlap script changed after audit')
    d0=ROOT/'experiments/2026-10-20_D0_protocol_calibration';audits=list(d0.glob('*_AUDIT.json'))
    if len(audits)!=7 or any(read_json(p)['status']!='PASS' or not read_json(p)['frozen_labels_verified'] or read_json(p)['script_sha']!=sha256(ROOT/'audit/verify.py') for p in audits):raise RuntimeError('Missing or stale independent audit')
    r=read_json(d0/'RESULT.json')
    if not r['preservation'] or any(v['independent_full_draw_maxdiff']>2e-12 for v in r['calibration'].values()):raise RuntimeError('Data/statistics audit failed')
    sources=[]
    for base in ['src','audit','config','tests','.github','TRACK_OVERLAP_AUDIT_20261020','experiments/2026-10-20_D0_protocol_calibration']:
        sources.extend(p for p in (ROOT/base).rglob('*') if p.is_file() and p.suffix in ['.py','.json','.toml','.yaml','.yml','.md'] and '__pycache__' not in p.parts and p.name!='paths.yaml')
    sources.extend(ROOT/p for p in ['PROTOCOL.md','DESIGN_AB.md','BASELINE.md','README.md','pyproject.toml','email_notifier.py'])
    env=environment_record(ROOT,ROOT/'config/protocol_v1.json',ROOT/'config/data_manifest.v1.json',sources)
    env['package_versions']={k:importlib.metadata.version(k) for k in ['numpy','scipy','torch','psutil','pytest']};env['source_manifest_is_post_implementation_not_historical_runtime_lock']=True
    write_json(out/'ENVIRONMENT_AND_SOURCE.json',env)
    localfreeze=subprocess.run([sys.executable,'-m','pip','freeze'],text=True,capture_output=True,check=True).stdout;(out/'environment.local.txt').write_text(localfreeze,encoding='utf-8')
    transitions={}
    for name in ['README.md','.gitignore','email_notifier.py']:
        old=subprocess.run(['git','show','HEAD:'+name],cwd=ROOT,capture_output=True,check=True).stdout
        transitions[name]=dict(previous_git_blob_content_sha256=hashlib.sha256(old).hexdigest(),current_sha256=sha256(ROOT/name),reason='Authorized protocol-era documentation/configuration/security change; not scientific result rewrite')
    copied={}
    downloads=Path.home()/'Downloads'
    for name in ['PROTOCOL.md','DESIGN_AB.md']:
        original=downloads/name;copy=ROOT/name
        if original.exists():
            same=original.read_text(encoding='utf-8-sig').replace('\r\n','\n').strip()==copy.read_text(encoding='utf-8-sig').replace('\r\n','\n').strip()
            if not same:raise RuntimeError('Author document content changed')
            copied[name]=dict(original_sha256=sha256(original),project_copy_sha256=sha256(copy),text_equal_ignoring_line_endings_and_terminal_whitespace=same)
    write_json(out/'VERIFICATION.json',dict(status='ENGINEERING_CHECKS_COMPLETE_SCIENTIFIC_FREEZE_BLOCKED',pytest=suite.attrib,security=dict(scanned_text_files=n,findings=0),independent_path_audits=7,independent_statistical_contrasts=3,bootstrap_draws_each=20000,author_documents=copied,authorized_root_transitions=transitions,budget=dict(formal_training_runs=0,gpu_training_hours=0),frozen_input_SHA_verified=True,formal_protocol_frozen=False,AB_training_started=False,old_directories_moved=0,limitations=['CPU synthetic optimizer/RNG test, not new A/B actual GPU resume.','No old checkpoint upgraded to complete-resume status.','C3/C6 and author decisions still OPEN.','New source manifest is a post-implementation record.']))
    print(json.dumps({'tests':suite.attrib,'independent_saved_audits':7,'security_findings':0,'formal_freeze':False,'AB_training':False}))

if __name__=='__main__':main()
