"""Explicit prospective source snapshots; never replace old source locks."""
import shutil,time
from env import *
from data_pipeline import CACHE,verify_prepared
def build_manifest():
    old=read_json(E0FINAL/'source_manifest.json');files=old['files'].copy()
    for p,ss in files.items():
        if sha256(ROOT/p)!=ss:raise ValueError('Inherited E0 source '+p)
    add=list(p for p in HERE.iterdir() if p.suffix in ('.py','.ps1','.md') or p.name=='protocol.json')
    add.extend([E0FINAL/p for p in ('RESULT_ANALYSIS.md','RESULT_ANALYSIS.json','RESULT.json','DECISION.json','ARITHMETIC_AUDIT.json','PRESERVATION.json','RESOURCE.json','worker_state.json','exit_status.json','source_manifest.json','completion_review/REVIEW.json','completion_review/ALL12_GROUPS.csv','completion_review/analyze.py')])
    add.extend([E0FINAL/f'seed{s}/ARITHMETIC_AUDIT.json' for s in SEEDS])
    add.extend(ROOT/f'COMPLEMENTARY_POOL_VALIDATION_20261018/artifacts/v1/kin_inner_seed{s}.npy' for s in SEEDS)
    ready=verify_prepared()
    add.append(CACHE/'ready.json');add.extend(CACHE/p for p in ready['outputs'])
    for p in add:
        key=p.relative_to(ROOT).as_posix();ss=sha256(p)
        if key in files and files[key]!=ss:raise ValueError('Cannot overwrite inherited digest')
        files.setdefault(key,ss)
    return dict(files=files,digest=e0.digest(files),inherited_count=len(old['files']),at=e0.now())
def verify(manifest=None):
    m=manifest or read_json(HERE/'source_manifest.json')
    if e0.digest(m['files'])!=m['digest']:raise ValueError('Source manifest canonical digest')
    for p,ss in m['files'].items():
        if sha256(ROOT/p)!=ss:raise ValueError('Current source SHA '+p)
    return m['digest']
def snapshot_candidate(manifest):
    folder=HERE/'preflight_versions'/str(time.time_ns());folder.mkdir(parents=True)
    for p in HERE.iterdir():
        if p.suffix in ('.py','.ps1','.md') or p.name=='protocol.json':shutil.copyfile(p,folder/p.name)
    write_json(folder/'CANDIDATE_SOURCE.json',manifest);return folder
