"""Execution safeguards outside the byte-preserved ETH-T1 scientific source."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import sys
import os

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'CONFIG.json').read_text(encoding='utf-8'))
WORKSPACE=Path(CONFIG['workspace'])
EXPLORE=Path(CONFIG['explore_dir'])
sys.path.insert(0,str(WORKSPACE/'qoder_work/FIVEFOLD_B0_PARALLEL_20261115/recovery_20261116'))
from safe_status import replace_json,transient


def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def verify(files):
    for p,digest in files.items():
        if sha(p)!=digest:raise ValueError('SOURCE_OR_OUTPUT_CHANGED: '+str(p))


def verify_source():verify(read(ROOT/'SOURCE_MANIFEST.json')['files'])


def status(path,value):
    obj=dict(at=now(),**value)
    try:replace_json(path,obj)
    except OSError as exc:
        if not transient(exc):raise
        rel=hashlib.sha256(str(path).encode()).hexdigest()[:16]
        try:replace_json(ROOT/'standalone/status_mirror'/f'{rel}.json',dict(original_path=str(path),value=obj,error=str(exc)))
        except OSError:
            # Monitoring alone must not terminate otherwise healthy training.
            try:
                p=ROOT/'standalone/warnings'/f'{os.getpid()}.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),path=str(path),error=str(exc)))+'\n')
            except OSError:pass


def imports():
    sys.path.insert(0,str(EXPLORE/'code'))
    import explore_common as ec
    # Keep fallback monitoring inside this new study, preserving the old archive.
    ec.status=status
    sys.path.insert(0,str(ROOT/'code'))
    import run_eth_t1 as original
    return ec,original


def write_seal(directory,name):
    files=[p for p in directory.iterdir() if p.is_file() and p.suffix in ('.json','.pt','.npz') and p.name not in (name,'EVALUATION_SEAL.json')]
    replace_json(directory/name,dict(at=now(),source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),files={str(p):sha(p) for p in files}))


def validate_seal(directory,name):
    record=read(directory/name)
    if record['source_manifest_sha256']!=sha(ROOT/'SOURCE_MANIFEST.json'):raise ValueError('RUN_SOURCE_MISMATCH')
    verify(record['files'])


def run_dir(arm):return ROOT/'runs'/f'{arm}_eth_E64_s42'
