"""Fail-closed stage/lock contracts. Does not confer human S4 authorization."""
import os
from pathlib import Path
from .io import sha256,read_json,write_json

def validate_registration(p):
    required=('experiment_id','hypothesis','primary','control','treatment','budget','protocol_version')
    missing=[k for k in required if k not in p]
    if missing:raise ValueError('Missing registration: '+','.join(missing))
    if not isinstance(p['primary'],str) or p['primary']!='delta_cov_outer_change':raise ValueError('Exactly one protocol PRIMARY required')
    if len(p['treatment'])>2:raise ValueError('At most two S3 treatments')
    if p['budget'].get('gpu_hours') is None or p['budget'].get('max_runs') is None:raise ValueError('Explicit task budget required')
    return True

def require_stage(stage,protocol,*,human_signature=None):
    if stage not in ('S1','S2','S3','S4'):raise ValueError('Unknown stage')
    if stage in ('S2','S3','S4'):
        if protocol.get('state')!='FROZEN' or not protocol.get('author_signed_freeze') or protocol.get('freeze_blockers'):raise PermissionError('Protocol is not author-frozen')
    if stage=='S4' and not human_signature:raise PermissionError('S4 requires separate human signature')

def consume_once(ledger_path,*,experiment,configuration_sha,split,direction,stage,protocol,human_signature=None):
    require_stage(stage,protocol,human_signature=human_signature)
    if split=='original_CAL' and stage!='S4':raise PermissionError('CAL is sealed outside S4')
    if split=='outer' and stage not in ('S3','S4'):raise PermissionError('outer outside preregistered evaluation stage')
    key=f'{direction}:CAL' if split=='original_CAL' else f'{experiment}:{configuration_sha}:{split}'
    path=Path(ledger_path);path.parent.mkdir(parents=True,exist_ok=True)
    lock=path.with_suffix(path.suffix+'.lock');fd=None
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        value=read_json(path) if path.exists() else {'uses':{}}
        if key in value['uses']:raise PermissionError('Evaluation already consumed (failed runs require review, not blind retry)')
        value['uses'][key]={'stage':stage,'experiment':experiment,'configuration_sha':configuration_sha,'split':split};write_json(path,value)
    finally:
        if fd is not None:os.close(fd);lock.unlink()

def lock_protocol(path,expected_sha):
    if sha256(path)!=expected_sha:raise ValueError('Locked protocol changed')
