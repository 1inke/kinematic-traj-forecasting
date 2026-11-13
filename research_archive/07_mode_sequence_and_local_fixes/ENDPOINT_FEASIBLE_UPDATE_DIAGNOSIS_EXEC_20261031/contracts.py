"""New execution namespace. Old sources are imported read-only, never old mains."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
PREV=ROOT.parent/'ENDPOINT_SET_UTILITY_RETENTION_EXEC_20261030'
OLD=ROOT.parent/'ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029'
sys.path.insert(0,str(PREV));sys.path.insert(0,str(OLD/'code'))
import common as old
from io_contract import sha,json_read,load_npz,npz_new
from action_geometry import space

def write(p,v):old.write(p,v)
def csv(p,rows):old.csv_new(p,rows)
def digest(v):return old.array_sha(v)
def resource(E=2*1024**3):return old.resource(E)
def data():
    ix=load_npz(OLD/'tiny_v2/indices.npz')
    d=old.data(indices=ix['indices'])
    assert np.array_equal(d['x']['sample_id'],ix['sample_id'])
    assert len(d['z'])==128 and len(np.unique(d['x']['track_key']))==97
    assert int((d['nj']<=1).any(1).sum())==69
    ports=load_npz(old.ATTR/'derived_ports.npz')
    for k in ('sample_id','native','past','gt','group','track_key'):
        v=d[k] if k=='gt' else d['x'][k]
        assert np.array_equal(v,ports[k]),k
    assert int((ports['safe']&(ports['J_matrix']<=1)).any((1,2)).sum())==90
    d['safe']=ports['safe']
    return d

def lock_inputs():
    g=resource();g.check(mode='startup',force=True);files={}
    def add(p,s):
        k=str(p.resolve())
        if k in files and files[k]!=s:raise ValueError('SHA conflict '+k)
        files[k]=s
    for name in ('source_manifest.json','FINAL_PRESERVATION_V2.json'):
        m=json_read(PREV/name)
        for k,s in m['files'].items():
            p=Path(k);add(p if p.is_absolute() else PREV/p,s)
        add(PREV/name,sha(PREV/name))
    for p in (PREV/'FINAL_SEAL_CHECK_V2.json',PROJECT/'docs/ENDPOINT_FEASIBLE_UPDATE_DIAGNOSIS_DESIGN_20261031.md'):
        add(p,sha(p))
    t=time.perf_counter();old.verify(files,g.check)
    write(ROOT/'INPUT_SOURCE_LOCK.json',dict(status='PASS_INHERITED_SHA',files=files,count=len(files),seconds=time.perf_counter()-t,
        inherited_production_count=json_read(PREV/'source_manifest.json')['source_count'],inherited_final_files=len(json_read(PREV/'FINAL_PRESERVATION_V2.json')['files'])))
    return files

def binding(d):return old.binding(d)|dict(safe=digest(d['safe']))
def protocol():return json_read(ROOT/'protocol.json')
def contract(d):
    code={p.name:sha(p) for p in ROOT.glob('*.py')}
    return dict(data=binding(d),protocol=sha(ROOT/'protocol.json'),code=code,input_lock=sha(ROOT/'INPUT_SOURCE_LOCK.json'))
def verify_contract(value,d):
    if value!=contract(d):raise ValueError('source/data/statistics/protocol mismatch')

def trusted_checkpoint(p,expected):
    if sha(p)!=expected:raise ValueError('checkpoint SHA mismatch')
    from train import checkpoint_load_trusted
    return checkpoint_load_trusted(p)

def atomic_checkpoint(p,state):
    """Commit-only, fsync, Windows no-overwrite rename; partial evidence retained."""
    import os
    import torch
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    tmp=p.with_name(p.name+'.partial')
    with tmp.open('xb') as f:
        torch.save(state,f);f.flush();os.fsync(f.fileno())
    os.rename(tmp,p)

ALPHAS=(1.,.5,.25,.125,.0625,.03125,.015625,.0078125)
