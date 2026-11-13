"""Source-bound shared IO and data, no import-time training or writes."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
OLD=ROOT.parent/'ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029'
P0=ROOT.parent/'ENDPOINT_TEMPORAL_OBSERVABILITY_EXEC_20261029'
ATTR=ROOT.parent/'ENDPOINT_SUPPORT_FAILURE_ATTRIBUTION_EXEC_20261030'
sys.path.insert(0,str(OLD/'code'))
from io_contract import load_npz,json_read,json_new,npz_new,sha
from features import transform
from action_geometry import space

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def write(p,x):json_new(p,clean(x))
def csv_new(p,rows):
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(clean(rows))
def array_sha(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes());return h.hexdigest()
def code_sha():return hashlib.sha256(json.dumps({p.name:sha(p) for p in sorted(ROOT.glob('*.py'))},sort_keys=True).encode()).hexdigest()

def resource(estimate=2*1024**3):
    spec=importlib.util.spec_from_file_location('set_utility_guard_v3',ROOT.parent/'K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod.Guard(estimate)

def verify(mapping,check=lambda:None):
    for i,(name,s) in enumerate(mapping.items()):
        if sha(name)!=s:raise ValueError('SOURCE_CHANGED '+str(name))
        if i%50==0:check()

def input_lock():
    g=resource();g.check(mode='startup',force=True);files={}
    def inherit(mapping,base):
        for name,s in mapping.items():
            p=Path(name);p=p if p.is_absolute() else base/p;p=p.resolve();key=str(p)
            if key in files and files[key]!=s:raise ValueError('inherited SHA conflict')
            files[key]=s
    for name in ('SOURCE_EXTENSION.json','FINAL_SOURCE_ADDITIONS.json'):
        inherit(json_read(ATTR/name)['files'],ATTR)
    inherit(json_read(ATTR/'FINAL_PRESERVATION_V2.json')['files'],ATTR)
    for p in (ATTR/'FINAL_PRESERVATION_V2.json',ATTR/'FINAL_SEAL_CHECK_V2.json',PROJECT/'docs/ENDPOINT_SET_UTILITY_RETENTION_DESIGN_20261030.md'):
        files[str(p.resolve())]=sha(p)
    started=time.perf_counter();verify(files,g.check)
    write(ROOT/'INPUT_SOURCE_LOCK.json',dict(status='PASS_ALL_INHERITED_INPUTS_INITIAL_SHA',files=files,
        inherited5140_unchanged=True,source_count=len(files),seconds=time.perf_counter()-started,
        no_old_PID_task_log_queries=True,no_old_file_mutation=True))
    return files

def data(pop='train',indices=None):
    x=load_npz(P0/f'prepared/X_{pop}.npz');y=load_npz(P0/f'prepared/GT_{pop}.npz')
    assert np.array_equal(x['sample_id'],y['sample_id'])
    labels={}
    for p in sorted((P0/f'P0_END_{pop}_labels').glob('[0-9]*.npz')):
        d=load_npz(p)
        for k,v in d.items():labels.setdefault(k,[]).append(v)
    labels={k:np.concatenate(v) for k,v in labels.items()}
    assert np.array_equal(labels['sample_id'],x['sample_id'])
    if indices is not None:
        x={k:v[indices] for k,v in x.items()};y={k:v[indices] for k,v in y.items()};labels={k:v[indices] for k,v in labels.items()}
    stats=load_npz(P0/'prepared/statistics.npz');z=transform(x['past'],x['native'],stats,'H')
    err=np.linalg.norm(x['native']-y['gt'][:,None],axis=-1)
    nj=np.maximum(err.mean(-1)/.2,err[...,-1]/.4)
    if not np.all(labels['best_J'].min(1)<=nj.min(1)+2e-12):raise AssertionError('safe oracle worse than identity')
    return dict(x=x,gt=y['gt'],z=z,target=labels['target'].astype(np.float32),nj=nj,best=labels['best_J'],labels=labels)

def subset(d,ids):
    return {k:({kk:vv[ids] for kk,vv in v.items()} if isinstance(v,dict) else v[ids]) for k,v in d.items()}

def binding(d):
    return {key:array_sha(d[key]) for key in ('z','target','gt','nj','best')}|dict(
        IDs=array_sha(d['x']['sample_id']),native=array_sha(d['x']['native']),
        original_support=array_sha(d['nj']<=1),statistics=sha(P0/'prepared/statistics.npz'))

def production_lock():
    files=dict(json_read(ROOT/'INPUT_SOURCE_LOCK.json')['files'])
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix not in ('.pyc',) and '__pycache__' not in p.parts and 'tiny' not in p.parts:
            files[str(p.resolve())]=sha(p)
    write(ROOT/'source_manifest.json',dict(status='PRODUCTION_SOURCE_LOCK',files=files,code_sha=code_sha(),
        source_count=len(files),no_old_digest_replacement=True))
    return files
