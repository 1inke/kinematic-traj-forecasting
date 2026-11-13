"""Finite discrete-utility execution. Only read-only imports from sealed ancestors."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
FEAS=ROOT.parent/'ENDPOINT_FEASIBLE_UPDATE_DIAGNOSIS_EXEC_20261031'
SET=ROOT.parent/'ENDPOINT_SET_UTILITY_RETENTION_EXEC_20261030'
OLD=ROOT.parent/'ENDPOINT_LEARNABILITY_CONTINUATION_EXEC_20261029'
P0=ROOT.parent/'ENDPOINT_TEMPORAL_OBSERVABILITY_EXEC_20261029'
sys.path.insert(0,str(FEAS));sys.path.insert(0,str(SET));sys.path.insert(0,str(OLD/'code'))
import common as legacy
from io_contract import sha,json_read,load_npz,npz_new
from action_geometry import space
from features import raw_features,transform as old_transform
import checks as reference

ACTIONS=space('END')
NONIDENTITY=np.array([k for k in range(49) if k!=ACTIONS.identity],dtype=np.int64)

def write(p,v):legacy.write(p,v)
def csv(p,rows):legacy.csv_new(p,rows)
def digest(x):return legacy.array_sha(x)
def resource(E=2*1024**3):return legacy.resource(E)
def verify(files,check=lambda:None):legacy.verify(files,check)

def inputs():
    g=resource();g.check(mode='startup',force=True);files={}
    def add(p,s):
        p=Path(p).resolve();k=str(p)
        if k in files and files[k]!=s:raise ValueError('inherited SHA conflict '+k)
        files[k]=s
    for name in ('source_manifest.json','FINAL_PRESERVATION_V2.json'):
        m=json_read(FEAS/name)
        for p,s in m['files'].items():add(Path(p) if Path(p).is_absolute() else FEAS/p,s)
        add(FEAS/name,sha(FEAS/name))
    for p in (FEAS/'FINAL_SEAL_CHECK_V2.json',PROJECT/'docs/ENDPOINT_DISCRETE_SET_UTILITY_DESIGN_20261101.md'):
        add(p,sha(p))
    t=time.perf_counter();verify(files,g.check)
    write(ROOT/'INPUT_SOURCE_LOCK.json',dict(status='PASS_ALL_INHERITED_ORIGINAL_DIGESTS',files=files,count=len(files),seconds=time.perf_counter()-t,no_replaced_digest=True))
    return files

def base_data(pop):
    if pop not in ('train','val'):raise ValueError('held data requires separately sealed formal admission')
    d=legacy.data(pop);m={}
    for p in sorted((P0/f'P0_END_{pop}_masks').glob('[0-9]*.npz')):
        z=load_npz(p)
        for k,v in z.items():m.setdefault(k,[]).append(v)
    m={k:np.concatenate(v) for k,v in m.items()}
    assert np.array_equal(m['sample_id'],d['x']['sample_id'])
    assert np.array_equal(m['safe'],m['legal']&~m['failed'].any(-1))
    d['valid']=m['safe'][:,:,NONIDENTITY].copy();d['safe']=m['safe'];d['mask_failed']=m['failed']
    assert np.array_equal(d['labels']['J_matrix'][:,:,ACTIONS.identity],d['nj'])
    return d

def subset(d,ix):
    return {k:({kk:vv[ix] for kk,vv in v.items()} if isinstance(v,dict) else v[ix]) for k,v in d.items()}

def binding(d):
    return dict(IDs=digest(d['x']['sample_id']),z=digest(d['z']),zG=digest(d['zG']),zH=digest(d['zH']),value=digest(d['value']),valid=digest(d['valid']),
        gt=digest(d['gt']),native=digest(d['x']['native']),old_stats=sha(P0/'prepared/statistics.npz'),new_stats=sha(ROOT/'prepared/statistics8.npz'),
        action_order=digest(ACTIONS.params),nonidentity_order=digest(NONIDENTITY))

def contract(d,arm,stage):
    return dict(stage=stage,arm=arm,protocol=sha(ROOT/'protocol.json'),input_lock=sha(ROOT/'INPUT_SOURCE_LOCK.json'),data=binding(d),
        code={p.name:sha(p) for p in sorted(ROOT.glob('*.py'))})

def atomic_checkpoint(p,state):
    import torch
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    temp=p.with_name(p.name+'.partial')
    with temp.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    os.rename(temp,p)

def trusted_checkpoint(p,s):
    if sha(p)!=s:raise ValueError('checkpoint SHA mismatch')
    from train import checkpoint_load_trusted
    return checkpoint_load_trusted(p)

def verify_prepared(folder):
    folder=Path(folder);receipt=json_read(folder/'done.json')
    for rel,s in receipt['files'].items():
        if sha(folder/rel)!=s:raise ValueError('prepared source/stat/action/split changed '+rel)
    return receipt

def read_prepared(pop='tiny'):
    if pop not in ('tiny','train','val'):raise ValueError('held not admitted')
    verify_prepared(ROOT/'prepared')
    d=base_data('train' if pop in ('train','tiny') else 'val')
    if pop=='tiny':
        ix=load_npz(OLD/'tiny_v2/indices.npz');d=subset(d,ix['indices']);assert np.array_equal(d['x']['sample_id'],ix['sample_id'])
    z=load_npz(ROOT/'prepared'/f'{pop}.npz')
    if not np.array_equal(z['sample_id'],d['x']['sample_id']):raise ValueError('prepared split/ID mismatch')
    if not np.array_equal(z['valid'],d['valid']):raise ValueError('prepared action mask mismatch')
    d.update(zH=z['zH'],zG=z['zG'],z=z['zH'],value=z['value'],q_edit=z['q_edit'])
    return d
