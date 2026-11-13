"""New execution utilities. No legacy runner imports or writes."""
import os, sys, json, hashlib, time, ctypes
from pathlib import Path
from datetime import datetime
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import psutil
H = Path(__file__).resolve().parent
S = H.parent
R = S.parent
RAW_ROOT = Path('<LOCAL_DESKTOP>/DRSE/ASTRA_ACCURACY_Q20')
INPUTS = {
    'cache': R/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz',
    'labels': R/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz',
    'encoder': R/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz',
    'observations': R/'GENERATION_UPSTREAM_TEMPORAL_SERIES/train_observations.npz',
    'metadata': R/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz',
    'split': R/'GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz',
    'raw': RAW_ROOT/'datasets/eth_ucy/processed_data_diverse/eth_data_train.npy',
}
def now(): return datetime.now().astimezone().isoformat()
def safe(x):
    if isinstance(x, np.ndarray): return safe(x.tolist())
    if isinstance(x, np.generic): return safe(x.item())
    if isinstance(x, Path): return str(x)
    if isinstance(x, dict): return {str(k): safe(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [safe(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x): return None
    return x
def write(name, value):
    p=Path(name); p=p if p.is_absolute() else H/p; p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.'+str(os.getpid())+'.'+str(time.time_ns())+'.tmp')
    with tmp.open('x',encoding='utf-8') as f:
        json.dump(safe(value),f,ensure_ascii=False,allow_nan=False,indent=2);f.flush();os.fsync(f.fileno())
    for i in range(40):
        try: os.replace(tmp,p); return
        except PermissionError:
            if i==39: raise
            time.sleep(.05)
def read(p):
    p=Path(p); return json.loads((p if p.is_absolute() else H/p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def digest(x): return hashlib.sha256(json.dumps(safe(x),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def array_sha(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def save(name, **arrays):
    p=H/name;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),('NO_OVERWRITE',str(p))
    np.savez_compressed(p,**arrays)
    with np.load(p,allow_pickle=False) as z:
        assert set(z.files)==set(arrays)
        for k,v in arrays.items(): assert np.array_equal(v,z[k],equal_nan=True) if np.asarray(v).dtype.kind in 'fc' else np.array_equal(v,z[k]),k
def verify(files):
    for i,(p,h) in enumerate(files.items()):
        if i%64==0: guard()
        assert sha(p)==h,('SOURCE_OR_OUTPUT_CHANGED',p)
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended')]
RESOURCE=dict(checks=0,peak_private_bytes=0,peak_RSS_bytes=0,min_physical_bytes=None,min_commit_bytes=None)
E=2*2**30
def guard(mode='runtime'):
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));p=psutil.Process().memory_info()
    limits=(1.3*E,1.3*(E if mode=='startup' else max(E-p.rss,0)),1.3*(E if mode=='startup' else max(E-p.private,0)))
    ok=max(p.rss,p.private,p.peak_wset,p.peak_pagefile)<=limits[0] and m.avail_phys>=limits[1] and m.avail_page>=limits[2]
    RESOURCE['checks']+=1
    RESOURCE['peak_private_bytes']=max(RESOURCE['peak_private_bytes'],int(p.peak_pagefile));RESOURCE['peak_RSS_bytes']=max(RESOURCE['peak_RSS_bytes'],int(p.peak_wset))
    for k,v in [('min_physical_bytes',int(m.avail_phys)),('min_commit_bytes',int(m.avail_page))]:RESOURCE[k]=v if RESOURCE[k] is None else min(v,RESOURCE[k])
    RESOURCE.update(estimated_peak_bytes=E,safety_factor=1.3,retained_lifetime_peaks=True)
    if not ok:
        write('RESOURCE_FAILURE.json',dict(at=now(),mode=mode,limits=limits,current=dict(RSS=p.rss,private=p.private,physical=m.avail_phys,commit=m.avail_page),resource=RESOURCE));raise MemoryError('STOP_RESOURCE_1P3')
    return dict(status='PASS',available_physical=int(m.avail_phys),available_commit=int(m.avail_page),resource=RESOURCE.copy())
def stage(name, **kw):
    guard();write('progress.json',dict(stage=name,at=now(),**kw));print(name,kw,flush=True)
def load():
    guard();cache=npz(INPUTS['cache']);enc=npz(INPUTS['encoder']);lab=npz(INPUTS['labels']);md=npz(INPUTS['metadata']);sp=npz(INPUTS['split'])
    ids=cache['sample_id'];assert len(ids)==28791 and len(np.unique(ids))==len(ids)
    assert np.array_equal(ids,enc['sample_id']) and np.array_equal(ids,lab['sample_id'])
    index={int(v):i for i,v in enumerate(md['sample_id'])};order=np.array([index[int(v)] for v in ids]);meta={k:v[order] for k,v in md.items()}
    with np.load(INPUTS['observations'],allow_pickle=False) as z:
        assert np.array_equal(ids,z['sample_id']);obs=z['observation'];assert np.array_equal(enc['past'].reshape(-1,16),obs[:,:16]) and np.array_equal(enc['past'],obs[:,160:176].reshape(-1,8,2))
    assert np.array_equal(sp['sample_id'],ids[:25912])
    split={n:sp[k] for n,k in [('FIT','grouped_train_index'),('outer','outer_eval_index'),('inner','inner_cal_index')]};split['original_CAL']=np.arange(25912,28791)
    for n,k in [('FIT','grouped_train_ids'),('outer','outer_eval_ids'),('inner','inner_cal_ids')]:assert np.array_equal(ids[split[n]],sp[k])
    assert tuple(map(len,split.values()))==(18909,1035,1856,2879)
    track=np.array([str(s)+'|'+str(int(p)) for s,p in zip(meta['sequence'],meta['pedestrian_id'])])
    for a,b in [('FIT','outer'),('FIT','inner'),('outer','inner')]:
        assert not np.intersect1d(ids[split[a]],ids[split[b]]).size and not np.intersect1d(track[split[a]],track[split[b]]).size
    past=enc['past'];assert past.shape==(28791,8,2) and np.count_nonzero(past[:,-1])==0
    data=dict(ids=ids,past=past,p2=enc['tokens'].mean(1).astype(np.float32),v0=(past[:,-1].astype(float)-past[:,-2].astype(float))/.4,gt=lab['future'].astype(float),native=cache['candidate'][...,:24].reshape(-1,20,12,2),groups=cache['change_group'].astype(int),track=track,scene=meta['scene_family'],meta=meta,split=split)
    assert all(np.isfinite(data[k]).all() for k in ('past','p2','v0','gt','native'));guard();return data
def masks(groups):return [('overall',np.ones(len(groups),bool))]+[(n,groups==i) for i,n in enumerate(('stable','transition','change'))]
