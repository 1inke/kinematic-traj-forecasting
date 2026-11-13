"""Strict saved-only Q0/Q1 admission utilities; import has no write or forward."""
import sys,json,hashlib,csv
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE/'runtime_deps'));sys.path.insert(0,str(ROOT/'src'))
from f16.resource import Guard
PREVIOUS=ROOT/'qoder_work/DECODER_THREE_DIRECTION_EXEC_20261024'
OLD=ROOT/'qoder_work/F16_CONTEXT_COMPLEMENTARITY_DESIGN_20261022/execution_20261022'
DESIGN=ROOT/'docs/DECODER_GEOMETRY_COMPLEMENTARITY_DESIGN_20261024'
OUT=HERE/'q0_attribution';GROUPS=('overall','stable','transition','change');SEEDS=(42,43,44)
E=3*2**30
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x))
def write(p,v):
    p=Path(p).resolve();assert p.is_relative_to(HERE)
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=plain,allow_nan=False),encoding='utf-8')
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def csvout(p,rows):
    with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def digest(files):return hashlib.sha256(json.dumps({k:v['sha256'] for k,v in files.items()},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def verify_manifest():
    m=read(HERE/'source_manifest.json');assert digest(m['files'])==m['digest']
    for v in m['files'].values():assert sha(v['path'])==v['sha256'],('SOURCE_CHANGED',v['path'])
    return m['digest']
def geometry(p,y):
    dist=np.sqrt(np.square(p.astype(float)-y[:,None].astype(float)).sum(-1));a=dist.mean(2);f=dist[:,:,11];j=np.maximum(a/.2,f/.4)
    good=(a<=.2)&(f<=.4)
    return dict(ADE=a.min(1),FDE=f.min(1),J=j.min(1),slot_J=j,slot_good=good,covered=good.any(1))
def physical(p):
    v=np.concatenate([p[:,:,:1].astype(float),np.diff(p.astype(float),axis=2)],axis=2)/.4;a=np.diff(v,axis=2)/.4;j=np.diff(a,axis=2)/.4
    return [np.sqrt(np.square(x).sum(-1)) for x in (v,a,j)]
def pool(arr,mask):
    if not mask.any():return {k:None for k in ('speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95')}
    s,a,j=arr
    return dict(speed_p95=float(np.percentile(s[mask],95)),accel_mean=float(a[mask].mean()),accel_p95=float(np.percentile(a[mask],95)),jerk_mean=float(j[mask].mean()),jerk_p95=float(np.percentile(j[mask],95)))
def clusters(p,good,independent=False):
    from scipy.cluster.hierarchy import linkage,fcluster
    from scipy.spatial.distance import pdist
    labels=[];distinct=[];effective=[]
    for row,g in zip(p.astype(float),good):
        if independent:
            mean=sum(pdist(row[:,t,:],'euclidean') for t in range(12))/12
            cond=np.maximum(mean/.10,pdist(row[:,-1,:],'euclidean')/.20)
        else:
            dd=np.sqrt(np.square(row[:,None]-row[None]).sum(-1));cond=np.maximum(dd.mean(-1)/.10,dd[:,:,-1]/.20)[np.triu_indices(20,1)]
        ll=fcluster(linkage(cond,method='complete'),1,criterion='distance').astype(np.int8)
        labels.append(ll);distinct.append(len(set(ll)));effective.append(len(set(ll[g])))
    return np.asarray(labels),np.asarray(distinct),np.asarray(effective)
def partitions(a,b):return dict(both_good=a&b,iso_only=a&~b,mix_only=~a&b,both_bad=~a&~b)
def assert_close(a,b):np.testing.assert_allclose(a,b,atol=2e-12,rtol=0)
