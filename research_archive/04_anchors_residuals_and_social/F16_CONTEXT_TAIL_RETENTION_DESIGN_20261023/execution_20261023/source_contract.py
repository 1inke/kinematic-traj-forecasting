"""Pure CPU source/identity utilities; no legacy main or Torch import."""
from pathlib import Path
import sys,os,json,hashlib,time
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OLD=ROOT/'qoder_work/F16_CONTEXT_COMPLEMENTARITY_DESIGN_20261022/execution_20261022'
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from f16.io import sha256,read_json,write_json
from f16.resource import Guard
SEEDS=(42,43,44)
E_DIAG=3*2**30

def now():
    from datetime import datetime,timezone,timedelta
    return datetime.now(timezone(timedelta(hours=8))).isoformat()

def stage(value,**kw):
    state=read_json(HERE/'runtime_state.json') if (HERE/'runtime_state.json').exists() else {}
    state.update(stage=value,at=now(),**kw);write_json(HERE/'runtime_state.json',state);print(value,kw,flush=True)

def inherit():
    parent=read_json(OLD/'source_manifest.json');files={}
    for key,v in parent['files'].items():
        p=Path(v['path']);assert sha256(p)==v['sha256'],('PRIOR_SOURCE_CHANGED',key)
        files[key]=dict(path=str(p),sha256=v['sha256'],inherited=True)
    ext=read_json(OLD/'source_extension_prelaunch.json');assert ext['base_source_digest']==parent['digest']
    for key,digest in ext['extension_files'].items():
        p=ROOT/key;assert sha256(p)==digest;files[p.relative_to(ROOT).as_posix()]=dict(path=str(p),sha256=digest,inherited=True)
    for p in OLD.rglob('*'):
        if not p.is_file() or '__pycache__' in str(p):continue
        key=p.relative_to(ROOT).as_posix();actual=sha256(p)
        if key in files:assert files[key]['sha256']==actual
        else:files[key]=dict(path=str(p),sha256=actual,inherited=False)
    for p in list(HERE.parent.glob('*.md'))+[HERE.parent/'protocol.draft.json']+list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))+[HERE/'protocol.json',HERE/'MEMORY_ESTIMATE.json']:
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=dict(path=str(p),sha256=sha256(p),inherited=False)
    digest=hashlib.sha256(json.dumps({k:v['sha256'] for k,v in sorted(files.items())},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write_json(HERE/'source_manifest.json',dict(digest=digest,parent_digest=parent['digest'],inherited_old_digest_count=len(parent['files']),prelaunch_extension_count=len(ext['extension_files']),files=files))
    return digest

def verify():
    mf=read_json(HERE/'source_manifest.json')
    for k,v in mf['files'].items():assert sha256(v['path'])==v['sha256'],('SOURCE_CHANGED',k)
    return mf['digest']

def job(seed,split):
    run=OLD/f'runs/A_seed{seed}';done=read_json(run/'done.json')
    for rel,digest in done['outputs'].items():assert sha256(OLD/rel)==digest,('OLD_OUTPUT_CHANGED',rel)
    with np.load(run/f'predictions_{split}.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    with np.load(OLD/'split_ids.npz') as s:assert np.array_equal(s[split],d['sample_id'])
    assert len(np.unique(d['sample_id']))==len(d['sample_id']) and set(np.unique(d['groups'])).issubset({0,1,2})
    with np.load(ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz') as z:
        loc={int(s):i for i,s in enumerate(z['sample_id'])};rows=[loc[int(s)] for s in d['sample_id']];d['past']=z['past'][rows].copy()
    with np.load(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz') as z:
        loc={int(s):i for i,s in enumerate(z['sample_id'])};rows=[loc[int(s)] for s in d['sample_id']];d['sequence']=z['sequence'][rows].astype(str);d['scene']=z['scene_family'][rows].astype(str)
        expected=np.array([str(s)+'::'+str(int(p)) for s,p in zip(z['sequence'][rows],z['pedestrian_id'][rows])]);assert np.array_equal(expected,d['track'])
    for k in ['paths','native','gt','past']:assert np.isfinite(d[k]).all(),k
    assert d['paths'].shape==(len(d['sample_id']),20,12,2) and d['past'].shape==(len(d['sample_id']),8,2) and np.count_nonzero(d['past'][:,-1])==0
    return d

def group_masks(groups):
    return [('overall',np.ones(len(groups),bool))]+[(g,groups==i) for i,g in enumerate(['stable','transition','change'])]

def stats(x,mask):
    x=np.asarray(x)[mask];finite=np.isfinite(x)
    if x.ndim>1:
        count=finite.sum(0);s=np.where(finite,x,0).sum(0);mean=np.divide(s,count,out=np.full(s.shape,np.nan),where=count>0)
        return dict(n=len(x),defined=count,mean=mean)
    y=x[finite];return dict(n=len(x),defined=len(y),undefined=int((~finite).sum()),mean=float(y.mean()) if len(y) else None,median=float(np.median(y)) if len(y) else None,p90=float(np.quantile(y,.9)) if len(y) else None,p95=float(np.quantile(y,.95)) if len(y) else None)
