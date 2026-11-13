from pathlib import Path
import os,sys,json,csv,hashlib,datetime
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'qoder_work/K20_MODE_COVERAGE_AUDIT_20261027'
sys.path.insert(0,str(HERE))
from resource_guard_v3 import Guard
SPLITS=('opt','hold','inner')
GROUPS=('overall','stable','transition','change')
ENTRY=('straight','left','right','undefined')
ONSET=('no-detected','early','middle','late','complex','undefined')
REASONS=('UNDEFINED_OR_BOUNDARY','ENTRY_MODE_ABSENT','ENTRY_PRESENT_ONSET_MISSING','ENTRY_ONSET_PRESENT_SPEED_MISSING','REFINED_MODE_PRESENT_BUT_INACCURATE','PARTIAL_VALID')
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def load(p,keys=None):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in (z.files if keys is None else keys)}
def save(p,**x):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
    if p.exists():raise FileExistsError('Preserve '+str(p))
    np.savez_compressed(p,**x)
def table(p,rows):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
def wrapped(a):return np.arctan2(np.sin(a),np.cos(a))
def finite(v):return float(v) if np.isfinite(v) else None
def quality(p,gt):
    d=np.linalg.norm(p-gt[:,None],axis=-1);a=d.mean(-1);f=d[:,:,-1];j=np.maximum(a/.2,f/.4);return a,f,j,(a<=.2)&(f<=.4)
def guard(start=False):
    g=Guard(2*1024**3);g.check('startup' if start else 'runtime');return g
def resource(p,g):write(p,dict(last=g.check(force=True),observations=g.observations))
def mark(stage,**kw):write(HERE/'progress.json',dict(stage=stage,at=now(),**kw));print(stage,kw,flush=True)
def xload(split):return load(OLD/f'inputs/X_{split}.npz')
def yload(split):return load(OLD/f'inputs/GT_{split}.npz')
def masks(x):
    for i,name in enumerate(GROUPS):yield name,np.ones(len(x['group']),bool) if i==0 else x['group']==i-1
def finish(folder,**kw):
    folder=Path(folder);write(folder/'done.json',dict(outputs={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='done.json'},**kw))
def verify_done(folder):
    folder=Path(folder);j=read(folder/'done.json')
    for rel,h in j['outputs'].items():assert sha(folder/rel)==h,rel
    return j
def verify_sources():
    j=read(HERE/'manifest.json')
    for rel,h in j['files'].items():assert sha(Path(rel) if Path(rel).is_absolute() else ROOT/rel)==h,rel
    return j['digest']
