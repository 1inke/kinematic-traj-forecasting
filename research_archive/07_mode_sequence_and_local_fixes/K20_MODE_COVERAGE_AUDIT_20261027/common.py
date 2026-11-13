from pathlib import Path
import sys,os,json,hashlib,datetime,csv
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import numpy as np
from resource_guard_v3 import Guard
PARENT=ROOT/'qoder_work/K20_COLLABORATIVE_REFINEMENT_RESOURCE_ADAPTER_20261027'
SPLITS=('opt','hold','inner');GROUPS=('overall','stable','transition','change')
COARSE=('low-motion','undefined','straight','left','right')
JOINT=('low-motion','undefined')+tuple(d+'-'+s for d in COARSE[2:] for s in ('decelerate','constant','accelerate'))
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def load(p,keys=None):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in (z.files if keys is None else keys)}
def save(p,**a):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError('Preserve existing '+str(p))
    np.savez_compressed(p,**a)
def table(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows])
def mark(stage,**kw):write(HERE/'progress.json',dict(stage=stage,at=now(),**kw))
def guard(start=False):
    g=Guard(2*1024**3);g.check('startup' if start else 'runtime');return g
def resource(p,g):write(p,dict(last=g.check(force=True),observations=g.observations))
def verify_sources():
    m=read(HERE/'config/sources_manifest.json')
    for p,h in m['files'].items():assert sha(Path(p) if Path(p).is_absolute() else ROOT/p)==h,p
    return m['digest']
def finish(p,**kw):
    p=Path(p);write(p/'done.json',dict(outputs={x.relative_to(p).as_posix():sha(x) for x in p.rglob('*') if x.is_file() and x.name!='done.json'},**kw))
def verify_done(p):
    d=read(Path(p)/'done.json')
    for n,h in d['outputs'].items():assert sha(Path(p)/n)==h,n
    return d
def wrapped(a):return np.arctan2(np.sin(a),np.cos(a))
def quality(p,gt):
    d=np.linalg.norm(p-gt[:,None],axis=-1);a=d.mean(-1);f=d[:,:,-1];j=np.maximum(a/.2,f/.4)
    return a,f,j,(a<=.2)&(f<=.4)
def group_masks(x):
    for gi,n in enumerate(GROUPS):yield n,np.ones(len(x['group']),bool) if gi==0 else x['group']==gi-1
