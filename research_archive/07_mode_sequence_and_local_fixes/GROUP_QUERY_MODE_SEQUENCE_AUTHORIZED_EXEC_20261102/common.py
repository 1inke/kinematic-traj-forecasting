"""Finite Q execution. Read-only legacy imports; all writes belong to this folder."""
from pathlib import Path
import copy,hashlib,importlib.util,json,os,random,sys,time
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
OLD=PROJECT/'qoder_work/F16_CONTEXT_COMPLEMENTARITY_DESIGN_20261022/execution_20261022'
PREVIOUS=PROJECT/'qoder_work/ENDPOINT_DISCRETE_SET_UTILITY_EXEC_20261101'
sys.path.insert(0,str(PROJECT/'src'))
import numpy as np
import torch
from f16.checkpoint import capture_rng,restore_rng

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
old=module('group_query_readonly_context',OLD/'core.py')
rg=module('group_query_resource_v3',PROJECT/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py')
Guard=rg.Guard
SEEDS=(42,43,44)
def now():
    import datetime
    return datetime.datetime.now().astimezone().isoformat()
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def write(path,v):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(clean(v),f,ensure_ascii=False,indent=2)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def digest(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes());return h.hexdigest()
def stage(name,**v):
    obj=dict(stage=name,at=now(),**v);tmp=HERE/'progress.json.new'
    with tmp.open('w',encoding='utf-8') as f:json.dump(clean(obj),f)
    os.replace(tmp,HERE/'progress.json');print(json.dumps(clean(obj)),flush=True)
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def setup():
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
def state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
equal=old.equal
def save_torch(path,v):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('xb') as f:torch.save(v,f);f.flush();os.fsync(f.fileno())
    os.rename(tmp,path)
def npz(path,**v):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.savez_compressed(f,**v)
def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def verify(files,check=lambda:None):
    for i,(p,s) in enumerate(files.items()):
        if sha(p)!=s:raise ValueError('SOURCE_OR_OUTPUT_SHA_MISMATCH '+str(p))
        if i%50==0:check()
def data():
    d=old.load_data();ids=load_npz(HERE/'split_ids.npz')
    for k in ('FIT','inner'):
        if not np.array_equal(d[k]['ids'],ids[k]):raise ValueError('SPLIT_ID_MISMATCH')
        d[k].pop('tokens');d[k].pop('registry_index')
    return d
def stats():
    s=load_npz(HERE/'statistics.npz');return s['mean'],s['std']
def new_model(s,groups,device='cuda'):
    from models import GroupContextK20
    seed(s);m=GroupContextK20(*stats(),groups=groups,aux_seed=2026110100+s)
    return m.to(device)
def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
def contract(arm,s):
    return dict(arm=arm,seed=s,groups={'Q1':1,'Q3':3}[arm],main_group=0,source=read(HERE/'source_manifest.json')['digest'],
        protocol=sha(HERE/'protocol.json'),stats=sha(HERE/'statistics.npz'),split=sha(HERE/'split_ids.npz'),
        initial=sha(HERE/f'initial/{arm}_{s}.pt'),orders=sha(HERE/f'orders/{s}.npz'))
def reject(actual,expected):
    if actual!=expected:raise ValueError('SOURCE_ID_STAT_INITIAL_ORDER_GROUP_CHECKPOINT_CONTRACT')
def trusted(p,expected_sha,c):
    if sha(p)!=expected_sha:raise ValueError('CHECKPOINT_SHA')
    z=torch.load(p,map_location='cpu',weights_only=False);reject(z['contract'],c);return z
def done(path,c):
    z=read(path);reject(z['contract'],c)
    verify({str(HERE/p):s for p,s in z['files'].items()});return z
