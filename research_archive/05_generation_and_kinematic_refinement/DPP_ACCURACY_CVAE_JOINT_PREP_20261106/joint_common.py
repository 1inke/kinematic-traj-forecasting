"""New namespace, strict opt-in, immutable evidence; never launch on import."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import copy, hashlib, importlib.util, json, os, random, sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'qoder_work/DPP_CURRENT_GENERATOR_DSF_EXEC_20261106'
BASE = ROOT/'qoder_work/E2E_MOTION_RESEARCH_RESET_EXEC_20261102'
STAGE = ROOT/'qoder_work/DPP_DUAL_FEASIBILITY_OFFLINE_EXEC_20261104'
GUARD = ROOT/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py'
sys.dont_write_bytecode = True

def now(): return datetime.now(timezone(timedelta(hours=8))).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2), b''): h.update(b)
    return h.hexdigest()
def write(p, value):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def verify(files):
    if not files: raise ValueError('EMPTY_MANIFEST')
    for p,d in files.items():
        if sha(p)!=d: raise ValueError('SHA_MISMATCH '+str(p))
def seal(paths): return {str(Path(p).resolve()):sha(p) for p in paths}
def module(name,path):
    if name in sys.modules: return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj)
    return obj
def legacy(): return module('joint_legacy_dsf_math',OLD/'run_common.py')
def resource_module(): return module('joint_resource_v3',GUARD)
def protocol(): return read(HERE/'protocol.json')
def authorize(action, execute=False):
    c=read(HERE/'control.json')
    key={'preflight':'allow_real_preflight','train':'allow_training','evaluate':'allow_training'}[action]
    if not execute or c.get(key) is not True:
        raise PermissionError('SCRIPTS_ONLY_HUMAN_HOLD: '+action+' requires later explicit human launch instruction and reviewed control change')
def inputs():
    d=dict(read(OLD/'PRODUCTION_SOURCE_LOCK.json')['files'])
    d.update(seal([BASE/'C0/fit.npz',BASE/'C0/dev.npz',OLD/'CACHE_RECEIPT.json']))
    for part in read(OLD/'CACHE_RECEIPT.json')['splits'].values(): d.update(part['files'])
    return d
def verify_package():
    receipt=read(HERE/'SCRIPT_MANIFEST.json');verify(receipt['files']);return receipt
def cache(split):
    if split not in ('fit','dev'): raise ValueError('SPLIT')
    verify(read(OLD/'CACHE_RECEIPT.json')['splits'][split]['files'])
    with np.load(OLD/'cache'/f'{split}.npz',allow_pickle=False) as f:
        return {k:f[k] for k in f.files}
def labels(split, ids, execution=None, preflight=False):
    if split=='dev' and (execution is None or not (Path(execution)/'ALL12_FINALS_LOCK.json').is_file()):
        raise PermissionError('DEV_GT_LOCKED_UNTIL_ALL12_FINALS')
    with np.load(BASE/'C0'/f'{split}.npz',allow_pickle=False) as f:
        allids=f['ids']; target=f['future']; groups=f['group']
    index={str(k):i for i,k in enumerate(allids)}
    ix=np.array([index[str(k)] for k in ids])
    if not np.array_equal(allids[ix],ids): raise ValueError('LABEL_ID_ORDER')
    return target[ix],groups[ix]
def batch(data,ix,device):
    import torch
    names=('context','raw_past','origin','matrix','past_loc','center')
    d={k:torch.as_tensor(data[k][ix],dtype=torch.float32,device=device) for k in names}
    d['sampler_context']=d['context'][:,0]
    d['transform']=(d['center'],d['origin'],d['matrix'])
    return d
def seed_all(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def cpu(value):
    import torch
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)
def exact(a,b):
    import torch
    if torch.is_tensor(a):
        if not torch.equal(a.cpu(),b.cpu()): raise ValueError('TENSOR_NOT_EXACT')
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b): raise ValueError('ARRAY_NOT_EXACT')
    elif isinstance(a,dict):
        if a.keys()!=b.keys(): raise ValueError('KEYS')
        for k in a: exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b): raise ValueError('LENGTH')
        for x,y in zip(a,b): exact(x,y)
    elif a!=b: raise ValueError('VALUE_NOT_EXACT')
def save_checkpoint(p,state):
    import torch
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise FileExistsError(p)
    tmp=p.with_name(p.name+'.partial')
    with tmp.open('xb') as f:
        torch.save(cpu(state),f);f.flush();os.fsync(f.fileno())
    os.rename(tmp,p)
def trusted_checkpoint(p,expected_sha):
    import torch
    if sha(p)!=expected_sha: raise ValueError('CHECKPOINT_SHA')
    # Local source-locked checkpoints contain Python/NumPy optimizer RNG objects.
    return torch.load(p,map_location='cpu',weights_only=False)
def run_names():
    return [f'{family}_{arm}_seed{seed}' for family in protocol()['families']
            for seed in protocol()['seeds'] for arm in protocol()['arms']]
