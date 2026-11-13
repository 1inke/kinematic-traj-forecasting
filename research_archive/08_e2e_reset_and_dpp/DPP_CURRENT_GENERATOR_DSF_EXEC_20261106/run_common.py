import sys,os,json,hashlib,copy,random,importlib.util
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import torch
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1]
SETUP=ROOT/'qoder_work/DPP_INTERFACES_SETUP_20261106';STAGE_A=ROOT/'qoder_work/DPP_DUAL_FEASIBILITY_OFFLINE_EXEC_20261104'
BASE=ROOT/'qoder_work/E2E_MOTION_RESEARCH_RESET_EXEC_20261102'
sys.path.insert(0,str(SETUP));sys.dont_write_bytecode=True
from interface_common import sha,read,write,check_files,seal
from current_h3_adapter import load_current
from dsf_interface import DSFSampler,kinematic_torch,ec_from_weights

def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def progress(stage,**kw):
    item=dict(stage=stage,at=now(),**kw);p=RUN/'progress.json';temp=RUN/'progress.new.json'
    with temp.open('w',encoding='utf-8') as f:json.dump(item,f,ensure_ascii=False,allow_nan=False)
    os.replace(temp,p);print(json.dumps(item,ensure_ascii=False),flush=True)
def rg_module():
    s=importlib.util.spec_from_file_location('dsf_resource_v3',ROOT/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def hash_indices(ids,tag,n):return np.array(sorted(range(len(ids)),key=lambda i:hashlib.sha256((tag+'|'+str(ids[i])).encode()).digest())[:n])
def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu(v) for v in x)
    return copy.deepcopy(x)
def exact(a,b,label='root'):
    if torch.is_tensor(a):
        if not torch.equal(a.cpu(),b.cpu()):raise ValueError('NOT_EXACT '+label)
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise ValueError('NOT_EXACT '+label)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('KEYS '+label)
        for k in a:exact(a[k],b[k],label+'/'+str(k))
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise ValueError('LEN '+label)
        for n,(x,y) in enumerate(zip(a,b)):exact(x,y,label+'/'+str(n))
    elif a!=b:raise ValueError('NOT_EXACT '+label)
def save_pt(p,state):
    p=Path(p)
    if p.exists():raise FileExistsError(p)
    temp=p.with_suffix('.new.pt')
    with temp.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    os.rename(temp,p)
def source_check():
    s=read(RUN/'PRODUCTION_SOURCE_LOCK.json');check_files(s['files']);return s
def contract(arm,seed,indices,cache):
    if arm not in ('uniform','kin') or seed not in (42,43,44):raise ValueError('ARM_SEED')
    s=read(RUN/'PRE_SOURCE.json')
    return dict(arm=arm,seed=seed,indices=cache['ids'][indices].tolist(),pre_source_sha=sha(RUN/'PRE_SOURCE.json'),cache_receipt=sha(RUN/'CACHE_RECEIPT.json'),protocol_sha=sha(RUN/'protocol.json'),normalization='original_past_only_heading_and_agentmean; no new fitted stats',frozen_sha=s['generator_checkpoint_sha'])
def verify_contract(actual,expected):exact(actual,expected,'CONTRACT')
def cache_read(split):
    receipt=read(RUN/'CACHE_RECEIPT.json');check_files(receipt['splits'][split]['files'])
    with np.load(RUN/'cache'/f'{split}.npz',allow_pickle=False) as f:return {k:f[k] for k in f.files}
def batch_bundle(cache,indices,device,dtype=torch.float32):
    names=('context','past_loc','center','origin','matrix','raw_past')
    d={k:torch.as_tensor(cache[k][indices],device=device,dtype=dtype) for k in names}
    return dict(context=d['context'],sampler_context=d['context'][:,0],past_loc=d['past_loc'],transform=(d['center'],d['origin'],d['matrix']),raw_past=d['raw_past'])
def new_sampler(seed,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    m=DSFSampler(64,64).to(device);torch.nn.init.zeros_(m.net[-1].weight);torch.nn.init.zeros_(m.net[-1].bias);return m
def decode_logits(adapter,control,bundle):
    from torch.nn import functional as F
    d=adapter.model.decoder;c=bundle['context'];past=bundle['past_loc']
    extra=torch.cat([control[:,None],control.new_zeros(len(control),8,20,64)],1)
    mode=d.mode_mlp(c[:,:,None]+d.mode_embedding[None,None]+extra)
    scale=F.softplus(d.proposal_scale(c))+.25
    delta=.75*torch.tanh(d.proposal_delta(mode)).reshape(*mode.shape[:-1],3,2)
    anchor=scale[:,:,None,None]*(d.anchor_templates[None,None]+delta)
    coarse=past[:,:,None,-1:]+torch.einsum('tp,bakpc->baktc',d.interpolation,anchor)
    logits=d.score_head(torch.cat([mode,anchor.flatten(-2)],-1)).squeeze(-1)
    return adapter.api.world_paths(coarse[:,:1],bundle['transform']),logits[:,0]
def objective(adapter,sampler,bundle,arm,cal):
    z=sampler(bundle['sampler_context']);y,logits=decode_logits(adapter,z,bundle)
    y64=y.double()
    q=kinematic_torch(bundle['raw_past'][:,0].double(),y64,cal)
    r=torch.ones_like(q) if arm=='uniform' else torch.sqrt(cal['epsilon_q']+(1-cal['epsilon_q'])*q)
    ec=ec_from_weights(y64,r,cal['k'])
    return -ec.mean(),dict(paths=y,logits=logits,control=z,qkin=q,EC=ec)
