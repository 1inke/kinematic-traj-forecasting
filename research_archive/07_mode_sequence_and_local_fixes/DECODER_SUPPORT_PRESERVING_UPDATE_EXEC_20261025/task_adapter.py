"""Read-only AST extraction; no old runner or top-level old writes."""
import os,sys,ast,copy,hashlib,time,random,json
from pathlib import Path
sys.dont_write_bytecode=True
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
OLD=ROOT/'qoder_work/F16_CONTEXT_COMPLEMENTARITY_DESIGN_20261022/execution_20261022'
DESIGN=ROOT/'docs/DECODER_SUPPORT_PRESERVING_UPDATE_DESIGN_20261025'
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch import nn
from f16.io import sha256,read_json,write_json
from f16.checkpoint import capture_rng,restore_rng
from f16.resource import Guard
from f16.eval import errors,evaluate
SEEDS=(42,43,44); ARMS=('U','N','P'); E=6*2**30; PAIR=np.triu_indices(20,1)
torch.set_num_threads(2)

def extract(path,names,scope):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'));nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)

_scope=dict(torch=torch,nn=nn,np=np,PAIR=PAIR)
extract(OLD/'core.py',['ContextK20','base_loss','equal','frozen_state','predict'],_scope)
ContextK20=_scope['ContextK20'];base_loss=_scope['base_loss'];equal=_scope['equal'];frozen_state=_scope['frozen_state'];predict=_scope['predict']

def stamp():
    from datetime import datetime,timezone,timedelta
    return datetime.now(timezone(timedelta(hours=8))).isoformat()

def stage(name,**kw):
    write_json(HERE/'runtime_state.json',dict(status='RUNNING',stage=name,at=stamp(),**kw));print(name,kw,flush=True)

def arraysha(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(a.tobytes()+str(a.shape).encode()+str(a.dtype).encode()).hexdigest()

def load_data():
    # Read only registered FIT/inner saved rows; other populations never evaluated.
    with np.load(ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz',allow_pickle=False) as z:
        registry=z['sample_id'];past=z['past'];tokens=z['tokens']
        x=np.concatenate((past.reshape(-1,16),(past[:,-1]-past[:,-2])/np.float32(.4),tokens[:,-1],tokens.mean(1)),1)
        lookup={int(v):i for i,v in enumerate(registry)}
    result={}
    with np.load(OLD/'split_ids.npz',allow_pickle=False) as ids:
        for split in ('FIT','inner'):
            with np.load(OLD/f'native_{split}.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
            assert np.array_equal(d['sample_id'],ids[split]);loc=np.array([lookup[int(v)] for v in d['sample_id']])
            assert np.array_equal(d.pop('paths'),d['native'])
            d['x']=x[loc].copy();d['ids']=d['sample_id'];result[split]=d
    assert [len(result[n]['ids']) for n in ('FIT','inner')]==[18909,1856]
    assert not np.intersect1d(result['FIT']['ids'],result['inner']['ids']).size
    assert not np.intersect1d(result['FIT']['track'],result['inner']['track']).size
    return result

def parent_blob(s):
    p=OLD/f'runs/A_seed{s}';b=torch.load(p/'resume.pt',map_location='cpu',weights_only=False)
    ck=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert equal(b,ck),'PARENT_CHECKPOINT_RESUME_NOT_EXACT'
    assert b['epoch']==8 and b['step']==592 and len(b['history'])==8
    assert b['contract']['arm']=='A' and b['contract']['seed']==s
    return b

def start(s,device='cuda'):
    b=parent_blob(s);m=ContextK20(b['model']['mean'],b['model']['std']).to(device);m.load_state_dict(b['model'],strict=True)
    assert sum(p.numel() for p in m.parameters())==82008
    o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);o.load_state_dict(b['optimizer'])
    g=torch.Generator();restore_rng(b['rng'],g);g.manual_seed(2026102500+s)
    assert equal(frozen_state(m),b['model'])
    assert all(v['step'].item()==592 for v in o.state.values())
    return m,o,g

def task(p,y):
    value,info=base_loss(p,y)
    return value+p.sum()*0,info  # Preserve original A zero-aux graph exactly.

def quality(p,y):
    e=torch.linalg.vector_norm(p-y[:,None],dim=-1);a=e.mean(-1);f=e[:,:,-1]
    parts=torch.stack((a/.2,f/.4),-1);j,component=parts.max(-1);jm,slot=j.min(1)
    return jm,dict(slot=slot,component=component,j_ties=(j==jm[:,None]).sum(1),component_ties=(parts[:,:,0]==parts[:,:,1]).sum(1))

def tensor_batch(d,idx,device):
    return torch.from_numpy(d['x'][idx].copy()).to(device),torch.from_numpy(d['gt'][idx].astype(np.float32)).to(device)

def parameter_order(m):return [(n,list(p.shape),str(p.dtype)) for n,p in m.named_parameters()]

def load_sets(s):
    with np.load(HERE/f'identity_audit/seed{s}.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def verify_manifest():
    m=read_json(HERE/'source_manifest.json')
    dd=hashlib.sha256(json.dumps({k:v['sha256'] for k,v in m['files'].items()},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert dd==m['digest']
    for v in m['files'].values():assert sha256(v['path'])==v['sha256'],('SOURCE_CHANGED',v['path'])
    return dd

def reject_contract(actual,expected):
    if actual!=expected:raise ValueError('SOURCE_ID_PARENT_STATS_ARM_CONTRACT_MISMATCH')

def checkpoint(m,o,g,hist,diag,step,contract):
    return dict(model=frozen_state(m),optimizer=copy.deepcopy(o.state_dict()),rng=capture_rng(g),history=copy.deepcopy(hist),diagnostics=copy.deepcopy(diag),new_step=step,parent_step=592,contract=copy.deepcopy(contract))

def restore(b,m,o,g,contract):
    reject_contract(b['contract'],contract);m.load_state_dict(b['model']);o.load_state_dict(b['optimizer']);restore_rng(b['rng'],g)
