"""Context-complementarity execution; all writes owned by this new execution."""
from __future__ import annotations
import sys,os,random,hashlib,importlib.util,time,copy
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from f16.io import sha256,read_json,write_json
from f16.resource import Guard
from f16.checkpoint import capture_rng,restore_rng
from f16.eval import evaluate,errors

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

prior=module('_context_readonly_evidence_util',ROOT/'qoder_work/E0E1_EVIDENCE_REPAIR_20261022/common.py')
independent=module('_context_independent_saved',ROOT/'audit/verify.py')
SEEDS=(42,43,44)
E=6*2**30  # Explicit conservative working-set inventory is saved in MEMORY_ESTIMATE.json.

def seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

def stamp():
    from datetime import datetime,timezone,timedelta
    return datetime.now(timezone(timedelta(hours=8))).isoformat()

def stage(name,**kw):
    old=read_json(HERE/'runtime_state.json') if (HERE/'runtime_state.json').exists() else {}
    old.update(stage=name,at=stamp(),**kw);write_json(HERE/'runtime_state.json',old);print(name,kw,flush=True)

def arraysha(a):return prior.array_sha(a)

def load_data():
    d=prior.data_load()
    with np.load(prior.PATHS['encoder'],allow_pickle=False) as z:
        assert np.array_equal(z['sample_id'],d['ids']);tokens=z['tokens'].copy();past=z['past'].copy()
    assert tokens.shape==(28791,8,96) and past.dtype==tokens.dtype==np.float32
    # Generate float32 production features once, before any float64 accumulation.
    x=np.concatenate((past.reshape(-1,16),(past[:,-1]-past[:,-2])/np.float32(.4),tokens[:,-1],tokens.mean(1)),1)
    assert x.shape==(28791,210) and np.isfinite(x).all()
    out={}
    for name,idx in d['split'].items():
        out[name]={k:d[k][idx] for k in ['ids','native','groups','gt','track']}
        out[name].update(x=x[idx].copy(),past=past[idx].copy(),tokens=tokens[idx].copy(),registry_index=idx.copy())
    assert set(out)=={'FIT','inner'}
    return out

class ContextK20(nn.Module):
    def __init__(self,mean,std):
        super().__init__();self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32).clone());self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32).clone())
        self.trunk=nn.Sequential(nn.Linear(210,128),nn.GELU(),nn.Linear(128,128),nn.GELU())
        self.queries=nn.Parameter(torch.randn(20,16)*.02)
        self.fuse=nn.Sequential(nn.Linear(144,128),nn.GELU(),nn.Linear(128,128),nn.GELU())
        self.head=nn.Linear(128,24);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        assert sum(p.numel() for p in self.parameters())==82008
    def forward(self,features):
        h=self.trunk((features-self.mean)/self.std);h=h[:,None].expand(-1,20,-1)
        z=self.fuse(torch.cat((h,self.queries[None].expand(len(h),-1,-1)),dim=-1))
        return self.head(z).reshape(-1,20,12,2)

PAIR=np.triu_indices(20,1)
def base_loss(p,gt):
    dist=torch.linalg.vector_norm(p-gt[:,None],dim=-1);q=dist.mean(-1)+.5*dist[:,:,-1]
    task,winner=q.min(1)
    ends=p[:,:,-1];pd=torch.linalg.vector_norm(ends[:,PAIR[0]]-ends[:,PAIR[1]],dim=-1)
    div=torch.relu(.3-pd).mean();return task.mean()+.05*div,dict(task=task.mean(),div=div,winner=winner,winner_ties=(q==q.min(1,keepdim=True).values).sum(1))

def teacher_sums(p,t,mask):
    # Only valid members; teacher-row/student-slot axes remain distinct.
    rows,slots=mask.nonzero(as_tuple=True)
    if not len(rows):return p.sum()*0,0,torch.empty(0,dtype=torch.long,device=p.device),0
    dd=torch.linalg.vector_norm(t[rows,slots,None]-p[rows],dim=-1)
    q=dd.mean(-1)+.5*dd[:,:,-1];v,w=q.min(1)
    return v.sum(),len(rows),w,int((q==q.min(1,keepdim=True).values).sum(1).sub(1).sum().detach().cpu())

def loss(p,gt,R,rm,N,nm,arm):
    b,info=base_loss(p,gt);sr,nr,wr,tr=teacher_sums(p,R,rm) if arm!='A' else (p.sum()*0,0,None,0)
    sn,nt,wn,tn=teacher_sums(p,N,nm) if arm!='A' else (p.sum()*0,0,None,0)
    if arm=='A':aux=p.sum()*0
    elif arm=='B':aux=(sr+sn)/max(nr+nt,1)
    elif arm=='E_bal':aux=.5*sr/nr+.5*sn/nt if nr and nt else sr/nr if nr else sn/nt if nt else p.sum()*0
    else:raise ValueError(arm)
    info.update(base=b,aux=aux,R_sum=sr,N_sum=sn,n_R=nr,n_N=nt,R_winner=wr,N_winner=wn,teacher_ties=tr+tn)
    return b+aux,info

def make_teacher(d,s):
    kin=np.load(prior.kin_path('FIT',s),allow_pickle=False)
    assert kin.shape==d['native'].shape and np.isfinite(kin).all()
    ne=errors(d['native'],d['gt']);ke=errors(kin,d['gt']);success=ne['coverage']
    return dict(R=d['native'].astype(np.float32),R_mask=ne['good']&success[:,None],N=kin.astype(np.float32),N_mask=ke['good']&~success[:,None],excluded_KIN_on_native_success=int((ke['good']&success[:,None]).sum()))

def new_model(s,device='cuda'):
    st=np.load(HERE/'statistics.npz');seed(s);return ContextK20(st['mean'],st['std']).to(device)

def frozen_state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def equal(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    return a==b

def predict(m,x,storage_chunk=256):
    # Inference compute always256; chunk affects storage only.
    result=np.empty((len(x),20,12,2),np.float32);m.eval()
    with torch.inference_mode():
        for i in range(0,len(x),256):
            n=min(256,len(x)-i);inp=np.zeros((256,210),np.float32);inp[:n]=x[i:i+n]
            y=m(torch.from_numpy(inp).to(next(m.parameters()).device)).cpu().numpy()[:n]
            for j in range(0,n,storage_chunk):result[i+j:i+min(j+storage_chunk,n)]=y[j:j+storage_chunk]
    assert np.isfinite(result).all();return result

def contract(arm,s):
    return dict(arm=arm,seed=s,source_digest=read_json(HERE/'source_manifest.json')['digest'],protocol=sha256(HERE/'protocol.json'),stats=sha256(HERE/'statistics.npz'),ids=sha256(HERE/'split_ids.npz'),initial=sha256(HERE/f'initial_states/seed{s}.pt'),orders=sha256(HERE/f'batch_orders/seed{s}.npz'),teacher=sha256(HERE/f'teachers/seed{s}.npz'))

def reject_contract(actual,expected):
    if actual!=expected:raise ValueError('SOURCE_ID_STATS_TEACHER_RESUME_CONTRACT_MISMATCH')

def checkpoint(m,opt,g,epoch,step,hist,assign,c):
    return dict(model=frozen_state(m),optimizer=copy.deepcopy(opt.state_dict()),rng=capture_rng(g),epoch=epoch,step=step,history=copy.deepcopy(hist),assignment=copy.deepcopy(assign),contract=copy.deepcopy(c),software=dict(torch=torch.__version__,numpy=np.__version__,device=str(next(m.parameters()).device),dtype='float32'))

def restore(blob,m,opt,g,c):
    reject_contract(blob['contract'],c);m.load_state_dict(blob['model']);opt.load_state_dict(blob['optimizer']);restore_rng(blob['rng'],g)

def verify_sources():
    mf=read_json(HERE/'source_manifest.json');bad=[]
    for key,v in mf['files'].items():
        p=Path(v['path']);
        if not p.is_file() or sha256(p)!=v['sha256']:bad.append(key)
    if bad:raise ValueError('SOURCE_CHANGED '+str(bad))
    return mf['digest']

def valid_done(path,c):
    b=read_json(path);reject_contract(b['contract'],c)
    for p,v in b['outputs'].items():
        if sha256(HERE/p)!=v:raise ValueError('COMPLETED_OUTPUT_SHA_MISMATCH')
    return b
