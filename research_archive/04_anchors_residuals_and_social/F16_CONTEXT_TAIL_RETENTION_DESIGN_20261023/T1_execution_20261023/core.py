"""T1 relative regression, immutable old implementation referenced read-only."""
import os,sys,copy,json,hashlib,random,time,importlib.util,shutil,ast,types
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OLD=ROOT/'qoder_work/F16_CONTEXT_COMPLEMENTARITY_DESIGN_20261022/execution_20261022'
DIAG=HERE.parent/'execution_20261023'
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from f16.io import sha256,read_json,write_json
from f16.resource import Guard
from f16.checkpoint import capture_rng,restore_rng
from f16.eval import evaluate,errors

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

# Reviewed imports only define utilities/classes. No legacy main/train is imported.
old=module('_tail_t1_readonly_original_core',OLD/'core.py')
diag=types.SimpleNamespace();scope={'np':np}
for filename,names in [('source_contract.py',{'group_masks','stats'}),('saved_diagnostics.py',{'quality','physics','temporal_transport','path_measures','compute','summary'})]:
    tree=ast.parse((DIAG/filename).read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==names
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(DIAG/filename),'exec'),scope)
for key in ('compute','summary'):setattr(diag,key,scope[key])
diag_review=module('_tail_t1_readonly_diagnostic_review',DIAG/'independent_diagnostics.py')
ContextK20=old.ContextK20;PAIR=old.PAIR;equal=old.equal;frozen_state=old.frozen_state;predict=old.predict
SEEDS=(42,43,44);ARMS=('A_replay','P_relative');E=6*2**30

def now():return old.stamp()
def stage(name,**kw):
    state=read_json(HERE/'runtime_state.json') if (HERE/'runtime_state.json').exists() else {}
    state.update(stage=name,at=now(),**kw);write_json(HERE/'runtime_state.json',state);print(name,kw,flush=True)

def load_data():
    out={}
    with np.load(ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz') as z:
        ids=z['sample_id'];past=z['past'];tokens=z['tokens']
        x=np.concatenate((past.reshape(-1,16),(past[:,-1]-past[:,-2])/np.float32(.4),tokens[:,-1],tokens.mean(1)),1)
        lookup={int(v):i for i,v in enumerate(ids)}
    with np.load(OLD/'split_ids.npz') as registry:
        for split in ('FIT','inner'):
            with np.load(OLD/f'native_{split}.npz') as z:d={k:z[k].copy() for k in z.files}
            assert np.array_equal(d['sample_id'],registry[split])
            loc=np.array([lookup[int(v)] for v in d['sample_id']]);d['x']=x[loc].copy();d['past']=past[loc].copy();d['ids']=d['sample_id']
            assert d['x'].dtype==np.float32 and len(np.unique(d['ids']))==len(d['ids'])
            assert d['paths'].shape==(len(loc),20,12,2) and np.array_equal(d.pop('paths'),d['native'])
            for k in ('x','past','native','gt'):assert np.isfinite(d[k]).all()
            out[split]=d
    assert [len(out[k]['ids']) for k in ('FIT','inner')]==[18909,1856]
    assert not np.intersect1d(out['FIT']['ids'],out['inner']['ids']).size
    assert not np.intersect1d(out['FIT']['track'],out['inner']['track']).size
    return out

def quality(p,y):
    dist=torch.linalg.vector_norm(p-y[:,None],dim=-1);a=dist.mean(-1);f=dist[:,:,-1]
    q=a+.5*f;parts=torch.stack((a/.2,f/.4),-1);j,component=parts.max(-1);jm,js=j.min(1)
    return dict(ADE=a,FDE=f,Q=q,J=j,Jmin=jm,Jslot=js,Jcomponent=component,Qslot=q.min(1).indices,
                Jties=(j==jm[:,None]).sum(1),component_ties=(parts[:,:,0]==parts[:,:,1]).sum(1))

def objective(p,y,native_j,arm):
    base,bi=old.base_loss(p,y);qi=quality(p,y);excess=torch.relu(qi['Jmin']-native_j.detach())
    # This +zero is exactly the old A branch, including its autograd expression.
    total=base+p.sum()*0 if arm=='A_replay' else base+.2*excess.mean() if arm=='P_relative' else None
    if total is None:raise ValueError(arm)
    return total,dict(**qi,base=base,task=bi['task'],div=bi['div'],excess=excess,
                     Q_extra_ties=bi['winner_ties']-1,J_extra_ties=qi['Jties']-1)

def new_model(s,device='cuda'):
    st=np.load(HERE/'statistics.npz');old.seed(s);m=ContextK20(st['mean'],st['std']).to(device)
    initial=torch.load(HERE/f'initial_states/seed{s}.pt',weights_only=True,map_location='cpu');m.load_state_dict(initial,strict=True);return m

def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)

def prepare(data):
    assert read_json(DIAG/'D0_D1/DECISION.json')['allow_T1_preflight']
    for name in ('initial_states','batch_orders'): (HERE/name).mkdir(parents=True,exist_ok=True)
    for name in ('statistics.npz','split_ids.npz','diagnostic_ids.npz'):
        shutil.copyfile(OLD/name,HERE/name);assert sha256(OLD/name)==sha256(HERE/name)
    for s in SEEDS:
        for rel in (f'initial_states/seed{s}.pt',f'batch_orders/seed{s}.npz'):
            shutil.copyfile(OLD/rel,HERE/rel);assert sha256(OLD/rel)==sha256(HERE/rel)
        orders=np.load(HERE/f'batch_orders/seed{s}.npz')['orders'];assert orders.shape==(8,18909)
        assert all(np.array_equal(np.sort(v),np.arange(18909)) for v in orders)
    for split,d in data.items():
        ref=[]
        for i in range(0,len(d['ids']),256):
            n=torch.from_numpy(d['native'][i:i+256].astype(np.float32)).cuda();y=torch.from_numpy(d['gt'][i:i+256].astype(np.float32)).cuda()
            with torch.no_grad():ref.append(quality(n,y)['Jmin'].cpu().numpy())
        d['native_j']=np.concatenate(ref);assert d['native_j'].dtype==np.float32
        np.savez_compressed(HERE/f'native_J_{split}.npz',sample_id=d['ids'],J=d['native_j'],training_dtype=np.array('float32'))

def contract(arm,s):
    return dict(arm=arm,seed=s,source_digest=read_json(HERE/'source_manifest.json')['digest'],
        protocol=sha256(HERE/'protocol.json'),statistics=sha256(HERE/'statistics.npz'),IDs=sha256(HERE/'split_ids.npz'),
        initial=sha256(HERE/f'initial_states/seed{s}.pt'),orders=sha256(HERE/f'batch_orders/seed{s}.npz'),native_J=sha256(HERE/'native_J_FIT.npz'))

def reject(actual,expected):
    if actual!=expected:raise ValueError('SOURCE_ID_STATS_INITIAL_ORDER_ARM_RESUME_MISMATCH')

def checkpoint(m,o,g,ep,steps,hist,assignment,con):
    return dict(model=frozen_state(m),optimizer=copy.deepcopy(o.state_dict()),rng=capture_rng(g),epoch=ep,step=steps,
        history=copy.deepcopy(hist),assignment=copy.deepcopy(assignment),contract=copy.deepcopy(con))

def restore(blob,m,o,g,con):
    reject(blob['contract'],con);m.load_state_dict(blob['model'],strict=True);o.load_state_dict(blob['optimizer']);restore_rng(blob['rng'],g)

def valid_done(path,con):
    b=read_json(path);reject(b['contract'],con)
    for rel,digest in b['outputs'].items():
        if sha256(HERE/rel)!=digest:raise ValueError('COMPLETED_OUTPUT_SHA_MISMATCH')
    return b

def lock_sources():
    parent=read_json(DIAG/'source_manifest.json');files=copy.deepcopy(parent['files'])
    for k,v in files.items():assert sha256(v['path'])==v['sha256'],('INHERITED_SOURCE_CHANGED',k)
    paths=list((DIAG/'D0_D1').rglob('*'))+list((DIAG/'completion_review').glob('*'))
    paths += [DIAG/n for n in ('RESULT_ANALYSIS.md','RESULT_ANALYSIS.json','RESULT_ANALYSIS_FINAL.md') if (DIAG/n).exists()]
    paths += list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))+list(HERE.glob('*.md'))+list((HERE/'failures').rglob('*'))
    paths += [HERE/n for n in ('protocol.json','MEMORY_ESTIMATE.json','statistics.npz','split_ids.npz','diagnostic_ids.npz','native_J_FIT.npz','native_J_inner.npz')]
    paths += list((HERE/'initial_states').glob('*'))+list((HERE/'batch_orders').glob('*'))
    for p in paths:
        if not p.is_file() or '__pycache__' in str(p):continue
        key=p.relative_to(ROOT).as_posix();val=sha256(p)
        if key in files:assert files[key]['sha256']==val
        else:files[key]=dict(path=str(p),sha256=val,inherited=False)
    digest=hashlib.sha256(json.dumps({k:v['sha256'] for k,v in sorted(files.items())},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write_json(HERE/'source_manifest.json',dict(digest=digest,parent_digest=parent['digest'],inherited=len(parent['files']),files=files));return verify_sources()

def verify_sources():
    mf=read_json(HERE/'source_manifest.json')
    for k,v in mf['files'].items():
        if sha256(v['path'])!=v['sha256']:raise ValueError('SOURCE_CHANGED '+k)
    return mf['digest']
