import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from datetime import datetime
import ast,json,hashlib,time,random,copy,gc,ctypes,traceback
import numpy as np,psutil,torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from scipy.cluster.hierarchy import linkage,fcluster
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
P=ROOT/'PED_FOUR_DIRECTION_SERIES_20261011/experiment1_motion_process'
SPLIT=ROOT/'GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz'
DEVICE=torch.device('cuda:0');SEEDS=(42,43,44);ARMS=('ordinary','projected')
torch.use_deterministic_algorithms(True)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()).encode()).hexdigest()
def write(p,obj):
    p=Path(p);p=p if p.is_absolute() else H/p;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8');os.replace(t,p)
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended')]
RESOURCE=dict(peak_private_GiB=0.,peak_RSS_GiB=0.,min_physical_GiB=100.,min_commit_GiB=100.,checks=0)
def guard():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));p=psutil.Process().memory_info()
    RESOURCE['peak_private_GiB']=max(RESOURCE['peak_private_GiB'],p.peak_pagefile/2**30);RESOURCE['peak_RSS_GiB']=max(RESOURCE['peak_RSS_GiB'],p.peak_wset/2**30)
    RESOURCE['min_physical_GiB']=min(RESOURCE['min_physical_GiB'],m.avail_phys/2**30);RESOURCE['min_commit_GiB']=min(RESOURCE['min_commit_GiB'],m.avail_page/2**30);RESOURCE['checks']+=1
    if max(p.private,p.rss)>6*2**30 or min(m.avail_phys,m.avail_page)<6*2**30:
        write('RESOURCE_FAILURE.json',dict(status='STOP_RESOURCE',at=datetime.now().astimezone().isoformat(),private_GiB=p.private/2**30,RSS_GiB=p.rss/2**30,physical_available_GiB=m.avail_phys/2**30,commit_available_GiB=m.avail_page/2**30,process_budget_GiB=6,reserve_each_GiB=6,lifecycle=RESOURCE))
        raise MemoryError('STOP_RESOURCE process6GiB/physical+commit reserve6GiB')
def stage(s,**kw):guard();write('progress.json',dict(stage=s,at=datetime.now().astimezone().isoformat(),**kw));print(s,json.dumps(kw),flush=True)
def verify(files):
    for i,(p,h) in enumerate(files.items()):
        if i%50==0:guard()
        assert sha(p)==h,('SHA_MISMATCH',p)
def definitions(path,names,ns):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'));nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert len(nodes)==len(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
_modelns=dict(np=np,torch=torch,nn=nn,ARMS=('position3','acceleration3'))
definitions(P/'run.py',['bases','decode_numpy','Refiner'],_modelns)
Refiner=_modelns['Refiner'];decode_numpy=_modelns['decode_numpy']
_lossns=dict(torch=torch,F=F)
definitions(ROOT/'PROTOCOL_RECONCILIATION_RUN_20261007/run.py',['objective'],_lossns)
canonical_objective=_lossns['objective']
_qualityns=dict(np=np,linkage=linkage,fcluster=fcluster,guard=guard,GROUPS=('overall','stable','transition','change'))
definitions(ROOT/'F16_CANONICAL_EQUAL_K20_REPLACEMENT_DIAGNOSIS_20261007/run.py',['errors','motion','clusters','masks'],_qualityns)
errors=_qualityns['errors'];motion=_qualityns['motion'];clusters=_qualityns['clusters'];masks=_qualityns['masks']
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def parent_checkpoint(s):return P/f'runs/position3/seed_{s}/checkpoint.pt'
def build(s,device=DEVICE):
    z=torch.load(parent_checkpoint(s),map_location='cpu',weights_only=False);assert z['epoch']==25
    st=(z['statistics']['mean'].numpy().copy(),z['statistics']['std'].numpy().copy());m=Refiner(st).to(device);m.load_state_dict(z['state_dict']);assert sum(p.numel() for p in m.parameters())==24134
    assert equal(m.state_dict(),z['state_dict']);return m
def load_data():
    guard();cache=npz(ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz');enc=npz(ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz');lab=npz(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz');meta=npz(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz')
    ids=cache['sample_id'];assert len(ids)==28791 and np.array_equal(ids,enc['sample_id']) and np.array_equal(ids,lab['sample_id'])
    order=np.argsort(meta['sample_id']);ix=order[np.searchsorted(meta['sample_id'][order],ids)];assert np.array_equal(meta['sample_id'][ix],ids)
    native=cache['candidate'][...,:24].reshape(-1,20,12,2).copy();rank=np.argsort(-cache['logits'],axis=1,kind='stable');slots=np.sort(rank[:,16:],axis=1);anchor=native[np.arange(len(ids))[:,None],slots].copy()
    p2=enc['tokens'].mean(1).astype(np.float32);f=np.concatenate((np.repeat(p2[:,None],4,axis=1),anchor.reshape(-1,4,24)),axis=-1).astype(np.float32)
    past=enc['past'];cv=(past[:,-1,None]-past[:,-2,None])*np.arange(1,13,dtype=np.float32)[None,:,None]
    data=dict(ids=ids,features=f,native=native,anchor=anchor,CV=cv,gt=lab['future'].astype(np.float32),groups=cache['change_group'],slots=slots,track=np.array([str(s)+'::'+str(int(p)) for s,p in zip(meta['sequence'][ix],meta['pedestrian_id'][ix])]),scene=meta['scene_family'][ix]);sp=npz(SPLIT)
    assert np.array_equal(sp['sample_id'],ids[:25912]);tr=sp['grouped_train_index'];assert len(tr)==18909
    x=f[tr].reshape(-1,120);mu=x.mean(0,dtype=np.float64).astype(np.float32);std=x.std(0,dtype=np.float64);std=np.where(std<1e-6,1.,std).astype(np.float32)
    for s in SEEDS:
        z=torch.load(parent_checkpoint(s),map_location='cpu',weights_only=False);assert np.array_equal(mu,z['statistics']['mean'].numpy()) and np.array_equal(std,z['statistics']['std'].numpy())
    return data,sp
def selection(data,sp):
    tr=sp['grouped_train_index'];tracks=sorted(set(data['track'][tr]),key=lambda t:(hashlib.sha256(('shared-map-d0-track-v1|'+t).encode()).hexdigest(),t));cut=int(np.floor(.8*len(tracks)));source=set(tracks[:cut]);probe=tr[np.array([t not in source for t in data['track'][tr]])]
    src=tr[np.array([t in source for t in data['track'][tr]])];src=np.array(sorted(src,key=lambda i:(hashlib.sha256(('shared-map-d0-source-v1|'+str(int(data['ids'][i]))).encode()).hexdigest(),int(data['ids'][i]))),np.int64)
    assert len(src)>=2048 and len(probe)>=500 and len(set(data['track'][probe]))>=30
    packs=src[:2048].reshape(8,256);assert not set(data['track'][packs.ravel()])&set(data['track'][probe])
    return packs,probe,dict(FIT_windows=len(tr),FIT_tracks=len(tracks),source_tracks=cut,source_pool_windows=len(src),selected_source_windows=2048,probe_windows=len(probe),probe_tracks=len(set(data['track'][probe])),whole_track_disjoint=True,previously_seen_FIT=True)
def batch(data,ix):return [torch.from_numpy(data[k][ix]).to(DEVICE) for k in ('features','anchor','CV','gt')]
def perwindow(out,gt):
    e=torch.linalg.vector_norm(out-gt[:,None],dim=-1);q=e.mean(-1)+.5*e[:,:,-1];div=torch.zeros(len(out),device=out.device,dtype=out.dtype)
    for i in range(4):
        for j in range(i+1,4):div=div+F.relu(.3-torch.linalg.vector_norm(out[:,i]-out[:,j],dim=-1).mean(-1))**2
    return q.min(1).values+.05*(div/6),q.argmin(1)
def aggregate(q,arm,step):
    assert arm in ARMS
    if arm=='ordinary':return q.sum(0),0
    h=q.clone();count=0;order=[step%3,(step+1)%3,(step+2)%3]
    for g in range(3):
        for j in order:
            if j==g:continue
            dot=torch.dot(h[g],q[j]);den=torch.dot(q[j],q[j])
            if dot<0 and den>0:h[g]=h[g]-(dot/den)*q[j];count+=1
    return h.sum(0),count
def vector(m):return torch.cat([p.detach().reshape(-1).double() for p in m.parameters()])
def apply_step(m,opt,data,ix,arm,step,check_direct=False):
    guard();m.train();f,a,c,gt=batch(data,ix);out,_=m(f,a,c,'position3');ell,winner=perwindow(out,gt);params=list(m.parameters());groups=data['groups'][ix];qs=[]
    for g in range(3):
        mask=torch.as_tensor(groups==g,device=DEVICE)
        if mask.any():
            # Mean group multiplied by n_g/B, preserving population weights.
            loss=ell[mask].mean()*(int(mask.sum())/len(ix));gr=torch.autograd.grad(loss,params,retain_graph=True)
            qs.append(torch.cat([x.reshape(-1).double() for x in gr]))
        else:qs.append(torch.zeros(sum(x.numel() for x in params),device=DEVICE,dtype=torch.float64))
    q=torch.stack(qs);scalar=float(ell.mean().detach());l2=None;scalar_error=None
    if check_direct:
        reference=canonical_objective(out,gt)[0];scalar_error=abs(float(reference.detach())-scalar);assert scalar_error<=2e-12
        grad=torch.autograd.grad(reference,params,retain_graph=True);direct=torch.cat([x.reshape(-1).double() for x in grad]);l2=float(torch.linalg.vector_norm(q.sum(0)-direct));assert l2<=1e-6+2e-4*float(direct.norm()),l2
    combined,projections=aggregate(q,arm,step);assert torch.isfinite(q).all() and torch.isfinite(combined).all();before=vector(m);opt.zero_grad(set_to_none=True);off=0
    for p in params:p.grad=combined[off:off+p.numel()].reshape_as(p).to(p.dtype);off+=p.numel()
    opt.step();delta=float((vector(m)-before).norm());u=m.net((f-m.mean)/m.std).reshape(-1,4,3,2).double().detach();u=torch.tanh(u);u=u/u.norm(dim=-1,keepdim=True).clamp_min(1.)
    _,_,lam=decode_numpy(data['anchor'][ix],u.cpu().numpy(),'position3')
    return dict(step=step+1,source_preupdate_L=scalar,group_counts=[int(np.sum(groups==g)) for g in range(3)],group_norms=q.norm(dim=1).cpu().tolist(),dot_matrix=(q@q.T).cpu().tolist(),negative_pair_count=int(((q@q.T)[torch.triu_indices(3,3,1,device=DEVICE).unbind()]<0).sum()),projections=projections,combined_gradient_norm=float(combined.norm()),parameter_displacement_norm=delta,winner=winner.detach().cpu().numpy(),lambda_active=lam<1,canonical_gradient_L2_error=l2,canonical_scalar_error=scalar_error)
def predict(m,data,ix):
    m.eval();out=[];res=[];u=[];before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    with torch.inference_mode():
        for start in range(0,len(ix),128):
            guard();ind=ix[start:start+128];f,a,c,gt=batch(data,ind);p,r,control=m(f,a,c,'position3',True);out.append(p.cpu().numpy());res.append(r.cpu().numpy());u.append(control.cpu().numpy())
    assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in m.state_dict().items())
    out=np.concatenate(out);res=np.concatenate(res);u=np.concatenate(u);expected,rr,lam=decode_numpy(data['anchor'][ix],u,'position3')
    assert np.max(abs(expected-out))<=2e-12 and np.max(abs(rr-res))<=2e-12
    assert (np.linalg.norm(res,axis=-1)<=.20*(np.arange(1,13)/12.)**2+2e-15).all()
    paths=data['native'][ix].astype(np.float64).copy();paths[np.arange(len(ix))[:,None],data['slots'][ix]]=out
    keep=np.ones((len(ix),20),bool);keep[np.arange(len(ix))[:,None],data['slots'][ix]]=False;assert np.array_equal(paths[keep],data['native'][ix][keep])
    return dict(sample_id=data['ids'][ix],groups=data['groups'][ix],track=data['track'][ix],scene=data['scene'][ix],gt=data['gt'][ix],paths=paths,refined4=out,residual=res,bounded_controls=u,slots=data['slots'][ix],lambda_value=lam)
def numpy_loss(p,gt):
    e=np.linalg.norm(p-gt[:,None],axis=-1);quality=np.min(e.mean(-1)+.5*e[:,:,-1],axis=1);d=np.zeros(len(p))
    for i in range(4):
        for j in range(i+1,4):d+=np.maximum(.3-np.linalg.norm(p[:,i]-p[:,j],axis=-1).mean(-1),0)**2
    return quality+.05*d/6
def metrics(z,baseline,ind=False,with_clusters=True):
    a,f,j,good,one=errors(z['paths'],z['gt'],ind);gg=good.any(1);bg=errors(baseline,z['gt'],ind)[3].any(1);counts=clusters(z['paths'],good,ind) if with_clusters else None;rows={}
    for name,mask in masks(z['groups']):
        if not mask.any():rows[name]=dict(n=0);continue
        row=dict(n=int(mask.sum()),tracks=len(set(z['track'][mask])),coverage=float(gg[mask].mean()),good=int(gg[mask].sum()),ADE=float(a[mask].min(1).mean()),FDE=float(f[mask].min(1).mean()),J=float(j[mask].min(1).mean()),rescued_from_step0=int((gg&~bg&mask).sum()),lost_step0good=int((~gg&bg&mask).sum()),step0good=int((bg&mask).sum()),K4_L=float(numpy_loss(z['refined4'][mask],z['gt'][mask]).mean()),single_Jclosest_early_middle_late=one[mask].mean(0),motion=motion(z['paths'][mask],ind))
        if counts is not None:row.update(effective_distinct=float(counts[mask,0].mean()),effective_good=float(counts[mask,1].mean()))
        rows[name]=row
    return rows
def locks():
    old=read(H.parent/'D0_execution_20261015/source_manifest.json')['files'];files=dict(old);verify(old)
    add=[H.parent/'EXPERIMENT_GUIDE.md',P/'run.py',P/'RESULT_ANALYSIS.md',P/'RESULT_ANALYSIS.json',P/'PRESERVATION.json',P/'ARITHMETIC_AUDIT.json',P/'statistics.npz',*H.glob('*.py'),*H.glob('*.ps1'),H/'README.md',H/'protocol.json']
    for p in (H.parent/'D0_execution_20261015').glob('worker_*.log'):
        add.append(p)
    parent=H.parent/'D0_execution_20261015'
    add.extend([H/'REPAIR_DESIGN.json',H/'SOURCE_EQUIVALENCE.json',parent/'source_manifest.json',parent/'worker_state.json',parent/'exit_status.json',parent/'halfway_check.json',parent/'failures/resource_gate_1/FAILURE_PROVENANCE.json',parent/'failures/resource_gate_1/REPORT.md'])
    for s in SEEDS:
        d=P/f'runs/position3/seed_{s}';done=read(d/'done.json');assert done['status']=='PASS'
        for p,h in done['files'].items():
            if p in files:assert files[p]==h
            files[p]=h
        add.append(d/'done.json')
    for p in add:
        key=str(p);h=sha(p)
        if key in files:assert files[key]==h,('OLD_SOURCE_CHANGED',key)
        files[key]=h
    verify(files);return files,len(old)
def reject_failure(kind):
    d=H/'failures'/str(time.time_ns());d.mkdir(parents=True,exist_ok=True)
    (d/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    for p in H.glob('*.py'):(d/p.name).write_bytes(p.read_bytes())
    write('FAILURE.json',dict(status=kind,source_directory=str(d),resource=RESOURCE));return d
