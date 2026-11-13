import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from datetime import datetime
import ast,json,time,ctypes,hashlib,traceback,gc
import numpy as np,psutil
from scipy.cluster.hierarchy import linkage,fcluster
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
D0=ROOT/'SHARED_MAPPING_GENERALIZATION_DESIGN_20261015/D0_recovery_resource_20261015'
CACHE=ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz'
LABELS=ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz'
META=ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz'
SPLITS=ROOT/'GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz'
PORTS=('native20','safe4','safe20');GROUPS=('overall','stable','transition','change')
SIZES=dict(FIT=18909,outer=1035,inner=1856,original_CAL=2879)
RESOURCE=dict(peak_private_GiB=0.,peak_RSS_GiB=0.,min_physical_GiB=100.,min_commit_GiB=100.,checks=0)
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended')]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,z):
    p=Path(p);p=p if p.is_absolute() else H/p;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8');os.replace(tmp,p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def digest(z):return hashlib.sha256(json.dumps(z,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def guard():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));p=psutil.Process().memory_info()
    RESOURCE['peak_private_GiB']=max(RESOURCE['peak_private_GiB'],p.peak_pagefile/2**30);RESOURCE['peak_RSS_GiB']=max(RESOURCE['peak_RSS_GiB'],p.peak_wset/2**30)
    RESOURCE['min_physical_GiB']=min(RESOURCE['min_physical_GiB'],m.avail_phys/2**30);RESOURCE['min_commit_GiB']=min(RESOURCE['min_commit_GiB'],m.avail_page/2**30);RESOURCE['checks']+=1
    if max(p.private,p.rss)>6*2**30 or min(m.avail_phys,m.avail_page)<6*2**30:
        write('RESOURCE_FAILURE.json',dict(status='STOP_RESOURCE',at=datetime.now().astimezone().isoformat(),private_GiB=p.private/2**30,RSS_GiB=p.rss/2**30,physical_available_GiB=m.avail_phys/2**30,commit_available_GiB=m.avail_page/2**30,lifecycle=RESOURCE));raise MemoryError('STOP_RESOURCE process6/physical+commit reserve6GiB')
def verify(files):
    for i,(p,h) in enumerate(files.items()):
        if i%32==0:guard()
        assert sha(p)==h,('SHA_MISMATCH',p)
def stage(name,**kw):guard();write('progress.json',dict(stage=name,at=datetime.now().astimezone().isoformat(),**kw));print(name,kw,flush=True)
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
quality_source=ROOT/'F16_CANONICAL_EQUAL_K20_REPLACEMENT_DIAGNOSIS_20261007/run.py'
ns=dict(np=np,linkage=linkage,fcluster=fcluster,guard=guard,GROUPS=GROUPS)
tree=ast.parse(quality_source.read_text(encoding='utf-8-sig'));names=('errors','motion','clusters','masks')
nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names];assert len(nodes)==4
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(quality_source),'exec'),ns)
errors,motion,clusters,masks=[ns[k] for k in names]
def load_data():
    guard()
    with np.load(CACHE,allow_pickle=False) as z:d=dict(sample_id=z['sample_id'].copy(),native=z['candidate'][...,:24].reshape(-1,20,12,2).astype(np.float64),groups=z['change_group'].copy(),logits=z['logits'].copy())
    with np.load(LABELS,allow_pickle=False) as z:assert np.array_equal(z['sample_id'],d['sample_id']);d['gt']=z['future'].astype(np.float64)
    with np.load(META,allow_pickle=False) as z:
        mid=z['sample_id'];order=np.argsort(mid);loc=order[np.searchsorted(mid[order],d['sample_id'])];assert np.array_equal(mid[loc],d['sample_id'])
        seq=z['sequence'][loc];ped=z['pedestrian_id'][loc];d.update(sequence=seq.copy(),track=np.array([str(s)+'::'+str(int(p)) for s,p in zip(seq,ped)]),scene=z['scene_family'][loc].copy())
    sp=npz(SPLITS);assert np.array_equal(sp['sample_id'],d['sample_id'][:25912])
    ix=dict(FIT=sp['grouped_train_index'],outer=sp['outer_eval_index'],inner=sp['inner_cal_index'],original_CAL=np.arange(25912,28791))
    assert len(d['sample_id'])==28791 and len(np.unique(d['sample_id']))==28791
    assert all(np.isfinite(d[k]).all() for k in ('native','gt','logits')) and set(d['groups']).issubset({0,1,2})
    d['slots']=np.sort(np.argsort(-d['logits'],axis=1,kind='stable')[:,16:],axis=1)
    for name,a in ix.items():assert len(a)==SIZES[name] and len(np.unique(a))==len(a)
    assert not any(set(ix[a])&set(ix[b]) for i,a in enumerate(ix) for b in list(ix)[i+1:])
    guard();return d,ix
def locks():
    parent=ROOT/'GENERATION_FEASIBILITY_CONDITIONING_DESIGN_20261015/A0_execution_20261015'
    inherited=read(parent/'source_manifest.json')['files'];assert len(inherited)==1823
    files=dict(inherited);verify(files)
    result=read(parent/'RESULT.json');assert result['status']=='PASS_DIAGNOSTIC_ONLY_ANALYTIC_ENVELOPE'
    extra=dict(result['outputs_SHA'])
    extra.update({str(p):sha(p) for p in parent.glob('*.json') if p.name not in ('runtime_state.json','progress.json','worker_state.json')})
    extra.update({str(p):sha(p) for p in parent.glob('*.md')})
    extra.update({str(p):sha(p) for p in (parent/'completion_review').iterdir() if p.is_file()})
    extra.update({str(p):sha(p) for p in (parent/'jobs').glob('*.done.json')})
    add=[H.parent/'EXPERIMENT_GUIDE.md',*H.glob('*.py'),*H.glob('*.ps1'),H/'README.md',H/'protocol.json']
    extra.update({str(p):sha(p) for p in add})
    for p,h in extra.items():
        if p in files:assert files[p]==h,('INHERITED_DIGEST_CHANGED',p)
        files[p]=h
    verify(files);return files,len(inherited)

def close(a,b,tol=2e-12):
    assert np.shape(a)==np.shape(b);err=float(np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float)))) if np.size(a) else 0.;assert err<=tol,err;return err
def compare(a,b):
    if isinstance(a,dict):assert a.keys()==b.keys();return max([compare(a[k],b[k]) for k in a] or [0.])
    if isinstance(a,(str,bool)):assert a==b;return 0.
    return close(a,b)
def valid_done(row,source):
    assert row['status']=='PASS' and row['source_digest']==source,'BAD_SOURCE_CONTRACT';verify(row['files']);return True
def failure(kind):
    d=H/'failures'/str(time.time_ns());d.mkdir(parents=True,exist_ok=True);(d/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    for p in H.glob('*.py'):(d/p.name).write_bytes(p.read_bytes())
    write('FAILURE.json',dict(status=kind,source=str(d),resource=RESOURCE))
