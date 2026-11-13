import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from datetime import datetime
import ast,json,hashlib,time,gc,traceback,ctypes
import numpy as np,psutil
from scipy.cluster.hierarchy import linkage,fcluster
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
ENVROOT=ROOT/'MOTION_INFORMATION_SAFE_REACHABILITY_DESIGN_20261016/F0_execution_authorized_20261017'
T1=ROOT/'GENERATION_TRANSFER_RETENTION_DESIGN_20261016/T1_execution_20261016'
GROUPS=('overall','stable','transition','change')
RESOURCE=dict(estimated_total_peak_bytes=4916164516,factor=1.3,checks=0,aggregate_current_max_private=0,aggregate_current_max_RSS=0,parent_peak_private=0,parent_peak_RSS=0,child_max_peak_private=0,child_max_peak_RSS=0,min_physical=None,min_commit=None)
CHILD=None
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended')]
def write(p,v):
    p=Path(p);p=p if p.is_absolute() else H/p;p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8');os.replace(temp,p)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),default=lambda a:a.tolist()).encode()).hexdigest()
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def save(p,**kw):
    p=H/p;p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**kw)
def guard(child=None,startup=False):
    global CHILD
    if child is not None:CHILD=child
    p=psutil.Process().memory_info();members=[p]
    try:
        if CHILD is not None:members.extend(x.memory_info() for x in [psutil.Process(CHILD),*psutil.Process(CHILD).children(recursive=True)])
    except psutil.NoSuchProcess:CHILD=None
    v=Mem();v.length=ctypes.sizeof(v);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(v))
    e=RESOURCE['estimated_total_peak_bytes'];rss=sum(x.rss for x in members);private=sum(x.private for x in members)
    RESOURCE['checks']+=1
    for key,value in dict(aggregate_current_max_private=private,aggregate_current_max_RSS=rss,parent_peak_private=p.peak_pagefile,parent_peak_RSS=p.peak_wset,child_max_peak_private=sum(x.peak_pagefile for x in members[1:]),child_max_peak_RSS=sum(x.peak_wset for x in members[1:])).items():RESOURCE[key]=max(RESOURCE[key],int(value))
    for key,value in (('min_physical',v.avail_phys),('min_commit',v.avail_page)):RESOURCE[key]=int(value) if RESOURCE[key] is None else min(RESOURCE[key],int(value))
    retained=max(RESOURCE['parent_peak_private']+RESOURCE['child_max_peak_private'],RESOURCE['parent_peak_RSS']+RESOURCE['child_max_peak_RSS'])
    needp=1.3*(e if startup else max(e-rss,0));needc=1.3*(e if startup else max(e-private,0))
    r=dict(physical_available=int(v.avail_phys),commit_available=int(v.avail_page),required_physical=needp,required_commit=needc,aggregateRSS=rss,aggregateprivate=private,owned_solver_tree_processes=len(members)-1,conservative_sum_retained_peaks=retained,observed_at=datetime.now().astimezone().isoformat())
    if v.avail_phys<needp or v.avail_page<needc or max(retained,rss,private)>1.3*e:
        write('RESOURCE_FAILURE.json',dict(status='STOP_RESOURCE',snapshot=r,resource=RESOURCE));raise MemoryError('TASK_1P3_RESOURCE')
    return r
def verify(files):
    for i,(p,h) in enumerate(files.items()):
        if i%32==0:guard()
        assert sha(p)==h,('SHA_MISMATCH',p)
def stage(name,**kw):
    guard();write('progress.json',dict(stage=name,at=datetime.now().astimezone().isoformat(),**kw));print(name,kw,flush=True)
def extract(path,names):
    nodes=[x for x in ast.parse(Path(path).read_text(encoding='utf-8-sig')).body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
    assert {x.name for x in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),globals())
extract(ROOT/'F16_CANONICAL_EQUAL_K20_REPLACEMENT_DIAGNOSIS_20261007/run.py',('errors','motion','clusters','masks'))
def close(a,b,tol=2e-12):
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
    err=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.
    assert err<=tol,('NUMERICAL_MISMATCH',err,tol);return err
def compare(a,b):
    if isinstance(a,dict):assert a.keys()==b.keys();return max([compare(a[k],b[k]) for k in a] or [0.])
    if isinstance(a,(tuple,list)):
        assert len(a)==len(b);return max([compare(x,y) for x,y in zip(a,b)] or [0.])
    if isinstance(a,(str,bool)) or a is None:assert a==b;return 0.
    return close(a,b)
def load_fit():
    row=read(T1/'JOBS.json')['native/FIT'];verify(row['files']);z=npz(row['output'])
    assert len(z['sample_id'])==18909 and len(set(z['sample_id']))==18909 and (z['groups']==2).sum()==4725
    assert np.array_equal(z['paths'],z['native']);assert np.array_equal(z['native'],z['native'].astype(np.float32).astype(float))
    return z
def locks():
    P=ROOT/'MOTION_INFORMATION_SAFE_REACHABILITY_DESIGN_20261016/F0_execution_solver_tree_recovery_20261017'
    files=dict(read(P/'source_manifest.json')['files']);assert len(files)==5732;verify(files)
    # Inherit completed F0 and I1 science without overwriting old digests.
    paths=list(H.glob('*.py'))+list(H.glob('*.ps1'))+list(H.glob('*.md'))+list(H.parent.glob('*.md'))
    paths += [H/n for n in ('protocol.json','solver_contract.json','sample_registry.json','MEMORY_ESTIMATE.json','DA1_REVIEW.json')]
    paths += [p for p in P.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.npz','.md') and '__pycache__' not in str(p)]
    I=ROOT/'MOTION_INFORMATION_SAFE_REACHABILITY_DESIGN_20261016/I1_execution_authorized_20261017'
    for p,h in read(I/'source_manifest.json')['files'].items():
        if p in files:assert files[p]==h,('INHERITED_CONFLICT',p)
        files[p]=h
    paths += [p for p in I.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.npz','.pt','.md') and '__pycache__' not in str(p) and '.solver_env' not in str(p)]
    for p in paths:
        if p.is_file():
            key=str(p);h=sha(p)
            if key in files:assert files[key]==h,('OLD_DIGEST_CHANGED',key)
            files[key]=h
    verify(files);return files
def failure(kind):
    folder=H/'failures'/str(time.time_ns());folder.mkdir(parents=True,exist_ok=True)
    (folder/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    for p in [*H.glob('*.py'),H/'protocol.json',H/'source_manifest.json',H/'RESOURCE_FAILURE.json']:
        if p.is_file():(folder/p.name).write_bytes(p.read_bytes())
    write('FAILURE.json',dict(status=kind,folder=str(folder),resource=RESOURCE))

