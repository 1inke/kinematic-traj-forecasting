import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from datetime import datetime
import ast,json,time,hashlib,traceback,gc,importlib.util
import numpy as np,psutil
from scipy.cluster.hierarchy import linkage,fcluster
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
PARENT=ROOT/'GENERATION_UNPROJECTED_TIME_COUPLED_REFINEMENT_20261015/U0_execution_20261015'
U1=ROOT/'U0_DIRECTION_RETENTION_AUDIT_DESIGN_20261015/execution_20261015'
CACHE=ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz'
LABELS=ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz'
META=ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz'
SPLITS=ROOT/'GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz'
GROUPS=('overall','stable','transition','change');SIZES=dict(FIT=18909,outer=1035,inner=1856,original_CAL=2879)
spec=importlib.util.spec_from_file_location('approved_memory',H/'memory_policy.py');policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy);policy.H=H
guard=policy.guard;RESOURCE=policy.RESOURCE

def extract(path,names,namespace):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'));nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)

# Explicit function whitelist, no legacy module top-level execution or old output writes.
extract(PARENT/'common.py',('read','write','sha','digest','npz','load_data','load','close','compare','valid_done'),globals())
quality_source=ROOT/'F16_CANONICAL_EQUAL_K20_REPLACEMENT_DIAGNOSIS_20261007/run.py'
extract(quality_source,('errors','motion','clusters','masks'),globals())

def compare(a,b):
    if a is None:assert b is None;return 0.
    if isinstance(a,dict):
        assert a.keys()==b.keys();return max([compare(a[k],b[k]) for k in a] or [0.])
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b);return max([compare(x,y) for x,y in zip(a,b)] or [0.])
    if isinstance(a,(str,bool)):assert a==b;return 0.
    return close(a,b)

def verify(files):
    for i,(p,h) in enumerate(files.items()):
        if i%32==0:guard()
        assert sha(p)==h,('SHA_MISMATCH',p)

def stage(name,**kw):
    guard();write('progress.json',dict(stage=name,at=datetime.now().astimezone().isoformat(),**kw));print(name,kw,flush=True)

def failure(kind):
    folder=H/'failures'/str(time.time_ns());folder.mkdir(parents=True,exist_ok=True)
    (folder/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    for p in H.iterdir():
        if p.is_file() and (p.suffix in ('.py','.ps1') or p.name in ('protocol.json','source_manifest.json','PREFLIGHT.json','RESOURCE_FAILURE.json')):(folder/p.name).write_bytes(p.read_bytes())
    write('FAILURE.json',dict(status=kind,at=datetime.now().astimezone().isoformat(),source=str(folder),resource=RESOURCE))

def stats(data,representation='position'):
    from features import numpy_features
    ix=data['split']['FIT'];s1=np.zeros(120);s2=np.zeros(120);n=np.zeros(120)
    for first in range(0,len(ix),256):
        guard();ii=ix[first:first+256];p=data['p2'][ii].astype(float);v=numpy_features(data['native'][ii],representation).reshape(-1,24).astype(float)
        s1[:96]+=p.sum(0);s2[:96]+=(p*p).sum(0);n[:96]+=len(p);s1[96:]+=v.sum(0);s2[96:]+=(v*v).sum(0);n[96:]+=len(v)
    mean=s1/n;sd=np.sqrt(np.maximum(s2/n-mean*mean,0))
    return dict(mean=mean,std=np.where(sd<1e-6,1.,sd),training_ids=data['ids'][ix],counts=n)

def locks():
    failed=H.parent/'P1_execution_20261017'
    files=dict(read(failed/'source_manifest.json')['files']);verify(files)
    for p in failed.rglob('*'):
        if p.is_file() and p.suffix in ('.py','.ps1','.md','.json','.npz','.pt') and '__pycache__' not in str(p):
            key=str(p);h=sha(p)
            if key in files:assert files[key]==h,('PARENT_SOURCE_CHANGED',key)
            files[key]=h
    parents=[ROOT/'MOTION_INFORMATION_SAFE_REACHABILITY_DESIGN_20261016/F0_execution_solver_tree_recovery_20261017',ROOT/'MOTION_INFORMATION_SAFE_REACHABILITY_DESIGN_20261016/I1_execution_authorized_20261017',ROOT/'GENERATION_TRANSFER_RETENTION_DESIGN_20261016/T2_execution_manifest_recovery_20261016']
    for parent in parents:
        for p,h in read(parent/'source_manifest.json')['files'].items():
            if p in files:assert files[p]==h,('INHERITED_CONFLICT',p)
            files[p]=h
        result=read(parent/'RESULT.json')
        for p,h in result['outputs_SHA'].items():
            if p in files:assert files[p]==h
            files[p]=h
        for name in ('RESULT.json','RESULT_ANALYSIS.md','RESULT_ANALYSIS.json','ARITHMETIC_AUDIT.json','PRESERVATION.json','RESOURCE.json'):
            p=parent/name
            if p.is_file():
                if str(p) in files:assert files[str(p)]==sha(p)
                files[str(p)]=sha(p)
    inherited=len(files);verify(files)
    paths=list(H.glob('*.py'))+list(H.glob('*.ps1'))+list(H.glob('*.md'))+list(H.parent.glob('*.md'))
    paths += [H/x for x in ('protocol.json','resource_policy.json','MEMORY_ESTIMATE.json','statistics_position.npz','responsibility_labels.npz','IDENTITY_AUDIT.json','DA1_REVIEW.json')]
    paths += [PARENT/x for x in ('common.py','model.py','metrics.py')]
    paths += [CACHE,LABELS,META,SPLITS,ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz',ROOT/'GENERATION_UPSTREAM_TEMPORAL_SERIES/train_observations.npz']
    for p in paths:
        if p.is_file():
            key=str(p);h=sha(p)
            if key in files:assert files[key]==h,('OLD_DIGEST_CHANGED',key)
            files[key]=h
    verify(files);return files,inherited


