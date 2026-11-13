import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
from datetime import datetime
import ast,json,time,hashlib,traceback,gc
import numpy as np,psutil
from scipy.cluster.hierarchy import linkage,fcluster
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
PARENT=ROOT/'GENERATION_UNPROJECTED_TIME_COUPLED_REFINEMENT_20261015/U0_execution_20261015'
T2=ROOT/'GENERATION_TRANSFER_RETENTION_DESIGN_20261016/T2_execution_manifest_recovery_20261016'
T1=T2.parent/'T1_execution_20261016'
CACHE=ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz'
LABELS=ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz'
META=ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz'
SPLITS=ROOT/'GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz'
GROUPS=('overall','stable','transition','change');SIZES=dict(FIT=18909,outer=1035,inner=1856,original_CAL=2879)
import memory_policy as policy
guard=policy.guard;RESOURCE=policy.RESOURCE
def extract(path,names,ns):
    t=ast.parse(Path(path).read_text(encoding='utf-8-sig'));a=[n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in a}==set(names);exec(compile(ast.Module(body=a,type_ignores=[]),str(path),'exec'),ns)
extract(PARENT/'common.py',('read','write','sha','digest','npz','load_data','close','valid_done'),globals())
extract(ROOT/'F16_CANONICAL_EQUAL_K20_REPLACEMENT_DIAGNOSIS_20261007/run.py',('errors','motion','clusters','masks'),globals())
def compare(a,b):
    if a is None:assert b is None;return 0.
    if isinstance(a,dict):assert a.keys()==b.keys();return max([compare(a[k],b[k]) for k in a] or [0.])
    if isinstance(a,(tuple,list)):
        assert len(a)==len(b);return max([compare(x,y) for x,y in zip(a,b)] or [0.])
    if isinstance(a,(str,bool)):assert a==b;return 0.
    return close(a,b)
def verify(files):
    for i,(p,h) in enumerate(files.items()):
        if i%32==0:guard()
        assert sha(p)==h,('SHA_MISMATCH',p)
def stage(name,**kw):
    policy.set_stage(name);guard();write('progress.json',dict(stage=name,at=datetime.now().astimezone().isoformat(),**kw));print(name,kw,flush=True)
def load():
    d,ix=load_data()
    with np.load(ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz',allow_pickle=False) as z:
        assert np.array_equal(z['sample_id'],d['sample_id']);d['p2']=z['tokens'].mean(1).astype(np.float32);d['past']=z['past'].copy()
    with np.load(ROOT/'GENERATION_UPSTREAM_TEMPORAL_SERIES/train_observations.npz',allow_pickle=False) as z:
        assert np.array_equal(z['sample_id'],d['sample_id']);assert np.array_equal(d['past'].reshape(-1,16),z['observation'][:,:16])
    assert d['past'].shape==(28791,8,2) and not np.count_nonzero(d['past'][:,-1]) and d['p2'].shape==(28791,96)
    d['split']=ix;d['ids']=d['sample_id'];assert np.isfinite(d['past']).all();guard();return d
def failure(kind):
    folder=H/'failures'/str(time.time_ns());folder.mkdir(parents=True,exist_ok=True);(folder/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    for p in list(H.glob('*.py'))+[H/'protocol.json',H/'source_manifest.json',H/'RESOURCE_FAILURE.json']:
        if p.is_file():(folder/p.name).write_bytes(p.read_bytes())
    write('FAILURE.json',dict(status=kind,at=datetime.now().astimezone().isoformat(),folder=str(folder),resource=RESOURCE))
def locks():
    prior=H.parent/'I1_execution_pipeline_recovery_20261017';inherited=read(prior/'source_manifest.json')['files'];assert len(inherited)==2717;files=dict(inherited);verify(files)
    additions=dict(read(T2/'RESULT.json')['outputs_SHA'])
    paths=[T2/n for n in ('RESULT.json','STATISTICS.json','ARITHMETIC_AUDIT.json','RESOURCE.json','PRESERVATION.json','RESULT_ANALYSIS.md','RESULT_ANALYSIS.json','ALL_GROUPS.md','NEXT_BRANCH.json','halfway_check.json')]
    paths+=[prior/n for n in ('PREFLIGHT.json','source_manifest.json','IDENTITY_AUDIT.json','PRODUCTION_MANIFEST_PREFLIGHT.json')]+list((prior/'preflight_smoke').rglob('*'))+list((prior/'resume_smoke').rglob('*'))+list((T2/'completion_review').glob('*'))+list(H.parent.glob('*.md'))+list((H.parent/'F0_execution_20261017').glob('*'))+list(H.glob('*.py'))+list(H.glob('*.ps1'))+list((H/'failures').rglob('*'))
    paths+=[H/n for n in ('README.md','protocol.json','resource_policy.json','MEMORY_ESTIMATE.json','statistics.npz','INITIALIZATIONS.json','IDENTITY_AUDIT.json','DA1_REVIEW.json','REPAIR_NOTES.md','PIPELINE_BUDGET_REVIEW.md','SOURCE_EQUIVALENCE.json','EXECUTION_AMENDMENT.json')]
    for p in paths:
        if p.is_file():additions[str(p)]=sha(p)
    for p,h in additions.items():
        if p in files:assert files[p]==h,('OLD_HASH_CHANGED',p)
        files[p]=h
    verify(files);return files
