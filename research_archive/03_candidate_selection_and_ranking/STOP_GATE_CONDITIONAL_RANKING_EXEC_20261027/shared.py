"""Read-only historical evidence; never import legacy executable entry points."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys,json,hashlib,datetime,ast
import numpy as np
SERIES=Path(__file__).resolve().parent;ROOT=SERIES.parents[1]
OLD=ROOT/'qoder_work/K20_SINGLE_EDIT_COMPLEMENTARITY_EXEC_20261025/E1_execution_20261026'
OBS=ROOT/'qoder_work/OBSERVABLE_INFORMATION_DIAGNOSIS_EXEC_20261026'
PREV=OBS/'step4_execution_20261027';P12=OBS/'execution_peak_measured_20261026';P3=OBS/'step3_execution_20261027'
sys.path.insert(0,str(ROOT))
from src.f16.io import read_json,write_json,sha256
from src.f16.resource import Guard
SEEDS=(42,43,44);SPLITS=('opt','hold','inner');GROUPS=('overall','stable','transition','change')
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def load_npz(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}
def save_npz(path,**x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Preserve old '+str(path))
    np.savez_compressed(path,**x)
def verify_manifest(folder):
    m=read_json(folder/'source_manifest.json');assert digest(m['files'])==m['digest']
    for p,h in m['files'].items():assert sha256(Path(p) if Path(p).is_absolute() else ROOT/p)==h,p
    return m['digest']
def source(folder):return read_json(folder/'source_manifest.json')['digest']
def finish(folder,contract,**kw):
    write_json(folder/'done.json',dict(contract=contract,outputs={p.name:sha256(p) for p in folder.iterdir() if p.is_file() and p.name!='done.json'},**kw))
def completed(folder,expected):
    d=read_json(folder/'done.json');assert d['contract']==expected
    for p,h in d['outputs'].items():assert sha256(folder/p)==h,p
    return d
def aux(seed,split):
    f=OLD/f'populations/seed{seed}/{split}';d=read_json(f/'done.json')
    for name in d['chunks']:
        with np.load(f/name) as q:yield name,{k:q[k] for k in q.files}
def metadata(split):
    z=load_npz(P12/f'prepared/observations_{split}.npz');return z
def pure(path,names,namespace):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'));nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace);return namespace
def add_source(files,path,expected=None):
    p=Path(path);key=p.relative_to(ROOT).as_posix();h=sha256(p)
    if expected is not None:assert h==expected,key
    if key in files:assert files[key]==h,key
    else:files[key]=h
def build_source(folder):
    if (folder/'PRODUCTION_SOURCE_LOCK.json').exists():raise ValueError('Cannot rebase locked production source')
    parent=read_json(PREV/'source_manifest.json');files=parent['files'].copy()
    assert len(files)==3536
    for name,h in read_json(PREV/'PRESERVATION.json')['outputs'].items():add_source(files,PREV/name,h)
    for p in (PREV/'completion_review').glob('*'):
        if p.is_file():add_source(files,p)
    for p in [PREV/'RESULT_ANALYSIS.md',PREV/'RESULT_ANALYSIS.json',OBS/'RESULT_ANALYSIS.md',OBS/'RESULT_ANALYSIS.json',SERIES/'shared.py',SERIES/'EXECUTION_AUTHORIZATION.md']:
        add_source(files,p)
    for p in (ROOT/'docs/STOP_GATE_CONDITIONAL_RANKING_DESIGN_20261027').glob('*.md'):add_source(files,p)
    for p in folder.iterdir():
        if p.is_file() and (p.suffix=='.py' or p.name in ('protocol.json','README.md','RESOURCE_STAGE_PLAN.json')):add_source(files,p)
    old=folder/'source_manifest.json'
    if old.exists():
        prev=read_json(old);hist=folder/'candidate_provenance';hist.mkdir(exist_ok=True);write_json(hist/(prev['digest']+'.json'),prev)
    m=dict(files=files,digest=digest(files),parent=parent['digest'],no_old_digests_replaced=True);write_json(old,m);verify_manifest(folder);return m
