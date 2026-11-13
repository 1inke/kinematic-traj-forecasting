from pathlib import Path
import sys,os,json,hashlib,datetime,ast,time,copy,random
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
import numpy as np
from src.f16.io import read_json,write_json,sha256
from resource_guard_v3 import Guard
OLD=ROOT/'qoder_work/K20_SINGLE_EDIT_COMPLEMENTARITY_EXEC_20261025/E1_execution_20261026'
OBS=ROOT/'qoder_work/OBSERVABLE_INFORMATION_DIAGNOSIS_EXEC_20261026/execution_peak_measured_20261026'
PARENT=ROOT/'qoder_work/STOP_GATE_CONDITIONAL_RANKING_EXEC_20261027/D0_execution_20261027'
DOC=Path('<LOCAL_DOWNLOADS>/K20_候选间协同细化_最小验证实验与创新路线_20261027.md')
SEEDS=(42,43,44);ARMS=('B1','B2');SPLITS=('opt','hold','inner');GROUPS=('overall','stable','transition','change')
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}
def save(path,**kw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Preserve '+str(path))
    np.savez_compressed(path,**kw)
def mark(stage,**kw):write_json(HERE/'progress.json',dict(stage=stage,at=now(),**kw))
def guard(cpu=False,startup=False):
    policy=read_json(HERE/'RESOURCE_STAGE_PLAN.json');g=Guard(policy['CPU_estimate_bytes' if cpu else 'GPU_estimate_bytes']);g.check('startup' if startup else 'runtime');return g
def resource(path,g):write_json(path,dict(last=g.check(force=True),observations=g.observations))
def finish(folder,contract,**kw):write_json(folder/'done.json',dict(contract=contract,outputs={p.name:sha256(p) for p in folder.iterdir() if p.is_file() and p.name!='done.json'},**kw))
def completed(folder,contract):
    d=read_json(folder/'done.json');assert d['contract']==contract
    for n,h in d['outputs'].items():assert sha256(folder/n)==h,n
    return d
def pure(path,names,ns):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'));nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names];assert {n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns
def source():return read_json(HERE/'source_manifest.json')['digest']
def verify():
    m=read_json(HERE/'source_manifest.json');assert digest(m['files'])==m['digest']
    for n,h in m['files'].items():assert sha256(Path(n) if Path(n).is_absolute() else ROOT/n)==h,n
    return m['digest']
def build_source():
    assert not (HERE/'PRODUCTION_SOURCE_LOCK.json').exists();parent=ROOT/'qoder_work/K20_COLLABORATIVE_REFINEMENT_EXEC_20261027';m=read_json(parent/'source_manifest.json');files=m['files'].copy()
    def add(p,h=None):
        try:k=p.relative_to(ROOT).as_posix()
        except ValueError:k=p.as_posix()
        hh=sha256(p);assert h is None or h==hh
        if k in files:assert files[k]==hh,k
        files[k]=hh
    for p in parent.rglob('*'):
        if p.is_file() and 'prepared' not in p.relative_to(parent).parts:add(p)
    add(DOC)
    for p in HERE.iterdir():
        if p.is_file() and (p.suffix in ('.py','.ps1','.md') or p.name in ('protocol.json','RESOURCE_STAGE_PLAN.json','SOURCE_EQUIVALENCE.json','MEASURED_MEMORY_ESTIMATE.json')):add(p)
    for root in ('prepared','failures','preflight_history','resource_policy_v2_snapshot'):
        for p in (HERE/root).rglob('*'):
            if p.is_file():add(p)
    old=HERE/'source_manifest.json'
    if old.exists():
        f=HERE/'candidate_provenance';f.mkdir(exist_ok=True);z=read_json(old);write_json(f/(z['digest']+'.json'),z)
    result=dict(files=files,digest=digest(files),parent=m['digest'],strict_old_digest_inheritance=True);write_json(old,result);verify();return result
def raw_population(split,include_gt):
    # Read only these explicit members, never KIN or old B/T/401 labels.
    out=[];f=OLD/f'populations/seed42/{split}';d=read_json(f/'done.json');keys=['native','sample_id','track_key','group']+(['gt'] if include_gt else [])
    for n in d['chunks']:
        assert sha256(f/n)==d['outputs'][n]
        with np.load(f/n) as z:out.append({k:z[k] for k in keys})
    return {k:np.concatenate([z[k] for z in out]) for k in keys}
def dataset(split,held=False):
    if split!='opt':
        assert held;lock=read_json(HERE/'ALL6_FINALS_LOCK.json');assert lock['source']==source() and len(lock['models'])==6
        for k,h in lock['models'].items():assert sha256(HERE/k)==h
    z=load(HERE/f'prepared/{split}.npz')
    if 'gt' not in z:
        r=raw_population(split,True);assert np.array_equal(r['sample_id'],z['sample_id']);z['gt']=r['gt']
    return z
def failure(stage,ex):
    import shutil,traceback
    f=HERE/'failures'/str(time.time_ns());f.mkdir(parents=True);write_json(f/'FAILURE.json',dict(stage=stage,error=str(ex),traceback=traceback.format_exc(),at=now(),no_delete=True))
    for p in HERE.glob('*.py'):shutil.copyfile(p,f/p.name)
def contract(seed,arm):return dict(source=source(),protocol=sha256(HERE/'protocol.json'),prepared=sha256(HERE/'prepared/ready.json'),seed=seed,arm=arm,epochs=8,batch=32,stats=sha256(HERE/'prepared/statistics.npz'),split=sha256(HERE/'prepared/split.npz'))
