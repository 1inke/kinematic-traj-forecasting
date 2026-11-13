"""New bounded execution; CPU modules do not import Torch or legacy mains."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import sys,json,hashlib,datetime,time,importlib.util
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OLD=ROOT/'qoder_work/K20_SINGLE_EDIT_COMPLEMENTARITY_EXEC_20261025/E1_execution_20261026'
E0=OLD.parent/'E0_execution_20261025';DESIGN=ROOT/'docs/OBSERVABLE_INFORMATION_DIAGNOSIS_DESIGN_20261026'
sys.path.insert(0,str(ROOT))
from src.f16.io import read_json,write_json,sha256
from src.f16.resource import Guard
SEEDS=(42,43,44);ARMS=('M_full','T_full','T_geom','T_perm')
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def guard(stage):
    g=Guard(read_json(HERE/'RESOURCE_STAGE_PLAN.json')['estimates_bytes'][stage]);g.check('startup');return g
def mark(stage,**kw):write_json(HERE/'progress.json',dict(stage=stage,at=now(),**kw))
def save_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Preserve existing science output '+str(path))
    np.savez_compressed(path,**arrays)
def source():return read_json(HERE/'source_manifest.json')['digest']
def verify(m=None):
    m=m or read_json(HERE/'source_manifest.json');assert digest(m['files'])==m['digest']
    for p,h in m['files'].items():
        if not (ROOT/p).is_file() or sha256(ROOT/p)!=h:raise ValueError('Source/input SHA '+p)
    return m['digest']
def completed(folder,contract):
    d=read_json(folder/'done.json')
    if d['contract']!=contract:raise ValueError('Completed contract mismatch')
    for p,h in d['outputs'].items():
        if sha256(folder/p)!=h:raise ValueError('Completed output SHA '+p)
    return d
def finish(folder,contract,**kw):
    files={p.name:sha256(p) for p in folder.iterdir() if p.is_file() and p.name!='done.json'}
    d=dict(status='COMPLETE',contract=contract,outputs=files,**kw);write_json(folder/'done.json',d);completed(folder,contract);return d
def resource_report(folder,g):write_json(folder/'RESOURCE.json',dict(observations=g.observations,last=g.check()))
def aux_chunks(seed,split):
    f=OLD/f'populations/seed{seed}/{split}';d=read_json(f/'done.json')
    for name in d['chunks']:
        with np.load(f/name) as z:yield {k:z[k] for k in z.files}
def stats(seed):
    with np.load(OLD/f'prepared_v1/stats_seed{seed}.npz') as z:return z['mean32'],z['std32']
def normalize(x,seed):
    mean,std=stats(seed);out=np.asarray(x,np.float32).copy();out[...,:323]=(out[...,:323]-mean)/std;return out
def old_features(seed,indices):
    mm=np.load(OLD/f'prepared_v1/features_seed{seed}.npy',mmap_mode='r');x=np.array(mm[indices],copy=True);mm._mmap.close();return x
def opt_labels(seed):
    with np.load(OLD/f'prepared_v1/labels_seed{seed}.npz') as z:return {k:z[k] for k in z.files}
def fail(exc):
    import traceback
    write_json(HERE/('FAILURE_'+str(time.time_ns())+'.json'),dict(at=now(),type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),preserve_all=True))
