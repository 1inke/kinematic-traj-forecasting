"""Step3 only; do not import legacy entry points or mutate old output roots."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys,json,hashlib,datetime,importlib.util
import numpy as np
HERE=Path(__file__).resolve().parent; SERIES=HERE.parent; ROOT=HERE.parents[2]
PREVIOUS=SERIES/'execution_peak_measured_20261026'
OLD=ROOT/'qoder_work/K20_SINGLE_EDIT_COMPLEMENTARITY_EXEC_20261025/E1_execution_20261026'
sys.path.insert(0,str(ROOT))
from src.f16.io import read_json,write_json,sha256
from src.f16.resource import Guard
SEEDS=(42,43,44);ARMS=('C_base','C_token','C_neighbor','C_both')
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def guard(stage):
    g=Guard(read_json(HERE/'RESOURCE_STAGE_PLAN.json')['estimates_bytes'][stage]);g.check('startup');return g
def require12():
    assert read_json(PREVIOUS/'RESULT_ANALYSIS.json')['status']=='ANALYZED_STEPS12_THEN_IMPLEMENT34'
    assert read_json(PREVIOUS/'completion_review/REVIEW.json')['status']=='PASS_SAVED_COMPLETION_SUPPLEMENT'
def save_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Preserve existing '+str(path))
    np.savez_compressed(path,**arrays)
def obs(split):
    with np.load(PREVIOUS/f'prepared/observations_{split}.npz') as z:return {k:z[k] for k in z.files}
def base_stats(seed):
    with np.load(OLD/f'prepared_v1/stats_seed{seed}.npz') as z:return z['mean32'],z['std32']
def old_features(seed,ix):
    mm=np.load(OLD/f'prepared_v1/features_seed{seed}.npy',mmap_mode='r');x=np.array(mm[ix],copy=True);mm._mmap.close();return x
def aux_chunks(seed,split):
    folder=OLD/f'populations/seed{seed}/{split}'
    d=read_json(folder/'done.json')
    for name in d['chunks']:
        with np.load(folder/name) as z:yield {k:z[k] for k in z.files}
def resource_report(path,g):write_json(path,dict(observations=g.observations,last=g.check()))
def verify():
    m=read_json(HERE/'source_manifest.json');assert digest(m['files'])==m['digest']
    for p,h in m['files'].items():assert sha256(ROOT/p if not Path(p).is_absolute() else Path(p))==h,p
    return m['digest']
def source():return read_json(HERE/'source_manifest.json')['digest']
def completed(folder,c):
    d=read_json(folder/'done.json');assert d['contract']==c
    for p,h in d['outputs'].items():assert sha256(folder/p)==h,p
    return d
def finish(folder,c,**kw):
    files={p.name:sha256(p) for p in folder.iterdir() if p.is_file() and p.name!='done.json'}
    write_json(folder/'done.json',dict(contract=c,outputs=files,**kw));return completed(folder,c)
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m


