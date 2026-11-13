"""Fourth and final finite diagnosis. CPU only, no legacy main imports."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,datetime,time,importlib.util
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];SERIES=HERE.parent
PREVIOUS=SERIES/'step3_execution_20261027';TWELVE=SERIES/'execution_peak_measured_20261026'
OLD=ROOT/'qoder_work/K20_SINGLE_EDIT_COMPLEMENTARITY_EXEC_20261025/E1_execution_20261026'
sys.path.insert(0,str(ROOT))
from src.f16.io import read_json,write_json,sha256
from src.f16.resource import Guard
SEEDS=(42,43,44)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def guard():
    g=Guard(read_json(HERE/'RESOURCE_STAGE_PLAN.json')['estimate_bytes']);g.check('startup');return g
def require3():
    assert read_json(PREVIOUS/'RESULT_ANALYSIS.json')['status']=='ANALYZED_STEP3_THEN_IMPLEMENT4'
    assert read_json(PREVIOUS/'completion_review/REVIEW.json')['status']=='PASS_SAVED_COMPLETION_SUPPLEMENT'
    assert read_json(PREVIOUS/'completion_review/CHECKPOINT_REVIEW.json')['status']=='PASS'
def save_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Preserve existing '+str(path))
    np.savez_compressed(path,**arrays)
def load_npz(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}
def mark(stage,**kw):write_json(HERE/'progress.json',dict(stage=stage,at=now(),**kw))
def resource(path,g):write_json(path,dict(observations=g.observations,last=g.check()))
def source():return read_json(HERE/'source_manifest.json')['digest']
def verify():
    m=read_json(HERE/'source_manifest.json');assert digest(m['files'])==m['digest']
    for p,h in m['files'].items():assert sha256(Path(p) if Path(p).is_absolute() else ROOT/p)==h,p
    return m['digest']
def completed(folder,c):
    d=read_json(folder/'done.json');assert d['contract']==c
    for p,h in d['outputs'].items():assert sha256(folder/p)==h,p
    return d
def finish(folder,c,**kw):
    files={p.name:sha256(p) for p in folder.iterdir() if p.is_file() and p.name!='done.json'}
    write_json(folder/'done.json',dict(contract=c,outputs=files,**kw));return completed(folder,c)
def old_columns(seed,split,keys):
    folder=OLD/f'populations/seed{seed}/{split}';d=read_json(folder/'done.json');blocks={k:[] for k in keys}
    for name in d['chunks']:
        with np.load(folder/name) as z:
            for k in keys:blocks[k].append(z[k])
    return {k:np.concatenate(v) for k,v in blocks.items()}
def rank(tag,value):return hashlib.sha256(f'{tag}|{value}'.encode()).digest()
def representative(ids,tracks,tag,windowtag):
    unique=sorted(set(tracks),key=lambda t:(rank(tag,str(t)),str(t)))[:128];rows=[]
    for t in unique:
        ix=np.flatnonzero(tracks==t);rows.append(min(ix,key=lambda i:(rank(windowtag,f'{t}|{int(ids[i])}'),int(ids[i]))))
    return np.asarray(rows,np.int32)
