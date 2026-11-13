from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from shared import *
HERE=Path(__file__).resolve().parent;D0=SERIES/'D0_execution_20261027';ARMS=('I','S','D');CACHE=OLD/'prepared_v1'
NAMES=('speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95')
def guard(kind='MLP'):
    g=Guard(read_json(HERE/'RESOURCE_STAGE_PLAN.json')['estimates_bytes'][kind]);g.check('startup');return g
def mark(stage,**kw):write_json(HERE/'progress.json',dict(stage=stage,at=now(),**kw))
def source0():return source(HERE)
def resource(path,g):write_json(path,dict(last=g.check(),observations=g.observations))
def contract(seed,arm,source_digest,count=4096,epochs=8):
    ready=read_json(CACHE/'ready.json');files=ready['outputs']
    return dict(source=source_digest,seed=seed,arm=arm,protocol=sha256(HERE/'protocol.json'),split=files['identity.npz'],statistics=files[f'stats_seed{seed}.npz'],initial=files[f'initial_seed{seed}.pt'],orders=files[f'orders_seed{seed}.npy'],features=files[f'features_seed{seed}.npy'],labels=files[f'labels_seed{seed}.npz'],candidate_pool=sha256(OLD.parent/f'E0_execution_20261025/seed{seed}/input_pools.npz'),population_windows=count,epochs=epochs,steps=epochs*((count+31)//32),parameters=11488,loss_version='registered-original-I-S-or-windowgate-editrank-D-v1')
def verify_prepared():
    m=read_json(CACHE/'ready.json')
    for name,h in m['outputs'].items():assert sha256(CACHE/name)==h,name
    assert read_json(D0/'RESULT_ANALYSIS.json')['status']=='ANALYZED_D0_FIXED_D1_ALLOWED_OWN_PREFLIGHT'
    return m
def build_current_source():
    if (HERE/'PRODUCTION_SOURCE_LOCK.json').exists():raise ValueError('Source already production locked')
    files=read_json(D0/'source_manifest.json')['files'].copy()
    for n,h in read_json(D0/'PRESERVATION.json')['outputs'].items():add_source(files,D0/n,h)
    for n in ('RESULT_ANALYSIS.md','RESULT_ANALYSIS.json'):add_source(files,D0/n)
    for p in (D0/'completion_review').glob('*'):
        if p.is_file():add_source(files,p)
    for p in HERE.iterdir():
        if p.is_file() and (p.suffix=='.py' or p.name in ('protocol.json','README.md','RESOURCE_STAGE_PLAN.json','PREFLIGHT_FIXTURE_REPAIR.md')):add_source(files,p)
    for folder in HERE.glob('failure_*'):
        if folder.is_dir():
            for p in folder.rglob('*'):
                if p.is_file():add_source(files,p)
    previous=HERE/'source_manifest.json'
    if previous.exists():
        old=read_json(previous);f=HERE/'candidate_provenance';f.mkdir(exist_ok=True);write_json(f/(old['digest']+'.json'),old)
    m=dict(files=files,digest=digest(files),parent=source(D0),strict_inheritance_no_old_digest_changes=True);write_json(previous,m);verify_manifest(HERE);return m
class OptData:
    def __init__(self,seed,count=4096):
        self.seed=seed;self.count=count;self.features=np.load(CACHE/f'features_seed{seed}.npy',mmap_mode='r');lab=load_npz(CACHE/f'labels_seed{seed}.npz')
        self.eligible=lab['eligible'][:count];self.target=lab['target'][:count];self.B=lab['B'][:count];self.ids=lab['sample_id'][:count];st=load_npz(CACHE/f'stats_seed{seed}.npz');self.mean=st['mean64'];self.std=st['std64'];self.orders=np.load(CACHE/f'orders_seed{seed}.npy')
        assert self.count<=4096 and np.array_equal(self.target[:,0],~self.B.any(1)) and not self.B[:,0].any() and not (self.B&~self.eligible).any()
    def batch(self,rows,device):
        import torch
        ix=np.asarray(rows);return tuple(torch.from_numpy(v).to(device) for v in (np.array(self.features[ix],copy=True),self.eligible[ix].copy(),self.target[ix].copy()))
    def close(self):self.features._mmap.close()
