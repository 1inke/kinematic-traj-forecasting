from pathlib import Path
import sys,random,hashlib,json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
E0=HERE.parent/'E0_execution_20261025';E0FINAL=HERE.parent/'E0_audit_resume_20261026'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(E0))
from src.f16.io import read_json,write_json,sha256
from src.f16.resource import Guard
from src.f16.checkpoint import capture_rng,restore_rng,save_checkpoint,load_checkpoint
import common as e0
import space as e0space
SEEDS=(42,43,44);ARMS=('I','S')
ESTIMATE=6*1024**3
def require_e0():
    result=read_json(E0FINAL/'RESULT_ANALYSIS.json')
    if result['status']!='ANALYZED_E0_PASS_ALLOW_E1_OWN_PREFLIGHT':raise ValueError('Full E0 gate is mandatory')
    if read_json(E0FINAL/'completion_review/REVIEW.json')['status']!='PASS_COMPLETION_SOURCE_OUTPUT_AND_GATES':raise ValueError('E0 completion review missing')
    return result
def seed_all(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def stage(name,**kw):write_json(HERE/'runtime_state.json',dict(status='RUNNING',stage=name,at=e0.now(),**kw))
