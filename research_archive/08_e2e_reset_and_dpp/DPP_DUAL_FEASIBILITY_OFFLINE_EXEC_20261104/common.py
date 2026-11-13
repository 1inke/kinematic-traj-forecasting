from pathlib import Path
import sys,os,json,csv,hashlib
from datetime import datetime,timezone,timedelta
sys.dont_write_bytecode=True
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
BASE=ROOT/'qoder_work/E2E_MOTION_RESEARCH_RESET_EXEC_20261102'
V2=ROOT/'qoder_work/E2E_DGEO_V2_DECOUPLED_EXEC_20261103'
DESIGN=Path('<LOCAL_DOWNLOADS>/DPP_双重可行性质量评分_改进实验设计报告.md')
MAP=ROOT/'qoder_work/decoder_geometry_complementarity_exec_20261024/q1_map_inventory/MAP_INVENTORY.json'
MODELS=('fresh-B0','DK-CenterDiff','DK-ManeuverDiff','DK-KinTaylorDiff','DK-FullDiff')
DT=.4
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def seal(paths):return {str(Path(p).resolve()):sha(p) for p in paths}
def verify(files):
    for p,d in files.items():
        if sha(p)!=d:raise ValueError('SHA_MISMATCH '+p)
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def table(p,rows):
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def population(name,split):
    return BASE/'populations'/split if name=='fresh-B0' else V2/'populations'/name/split
def authorize(execute,stage):
    control=read(RUN/'execution_control.json')
    if not execute:
        print('DRY_RUN_ONLY: '+stage+'; no real-data reading/calibration/training/worker started.')
        return False
    if control.get('status')!='RUN_APPROVED_BY_NEW_HUMAN_REQUEST' or control.get('allow_experiment_start') is not True:
        raise PermissionError('LATEST_HUMAN_SAYS_DO_NOT_START; need a later explicit user start request')
    return True
def checked_population(name,split):
    import numpy as np
    folder=population(name,split)
    receipt=read(folder/'done.json');verify(receipt['files'])
    with np.load(folder/'predictions.npz',allow_pickle=False) as f:p={k:f[k] for k in f.files}
    with np.load(folder/'window_metrics.npz',allow_pickle=False) as f:m={k:f[k] for k in f.files}
    with np.load(BASE/'C0'/f'{split}.npz',allow_pickle=False) as f:
        if not np.array_equal(p['ids'],f['ids']) or not np.array_equal(p['future'],f['future']) or not np.array_equal(p['groups'],f['group']):
            raise ValueError('POPULATION_ID_GT_GROUP_CONTRACT')
        if not np.array_equal(p['past'],f['past'][:,0]):raise ValueError('PAST_CONTRACT')
    if p['paths'].shape!=(len(p['ids']),20,12,2) or len(set(p['ids'].tolist()))!=len(p['ids']):raise ValueError('K20_OR_UNIQUE_ID')
    return p,m
def guard_module():
    import importlib.util
    p=ROOT/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py'
    s=importlib.util.spec_from_file_location('dpp_guard_v3',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
