"""New repair utilities. No import of the old experiment main/train entry."""
from __future__ import annotations
import os,sys,json,csv,hashlib,importlib.util,types,ast
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('F16_ROOT',HERE.parents[1])).resolve()
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from f16.io import sha256,read_json,write_json
from f16.resource import Guard

E0=ROOT/'qoder_work/E0_CALV11_ORACLE_OPPORTUNITY_20261020'
E1=ROOT/'qoder_work/E1_STUDENT_DISTILL_SCREENING_20261020'
MODEL_DIR=ROOT/'HYBRID_NATIVE_KIN_UNION_20261018/kdec'
SEEDS=(42,43,44)
RUNS=[(42,a) for a in 'ABCDE']+[(s,a) for s in [43,44] for a in 'ABE']
PATHS={
 'cache':ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz',
 'labels':ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz',
 'encoder':ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz',
 'splits':ROOT/'GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz',
 'metadata':ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz'}

def array_sha(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(a.tobytes());h.update(str(a.shape).encode());h.update(str(a.dtype).encode());return h.hexdigest()

def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['empty']);w.writeheader();w.writerows(rows)

def kin_path(split,seed):
    if split=='FIT':return ROOT/f'UNION_SELECTOR_K40_TO_K20_20261018/results/cache_pred/kin_FIT_seed{seed}.npy'
    if split=='inner':return ROOT/f'COMPLEMENTARY_POOL_VALIDATION_20261018/artifacts/v1/kin_inner_seed{seed}.npy'
    raise ValueError(split)

def checkpoint(seed,arm):return E1/f'student_{arm}_seed{seed}.pt'

def old_result(seed):return read_json(E1/('RESULT.json' if seed==42 else f'RESULT_seed{seed}.json'))

def current_sources():
    paths=list(PATHS.values())+[E0/'RESULT.json',E0/'run_e0.py',E1/'run_e1.py',E1/'DESIGN.md',E1/'FATAL.txt',ROOT/'qoder_work/互补蒸馏_E0E1_实验分析报告_20261020.md',ROOT/'qoder_work/E0E1_REVIEW_20261020/IMPLEMENTATION_GUIDE_20261022.md',ROOT/'qoder_work/E0E1_REVIEW_20261020/SAVED_REVIEW.json',ROOT/'qoder_work/E0E1_REVIEW_20261020/ANALYSIS.md',ROOT/'qoder_work/v1_1/SPLIT_V1_1.json',ROOT/'qoder_work/v1_1/cal_v11_index.npy',ROOT/'qoder_work/v1_1/cal_v11_windows.csv',ROOT/'TRACK_OVERLAP_AUDIT_20261020/OVERLAP_REPORT.json',ROOT/'config/data_manifest.v1.json']
    paths += [E1/('RESULT.json' if seed==42 else f'RESULT_seed{seed}.json') for seed in SEEDS]
    paths += [checkpoint(s,a) for s,a in RUNS]
    for base in [MODEL_DIR,ROOT/'CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018/k0',ROOT/'KINEMATIC_DECODER_SPEED_CLAMP_20261018/kdec',ROOT/'UNION_SELECTOR_K40_TO_K20_20261018/kdec']:
        paths += list(base.glob('*.py'))
    paths += [ROOT/'COMPLEMENTARY_POOL_VALIDATION_20261018/run_v1.py',ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/run_formal_regime_selection.py',ROOT/'GENERATION_NATIVE_DECODER_SERIES/run_native.py']
    for seed in SEEDS:
        paths += [kin_path(split,seed) for split in ['FIT','inner']]
        paths += [Path(read_json(E0/'RESULT.json')['results'][str(seed)]['kin_src']),ROOT/f'KINEMATIC_DECODER_SPEED_CLAMP_20261018/results/short/KIN_seed{seed}.pt',ROOT/f'KINEMATIC_DECODER_SPEED_CLAMP_20261018/results/short/KIN_seed{seed}_RESULT.json']
        paths += list(E0.glob(f'*seed{seed}.npy'))
    for stage in ['h3','v2']:
        base=ROOT/f'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/EXP_156B_FOLD_LOCAL_CURRENT_DECODER/results/eth/{stage}/f16_temporal_social_cross/seed_1/val_scene_-1'
        paths += [base/'split_indices.npz',base/'best_selected.pth']
    paths += list((ROOT/'src/f16').glob('*.py'))+[ROOT/'audit/verify.py']
    paths += list(HERE.glob('*.py'))+[HERE/'protocol.json']
    return sorted(set(p.resolve() for p in paths if p.exists()))

def capture(paths):return {p.relative_to(ROOT).as_posix():{'absolute_path':str(p),'size':p.stat().st_size,'sha256':sha256(p)} for p in paths}

def verify_sources():
    manifest=read_json(HERE/'source_manifest_before.json')['files'];bad=[]
    for key,v in manifest.items():
        if not (ROOT/key).exists() or sha256(ROOT/key)!=v['sha256']:bad.append(key)
    if bad:raise RuntimeError('SOURCE_CHANGED: '+str(bad))
    return manifest

def data_load():
    # Explicit IDs, not a guess based on equal array lengths.
    with np.load(PATHS['cache'],allow_pickle=False) as z:
        ids=z['sample_id'].copy();native=z['candidate'][...,:24].reshape(-1,20,12,2).astype(np.float64);groups=z['change_group'].copy()
    with np.load(PATHS['labels'],allow_pickle=False) as z:
        if not np.array_equal(z['sample_id'],ids):raise ValueError('GT_ID_MISMATCH')
        gt=z['future'].astype(np.float64)
    with np.load(PATHS['encoder'],allow_pickle=False) as z:
        if not np.array_equal(z['sample_id'],ids):raise ValueError('PAST_ID_MISMATCH')
        past=z['past'].astype(np.float64)
    if not np.isfinite(past).all() or not np.allclose(past[:,-1],0,atol=1e-7,rtol=0):raise ValueError('FOCAL_LOCAL_PAST_ORIGIN')
    with np.load(PATHS['splits'],allow_pickle=False) as z:
        if not np.array_equal(z['sample_id'],ids[:25912]):raise ValueError('SPLIT_REGISTRY_MISMATCH')
        split={'FIT':z['grouped_train_index'].copy(),'inner':z['inner_cal_index'].copy()}
        for name,key in [('FIT','grouped_train_ids'),('inner','inner_cal_ids')]:
            if not np.array_equal(ids[split[name]],z[key]):raise ValueError('SPLIT_EXPLICIT_ID_MISMATCH')
    with np.load(PATHS['metadata'],allow_pickle=False) as z:
        lookup={int(s):i for i,s in enumerate(z['sample_id'])};loc=[lookup[int(s)] for s in ids];md={k:z[k][loc].copy() for k in z.files}
    track=np.array([str(s)+'::'+str(int(p)) for s,p in zip(md['sequence'],md['pedestrian_id'])])
    if len(ids)!=28791 or len(np.unique(ids))!=len(ids) or len(split['FIT'])!=18909 or len(split['inner'])!=1856 or np.intersect1d(track[split['FIT']],track[split['inner']]).size:raise ValueError('IDENTITY_CONTRACT')
    return dict(ids=ids,native=native,groups=groups,gt=gt,past=past,metadata=md,track=track,split=split)

def model_module(directory=MODEL_DIR,alias='_evidence_repair_original_kdec'):
    # Only inspected models/integrate modules; no old runner, train or __init__.
    import importlib
    for fname in ['models.py','integrate.py']:
        tree=ast.parse((directory/fname).read_text(encoding='utf-8-sig'))
        allowed=(ast.Import,ast.ImportFrom,ast.Assign,ast.AnnAssign,ast.FunctionDef,ast.ClassDef)
        for i,node in enumerate(tree.body):
            if i==0 and isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant):continue
            if not isinstance(node,allowed):raise ValueError('UNREVIEWED_TOPLEVEL '+fname)
    package=types.ModuleType(alias);package.__path__=[str(directory)];sys.modules[alias]=package
    spec=importlib.util.spec_from_file_location(alias+'.models',directory/'models.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);return mod

def load_student(seed,arm,device):
    import torch
    m=model_module();blob=torch.load(checkpoint(seed,arm),map_location='cpu',weights_only=True)
    if blob.get('seed')!=seed or blob.get('arm')!=arm:raise ValueError('CHECKPOINT_ARM_SEED')
    net=m.build('BASE');net.load_state_dict(blob['state_dict'],strict=True);net.to(device).eval();net.requires_grad_(False)
    if sum(p.numel() for p in net.parameters())!=57432:raise ValueError('STUDENT_PARAM_COUNT')
    return net

def frozen_state(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}

def assert_frozen(net,state):
    import torch
    if set(net.state_dict())!=set(state) or any(not torch.equal(v.detach().cpu(),state[k]) for k,v in net.state_dict().items()):raise RuntimeError('PARAMETER_OR_BUFFER_CHANGED')
    if any(p.requires_grad or p.grad is not None for p in net.parameters()):raise RuntimeError('FROZEN_HEAD_GRADIENT')

def predict(net,past,device,batch=256):
    import torch
    answer=[]
    with torch.inference_mode():
        for i in range(0,len(past),batch):answer.append(net(torch.from_numpy(past[i:i+batch].astype(np.float32)).to(device))['paths'].cpu().numpy())
    out=np.concatenate(answer)
    if out.shape!=(len(past),20,12,2) or not np.isfinite(out).all():raise ValueError('INVALID_PREDICTION')
    return out

def reconcile(metrics,old,tolerance=2e-6):
    rows=[]
    keys={'N':'n','good_count':'good','native_good_count':'native_good','coverage':'coverage','rescue':'rescue','lost':'lost','minADE':'ADE','minFDE':'FDE','motion_gate':'motion_pass'}
    for group in ['overall','stable','transition','change']:
        for oldkey,key in keys.items():
            a=old[group][oldkey];b=metrics[group][key]
            if oldkey=='motion_gate':b='PASS' if b else 'FAIL'
            discrete=oldkey in ['N','good_count','native_good_count','rescue','lost','motion_gate'];diff=0 if a==b else (abs(a-b) if isinstance(a,(int,float)) and isinstance(b,(int,float)) else None);ok=a==b if discrete else diff is not None and diff<=tolerance
            rows.append(dict(group=group,field=oldkey,old=a,replay=b,abs_diff=diff,pass_=bool(ok),tolerance=0 if discrete else tolerance))
    return dict(status='RECONCILED' if all(r['pass_'] for r in rows) else 'REPLAY_NOT_RECONCILED',fields=rows)

def progress(stage,**kwargs):write_json(HERE/'runtime_state.json',dict(stage=stage,**kwargs))
