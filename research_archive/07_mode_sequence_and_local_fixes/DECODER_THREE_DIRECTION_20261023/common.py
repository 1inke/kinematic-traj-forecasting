"""Read-only inputs and common evaluation for the three NEW directions.

Importing this module never loads arrays, trains, creates a worker, or writes.
Only FIT/inner archives are accepted. Source hashes do not imply model replay.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from f16.io import sha256, read_json, write_json
from f16.eval import errors, evaluate, motion

OLD = ROOT / 'qoder_work/F16_CONTEXT_COMPLEMENTARITY_DESIGN_20261022/execution_20261022'
TAIL = ROOT / 'qoder_work/F16_CONTEXT_TAIL_RETENTION_DESIGN_20261023/T1_execution_20261023'
DESIGN = ROOT / 'docs/解码端三方向单因素实验设计合同_20261023.md'
ENCODER = ROOT / 'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz'
ENCODER_SHA = 'f6c232e4b05e4e8db2a01c7fd0880beecd835b59352b81ba08132c8b610e87f0'
CHECKPOINT_SHA = 'f245c1ba7277b9b843c891f7f3902a66f3c4005108fe1f729a74be3c7167bc43'
SIZES = {'FIT': 18909, 'inner': 1856}
SEEDS = (42, 43, 44)

def owned_output(path,direction):
    target=Path(path).resolve();allowed=(HERE/direction).resolve()
    if target==allowed or not target.is_relative_to(allowed):
        raise ValueError('All new artifacts must stay in a NEW child of '+str(allowed))
    return target

def require_split(split):
    if split not in SIZES:
        raise ValueError('PROTOCOL_VIOLATION: FIT/inner only')

def digest_dict(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def source_inputs():
    parent = read_json(TAIL / 'source_manifest.json')
    matching = [Path(v['path']) for v in parent['files'].values() if v['sha256'] == CHECKPOINT_SHA]
    if not matching:
        raise ValueError('Required upstream checkpoint not registered')
    return [DESIGN, ROOT/'PROTOCOL.md', ROOT/'DESIGN_AB.md', ENCODER, matching[0], OLD / 'split_ids.npz', OLD / 'statistics.npz'] + [
        OLD / f'native_{s}.npz' for s in SIZES
    ] + [OLD / f'runs/A_seed{k}/predictions_{s}.npz' for k in SEEDS for s in SIZES] + [
        OLD / f'initial_states/seed{k}.pt' for k in SEEDS
    ] + [OLD / f'batch_orders/seed{k}.npz' for k in SEEDS]

def lock_sources(output):
    """Explicit opt-in preparation; inherit old digests without overwriting them."""
    output = Path(output)
    if output.exists():
        verify_sources(output)
        return read_json(output)
    parent = read_json(TAIL / 'source_manifest.json')
    files = {k: dict(v) for k, v in parent['files'].items()}
    for v in files.values():
        if sha256(v['path']) != v['sha256']:
            raise ValueError('INHERITED_SOURCE_CHANGED: ' + v['path'])
    paths = source_inputs() + list(HERE.rglob('*.py')) + list(HERE.rglob('*.md'))
    paths += list(HERE.rglob('protocol.json'))
    paths += list((ROOT/'src/f16').glob('*.py')) + [ROOT/'audit/verify.py']
    for p in paths:
        key = p.relative_to(ROOT).as_posix()
        val = {'path': str(p), 'sha256': sha256(p)}
        if key in files and files[key]['sha256'] != val['sha256']:
            raise ValueError('Cannot overwrite inherited digest: ' + key)
        files.setdefault(key, val)
    if sha256(ENCODER) != ENCODER_SHA:
        raise ValueError('ENCODER_SHA_MISMATCH')
    result = dict(parent_digest=parent['digest'], inherited_count=len(parent['files']),
                  files=files, digest=digest_dict({k:v['sha256'] for k,v in files.items()}))
    write_json(output, result)
    return result

def verify_sources(path):
    m = read_json(path)
    if m['digest'] != digest_dict({k:v['sha256'] for k,v in m['files'].items()}):
        raise ValueError('SOURCE_MANIFEST_DIGEST_MISMATCH')
    for v in m['files'].values():
        if sha256(v['path']) != v['sha256']:
            raise ValueError('SOURCE_CHANGED: ' + v['path'])
    return m['digest']

def load_data():
    if sha256(ENCODER) != ENCODER_SHA:
        raise ValueError('ENCODER_SHA_MISMATCH')
    with np.load(OLD / 'split_ids.npz', allow_pickle=False) as z:
        registry = {s:z[s].copy() for s in SIZES}
    # The encoder archive contains other rows. Select permitted IDs BEFORE reading
    # any GT/native archives; no outer/CAL file or label is opened.
    with np.load(ENCODER, allow_pickle=False) as z:
        lookup = {int(v):i for i,v in enumerate(z['sample_id'])}
        out = {}
        for s, ids in registry.items():
            require_split(s)
            ix = np.array([lookup[int(i)] for i in ids])
            past = z['past'][ix].copy()
            tok = z['tokens'][ix].copy()
            x = np.concatenate((past.reshape(-1,16), (past[:,-1]-past[:,-2])/np.float32(.4),
                                tok[:,-1], tok.mean(1)), axis=1)
            with np.load(OLD / f'native_{s}.npz', allow_pickle=False) as n:
                d = {k:n[k].copy() for k in n.files if k != 'paths'}
            assert np.array_equal(d['sample_id'], ids)
            assert len(ids) == SIZES[s] and len(np.unique(ids)) == len(ids)
            assert x.shape == (len(ids),210) and x.dtype == np.float32
            d.update(x=x, past=past, ids=ids)
            for k in ('x','past','gt','native'):
                assert np.isfinite(d[k]).all()
            out[s] = d
    assert not np.intersect1d(out['FIT']['ids'], out['inner']['ids']).size
    assert not np.intersect1d(out['FIT']['track'], out['inner']['track']).size
    assert int((out['FIT']['groups']==2).sum())==4725
    assert int((out['inner']['groups']==2).sum())==497
    assert int(errors(out['inner']['native'],out['inner']['gt'])['coverage'][out['inner']['groups']==2].sum())==166
    assert len(np.unique(out['inner']['track'][out['inner']['groups']==2]))==35
    return out

def old_paths(seed, split):
    require_split(split)
    if seed not in SEEDS:
        raise ValueError(seed)
    with np.load(OLD / f'runs/A_seed{seed}/predictions_{split}.npz', allow_pickle=False) as z:
        return z['sample_id'].copy(), z['paths'].copy()

def fit_stats(x):
    mean=x.astype(np.float64).mean(0); std=x.astype(np.float64).std(0, ddof=0)
    std[std < 1e-6]=1.
    return mean, std

def paired_ci(values, tracks, *, draws=20000, seed=20261023, resample_seeds=True):
    """values: paired delta (model seeds,N) or (N,); shared track draws.
    Each draw is window-weighted, then averages drawn paired model seeds.
    """
    v=np.atleast_2d(values).astype(float)
    if v.shape[1] != len(tracks) or not np.isfinite(v).all():
        raise ValueError('CI identity/finite contract')
    _, inv=np.unique(tracks, return_inverse=True); nt=int(inv.max())+1
    if nt < 20:
        return dict(status='INSUFFICIENT_TRACKS', tracks=nt, point=float(v.mean()))
    counts=np.bincount(inv); sums=np.array([np.bincount(inv, weights=row) for row in v])
    rng=np.random.default_rng(seed); result=np.empty(draws)
    for i in range(draws):
        sd=rng.integers(len(v),size=len(v)) if resample_seeds else np.arange(len(v))
        td=rng.integers(nt,size=nt)
        result[i]=(sums[sd][:,td].sum(1)/counts[td].sum()).mean()
    return dict(status='OK', tracks=nt, valid=draws, point=float(v.mean()),
                CI95=np.quantile(result,[.025,.975]).tolist(),
                resample_seeds=resample_seeds, seed=seed)

def report_paths(path, d, output):
    rows,e=evaluate(path,d['gt'],d['groups'],d['native'],d['ids'],d['track'])
    p=np.asarray(path,np.float64); velocity=np.diff(p,axis=2,prepend=np.zeros_like(p[:,:,:1]))/.4
    gt_v=np.diff(d['gt'],axis=1,prepend=np.zeros_like(d['gt'][:,:1]))/.4
    aa=np.diff(velocity,axis=2)/.4; ga=np.diff(gt_v,axis=1)/.4
    row=np.arange(len(p)); selected=aa[row,e['J_winner']]
    join=np.linalg.norm(velocity[:,:,0]-(d['past'][:,-1]-d['past'][:,-2])[:,None]/.4,axis=-1)
    for i,g in enumerate(('overall','stable','transition','change')):
        mask=np.ones(len(p),bool) if i==0 else d['groups']==i-1
        if mask.any():
            rows[g]['Jslot_GT_accel_error']=float(np.linalg.norm(selected[mask]-ga[mask],axis=-1).mean())
            rows[g]['velocity_join']=float(join[mask].mean())
            rows[g]['G_base']=bool(rows[g]['coverage']>=.99*rows[g]['native_good']/rows[g]['n'] and
                rows[g]['ADE']<=1.01*errors(d['native'][mask],d['gt'][mask])['ADE'].min(1).mean() and
                rows[g]['FDE']<=1.01*errors(d['native'][mask],d['gt'][mask])['FDE'].min(1).mean() and rows[g]['motion_pass'])
    write_json(output,rows)
    return rows,e

def frozen_old_subsets(paths,d,old):
    """Populations fixed using old A/native, never redefined by the new arm."""
    e=errors(paths,d['gt']);a=errors(old,d['gt']);n=errors(d['native'],d['gt']);out={}
    ac=a['coverage'];nc=n['coverage']
    fixed={'retained':ac&nc,'rescued':ac&~nc,'lost':~ac&nc,'neither':~ac&~nc}
    for group,num in [('overall',None),('stable',0),('transition',1),('change',2)]:
        gm=np.ones(len(paths),bool) if num is None else d['groups']==num
        out[group]={}
        for label,base in fixed.items():
            take=gm&base;count=int(take.sum())
            row=dict(n=count,tracks=len(np.unique(d['track'][take])),IDs=d['ids'][take],
                new_good=int(e['coverage'][take].sum()))
            if count:
                for key in ('ADE','FDE','J'):
                    delta=e[key].min(1)-a[key].min(1)
                    row[key+'_delta_mean']=float(delta[take].mean())
                    row[key+'_whole_group_signed_contribution']=float(delta[take].sum()/gm.sum())
            out[group][label]=row
    return out
