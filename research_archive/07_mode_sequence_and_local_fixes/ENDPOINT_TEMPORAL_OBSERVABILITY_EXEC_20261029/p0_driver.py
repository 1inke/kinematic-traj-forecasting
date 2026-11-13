"""Finite real-data P0 coordinator. Original files are strictly read-only.

No Torch, held GT, old process lookup or model training in this program.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
DESIGN = PROJECT/'docs/ENDPOINT_TEMPORAL_OBSERVABILITY_DESIGN_20261028'
sys.path.insert(0, str(ROOT/'code'))
import numpy as np
from action_geometry import space, enumerate_actions, progress
from build_teacher import masks_for_window, teacher_for_window, write_masks, write_labels
from io_contract import (json_new, json_read, npz_new, load_npz, sha, seal,
                         finish_stage, checked_seal, verify_manifest, code_digest)
from freeze_sources import freeze
from input_audit import prepare
from motion_guard import inspect


def now():
    return datetime.now().astimezone().isoformat()


def emit(stage, **more):
    print(json.dumps(dict(at=now(), stage=stage, **more), ensure_ascii=False), flush=True)


def guard():
    path = PROJECT/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py'
    spec = importlib.util.spec_from_file_location('registered_guard_v3', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    g = mod.Guard(2*1024**3)
    g.check('runtime', force=True)
    return g


def resource(g):
    g.check(force=True)
    return dict(policy='warning_first_remaining_v3', estimate=g.estimate,
                observations=g.observations, sampled_not_complete_lifecycle=True)


def terminal(folder, g):
    verify_manifest(ROOT/'source_manifest.json')
    json_new(folder/'RESOURCE_AND_SOURCE_FINAL.json', dict(
        resource=resource(g), source_lock_sha=sha(ROOT/'source_manifest.json'),
        final_sources_unchanged=True))
    finish_stage(folder, 'RESOURCE_AND_SOURCE_FINAL.json')


def independent_geometry(path, s):
    """All-action reconstruction, independent of production np.interp."""
    p = np.asarray(path, float)
    out = np.repeat(p[None], len(s.params), axis=0)
    legal = np.ones(len(out), bool)
    if s.branch == 'END':
        out = p[None] + np.array([(t+1)/12 for t in range(12)])[None,:,None]*s.params[:,None,:]
    else:
        z = np.vstack((np.zeros((1,2)), p))
        length = np.sqrt(np.sum((z[1:]-z[:-1])**2, axis=1))
        legal[:] = False
        if np.all(length > 0):
            q = np.cumsum(length)/length.sum(); q[-1] = 1.
            knots = np.r_[0.,q]
            if np.all(knots[1:] > knots[:-1]):
                a,b = s.params[:,0:1],s.params[:,1:2]
                w1,w2,w3 = q[3],q[7]-q[3],1-q[7]
                c=(1-a*w1-b*w2)/w3
                legal = np.isfinite(c[:,0]) & (c[:,0] >= .75) & (c[:,0] <= 1.25)
                u = np.concatenate((a*q[:4],a*w1+b*(q[4:8]-w1),a*w1+b*w2+c*(q[8:]-q[7])),1)
                u[:,-1]=1.
                indices=np.clip(np.searchsorted(knots,u,side='right')-1,0,11)
                fraction=(u-knots[indices])/(knots[indices+1]-knots[indices])
                alt=z[indices]+fraction[...,None]*(z[indices+1]-z[indices])
                out[legal]=alt[legal]
                out[:,-1]=p[-1]
    out[s.identity]=p
    legal[s.identity]=True
    return out,legal


def independent_six(past, paths):
    """Independent time-coordinate construction; same registered association."""
    p=np.asarray(paths,float)
    v=np.stack([(p[...,t,:]-(p[...,t-1,:] if t else np.zeros(p.shape[:-2]+(2,))))/.4 for t in range(12)],-2)
    a=np.stack([(v[...,t+1,:]-v[...,t,:])/.4 for t in range(11)],-2)
    j=np.stack([((v[...,t+2,:]-v[...,t+1,:])-(v[...,t+1,:]-v[...,t,:]))/(.4**2) for t in range(10)],-2)
    vn,an,jn=[np.sqrt(np.sum(t*t,axis=-1)) for t in (v,a,j)]
    return np.stack((np.quantile(vn,.95,axis=-1),an.mean(-1),np.quantile(an,.95,axis=-1),
                     jn.mean(-1),np.quantile(jn,.95,axis=-1),
                     np.linalg.norm(v[...,0,:]-(np.asarray(past,float)[-1]-past[-2])/.4,axis=-1)/.4),-1)


def original_module():
    spec=importlib.util.spec_from_file_location('original_endpoint_geometry',DESIGN/'code/action_geometry.py')
    m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)
    return m


def compare_code():
    result={}
    for p in (ROOT/'code').glob('*.py'):
        old=DESIGN/'code'/p.name
        aa,bb=ast.parse(old.read_text(encoding='utf-8')),ast.parse(p.read_text(encoding='utf-8'))
        if p.name=='action_geometry.py':
            aa.body=[n for n in aa.body if not isinstance(n,ast.FunctionDef) or n.name!='enumerate_actions']
            bb.body=[n for n in bb.body if not isinstance(n,ast.FunctionDef) or n.name!='enumerate_actions']
        if ast.dump(aa)!=ast.dump(bb): raise AssertionError('unregistered code difference '+p.name)
        result[p.name]=dict(original_sha=sha(old), execution_sha=sha(p),
                           unchanged_AST_except_enumerate_actions=p.name=='action_geometry.py',
                           other_AST_exact=True)
    return result


def initialize():
    extras=[p for p in DESIGN.rglob('*') if p.suffix in ('.md','.py')]
    extras += [ROOT/'p0_driver.py',ROOT/'protocol.json',ROOT/'README.md',
               PROJECT/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py']
    for p in (PROJECT/'qoder_work/K20_MODE_COVERAGE_AUDIT_20261027/inputs').glob('*'):
        # Bind X for identity provenance, but never open held GT here.
        if p.name.startswith('X_') or p.name=='GT_opt.npz': extras.append(p)
    extras.append(PROJECT/'qoder_work/K20_COLLABORATIVE_REFINEMENT_RESOURCE_ADAPTER_20261027/prepared/split.npz')
    json_new(ROOT/'SOURCE_EQUIVALENCE.json',dict(status='PASS_AST_ONLY_PENDING_REAL_EQUIVALENCE',modules=compare_code()))
    extras.append(ROOT/'SOURCE_EQUIVALENCE.json')
    freeze(PROJECT/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/manifest.json', PROJECT, extras, ROOT/'source_manifest.json')
    emit('SOURCES_LOCKED', count=len(json_read(ROOT/'source_manifest.json')['files']))
    g=guard()
    src=PROJECT/'qoder_work/K20_MODE_COVERAGE_AUDIT_20261027/inputs'
    split=PROJECT/'qoder_work/K20_COLLABORATIVE_REFINEMENT_RESOURCE_ADAPTER_20261027/prepared/split.npz'
    prepare(src/'X_opt.npz',src/'GT_opt.npz',split,ROOT/'prepared')
    terminal(ROOT/'prepared',g)
    x=load_npz(src/'X_opt.npz')
    json_new(ROOT/'IDENTITY_AUDIT.json',dict(status='PASS_REGISTERED_SPLIT_AND_SOURCE_SHA',
        counts=json_read(ROOT/'prepared/INPUT_AUDIT.json'),
        ambiguous_source_time_opt=int(x['ambiguous_source_time'].sum()),
        ambiguous_records_retained=True, held_GT_read=False))
    emit('PREPARED_CANONICAL_TRAIN_VAL')


def preflight():
    checked_seal(ROOT/'prepared')
    verify_manifest(ROOT/'source_manifest.json')
    out=ROOT/'p0_preflight'; out.mkdir(exist_ok=False)
    g=guard(); original=original_module()
    x=load_npz(ROOT/'prepared/X_train.npz'); gt=load_npz(ROOT/'prepared/GT_train.npz')['gt']
    ids=sorted(range(len(gt)),key=lambda i:hashlib.sha256(('endpoint-temporal-preflight-v1|'+str(x['sample_id'][i])).encode()).hexdigest())[:128]
    rows=[]
    for branch in ('END','TIME'):
        s=space(branch); old_s=original.space(branch); begin=time.perf_counter()
        maxerr=0.; total_actions=0; actual=[]
        for index,i in enumerate(ids):
            g.check(); masks=masks_for_window(x['past'][i],x['native'][i],s)
            label=teacher_for_window(x['native'][i],gt[i],s,masks)
            for k,p in enumerate(x['native'][i]):
                paths,legal=enumerate_actions(p,s)
                oldpaths,oldlegal=original.enumerate_actions(p,old_s)
                if not np.array_equal(paths,oldpaths) or not np.array_equal(legal,oldlegal):
                    raise AssertionError(('vectorized/reference NOT EXACT',branch,int(i),k))
                independent,il=independent_geometry(p,s)
                err=float(np.max(np.abs(independent-paths)))
                maxerr=max(maxerr,err)
                if err>2e-12 or not np.array_equal(il,legal): raise AssertionError(('independent geometry',err))
                m=independent_six(x['past'][i],paths); base=independent_six(x['past'][i],p)
                failed=np.c_[~(m<=base*1.01),~np.isfinite(paths).all(axis=(1,2))]
                failed[np.all(paths==p,axis=(1,2))]=False
                if not np.array_equal(failed,masks['failed'][k]): raise AssertionError('literal fail-bit mismatch')
                safe=legal & ~failed.any(1)
                if not np.array_equal(safe,masks['safe'][k]): raise AssertionError('safe mismatch')
                d=np.sqrt(np.sum((paths-gt[i])**2,axis=2)); js=np.maximum(d.mean(1)/.2,d[:,-1]/.4)
                if not np.array_equal(js,label['J_matrix'][k]): raise AssertionError('J mismatch')
                win=min(np.flatnonzero(safe),key=lambda j:(js[j],j!=s.identity,np.sum(s.normalized[j]**2),int(j)))
                values=np.sort(js[safe]); tied=safe & (js==js[win]); near=safe & (js<=js[win]+1e-6)
                if (win!=label['action_id'][k] or js[win]!=label['best_J'][k] or
                    tied.sum()!=label['exact_ties'][k] or near.sum()!=label['near_ties'][k] or
                    (len(values)>1 and values[1]-values[0]!=label['second_gap'][k])):
                    raise AssertionError('teacher/tie mismatch')
                total_actions+=len(paths)
            actual.append(dict(sample_id=x['sample_id'][i],**masks,**label))
            if index%16==15: emit('P0_PREFLIGHT_ACTUAL_128',branch=branch,windows=index+1)
        npz_new(out/(branch+'_actual128.npz'),**{k:np.stack([r[k] for r in actual]) for k in actual[0]})
        read=load_npz(out/(branch+'_actual128.npz'))
        for k in actual[0]:
            if not np.array_equal(read[k],np.stack([r[k] for r in actual]),equal_nan=True) if read[k].dtype.kind not in 'US' else not np.array_equal(read[k],np.stack([r[k] for r in actual])):
                raise AssertionError('saved readback mismatch '+k)
        rows.append(dict(branch=branch,windows=128,actions=total_actions,
                         reference_all_paths_bitwise=True,independent_geometry_max_error=maxerr,
                         literal_all_fail_bits_and_teacher_exact=True,seconds=time.perf_counter()-begin))
    verify_manifest(ROOT/'source_manifest.json')
    json_new(out/'PREFLIGHT.json',dict(status='PASS_REAL128_P0_ONLY_NOT_GPU_TRAINING',branches=rows,
        source_unchanged=True, held_GT_read=False,resource=resource(g),no_Torch=True))
    emit('P0_PREFLIGHT_PASS',branches=rows)


def run_p0():
    pf=json_read(ROOT/'p0_preflight/PREFLIGHT.json')
    if pf['status']!='PASS_REAL128_P0_ONLY_NOT_GPU_TRAINING': raise ValueError('missing actual P0 preflight')
    checked_seal(ROOT/'prepared'); verify_manifest(ROOT/'source_manifest.json')
    g=guard(); result={}
    for branch in ('END','TIME'):
        result[branch]={}
        for population in ('train','val'):
            x=ROOT/f'prepared/X_{population}.npz'; y=ROOT/f'prepared/GT_{population}.npz'
            mask=ROOT/f'P0_{branch}_{population}_masks'; labels=ROOT/f'P0_{branch}_{population}_labels'
            started=time.perf_counter(); emit('P0_MASKS_START',branch=branch,population=population)
            write_masks(x,mask,branch,g.check); terminal(mask,g)
            emit('P0_MASKS_SEALED',branch=branch,population=population,seconds=time.perf_counter()-started)
            write_labels(x,y,mask,labels,g.check); terminal(labels,g)
            result[branch][population]=json_read(labels/'opportunity.json')
            emit('P0_TEACHERS_SEALED',branch=branch,population=population,opportunity=result[branch][population])
    json_new(ROOT/'P0_RESULT.json',dict(status='P0_SAVED_OUTPUTS_REQUIRE_COMPLETION_REVIEW',
        branches=result, held_GT_read=False,no_training=True,resource=resource(g)))
    emit('P0_COMPLETE_REVIEW_REQUIRED',branches=result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=('initialize','preflight','run'))
    args=parser.parse_args()
    try:
        {'initialize':initialize,'preflight':preflight,'run':run_p0}[args.stage]()
    except Exception:
        emit('FAILURE',traceback=traceback.format_exc())
        failed=ROOT/('FAILURE_'+args.stage+'_'+datetime.now().strftime('%H%M%S')+'.json')
        json_new(failed,dict(stage=args.stage,at=now(),traceback=traceback.format_exc()))
        raise
