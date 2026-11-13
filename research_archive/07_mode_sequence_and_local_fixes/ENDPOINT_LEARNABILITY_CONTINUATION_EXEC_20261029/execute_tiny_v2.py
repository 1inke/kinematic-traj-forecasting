"""Explicit authorized V2 selection, dependency bridge and one fixed tiny fit.

No formal training or held GT access. Existing P0/GPU/resume files are read-only.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from datetime import datetime
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
OLD=ROOT.parent/'ENDPOINT_TEMPORAL_OBSERVABILITY_EXEC_20261029'
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
from io_contract import (json_read,json_new,npz_new,load_npz,sha,verify_manifest,
                         checked_seal,code_digest)
from build_teacher import read_targets
from action_geometry import space
from features import transform


def emit(stage,**kwargs):
    print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),stage=stage,**kwargs),ensure_ascii=False),flush=True)


def dynamic(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def select_v2(ids,active):
    if len(ids)!=len(active) or len(np.unique(ids))!=len(ids):raise ValueError('unique aligned IDs required')
    hashes=[hashlib.sha256(('endpoint-temporal-tiny-v1|'+str(s)).encode('utf-8')).hexdigest() for s in ids]
    ranked=sorted(range(len(ids)),key=lambda i:(hashes[i],int(ids[i])))
    first=[i for i in ranked if active[i]][:64]
    second=[i for i in ranked if i not in set(first)][:64]
    if len(first)!=64 or len(second)!=64:raise ValueError('V2 actual insufficient; no replacement')
    chosen=np.array(first+second,dtype=np.int64)
    if len(set(chosen.tolist()))!=128:raise AssertionError('duplicate selected window')
    return chosen,hashes


def selection_tests():
    ids=np.arange(140);active=np.arange(140)<70
    selected,_=select_v2(ids,active)
    if not active[selected[:64]].all() or len(set(selected))!=128:raise AssertionError('mixed fixture')
    allactive,_=select_v2(ids,np.ones(140,bool))
    ranked=sorted(range(140),key=lambda i:(hashlib.sha256(('endpoint-temporal-tiny-v1|'+str(i)).encode()).hexdigest(),i))
    if allactive.tolist()!=ranked[:128]:raise AssertionError('all-active fixture')
    try:select_v2(ids,np.zeros(140,bool))
    except ValueError:pass
    else:raise AssertionError('insufficient fixture not rejected')
    return dict(status='PASS_NEW_SELECTION_FIXTURES',all_active_is_first128=True,duplicates_rejected=True,no_replacement=True)


def initialize():
    files={};old_maps=[json_read(OLD/'source_manifest.json')['files'],json_read(OLD/'learning_source_extension.json')['files']]
    for mapping in old_maps:
        for name,digest in mapping.items():
            p=Path(name).resolve()
            if str(p) in files and files[str(p)]!=digest:raise ValueError('inherited SHA conflict')
            files[str(p)]=digest
    final=json_read(OLD/'FINAL_PRESERVATION.json')
    for name,digest in final['files'].items():
        p=OLD/name
        if str(p) in files and files[str(p)]!=digest:raise ValueError('completed dependency conflict')
        files[str(p)]=digest
    for name,digest in files.items():
        if sha(name)!=digest:raise ValueError('old source/output changed: '+name)
    compare={}
    for p in (ROOT/'code').glob('*.py'):
        old=OLD/'code'/p.name
        if sha(p)!=sha(old) or ast.dump(ast.parse(p.read_text(encoding='utf-8')))!=ast.dump(ast.parse(old.read_text(encoding='utf-8'))):raise AssertionError('core dependency change '+p.name)
        compare[p.name]=dict(old_sha=sha(old),current_sha=sha(p),byte_exact=True,AST_exact=True)
        files[str(p)]=sha(p)
    if len(compare)!=18:raise AssertionError('core module count')
    for directory in ('prepared','P0_END_train_labels','P0_END_val_labels'):checked_seal(OLD/directory)
    x=load_npz(OLD/'prepared/X_train.npz');y=read_targets(OLD/'P0_END_train_labels',x['sample_id'])
    actions=np.concatenate([load_npz(f,('action_id',))['action_id'] for f in sorted((OLD/'P0_END_train_labels').glob('[0-9]*.npz'))])
    active=(actions!=space('END').identity).any(1)
    if not np.array_equal(active,np.any(y!=0,axis=(1,2))):raise AssertionError('action / target active definition')
    chosen,hashes=select_v2(x['sample_id'],active)
    old_first=sorted(np.flatnonzero(active),key=lambda i:hashes[i])[:64]
    if chosen[:64].tolist()!=old_first:raise AssertionError('old first64 compatibility')
    tiny=ROOT/'tiny_v2';tiny.mkdir(exist_ok=False)
    selection=dict(status='SEALED_BEFORE_ANY_TINY_UPDATE',amendment='END_TINY_SAMPLING_AMENDMENT_V2_20261029',
        selector_sha=sha(__file__),protocol_sha=sha(ROOT/'protocol.json'),
        registered_train_observed_sha=sha(OLD/'prepared/X_train.npz'),old_teacher_done_sha=sha(OLD/'P0_END_train_labels/done.json'),
        tests=selection_tests(),old_first64_preserved=True,positive_negative_balanced=False,
        unique_windows=128,unique_tracks=int(len(np.unique(x['track_key'][chosen]))),
        active_candidates=int(np.any(y[chosen]!=0,axis=2).sum()),identity_candidates=int((~np.any(y[chosen]!=0,axis=2)).sum()),
        candidate_denominator=2560,
        rows=[dict(index=int(i),sample_id=int(x['sample_id'][i]),hash=hashes[i],track=str(x['track_key'][i]),group=int(x['group'][i]),
                   sampling_group='first64_active' if row<64 else 'remaining64_train',active=bool(active[i])) for row,i in enumerate(chosen)])
    json_new(tiny/'SELECTION.json',selection)
    npz_new(tiny/'indices.npz',indices=chosen,sample_id=x['sample_id'][chosen])
    review=dict(status='INHERITED_EXACT_DEPENDENCY_MATCH',core_modules=compare,code_digest_current=code_digest(),
        code_digest_old=json_read(OLD/'source_manifest.json')['code_digest'],
        inherited_checks={'P0':'INHERITED_EXACT_DEPENDENCY_MATCH','actual_END_G_H_GPU8_gradient':'INHERITED_EXACT_DEPENDENCY_MATCH',
                          'both_real513_12_all_RNG_resume':'INHERITED_EXACT_DEPENDENCY_MATCH','reader_bad_source_stat_split_output':'INHERITED_EXACT_DEPENDENCY_MATCH'},
        newly_required=['V2 selection and source lock','fixed real8 bridge','one H42 fixed256 tiny','remaining saved quality/CI/benchmark preflight if tiny passes'],
        no_old_evidence_relabelled_as_rerun=True,old_tiny_STOP_unchanged=True)
    if review['code_digest_current']!=review['code_digest_old']:raise AssertionError('exact code digest bridge')
    json_new(ROOT/'SOURCE_REUSE_REVIEW.json',review)
    for p in [ROOT/'SOURCE_REUSE_REVIEW.json',ROOT/'protocol.json',ROOT/'README.md',ROOT/'AMENDMENT_TINY_V2.md',Path(__file__),
              PROJECT/'docs/ENDPOINT_LEARNABILITY_CONTINUATION_DESIGN_20261029.md',tiny/'SELECTION.json',tiny/'indices.npz',
              OLD/'source_manifest.json',OLD/'learning_source_extension.json',OLD/'FINAL_PRESERVATION.json']:
        files[str(p.resolve())]=sha(p)
    json_new(ROOT/'source_manifest.json',dict(status='SOURCE_LOCK_ONLY_NOT_PREFLIGHT_PASS',files=files,
        inherited_base_count=4451,inherited_full_dependency_count=len(files),code_digest=code_digest(),no_old_digest_replaced=True))
    emit('V2_SOURCE_AND_SELECTION_SEALED',sources=len(files),unique_tracks=selection['unique_tracks'],active_candidates=selection['active_candidates'])


def bridge(features,target):
    import torch
    from model import ParameterRegressor,window_mse
    from train import seed_everything
    old_model=dynamic('old_exact_4770_model',OLD/'code/model.py')
    results={}
    for arm in ('G','H'):
        seed_everything(42,True)
        old=old_model.ParameterRegressor().to('cuda:0');new=ParameterRegressor().to('cuda:0')
        new.load_state_dict(old.state_dict())
        z=torch.from_numpy(features[arm][:8]).cuda();t=torch.from_numpy(target[:8]).cuda()
        if not torch.equal(old(z),new(z)) or not (new(z)==0).all():raise AssertionError('identity bridge')
        with torch.no_grad():
            torch.manual_seed(20261029)
            weight=torch.randn_like(old.layers[-1].weight)*.01
            old.layers[-1].weight.copy_(weight);new.layers[-1].weight.copy_(weight)
        lo=old_model.window_mse(old(z),t);ln=window_mse(new(z),t)
        lo.backward();ln.backward()
        if not torch.equal(lo,ln) or not torch.equal(old(z),new(z)):raise AssertionError('GPU bridge output/scalar')
        for a,b in zip(old.parameters(),new.parameters()):
            if not torch.equal(a.grad,b.grad):raise AssertionError('GPU bridge gradients')
        results[arm]=dict(actual_windows=8,old_new_output_scalar_allparametergrad_bitwise=True,optimizer_updates=0)
    return dict(status='PASS_CURRENT_REAL8_EXACT_OLD_NEW_GPU_BRIDGE',arms=results)


def report_metrics(x,gt,prediction,labels,g):
    from action_geometry import space
    from audit_saved import check_saved
    from evaluate import evaluate_population
    checked=check_saved(x,dict(sample_id=x['sample_id'],**prediction),'END',check=g.check)
    yy=labels['target'].astype(float)
    error=((prediction['continuous_param'].astype(float)-yy)**2).mean(-1)
    active=np.any(yy!=0,axis=-1);identity=~active
    result=dict(all_candidate_MSE=float(error.mean()),active_candidate_MSE=float(error[active].mean()),
        identity_candidate_MSE=float(error[identity].mean()) if identity.any() else None,
        active_candidate_denominator=int(active.sum()),identity_candidate_denominator=int(identity.sum()),
        candidate_denominator=int(error.size),guard_rejections=int(prediction['rejection_reason'].any(-1).sum()),
        seven_failure_bit_counts=prediction['rejection_reason'].sum(axis=(0,1)).tolist(),
        nonidentity_executed_fraction=float((prediction['executed_action_id']!=space('END').identity).mean()),
        identity_J_regret=float((labels['identity_J']-labels['best_J']).mean()),ports={})
    saved={};rowsall=[]
    for name,key,ids in [('raw','raw_paths','raw_action_id'),('guarded','guarded_paths','executed_action_id')]:
        jj=np.take_along_axis(labels['J_matrix'],prediction[ids][...,None],axis=2)[...,0]
        regret=jj-labels['best_J']
        if name=='guarded' and (regret < -2e-12).any():raise AssertionError('safe regret')
        sample,rows=evaluate_population(x,gt,prediction[key],g.check)
        # Supplementary full saved metrics are interpreted independently by the
        # completion reviewer; no second model pass is needed.
        result['ports'][name]=dict(exact_teacher_action_hit=float((prediction[ids]==labels['action_id']).mean()),
                                  J_regret=float(regret.mean()),groups=rows)
        saved.update({name+'_'+k:v for k,v in sample.items()})
        rowsall+=rows
    result['scalar_geometry_guard_review']=checked['status']
    return result,saved


def run_tiny():
    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('required actual CUDA unavailable')
    from train import Engine,infer,checkpoint_new
    from inference import predict_paths
    helper=dynamic('old_metrics_only_helper',OLD/'learning_preflight.py')
    verify_manifest(ROOT/'source_manifest.json')
    selection=json_read(ROOT/'tiny_v2/SELECTION.json');indices=load_npz(ROOT/'tiny_v2/indices.npz')['indices']
    x=load_npz(OLD/'prepared/X_train.npz');y=read_targets(OLD/'P0_END_train_labels',x['sample_id']);stats=load_npz(OLD/'prepared/statistics.npz')
    features={arm:transform(x['past'],x['native'],stats,arm) for arm in ('G','H')}
    original_features=dynamic('old_exact_features',OLD/'code/features.py')
    for arm in features:
        if not np.array_equal(features[arm],original_features.transform(x['past'],x['native'],stats,arm)):raise AssertionError('feature/stats bridge')
    runtime=dynamic('registered_v3',PROJECT/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py')
    estimate=4030148608+sum(a.nbytes for a in features.values())+4*128*20*12*2*8+256*1024**2
    g=runtime.Guard(estimate);g.check(force=True)
    inventory=dict(estimate_bytes=estimate,measured_previous_GPU_private_peak=4030148608,
        current_full_feature_bytes=sum(a.nbytes for a in features.values()),saved_tiny_payload_bytes=4*128*20*12*2*8,
        allocator_and_checkpoint_reserve=256*1024**2,policy='warning_first_remaining_v3',not_from_available_memory=True)
    if shutil.disk_usage(ROOT).free<1.3*64*1024**2:raise OSError('Insufficient disk for bounded tiny/checkpoint evidence; do not delete old files')
    json_new(ROOT/'RESOURCE_INVENTORY.json',inventory)
    json_new(ROOT/'REAL8_BRIDGE.json',bridge(features,y))
    subset={k:v[indices] for k,v in x.items()};labels=helper.label_subset(OLD/'P0_END_train_labels',indices)
    if not np.array_equal(labels['target'].astype(np.float32),y[indices]) or not np.array_equal(subset['sample_id'],np.array([r['sample_id'] for r in selection['rows']])):raise AssertionError('tiny selection/teacher ID')
    gt_all=load_npz(OLD/'prepared/GT_train.npz')
    if not np.array_equal(gt_all['sample_id'],x['sample_id']):raise AssertionError('trainGT ID')
    target=y[indices]
    contract=dict(amendment='END_TINY_SAMPLING_AMENDMENT_V2_20261029',source_lock_sha=sha(ROOT/'source_manifest.json'),
        selection_sha=sha(ROOT/'tiny_v2/SELECTION.json'),old_teacher_done_sha=sha(OLD/'P0_END_train_labels/done.json'),
        stats_sha=sha(OLD/'prepared/statistics.npz'),seed=42,arm='H',not_formal=True)
    engine=Engine(features['H'][indices],target,contract,seed=42,device='cuda:0',epochs=64)
    tiny=ROOT/'tiny_v2';checkpoint_new(tiny/'initial_checkpoint.pt',engine.snapshot())
    before=infer(engine.model,features['H'][indices],'cuda:0',g.check)
    bpred=predict_paths(subset['past'],subset['native'],before,'END',g.check)
    bm,bsaved=report_metrics(subset,gt_all['gt'][indices],bpred,labels,g)
    npz_new(tiny/'initial.npz',sample_id=subset['sample_id'],**bpred,**bsaved)
    begin=time.perf_counter()
    for i in range(256):
        g.check();engine.step()
        if i%64==63:emit('TINY_V2_FIXED_UPDATES',updates=i+1)
    torch.cuda.synchronize();train_seconds=time.perf_counter()-begin
    after=infer(engine.model,features['H'][indices],'cuda:0',g.check)
    apred=predict_paths(subset['past'],subset['native'],after,'END',g.check)
    am,asaved=report_metrics(subset,gt_all['gt'][indices],apred,labels,g)
    npz_new(tiny/'final.npz',sample_id=subset['sample_id'],**apred,**asaved)
    checkpoint_new(tiny/'checkpoint.pt',engine.snapshot());json_new(tiny/'history.json',engine.history)
    active=np.any(target!=0,axis=-1)
    initial=float(((before.astype(float)-target)**2).mean(-1)[active].mean())
    final=float(((after.astype(float)-target)**2).mean(-1)[active].mean())
    if initial<=0:raise AssertionError('unexpected zero active initial MSE; no epsilon')
    drop=1-final/initial;status='PASS_TINY_FIT_V2' if drop>=.9 else 'STOP_TINY_FIT_V2'
    json_new(tiny/'METRICS.json',dict(status=status,updates=256,model_epochs=64,seed=42,arm='H',train_seconds=train_seconds,
        gate_initial_active_MSE=initial,gate_final_active_MSE=final,active_MSE_relative_drop=drop,
        required_drop=.9,initial=bm,final=am,not_generalization=True))
    verify_manifest(ROOT/'source_manifest.json');g.check(force=True)
    json_new(ROOT/'RESOURCE.json',dict(policy='warning_first_remaining_v3',estimate_bytes=estimate,observations=g.observations))
    json_new(ROOT/'TINY_EXECUTION_RESULT.json',dict(status=status,tiny_metrics_sha=sha(tiny/'METRICS.json'),
        core_evidence='INHERITED_EXACT_DEPENDENCY_MATCH',new_bridge='PASS_REAL8',new_tiny_updates=256,
        formal_models=0,held_GT_opened=False,source_final_unchanged=True,
        next='FINAL_SAVED_PIPELINE_BENCHMARK_PREFLIGHT_REQUIRED' if drop>=.9 else 'STOP_NO_FORMAL_MODELS_NO_RETRY'))
    emit('TINY_V2_COMPLETED',status=status,relative_drop=drop)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('initialize','tiny'));args=p.parse_args()
    try:{'initialize':initialize,'tiny':run_tiny}[args.stage]()
    except Exception:
        failed=ROOT/('FAILURE_'+args.stage+'_'+datetime.now().strftime('%H%M%S')+'.json')
        json_new(failed,dict(stage=args.stage,traceback=traceback.format_exc()));raise
