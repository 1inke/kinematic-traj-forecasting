"""Independent CPU review of fixed tiny saved outputs; no forward/backward."""
import csv
import hashlib
import importlib.util
from pathlib import Path
import sys
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'ENDPOINT_TEMPORAL_OBSERVABILITY_EXEC_20261029'
sys.path.insert(0,str(ROOT/'code'))
from io_contract import load_npz,json_read,json_new,npz_new,sha,verify_manifest
from audit_saved import check_saved
from features import transform


def digest(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes());return h.hexdigest()


def main():
    spec=importlib.util.spec_from_file_location('tiny_CPU_review_guard',ROOT.parents[1]/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py')
    rg=importlib.util.module_from_spec(spec);spec.loader.exec_module(rg)
    # Conservative bounded review inventory: previous measured GPU-preflight
    # private peak plus six 128-window path payloads; no available-memory tuning.
    guard=rg.Guard(4030148608+6*128*20*12*2*8);guard.check(force=True)
    verify_manifest(ROOT/'source_manifest.json');verify_manifest(ROOT/'namespace_adapter_source_extension.json')
    result=json_read(ROOT/'TINY_EXECUTION_RESULT.json');m=json_read(ROOT/'tiny_v2/METRICS.json')
    if result['tiny_metrics_sha']!=sha(ROOT/'tiny_v2/METRICS.json'):raise AssertionError('result hash')
    out=ROOT/'completion_review';out.mkdir(exist_ok=False)
    json_new(out/'SOURCE_EXTENSION.json',dict(files={str(Path(__file__).resolve()):sha(__file__),
        str(ROOT/'source_manifest.json'):sha(ROOT/'source_manifest.json'),str(ROOT/'namespace_adapter_source_extension.json'):sha(ROOT/'namespace_adapter_source_extension.json')},
        status='EXPLICIT_CPU_SAVED_REVIEW_EXTENSION'))
    xall=load_npz(OLD/'prepared/X_train.npz');gtall=load_npz(OLD/'prepared/GT_train.npz')
    ids=load_npz(ROOT/'tiny_v2/indices.npz')['indices'];x={k:v[ids] for k,v in xall.items()};gt=gtall['gt'][ids]
    if not np.array_equal(gtall['sample_id'],xall['sample_id']):raise AssertionError('GT IDs')
    y=np.concatenate([load_npz(p,('target',))['target'] for p in sorted((OLD/'P0_END_train_labels').glob('[0-9]*.npz'))]).astype(np.float32)[ids]
    active=np.any(y!=0,axis=-1);metadata=json_read(ROOT/'tiny_v2/SELECTION.json')
    if len(set(x['sample_id'].tolist()))!=128 or metadata['active_candidates']!=int(active.sum()):raise AssertionError('selection denominators')
    left,right=np.triu_indices(20,1);reports={};events=[];window_records=[]
    for readout,filename in [('initial','initial.npz'),('final','final.npz')]:
        guard.check()
        z=load_npz(ROOT/'tiny_v2'/filename)
        check_saved(x,z,'END')
        err=((z['continuous_param'].astype(float)-y)**2).mean(-1)
        reports[readout]=dict(all_candidate_MSE=float(err.mean()),active_candidate_MSE=float(err[active].mean()),
            identity_candidate_MSE=float(err[~active].mean()) if (~active).any() else None)
        for key,val in reports[readout].items():
            if abs(val-m[readout][key])>2e-12:raise AssertionError('MSE saved arithmetic')
        for port in ('raw','guarded'):
            p=z[port+'_paths'];d=np.sqrt(((p-gt[:,None])**2).sum(-1));a,f=d.mean(-1),d[:,:,-1]
            j=np.maximum(a/.2,f/.4);good=(a<=.2)&(f<=.4);covered=good.any(1)
            for name,val in [('ADE',a),('FDE',f),('J',j),('good',good),('covered',covered)]:
                if not np.array_equal(val,z[port+'_'+name]):raise AssertionError('saved quality '+name)
            for i in range(128):
                guard.check()
                dd=np.sqrt(((p[i,left]-p[i,right])**2).sum(-1))
                condensed=np.maximum(dd.mean(1)/.1,dd[:,-1]/.2)
                cl=fcluster(linkage(condensed,method='complete'),1.,criterion='distance')
                if len(np.unique(cl))!=z[port+'_distinct'][i] or len(np.unique(cl[good[i]]))!=z[port+'_effective_good'][i]:raise AssertionError('complete-link saved')
                b=bool(z[port+'_native_covered'][i]);c=bool(covered[i])
                event='retained' if b and c else 'lost' if b else 'rescue' if c else 'unresolved'
                events.append(dict(readout=readout,port=port,sample_id=int(x['sample_id'][i]),track=str(x['track_key'][i]),group=int(x['group'][i]),event=event))
                for subset,mask in [('active',active[i]),('identity',~active[i]),('all',np.ones(20,bool))]:
                    window_records.append(dict(readout=readout,port=port,sample_id=int(x['sample_id'][i]),subset=subset,
                        candidate_N=int(mask.sum()),parameter_MSE=float(err[i,mask].mean()) if mask.any() else None))
            for row in m[readout]['ports'][port]['groups']:
                mask=np.ones(128,bool) if row['group']=='overall' else x['group']==['stable','transition','change'].index(row['group'])
                if not mask.any():continue
                pp=p[mask];v=(pp-np.concatenate((np.zeros(pp.shape[:-2]+(1,2)),pp[...,:-1,:]),-2))/.4
                aa=(v[...,1:,:]-v[...,:-1,:])/.4
                dv=v[...,1:,:]-v[...,:-1,:];jj=(dv[...,1:,:]-dv[...,:-1,:])/(.4**2)
                vn,an,jn=[np.sqrt((q*q).sum(-1)) for q in (v,aa,jj)]
                motion=[np.quantile(vn,.95),an.mean(),np.quantile(an,.95),jn.mean(),np.quantile(jn,.95)]
                if not np.array_equal(motion,list(row['motion'].values())):raise AssertionError('full pooled motion')
                if int(covered[mask].sum())!=row['good']:raise AssertionError('group counts')
    drop=1-reports['final']['active_candidate_MSE']/reports['initial']['active_candidate_MSE']
    expected='PASS_TINY_FIT_V2' if drop>=.9 else 'STOP_TINY_FIT_V2'
    if expected!=m['status'] or abs(drop-m['active_MSE_relative_drop'])>2e-12:raise AssertionError('fixed90 gate')
    import torch
    ck=torch.load(ROOT/'tiny_v2/checkpoint.pt',map_location='cpu',weights_only=False)
    initial=torch.load(ROOT/'tiny_v2/initial_checkpoint.pt',map_location='cpu',weights_only=False)
    stats=load_npz(OLD/'prepared/statistics.npz');xx=transform(x['past'],x['native'],stats,'H')
    if ck['epoch']!=64 or ck['updates']!=256 or ck['cursor']!=0 or len(ck['history'])!=64 or len(ck['batch_order'])!=256:raise AssertionError('fixed final budget')
    if ck['binding']['x']!=digest(xx) or ck['binding']['y']!=digest(y):raise AssertionError('checkpoint data binding')
    if ck['contract']['source_lock_sha']!=sha(ROOT/'source_manifest.json') or ck['contract']['selection_sha']!=sha(ROOT/'tiny_v2/SELECTION.json'):raise AssertionError('checkpoint source binding')
    if sum(t.numel() for t in ck['model'].values())!=4770 or not all(torch.isfinite(t).all() for t in ck['model'].values()):raise AssertionError('parameters')
    for key in ck['initial_state']:
        if not torch.equal(ck['initial_state'][key],initial['initial_state'][key]) or not torch.equal(initial['model'][key],initial['initial_state'][key]):raise AssertionError('fresh initialization evidence')
    for state in ck['optimizer']['state'].values():
        if int(state['step'])!=256 or not torch.isfinite(state['exp_avg']).all() or not torch.isfinite(state['exp_avg_sq']).all():raise AssertionError('optimizer')
    if any(key not in ck for key in ('sampler','python_rng','numpy_rng','torch_rng','cuda_rng')):raise AssertionError('full RNG payload')
    guard.check(force=True)
    json_new(out/'RESOURCE.json',dict(policy='warning_first_remaining_v3',estimate_bytes=guard.estimate,observations=guard.observations))
    for filename,rows in [('TINY_WINDOW_EVENTS.csv',events),('TINY_WINDOW_CANDIDATE_STRATA.csv',window_records)]:
        with (out/filename).open('x',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    verify_manifest(ROOT/'source_manifest.json');verify_manifest(ROOT/'namespace_adapter_source_extension.json');verify_manifest(out/'SOURCE_EXTENSION.json')
    files={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts}
    json_new(out/'TINY_REVIEW.json',dict(status='PASS_INDEPENDENT_SAVED_TINY_V2',files=files,paths_relative_to_execution_root=True,
        fixed_gate_status=expected,relative_drop=drop,arithmetic=reports,window_ports=512,candidate_ports=10240,
        checkpoint_metadata_optimizer_RNG_payload_checked=True,no_forward_backward_optimizer_replay=True,
        same_assistant_not_external=True,all_inherited_sources_unchanged=True,namespace_failure_preserved=True))
    print('TINY CPU saved review PASS',expected,drop,flush=True)


if __name__=='__main__':main()
