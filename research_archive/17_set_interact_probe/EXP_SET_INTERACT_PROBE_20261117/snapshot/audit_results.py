"""Independent saved-geometry/statistics audit plus frozen-cache final-weight replay."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from probe_io import read,write,arrays,sha,verify,seal,now

def equal(a,b,path=''):
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        for k in a:equal(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),path
        for j,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(j))
    elif isinstance(a,(int,float,np.number)) and not isinstance(a,(bool,np.bool_)):
        np.testing.assert_allclose(a,b,atol=3e-7 if path.endswith('/mean_shift') else 3e-11,rtol=3e-11,err_msg=path)
    else:assert a==b,(path,a,b)

def cluster_count(pair):
    d=pair.copy();np.fill_diagonal(d,np.inf);count=len(d)
    while True:
        index=d.argmin();i,j=divmod(index,len(d))
        if d[i,j]>.2:break
        merged=np.maximum(d[i],d[j]);d[i]=merged;d[:,i]=merged;d[j]=np.inf;d[:,j]=np.inf;d[i,i]=np.inf;count-=1
    return count

def independently_measure(pred,y,valid):
    p=pred['paths'].astype(float);z=pred['logits'].astype(float);delta=p-y[:,None].astype(float)
    dist=np.hypot(delta[:,:,:,0],delta[:,:,:,1]);ade=dist.sum(-1)/12;fde=dist[:,:,-1];row=np.arange(len(p));winner=np.argmin(ade+.25*fde,axis=1);top=np.argmax(z,axis=1)
    end=p[:,:,-1];d=end[:,:,None]-end[:,None];pair=np.hypot(d[:,:,:,0],d[:,:,:,1]);i,j=np.triu_indices(20,1)
    low=end[:,:,0].min(1);high=end[:,:,0].max(1);gt=y[:,-1,0].astype(float);below=gt<low;above=gt>high
    return dict(minADE=ade.min(1),minFDE=fde.min(1),top1FDE=fde[row,top],winner=winner,winner_FDE=fde[row,winner],uncovered=fde.min(1)>.4,allocation_gate=fde[row,winner]>.4,redundant_pairs=np.count_nonzero(pair[:,i,j]<.2,axis=1),effective_candidates=np.array([cluster_count(d) for d in pair]),longitudinal_outside=below|above,longitudinal_gt_below_all=below,longitudinal_gt_above_all=above,longitudinal_inside_gap=(~below&~above)&(np.abs(end[:,:,0]-gt[:,None]).min(1)>.4),direction_valid=valid,mean_shift=np.hypot(pred['shift'][:,:,:,0].astype(float),pred['shift'][:,:,:,1].astype(float)).mean((-1,-2)))

def independent_groups(w,meta):
    masks=dict(all=np.ones(len(meta['group']),bool),stable=meta['group']==0,transition=meta['group']==1,change=meta['group']==2,decel_stop=np.isin(meta['labels'],['deceleration','stop']))
    output={}
    for name,mask in masks.items():
        n=int(sum(mask));b=dict(n=n,small_group=n<20)
        if n:
            for k in ('minADE','minFDE','top1FDE','uncovered','allocation_gate','redundant_pairs','effective_candidates','mean_shift'):b[k]=float(np.sum(w[k][mask])/n)
            b['minFDE_p90']=float(np.percentile(w['minFDE'][mask],90));v=mask&w['direction_valid'];b['direction_valid_n']=int(sum(v))
            for k in ('longitudinal_outside','longitudinal_gt_below_all','longitudinal_gt_above_all','longitudinal_inside_gap'):b[k]=float(np.count_nonzero(w[k][v])/sum(v)) if v.any() else None
            counts=[int(sum(w['winner'][mask]==j)) for j in range(20)];b['winner_counts']=counts;b['winner_frequency']=[v/n for v in counts];b['rarely_winning_slots_below_0p1pct']=sum(v/n<.001 for v in counts)
        output[name]=b
    return output

def independent_decision(baseline,runs):
    def gains(arm,ref,metric='minFDE',group='all'):
        values=[]
        for seed in ('0','1','2'):
            a=runs[arm][seed]['groups'][group][metric];b=baseline['groups'][group][metric] if ref=='A0' else runs[ref][seed]['groups'][group][metric];values.append((b-a)/b)
        return values
    first=gains('A1','A0');sigma=float(np.std(first,ddof=1));thr=max(.005,2*sigma)
    result=dict(Q1_proxy_gain=float(np.mean(first)),A1_paired_relative_FDE_gains=first,sigma=sigma,threshold=thr,protection_mode='separate_status')
    if result['Q1_proxy_gain']<=0:return dict(result,verdict='PROXY_INVALID',raw_verdict='PROXY_INVALID',protection_status='NOT_EVALUATED',eligible_to_advance=False,stopped_after_A1=True)
    comparisons={}
    for name,a,b in [('Q2_sparse','A3','A1'),('Q2_generic','A2','A1'),('Q3','A5','A4'),('Q4','A5','A6'),('Q5','A4','A1')]:
        f=gains(a,b);e=gains(a,b,'minADE');mf=float(np.mean(f));me=float(np.mean(e));comparisons[name]=dict(arm=a,reference=b,FDE_paired_gains=f,ADE_paired_gains=e,mean_FDE_gain=mf,mean_ADE_gain=me,passes=mf>=thr and me>=-.002)
    q=comparisons;raw='H_SUPPORTED' if q['Q3']['passes'] and q['Q4']['passes'] else 'GENERIC_ONLY' if q['Q3']['passes'] else 'LOSS_ONLY' if q['Q5']['passes'] else 'NO_SET_GAIN'
    protections={}
    for arm in runs:
        checks={}
        for key,metric,group,tol in [('top1','top1FDE','all',.02),('stable','minFDE','stable',.01),('decel_stop','minFDE','decel_stop',.01)]:
            harm=-float(np.mean(gains(arm,'A0',metric,group)));checks[key]=dict(relative_harm=harm,max_harm=tol,passes=harm<=tol)
        protections[arm]=dict(checks=checks,passes=all(b['passes'] for b in checks.values()))
    selected='A5' if raw in ('H_SUPPORTED','GENERIC_ONLY') else 'A4' if raw=='LOSS_ONLY' else None
    protection='PASS' if selected and protections[selected]['passes'] else 'PROTECTION_FAILED' if selected else 'NO_EFFECTIVE_ARM'
    return dict(result,verdict=raw,raw_verdict=raw,comparisons=comparisons,protection_status=protection,protections=protections,selected_arm=selected,eligible_to_advance=raw=='H_SUPPORTED' and protection=='PASS',stopped_after_A1=False)

def main():
    import torch
    from probe_cache import load_cache
    from probe_model import Refiner,predict
    from probe_io import OLD
    started=time.monotonic();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    verify(read(ROOT/'SOURCE_MANIFEST.json')['files']);verify(read(ROOT/'DONE.json')['files']);verify(read(ROOT/'cache/DONE.json')['files'])
    result=read(ROOT/'RESULT.json');data,head,criterion=load_cache(result['device']);dev=data['dev'];meta=dev['meta'];y=dev['tensors']['y'].cpu().numpy()
    # Recompute cache target and labels from original C0 arrays without encoder execution.
    raw=arrays(OLD/'C0/dev.npz');np.testing.assert_array_equal(raw['ids'],meta['ids']);p=raw['past'][:,0].astype(float);disp=p[:,-1]-p[:,-4];norm=np.linalg.norm(disp,axis=1);direction=np.tile([1.,0.],(len(p),1));good=norm>.03;direction[good]=disp[good]/norm[good,None]
    delta=raw['future'].astype(float)-p[:,-1,None];target=np.stack([delta[:,:,0]*direction[:,0,None]+delta[:,:,1]*direction[:,1,None],-delta[:,:,0]*direction[:,1,None]+delta[:,:,1]*direction[:,0,None]],axis=-1)
    np.testing.assert_allclose(y,target,atol=2e-5,rtol=2e-5);np.testing.assert_array_equal(meta['heading_valid'],good);np.testing.assert_array_equal(raw['group'],meta['group'])
    sys.path.insert(0,'C:/Users/linke/Downloads/1/EXP_ETH_SPEED_20261116/code');module=__import__('eth_speed_math');np.testing.assert_array_equal(meta['labels'],module.maneuver_labels(raw['past'],raw['future']))
    base_pred=arrays(ROOT/'A0/predictions_local.npz');w=independently_measure(base_pred,y,good);base_groups=independent_groups(w,meta)
    base_saved=arrays(ROOT/'A0/window_metrics.npz')
    for name,value in w.items():np.testing.assert_allclose(value,base_saved[name],atol=3e-11,rtol=3e-11,err_msg='A0/'+name)
    equal(base_groups,result['baseline']['groups'],'A0/groups');baseline=dict(result['baseline'],groups=base_groups)
    checked=[];runs={}
    expected_orders={}
    for seed in (0,1,2):
        generator=torch.Generator().manual_seed(20261117+seed);expected_orders[str(seed)]=[hashlib.sha256(torch.randperm(21719,generator=generator).numpy().astype('<i8').tobytes()).hexdigest() for _ in range(20)]
    for arm,items in result['runs'].items():
        runs[arm]={}
        for seed,row in items.items():
            print('AUDIT '+arm+' s'+seed,flush=True);folder=ROOT/'runs'/f'{arm}_s{seed}';verify(read(folder/'DONE.json')['files'])
            pred=arrays(folder/'predictions_local.npz');saved=arrays(folder/'window_metrics.npz');np.testing.assert_array_equal(pred['ids'],meta['ids']);np.testing.assert_array_equal(saved['ids'],meta['ids']);np.testing.assert_array_equal(saved['group'],meta['group']);np.testing.assert_array_equal(saved['labels'],meta['labels'])
            measured=independently_measure(pred,y,good)
            for name,value in measured.items():np.testing.assert_allclose(value,saved[name],atol=3e-7 if name=='mean_shift' else 3e-11,rtol=3e-11,err_msg=arm+'/'+seed+'/'+name)
            groups=independent_groups(measured,meta);equal(groups,row['groups'],arm+'/'+seed+'/groups');runs[arm][seed]=dict(row,groups=groups)
            training=read(folder/'TRAINING.json');assert training['epochs']==20 and training['updates']==1700
            assert [b['order_hash'] for b in training['history']]==expected_orders[seed]
            state=torch.load(folder/'final.pt',map_location='cpu',weights_only=False);assert state['epoch']==20 and state['updates']==1700 and state['scheduler']['last_epoch']==1700
            initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False);paired=torch.load(ROOT/'runs'/f'A1_s{seed}'/'initial.pt',map_location='cpu',weights_only=False)
            for key in initial:assert torch.equal(initial[key],paired[key])
            model=Refiner(arm).to(result['device']);model.load_state_dict(state['model']);model.eval();max_p=max_z=max_s=0.
            with torch.no_grad():
                for first in range(0,len(y),256):
                    index=slice(first,first+256);pp,zz,ss=predict(model,head,dev['tensors'],index)
                    max_p=max(max_p,float(np.max(np.abs(pp.cpu().numpy()-pred['paths'][index]))));max_z=max(max_z,float(np.max(np.abs(zz.cpu().numpy()-pred['logits'][index]))));max_s=max(max_s,float(np.max(np.abs(ss.cpu().numpy()-pred['shift'][index]))))
            assert max_p<=1e-5 and max_z<=1e-5 and max_s<=1e-5,(arm,seed,max_p,max_z,max_s)
            checked.append(dict(arm=arm,seed=int(seed),all_4753_windows_geometry_and_groups='PASS',final_cache_replay_max_path_difference=max_p,final_cache_replay_max_logit_difference=max_z,final_cache_replay_max_shift_difference=max_s,batch_order_and_init_pairing='PASS',epochs=20,updates=1700))
    decision=independent_decision(baseline,runs);equal(decision,result['decision'],'decision')
    assert result['completed_runs']==len(checked) and result['skipped_runs']==18-len(checked)
    assets=torch.load(ROOT/'cache/frozen_assets.pt',map_location='cpu',weights_only=False);old=torch.load(OLD/'runs/E1C_E64_s42/final.pt',map_location='cpu',weights_only=False)
    for key,value in assets['score_head'].items():assert torch.equal(value,old['ema']['decoder.score_head.'+key])
    verify(read(ROOT/'SOURCE_MANIFEST.json')['files']);verify(read(ROOT/'DONE.json')['files'])
    additional={}
    if len(runs)==6:
        b=baseline['groups']['all']['minFDE']
        for name,old_arm,new_arm in [('sparse','A3','A5'),('generic','A2','A6')]:
            values=[((runs['A4'][s]['groups']['all']['minFDE']-runs[new_arm][s]['groups']['all']['minFDE'])-(runs['A1'][s]['groups']['all']['minFDE']-runs[old_arm][s]['groups']['all']['minFDE']))/b for s in ('0','1','2')]
            additional[name]=dict(paired_additional_interaction_gain_with_new_loss=values,mean=float(np.mean(values)),decision_use=False)
    write(ROOT/'ADDITIONAL_DESCRIPTIVE.json',dict(interaction_loss_difference_in_differences=additional,explanation='Positive means the interaction advantage increases with reallocation; descriptive only, no new decision gate.'))
    write(ROOT/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',at=now(),seconds=time.monotonic()-started,script_sha256=sha(__file__),completed_runs=len(checked),checks=checked,independent_complete_link='GREEDY_MAX_DISTANCE_MERGING_ALL_WINDOWS',cached_targets_groups_labels='PASS',all_per_window_metrics_and_groups='PASS',decision_sigma_and_protection='PASS',frozen_score_head_equals_original_EMA=True,encoder_executed_in_audit=False,new_training_in_audit=False,final_cache_replay=True,raw_verdict=decision['raw_verdict'],protection_status=decision['protection_status']))
    print('AUDIT_PASS '+decision['verdict'],flush=True)

if __name__=='__main__':main()
