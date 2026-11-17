import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster

def metrics(paths,y,logits,heading_valid,shift=None):
    p,y,z=[np.asarray(v,np.float64) for v in (paths,y,logits)];n=len(p);row=np.arange(n)
    dist=np.linalg.norm(p-y[:,None],axis=-1);ade=dist.mean(-1);fde=dist[:,:,-1];winner=(ade+.25*fde).argmin(-1);top=z.argmax(-1)
    end=p[:,:,-1];pair=np.linalg.norm(end[:,:,None]-end[:,None],axis=-1);upper=np.triu_indices(20,1)
    condensed=pair[:,upper[0],upper[1]]
    effective=np.array([len(np.unique(fcluster(linkage(v,method='complete'),t=.2,criterion='distance'))) for v in condensed])
    prog=end[:,:,0];gt=y[:,-1,0];over=gt<prog.min(-1);under=gt>prog.max(-1);inside=~(over|under)
    return dict(minADE=ade.min(-1),minFDE=fde.min(-1),top1FDE=fde[row,top],winner=winner,
        winner_FDE=fde[row,winner],uncovered=fde.min(-1)>.4,allocation_gate=fde[row,winner]>.4,
        redundant_pairs=(condensed<.2).sum(-1),effective_candidates=effective,
        longitudinal_outside=over|under,longitudinal_gt_below_all=over,longitudinal_gt_above_all=under,
        longitudinal_inside_gap=inside&(np.abs(prog-gt[:,None]).min(-1)>.4),direction_valid=np.asarray(heading_valid,bool),
        mean_shift=np.linalg.norm(shift,axis=-1).mean(axis=(-1,-2)) if shift is not None else np.zeros(n))
def masks(group,labels):
    return dict(all=np.ones(len(group),bool),stable=group==0,transition=group==1,change=group==2,decel_stop=np.isin(labels,['deceleration','stop']))
def aggregate(w,group,labels):
    result={}
    for name,mask in masks(group,labels).items():
        n=int(mask.sum());b=dict(n=n,small_group=n<20)
        if not n:result[name]=b;continue
        for key in ('minADE','minFDE','top1FDE','uncovered','allocation_gate','redundant_pairs','effective_candidates','mean_shift'):b[key]=float(w[key][mask].mean())
        b['minFDE_p90']=float(np.quantile(w['minFDE'][mask],.9));valid=mask&w['direction_valid'];b['direction_valid_n']=int(valid.sum())
        for key in ('longitudinal_outside','longitudinal_gt_below_all','longitudinal_gt_above_all','longitudinal_inside_gap'):b[key]=float(w[key][valid].mean()) if valid.any() else None
        counts=np.bincount(w['winner'][mask],minlength=20);b['winner_counts']=counts.tolist();b['winner_frequency']=(counts/n).tolist();b['rarely_winning_slots_below_0p1pct']=int((counts/n<.001).sum())
        result[name]=b
    return result

def decide(baseline,runs,protection_mode='separate_status'):
    def gains(arm,reference,metric='minFDE',group='all'):
        x=[runs[arm][s]['groups'][group][metric] for s in ('0','1','2')]
        ref=[baseline['groups'][group][metric]]*3 if reference=='A0' else [runs[reference][s]['groups'][group][metric] for s in ('0','1','2')]
        return [(b-a)/b for a,b in zip(x,ref)]
    gain1=gains('A1','A0');sigma=float(np.std(gain1,ddof=1));threshold=max(.005,2*sigma)
    out=dict(Q1_proxy_gain=float(np.mean(gain1)),A1_paired_relative_FDE_gains=gain1,sigma=sigma,threshold=threshold,protection_mode=protection_mode)
    if out['Q1_proxy_gain']<=0:return dict(out,verdict='PROXY_INVALID',raw_verdict='PROXY_INVALID',protection_status='NOT_EVALUATED',eligible_to_advance=False,stopped_after_A1=True)
    if len(runs)<6:return dict(out,verdict='PENDING',raw_verdict='PENDING',eligible_to_advance=False)
    comparisons={}
    for name,arm,ref in [('Q2_sparse','A3','A1'),('Q2_generic','A2','A1'),('Q3','A5','A4'),('Q4','A5','A6'),('Q5','A4','A1')]:
        f=gains(arm,ref);a=gains(arm,ref,'minADE');fg=float(np.mean(f));ag=float(np.mean(a))
        comparisons[name]=dict(arm=arm,reference=ref,FDE_paired_gains=f,ADE_paired_gains=a,mean_FDE_gain=fg,mean_ADE_gain=ag,passes=fg>=threshold and ag>=-.002)
    c=comparisons;raw='H_SUPPORTED' if c['Q3']['passes'] and c['Q4']['passes'] else 'GENERIC_ONLY' if c['Q3']['passes'] else 'LOSS_ONLY' if c['Q5']['passes'] else 'NO_SET_GAIN'
    protections={}
    for arm in runs:
        harm={}
        for key,metric,group,tol in [('top1','top1FDE','all',.02),('stable','minFDE','stable',.01),('decel_stop','minFDE','decel_stop',.01)]:
            value=-float(np.mean(gains(arm,'A0',metric,group)));harm[key]=dict(relative_harm=value,max_harm=tol,passes=value<=tol)
        protections[arm]=dict(checks=harm,passes=all(v['passes'] for v in harm.values()))
    selected='A5' if raw in ('H_SUPPORTED','GENERIC_ONLY') else 'A4' if raw=='LOSS_ONLY' else None
    protection='PASS' if selected and protections[selected]['passes'] else 'PROTECTION_FAILED' if selected else 'NO_EFFECTIVE_ARM'
    return dict(out,verdict=raw,raw_verdict=raw,comparisons=comparisons,protection_status=protection,protections=protections,selected_arm=selected,eligible_to_advance=raw=='H_SUPPORTED' and protection=='PASS',stopped_after_A1=False)
