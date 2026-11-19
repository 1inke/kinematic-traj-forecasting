"""Local anchor feasibility and objective-gradient diagnostics from CPU arrays."""
import numpy as np
from loss_independent import loss_grad, softmax


def feasibility(templates, scale, target, labels):
    result = {}
    for tag, scal in [('current_scale',scale),('remove_floor_hold_softplus_fixed',scale-.25)]:
        lower = scal[:,None,None]*(templates[None]-.75)
        upper = scal[:,None,None]*(templates[None]+.75)
        violation = np.maximum(np.maximum(lower-target[:,None], target[:,None]-upper), 0)
        nearest_end = np.linalg.norm(violation[:,:,-1],axis=-1).min(1)
        feasible = (violation<=1e-7).all((2,3)).any(1)
        result[tag] = {}
        for name, mask in dict(all=np.ones(len(labels),bool),stationary=labels=='stationary',
                              steady=labels=='steady',decel_stop=np.isin(labels,['deceleration','stop'])).items():
            if mask.any():
                result[tag][name] = dict(n=int(mask.sum()),all_anchors_reachable_n=int(feasible[mask].sum()),
                                        fraction=float(feasible[mask].mean()),
                                        endpoint_box_lower_bound_mean_m=float(nearest_end[mask].mean()),
                                        endpoint_box_lower_bound_p95_m=float(np.quantile(nearest_end[mask],.95)))
    return result


def gradient_components(p, y, z, weights, basis, verify):
    n=len(y)
    row=np.arange(n)
    error=p-y[:,None]
    d=np.linalg.norm(error,axis=-1)
    quality=d.mean(-1)+.25*d[:,:,-1]
    winner=quality.argmin(1)
    derivative=np.clip(error[row,winner],-1,1)
    regression=np.zeros_like(p)
    regression[row,winner]=derivative*weights[None,:,None]/12
    regression[row,winner,-1]+=.25*derivative[:,-1]/2
    regression[row,winner,:2]+=.2*derivative[:,:2]/4
    unit=np.divide(error,d[...,None],out=np.zeros_like(error),where=d[...,None]>0)
    responsibility=softmax(-quality/.1)
    soft=.2*responsibility[:,:,None,None]*unit/12
    soft[:,:,-1]+=.05*responsibility[:,:,None]*unit[:,:,-1]
    diversity=np.zeros_like(p)
    end=p[:,:,-1]
    delta=end[:,:,None]-end[:,None]
    dist=np.linalg.norm(delta,axis=-1)
    active=(dist>0)&(dist<.4)
    direction=np.divide(delta,dist[...,None],out=np.zeros_like(delta),where=active[...,None])
    diversity[:,:,-1]=-.05*direction.sum(2)/190
    audit_error=None
    if verify:
        ncheck=min(n,32)
        _,reference,_,_=loss_grad(p[:ncheck,None],y[:ncheck,None],z[:ncheck,None],weights)
        summed=(regression[:ncheck]+soft[:ncheck]+diversity[:ncheck])/ncheck
        audit_error=float(np.max(np.abs(summed-reference[:,0])))
        np.testing.assert_allclose(summed,reference[:,0],atol=2e-14,rtol=2e-12)
    # Project into candidate anchor coordinates. The score-head-mediated path
    # derivative is deliberately excluded from this comparison of objective terms.
    projected=[]
    for v in (regression,soft,diversity):
        projected.append(np.einsum('ta,ntd->nad',basis,v[row,winner]).reshape(n,6))
    reg,cov,div=projected
    main=reg+cov
    norm=np.linalg.norm
    dot=(main*div).sum(-1)
    rdot=(reg*div).sum(-1)
    main_norm=norm(main,axis=-1)
    div_norm=norm(div,axis=-1)
    return dict(dot=dot, regression_dot=rdot,main_norm=main_norm,diversity_norm=div_norm,
                winner_joint_good=(d[row,winner].mean(-1)<=.2)&(d[row,winner,-1]<=.4)),audit_error


def analyze_mechanism(old, dec, data, labels):
    import torch
    torch.set_num_threads(1)
    checkpoint=torch.load(old/'runs/E1C_E64_s42/final.pt',map_location='cpu',weights_only=False)
    template=checkpoint['ema']['decoder.anchor_templates'].numpy().astype(float)
    basis=checkpoint['ema']['decoder.interpolation'].numpy().astype(float)
    assert template.shape==(20,3,2) and basis.shape==(12,3)
    assets=torch.load(dec/'cache/frozen_assets.pt',map_location='cpu',weights_only=False)
    config=assets['criterion_recipe']['decoder']
    for k,v in dict(WINNER_FDE_WEIGHT=.25,ENDPOINT_LOSS_WEIGHT=.25,DIVERSITY_WEIGHT=.05,
                    DIVERSITY_MARGIN=.4,SHORT_HORIZON_WEIGHT=.2,SHORT_HORIZON_STEPS=2).items():
        assert config[k]==v,(k,config[k])
    weights=assets['loss_buffers']['regression.weights'].numpy().astype(float)
    reach={}
    gradients={}
    for split in ('fit','dev'):
        c=torch.load(dec/f'cache/{split}.pt',map_location='cpu',weights_only=False)
        np.testing.assert_array_equal(c['meta']['ids'],data[split]['ids'])
        np.testing.assert_array_equal(c['meta']['labels'],labels[split])
        t=c['tensors']
        scale=t['scale'].numpy().astype(float)
        assert (scale>.25).all()
        anchors=t['relative'].numpy().astype(float)
        y=t['y'].numpy().astype(float)
        z=t['logits'].numpy().astype(float)
        cv=t['self_vel'].numpy().astype(float)[:,None]*np.array([1.6,3.2,4.8])[None,:,None]
        normalized=anchors/scale[:,None,None]-template[None]
        assert np.max(np.abs(normalized))<=.750002
        nearest=np.linalg.norm(anchors[:,:,-1],axis=-1).min(1)
        stat=labels[split]=='stationary'
        reach[split]=dict(CV=feasibility(template,scale,cv,labels[split]),
                         stationary=feasibility(template,scale,np.zeros_like(cv),labels[split]),
                         current_stationary_nearest_endpoint_m=float(nearest[stat].mean()),
                         stationary_scale_min_per_axis=scale[stat].min(0).tolist(),
                         stationary_scale_quantiles_per_axis=np.quantile(scale[stat],[.01,.5,.99],axis=0).tolist(),
                         stationary_with_either_scale_le_0p26=float((scale[stat]<=.26).any(1).mean()))
        if split!='fit':
            continue
        chunks=[]
        for lo in range(0,len(y),256):
            i=slice(lo,lo+256)
            p=np.einsum('ta,nkad->nktd',basis,anchors[i])
            props,audit=gradient_components(p,y[i],z[i],weights,basis,lo==0)
            chunks.append(props)
            if audit is not None:
                audit_error=audit
        all_v={k:np.concatenate([v[k] for v in chunks]) for k in chunks[0]}
        good=all_v['winner_joint_good']
        stat=labels[split]=='stationary'
        for name,mask in dict(all=np.ones(len(y),bool),joint_good_winner=good,
                              stationary=stat,stationary_joint_good_winner=stat&good,
                              steady=labels[split]=='steady',change=data[split]['group']==2).items():
            active=mask&(all_v['diversity_norm']>1e-12)&(all_v['main_norm']>1e-12)
            n=int(active.sum())
            ratios=all_v['diversity_norm'][active]/all_v['main_norm'][active]
            gradients[name]=dict(n=int(mask.sum()),both_nonzero_n=n,
                                 opposing_n=int((all_v['dot'][active]<0).sum()),
                                 opposing_fraction_of_active=float((all_v['dot'][active]<0).mean()) if n else None,
                                 direct_regression_opposing_n=int((all_v['regression_dot'][active]<0).sum()),
                                 median_diversity_over_geometric_attraction=float(np.median(ratios)) if n else None,
                                 diversity_larger_and_opposing_n=int(((ratios>1)&(all_v['dot'][active]<0)).sum()))
    return dict(seed=42,base='B0@64 EMA',reachability=reach,fit_gradients=gradients,
                gradient_max_difference_vs_original_independent=float(audit_error),
                gradient_scope='Direct winner-anchor coordinate terms. Classification via score head, model parameter derivatives and training dynamics excluded.',
                feasibility_scope='Same frozen template and current scale. Removing +0.25 holds learned softplus output fixed. No inference or counterfactual retraining.')
