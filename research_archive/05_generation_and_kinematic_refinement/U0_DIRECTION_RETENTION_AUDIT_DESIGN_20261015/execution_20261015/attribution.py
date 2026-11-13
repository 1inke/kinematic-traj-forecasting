import numpy as np
import common as c
EVENTS=('unresolved','retained','lost','rescue')

def acceleration(paths,independent=False):
    if independent:
        parts=[]
        for t in range(1,12):
            previous=paths[:,:,t-2] if t>=2 else 0.
            a=((paths[:,:,t]-paths[:,:,t-1])/.4-(paths[:,:,t-1]-previous)/.4)/.4
            parts.append(np.sqrt(a[...,0]**2+a[...,1]**2))
        return np.stack(parts,axis=-1).mean((1,2))
    v=np.diff(paths,axis=2,prepend=np.zeros_like(paths[:,:,:1]))/.4;a=np.diff(v,axis=2)/.4
    return np.sqrt((a*a).sum(-1)).mean((1,2))

def describe(w,d,sel):
    n=int(sel.sum());tracks=len(set(w['track'][sel]));c0=w['C0'][sel];ev=w['event'][sel];g0=w['g0'][sel];lost=ev==2
    base=dict(n=n,tracks=tracks,thin=n<20 or tracks<5)
    if not n:return base
    definite=~w['uncertain_fixed'][sel];ratios=w['G0_type_fraction'][sel];hasg=g0>0
    base.update(native_good=int(c0.sum()),native_failure=int((~c0).sum()),good=int(w['C1'][sel].sum()),retained=int((ev==1).sum()),lost=int(lost.sum()),rescue=int((ev==3).sum()),unresolved=int((ev==0).sum()),lost_rate=float(lost.sum()/c0.sum()) if c0.any() else None,rescue_rate=float((ev==3).sum()/(~c0).sum()) if (~c0).any() else None,native_ADE_mean=c.scalar_mean(w['native_ADE'][sel]),native_FDE_mean=c.scalar_mean(w['native_FDE'][sel]),native_minJ_mean=c.scalar_mean(w['J0_min'][sel]),dJ_fixed_mean=c.scalar_mean(w['dJ_fixed'][sel]),dJ_fixed_certain_mean=c.scalar_mean(w['dJ_fixed'][sel][definite]),dJ_fixed_certain_n=int(definite.sum()),deltaJ_fixed_mean=c.scalar_mean(w['deltaJ_fixed'][sel]),deltaA_fixed_mean=c.scalar_mean(w['deltaA_fixed'][sel]),deltaF_fixed_mean=c.scalar_mean(w['deltaF_fixed'][sel]),fixed_type_counts=np.bincount(w['kind_fixed'][sel],minlength=4),all20_candidate_denominator=n*20,all20_type_counts=w['all20_type_counts'][sel].sum(0),all20_windowweighted_type_fraction=w['all20_type_fraction'][sel].mean(0),all20_dJ_mean=float(d['dJ'][sel].mean()),all20_actual_deltaJ_mean=float(d['deltaJ'][sel].mean()),G0_slots=int(g0.sum()),G0_failed_slots=int(w['G0_failed_slots'][sel].sum()),G0_type_counts=w['G0_type_counts'][sel].sum(0),G0_windowweighted_type_fraction=ratios[hasg].mean(0) if hasg.any() else None,no_definite_descent_all20=int(w['no_definite_descent_all20'][sel].sum()),no_definite_descent_all20_and_allcertain=int((w['no_definite_descent_all20'][sel]&w['all20_certain'][sel]).sum()),uncertain_slots=int(d['uncertain'][sel].sum()),near_tie_slots=int(d['near_tie'][sel].sum()),cosine_defined_n=int(w['cosine_defined_fixed'][sel].sum()),cosine_fixed_mean=c.scalar_mean(w['cosine_fixed'][sel][w['cosine_defined_fixed'][sel]]),correction_mean=c.scalar_mean(w['correction_mean'][sel]),correction_endpoint=c.scalar_mean(w['correction_endpoint'][sel]),Jslot_switch_rate=float(w['Jslot_switched'][sel].mean()),J_segments_native_fixed=w['J_segments_native_fixed'][sel].mean(0),J_segments_refined_fixed=w['J_segments_refined_fixed'][sel].mean(0),last_G0_exit_le_half=int(w['last_G0_exit_le_half'][sel].sum()),last_G0_exit_gt_half=int(w['last_G0_exit_gt_half'][sel].sum()),last_G0_exit_half_uncertain=int(w['last_G0_exit_half_bracket_uncertain'][sel].sum()))
    if lost.any():
        ls=sel&(w['event']==2);g=w['g0'][ls];num=w['G0_type_counts'][ls,2]
        base['lost_G0_overshoot_slot_fraction']=float(num.sum()/g.sum())
        base['lost_G0_overshoot_windowweighted_fraction']=float((num/g).mean())
        base['lost_native_margin_le_002_fraction']=float((w['margin_bin'][ls]==0).mean())
        base['lost_last_G0_exit_lo_quantiles']=np.quantile(w['last_G0_exit_lo'][ls],[0,.25,.5,.75,1])
        base['lost_last_G0_exit_hi_quantiles']=np.quantile(w['last_G0_exit_hi'][ls],[0,.25,.5,.75,1])
    return base

def summarize(w,d):
    result={}
    for group,mask in c.masks(w['groups']):
        allrow=describe(w,d,mask);strata=[]
        for success in (False,True):
            pop=mask&(w['C0']==success)
            strata.append(dict(stratum='native_success' if success else 'native_failure',**describe(w,d,pop)))
            for b in range(3):
                ss=pop&(w['margin_bin']==b);strata.append(dict(stratum='margin',native_success=success,margin_bin=b,**describe(w,d,ss)))
                if success:
                    for g in (1,2,3):strata.append(dict(stratum='margin_x_g0',native_success=True,margin_bin=b,g0_bin=g,**describe(w,d,ss&(w['g0_bin']==g))))
            if success:
                for g in (1,2,3):strata.append(dict(stratum='g0',g0_bin=g,**describe(w,d,pop&(w['g0_bin']==g))))
        for e,name in enumerate(EVENTS):strata.append(dict(stratum='event',event=name,**describe(w,d,mask&(w['event']==e))))
        n=int(mask.sum());ng=int((mask&w['C0']).sum());R=int((mask&(w['event']==3)).sum());L=int((mask&(w['event']==2)).sum());target=(5*n+99)//100;allowed=ng//100
        ceiling=dict(n=n,native_good=ng,rescue=R,lost=L,actual_delta_coverage=(R-L)/n if n else None,privileged_keep_all_old_delta_coverage=R/n if n else None,privileged_keep_all_old_good=ng+R,required_rescues_for_5pp=target,allowed_lost_under_1pct=allowed,missing_rescues_if_zero_lost=max(0,target-R),current_R_reaches_5pp_if_zero_lost=R>=target,current_R_reaches_5pp_with_max_allowed_lost=R-allowed>=target,not_bound_for_new_training_changing_R=True)
        contrib=[]
        for e,name in enumerate(EVENTS):
            ss=mask&(w['event']==e);nn=int(ss.sum());delta=c.scalar_mean(w['delta_accel'][ss]);contrib.append(dict(event=name,n=nn,delta_accel_mean=delta,weighted_signed_contribution=nn/n*delta if nn else 0.,native_accel_mean=c.scalar_mean(w['native_accel'][ss]),refined_accel_mean=c.scalar_mean(w['refined_accel'][ss])))
        total=c.scalar_mean(w['delta_accel'][mask]);reconstructed=sum(x['weighted_signed_contribution'] for x in contrib)
        if n:c.close(total,reconstructed)
        else:reconstructed=None
        acc=dict(n=n,native_mean=c.scalar_mean(w['native_accel'][mask]),refined_mean=c.scalar_mean(w['refined_accel'][mask]),delta_mean=total,reconstructed_delta=reconstructed,events=contrib,p95_not_additive=True)
        result[group]=dict(overall=allrow,strata=strata,retention_ceiling=ceiling,accel_attribution=acc)
    return result

def fragile_check(w):
    m=w['groups']==2;lost=m&(w['event']==2);near=m&w['C0']&(w['margin_bin']==0);other=m&w['C0']&(w['margin_bin']!=0)
    def rate(s):return float(((w['event']==2)&s).sum()/s.sum()) if s.any() else None
    fraction=float((lost&(w['margin_bin']==0)).sum()/lost.sum()) if lost.any() else None
    adequate=all(int(s.sum())>=20 and len(set(w['track'][s]))>=5 for s in (near,other))
    return dict(fraction_lost_near_margin=fraction,near_n=int(near.sum()),other_n=int(other.sum()),near_lost_rate=rate(near),other_lost_rate=rate(other),adequate=adequate,supported=bool(adequate and fraction is not None and fraction>=.8 and rate(near)>rate(other)))
