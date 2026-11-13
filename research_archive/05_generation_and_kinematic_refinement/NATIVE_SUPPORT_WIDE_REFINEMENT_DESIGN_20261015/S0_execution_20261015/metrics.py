import common as c
import numpy as np

def metrics(z,groups,tracks,ind=False):
    ports=(z['native'],z['safe4'],z['safe20']);raw=[c.errors(p,z['gt'],ind) for p in ports];ng=raw[0][3].any(1);out={}
    for port,p,(ade,fde,j,good,temporalJ) in zip(c.PORTS,ports,raw):
        gg=good.any(1);cc=c.clusters(p,good,ind);e=np.sqrt(((p-z['gt'][:,None])**2).sum(-1));qslot=(ade+.5*fde).argmin(1);qt=e[np.arange(len(p)),qslot].reshape(-1,3,4).mean(-1);out[port]={}
        for name,mask in c.masks(groups):
            if not mask.any():out[port][name]=dict(n=0);continue
            mot=c.motion(p[mask],ind);base=c.motion(ports[0][mask],ind)
            out[port][name]=dict(n=int(mask.sum()),tracks=len(set(tracks[mask])),good=int(gg[mask].sum()),coverage=float(gg[mask].mean()),ADE=float(ade[mask].min(1).mean()),FDE=float(fde[mask].min(1).mean()),J=float(j[mask].min(1).mean()),Q=float((ade+.5*fde)[mask].min(1).mean()),rescue=int((gg&~ng&mask).sum()),lost=int((~gg&ng&mask).sum()),retained=int((gg&ng&mask).sum()),effective_distinct=float(cc[mask,0].mean()),effective_good=float(cc[mask,1].mean()),single_Jclosest_early_middle_late=temporalJ[mask].mean(0),single_Qclosest_early_middle_late=qt[mask].mean(0),motion=mot,native_motion=base,motion_literal_1pct=all(mot[k]<=1.01*base[k] for k in mot),privileged_witness_not_deployable=port!='native20')
    for name in c.GROUPS:
        n=out['native20'][name]['n']
        if n:
            a,b,d=[out[p][name] for p in c.PORTS];assert a['good']<=b['good']<=d['good'] and b['lost']==d['lost']==0
    return out
def certificate_summary(z,groups):
    out={};good=c.errors(z['safe20'],z['gt'])[3];nativegood=c.errors(z['native'],z['gt'])[3].any(1)
    ub=c.errors(z['native']+z['ball_residual'],z['gt'])[3]
    for name,m in c.masks(groups):
        if not m.any():continue
        alpha=z['alpha'][m];out[name]=dict(candidates=int(alpha.size),alpha_quantiles=np.quantile(alpha,[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist(),zero_alpha=int((alpha==0).sum()),actual_nonzero_correction=int(np.any(z['safe20'][m]!=z['native'][m],axis=(-1,-2)).sum()),fallback=int(z['fallback'][m].sum()),fallback_reason_counts_33=z['fallback_reasons'][m].sum((0,1)).tolist(),binding_counts_33=z['binding'][m].sum((0,1)).tolist(),zero_budget_counts_33=z['zero_budget'][m].sum((0,1)).tolist(),per_slot_rescue_windows=((~nativegood[m,None])&good[m]).sum(0).tolist(),relaxed_ball_good_windows=int(ub[m].any(1).sum()),safe20_minus_safe4_good_windows=int(good[m].any(1).sum()-c.errors(z['safe4'][m],z['gt'][m])[3].any(1).sum()))
    return out

def gate(records):
    rows=records['FIT']['metrics'];a=rows['native20']['change'];b=rows['safe4']['change'];d=rows['safe20']['change']
    assert a['n']==4725 and a['good']==1937
    coverage=100*(d['good']-1937)>=5*4725
    safety=all(r['motion_literal_1pct'] and r['lost']==0 for port in c.PORTS for r in rows[port].values() if r['n'])
    return dict(status='S0_PASS_ALLOW_S1_PREFLIGHT' if coverage and safety else 'STOP_SAFE_OPPORTUNITY_UNDER_REGISTERED_MAP',FIT_change_n=4725,native_good=1937,safe4_good=b['good'],safe20_good=d['good'],safe20_gain_pp=100*(d['good']-1937)/4725,integer_lhs=100*(d['good']-1937),integer_rhs=5*4725,coverage_pass=coverage,all_FIT_literal_motion_and_zero_lost=safety,S1_preflight_allowed=coverage and safety,S1_started=False,new_formal_count=0,not_global_safe_space_optimum=True)
