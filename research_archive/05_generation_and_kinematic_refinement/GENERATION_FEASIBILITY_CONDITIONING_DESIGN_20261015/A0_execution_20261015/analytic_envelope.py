import common as c
import numpy as np
RADIUS=.20*(np.arange(1,13,dtype=np.float64)/12.)**2
def construct(native,gt,slots,ind=False):
    p=native.astype(np.float64);y=gt.astype(np.float64);delta=y[:,None]-p
    e=np.sqrt((delta*delta).sum(-1)) if ind else np.linalg.norm(delta,axis=-1)
    if ind:
        r=np.empty_like(delta)
        for t,b in enumerate(RADIUS):
            scale=np.zeros_like(e[:,:,t]);positive=e[:,:,t]>0;scale[positive]=np.minimum(e[:,:,t][positive],b)/e[:,:,t][positive];r[:,:,t]=delta[:,:,t]*scale[...,None]
    else:r=delta*np.minimum(np.divide(RADIUS,e,out=np.zeros_like(e),where=e>0),1.)[...,None]
    lb=np.maximum(e-RADIUS,0.);ade=e.mean(-1);fde=e[:,:,-1];ba=lb.mean(-1);bf=lb[:,:,-1]
    mask=np.zeros((len(p),20),bool);mask[np.arange(len(p))[:,None],slots]=True
    allp=p+r;bottom=np.where(mask[...,None,None],allp,p)
    nativeJ=np.maximum(ade/.2,fde/.4);ballJ=np.maximum(ba/.2,bf/.4);bottomJ=np.where(mask,ballJ,nativeJ).min(1)
    assert np.all(nativeJ.min(1)>=bottomJ) and np.all(bottomJ>=ballJ.min(1))
    assert np.array_equal(bottom[~mask],p[~mask])
    return dict(native=p,gt=y,slots=slots,lower_distance=lb,native_slot_ADE=ade,native_slot_FDE=fde,native_slot_J=nativeJ,ball_slot_ADE=ba,ball_slot_FDE=bf,ball_slot_J=ballJ,bottom_slot_J=np.where(mask,ballJ,nativeJ),ideal_ball_paths=allp,bottom4_paths=bottom)
def metrics(z,groups,tracks,ind=False):
    ports=(z['native'],z['bottom4_paths'],z['ideal_ball_paths']);raw=[c.errors(p,z['gt'],ind) for p in ports];ng=raw[0][3].any(1);out={}
    for port,p,(ade,fde,j,good,temporalJ) in zip(c.PORTS,ports,raw):
        gg=good.any(1);cc=c.clusters(p,good,ind);e=np.sqrt(((p-z['gt'][:,None])**2).sum(-1));qslot=(ade+.5*fde).argmin(1);qt=e[np.arange(len(p)),qslot].reshape(-1,3,4).mean(-1);out[port]={}
        for name,mask in c.masks(groups):
            if not mask.any():out[port][name]=dict(n=0);continue
            mot=c.motion(p[mask],ind);base=c.motion(ports[0][mask],ind)
            out[port][name]=dict(n=int(mask.sum()),tracks=len(set(tracks[mask])),good=int(gg[mask].sum()),coverage=float(gg[mask].mean()),ADE=float(ade[mask].min(1).mean()),FDE=float(fde[mask].min(1).mean()),J=float(j[mask].min(1).mean()),Q=float((ade+.5*fde)[mask].min(1).mean()),rescue=int((gg&~ng&mask).sum()),lost=int((~gg&ng&mask).sum()),retained=int((gg&ng&mask).sum()),effective_distinct=float(cc[mask,0].mean()),effective_good=float(cc[mask,1].mean()),single_Jclosest_early_middle_late=temporalJ[mask].mean(0),single_Qclosest_early_middle_late=qt[mask].mean(0),motion=mot,native_motion=base,motion_literal_1pct=all(mot[k]<=1.01*base[k] for k in mot),oracle_not_safe_prediction=port!='native20')
    for name in c.GROUPS:
        n=out['native20'][name]['n']
        if n:
            a,b,d=[out[p][name] for p in c.PORTS];assert a['good']<=b['good']<=d['good'] and b['lost']==d['lost']==0
    return out
def gate(records):
    row=records['FIT']['metrics'];a,b,d=[row[p]['change'] for p in c.PORTS];n=a['n'];gb=b['good']-a['good'];ga=d['good']-a['good']
    allpass=100*ga>=5*n;bottompass=100*gb>=5*n
    verdict='STOP_LOCAL_BALL_PLATFORM' if not allpass else 'STOP_FIXED_BOTTOM4_OPPORTUNITY' if not bottompass else 'A0_PASS_CONTINUE_A1_PREFLIGHT'
    return dict(status=verdict,gate_population='FIT change ONLY',n=n,native_good=a['good'],UB_bottom4_good=b['good'],UB_all20_good=d['good'],bottom4_gain_pp=100*gb/n,all20_gain_pp=100*ga/n,bottom4_integer_lhs=100*gb,all20_integer_lhs=100*ga,threshold_rhs=5*n,bottom4_pass=bottompass,all20_pass=allpass,A1_preflight_triggered=bottompass and allpass,B_training_started=False,new_formal_count=0,exact_relaxed_ball_bound_not_safe_model_optimum=True)
