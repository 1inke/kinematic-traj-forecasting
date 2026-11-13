"""Posthoc final-output assignment, teacher subsets and population losses."""
from common import *

def analyze(paths,native,gt,groups,kin):
    from f16.eval import errors
    pe=errors(paths,gt);ne=errors(native,gt);ke=errors(kin,gt) if kin is not None else None;ng=ne['coverage'];pg=pe['coverage'];good=pe['good'];q=pe['ADE']+.5*pe['FDE'];n=len(paths);slot={};loss={};teacher_stats={}
    # The historical WTA uses ADE+.5FDE, not J's slot, and first physical tie.
    winner=q.argmin(1);teacher_R=np.zeros(20,np.int64);teacher_N=np.zeros(20,np.int64);teacher_B=np.zeros(20,np.int64);SR=SN=SB=0.;nR=nN=nB=0;teacher_ties_R=teacher_ties_N=teacher_ties_B=0
    if kin is not None:
        for start in range(0,n,64):
            end=min(n,start+64);pool=np.concatenate([native[start:end],kin[start:end]],1).astype(np.float32).astype(np.float64);student=paths[start:end].astype(np.float64);d=np.linalg.norm(pool[:,:,None]-student[:,None],axis=-1);distance=d.mean(-1)+.5*d[:,:,:,-1];nearest=distance.argmin(-1);minimum=distance.min(-1);R=np.concatenate([ne['good'][start:end],np.zeros((end-start,20),bool)],1)&ng[start:end,None];N=np.concatenate([np.zeros((end-start,20),bool),ke['good'][start:end]],1)&~ng[start:end,None];B=np.concatenate([ne['good'][start:end],ke['good'][start:end]],1)
            for mask,key in [(R,'R'),(N,'N'),(B,'B')]:
                count=int(mask.sum());total=float(minimum[mask].sum());responsibility=np.bincount(nearest[mask],minlength=20);ties=int(((distance==minimum[:,:,None]).sum(-1)[mask]>1).sum())
                if key=='R':SR+=total;nR+=count;teacher_R+=responsibility;teacher_ties_R+=ties
                elif key=='N':SN+=total;nN+=count;teacher_N+=responsibility;teacher_ties_N+=ties
                else:SB+=total;nB+=count;teacher_B+=responsibility;teacher_ties_B+=ties
    for name,mask in [('overall',np.ones(n,bool)),('stable',groups==0),('transition',groups==1),('change',groups==2)]:
        counts=np.bincount(winner[mask],minlength=20);firstgood=good.argmax(1);success=mask&pg
        slot[name]=dict(N=int(mask.sum()),WTA_Q_winner_counts=counts.tolist(),WTA_exact_tie_windows=int(((q==q.min(1)[:,None]).sum(1)[mask]>1).sum()),any_good_candidate_counts=good[mask].sum(0).tolist(),first_good_physical_slot_distribution=np.bincount(firstgood[success],minlength=20).tolist(),success_J_closest_slot_distribution=np.bincount(pe['J_winner'][success],minlength=20).tolist(),warning='Final checkpoint responsibility, not historical training usage; teacher matching is separate below.')
    if kin is not None:
        teacher_stats=dict(available=True,teacher_cache_is_GT_posthoc_label_only=True,R_native_success_teacher_count=nR,N_native_failed_KIN_good_teacher_count=nN,B_union_good_teacher_count=nB,B_extra_KIN_in_native_success_count=int((ke['good']&ng[:,None]).sum()),R_nearest_student_counts=teacher_R.tolist(),N_nearest_student_counts=teacher_N.tolist(),B_nearest_student_counts=teacher_B.tolist(),R_exact_teacher_ties=teacher_ties_R,N_exact_teacher_ties=teacher_ties_N,B_exact_teacher_ties=teacher_ties_B)
    else:teacher_stats=dict(available=False,reason='No verified aligned KIN cache; never substitute FIT rows')
    end=paths[:,:,-1].astype(float);ii,jj=np.triu_indices(20,1);hinge=np.maximum(.3-np.linalg.norm(end[:,ii]-end[:,jj],axis=-1),0).mean();base=float(q.min(1).mean())+.05*float(hinge)
    loss=dict(WTA_Q=float(q.min(1).mean()),endpoint_linear_diversity=float(hinge),base=base,R_population_teacher_mean=SR/nR if nR else 0.,N_population_teacher_mean=SN/nN if nN else 0.,B_population_teacher_mean=SB/nB if nB else 0.,hypothetical_full_population_A=base,hypothetical_full_population_B=base+(SB/nB if nB else 0.),hypothetical_full_population_C=base+(SR/nR if nR else 0.),hypothetical_full_population_D=base+(SN/nN if nN else 0.),hypothetical_full_population_E=base+(SR/nR if nR else 0.)+(SN/nN if nN else 0.),warning='Full-population final diagnostic, not pre-update epoch history; per-batch mean-of-ratios differs.')
    subsets={};perwindow={}
    labels={'native_success':ng,'native_failure_with_good_KIN':~ng&ke['coverage'] if ke is not None else None,'native_failure_without_good_KIN':~ng&~ke['coverage'] if ke is not None else None}
    for name,mask in labels.items():
        if mask is None:subsets[name]={'available':False};continue
        sub={}
        for group,gmask in [('overall',np.ones(n,bool)),('stable',groups==0),('transition',groups==1),('change',groups==2)]:
            take=mask&gmask;den=int(take.sum());num=int(pg[take].sum());sub[group]=dict(N=den,good=num,coverage=num/den if den else None,lost=int((ng&~pg&take).sum()),rescue=int((~ng&pg&take).sum()),ADE=float(pe['ADE'][take].min(1).mean()) if den else None,FDE=float(pe['FDE'][take].min(1).mean()) if den else None,J=float(pe['J'][take].min(1).mean()) if den else None)
        subsets[name]=dict(available=True,groups=sub)
    perwindow=dict(ADE=pe['ADE'],FDE=pe['FDE'],J=pe['J'],good=pe['good'],covered=pg,native_covered=ng,Q_winner=winner,J_winner=pe['J_winner'],Q_segments=pe['Q_segments'],J_segments=pe['J_segments'],teacher_KIN_good=ke['good'] if ke is not None else np.empty((n,0),bool))
    return dict(assignment=slot,teacher=teacher_stats,population_loss=loss,subsets=subsets),perwindow
