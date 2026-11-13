from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from geometry_oracles import time_from_rates
def main():
    g=guard(True);verify_sources();rows=[];phase=[];reviewed_error_rows=[]
    for split in SPLITS:
        x=xload(split);gt=yload(split)['gt'];p=x['native'];N=len(p)
        # Progress of an exactly zero-length curve has no temporal phase.
        prev=np.concatenate((np.zeros((N,20,1,2)),p[:,:,:-1]),axis=2);plen=np.linalg.norm(p-prev,axis=-1).sum(-1)
        gprev=np.concatenate((np.zeros((N,1,2)),gt[:,:-1]),axis=1);glen=np.linalg.norm(gt-gprev,axis=-1).sum(-1)
        phasevalid=(plen>0)&(glen[:,None]>0)
        original=load(HERE/f'E4_error/{split}/candidate_error_metrics.npz')
        corrected=original.copy();corrected['phase_progress_valid']=phasevalid;corrected['phase_progress_diff']=np.where(phasevalid,original['phase_progress_diff'],np.nan)
        save(HERE/f'completion_review/error_reviewed/{split}.npz',**corrected)
        quality_=load(HERE/f'E3_gt_audit/{split}/native_quality.npz');assoc=load(HERE/f'E3_gt_audit/{split}/a20_history3_s2/presence.npz')
        for group,mask in masks(x):
            for state,stateMask in [('success',quality_['native_good']),('failure',~quality_['native_good']),('refined_valid_failure',(~quality_['native_good'])&(assoc['reason']==4))]:
                base=mask&stateMask
                for ref,key in [('kJ','k_J'),('kA','k_A'),('kF','k_F'),('kMode','k_mode'),('ALL20',None)]:
                    if key is not None:
                        k=corrected[key];active=base&(k>=0);ix=np.flatnonzero(active);v=corrected['phase_progress_diff'][ix,k[ix]];Nref=len(ix)
                    else:v=corrected['phase_progress_diff'][base].ravel();Nref=int(base.sum()*20)
                    ok=np.isfinite(v);vv=v[ok];reviewed_error_rows.append(dict(population=split,group=group,state=state,reference=ref,metric='phase_progress_diff',N_all=int(base.sum()),N_reference=Nref,N_valid=int(ok.sum()),mean=float(vv.mean()) if len(vv) else None,q25=float(np.quantile(vv,.25)) if len(vv) else None,median=float(np.median(vv)) if len(vv) else None,q75=float(np.quantile(vv,.75)) if len(vv) else None))
        phase.append(dict(population=split,all_candidate_cases=int(N*20),phase_undefined_cases=int((~phasevalid).sum()),change_phase_undefined=int((~phasevalid)[x['group']==2].sum()),source='original E4_error retained; only exact-zero-length phase marked undefined'))
        hv=(x['past'][:,-1]-x['past'][:,-2])/.4;params=load(HERE/f'E4_oracle/{split}/time_parameters.npz')
        nativejunction=np.linalg.norm(p[:,:,0]/.4-hv[:,None],axis=-1)/.4
        for port in ('Native','END_unbounded','END_b02','END_b04','TIME_grid'):
            if port=='Native':first=p[:,:,0]
            elif port=='TIME_grid':
                tmp,_,_=time_from_rates(p.reshape(-1,12,2),params['rate_a'].ravel(),params['rate_b'].ravel(),params['time_feasible'].ravel());first=tmp.reshape(p.shape)[:,:,0]
            else:
                d=gt[:,None,-1]-p[:,:,-1];b={'END_unbounded':np.inf,'END_b02':.2,'END_b04':.4}[port];ok=np.linalg.norm(d,axis=-1)<=b;first=p[:,:,0]+np.where(ok[...,None],d/12,0)
            junction=np.linalg.norm(first/.4-hv[:,None],axis=-1)/.4
            for pop in (('opt','train','val') if split=='opt' else (split,)):
                sel=load(OLD/'inputs/train_val.npz')[pop] if pop in ('train','val') else np.arange(N);xx={k:v[sel] for k,v in x.items()}
                for group,mask in masks(xx):
                    a=junction[sel][mask];ref=nativejunction[sel][mask];rows.append(dict(population=pop,port=port,group=group,N=int(mask.sum()),candidate_den=int(mask.sum()*20),junction_acc_mean=float(a.mean()),junction_acc_p95=float(np.quantile(a,.95)),native_mean=float(ref.mean()),native_p95=float(np.quantile(ref,.95)),not_old_five_motion_gate=True,definition='norm(future_v1 - last_observed_v)/.4; diagnostic only'))
        g.check()
    table(HERE/'completion_review/JUNCTION_DIAGNOSTIC.csv',rows);table(HERE/'completion_review/PHASE_ERROR_REVIEWED.csv',reviewed_error_rows);write(HERE/'completion_review/PHASE_VALIDITY_ERRATUM.json',dict(status='EXPLICIT_NEW_REVIEWED_ARRAYS_OLD_PRESERVED',only_metric='phase_progress_diff',rule='exactly zero candidate or GT arc length has undefined normalized progress; preserve window, set only this descriptive value NaN',counts=phase,no_quality_or_mode_or_oracle_or_bootstrap_changes=True))
    resource(HERE/'completion_review/JUNCTION_RESOURCE.json',g)
if __name__=='__main__':main()
