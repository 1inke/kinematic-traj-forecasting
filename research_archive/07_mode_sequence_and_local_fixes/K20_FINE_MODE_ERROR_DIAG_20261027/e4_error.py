from common import *
from fine_features import *
from geometry_oracles import *
def main():
    g=guard(True);rows=[]
    for split in SPLITS:
        x=xload(split);y=yload(split)['gt'];p=x['native'];n=len(p);f=load(HERE/f'E3_gt_free/{split}/a20_history3_s2.npz');gt=load(HERE/f'E3_gt_audit/{split}/a20_history3_s2/gt_features.npz');assoc=load(HERE/f'E3_gt_audit/{split}/a20_history3_s2/presence.npz');q=load(HERE/f'E3_gt_audit/{split}/native_quality.npz')
        kj=q['J'].argmin(1);ka=q['ADE'].argmin(1);kf=q['FDE'].argmin(1);mm=assoc['full_match'];modeJ=np.where(mm,q['J'],np.inf);km=modeJ.argmin(1);km[~mm.any(1)]=-1
        distances=np.linalg.norm(p-y[:,None],axis=-1);end=y[:,None,-1]-p[:,:,-1];ha=f['history_heading'][:,None];valid=f['history_heading_valid'][:,None]
        longitudinal=np.where(valid,end[...,0]*np.cos(ha)+end[...,1]*np.sin(ha),np.nan);lateral=np.where(valid,-end[...,0]*np.sin(ha)+end[...,1]*np.cos(ha),np.nan)
        shapeP=shape_samples(p);shapeGT=np.repeat(shape_samples(y[:,None]),20,axis=0);shape=np.linalg.norm(shapeP-shapeGT,axis=-1).mean(1).reshape(n,20)
        _,l,tot,progress=native_progress(p.reshape(-1,12,2));_,lg,totalg,pg=native_progress(y)
        phase=np.abs(progress.reshape(n,20,12)-pg[:,None]).mean(-1)
        speed=np.linalg.norm(np.diff(np.concatenate((np.zeros((n,20,1,2)),p),axis=2),axis=2),axis=-1)/.4;sg=np.linalg.norm(np.diff(np.concatenate((np.zeros((n,1,2)),y),axis=1),axis=1),axis=-1)/.4
        onsetok=(f['onset_frame']>0)&(gt['onset_frame']>0)&f['onset_reliable']&gt['onset_reliable']
        metrics=dict(ADE_first=distances[:,:,:4].mean(-1),ADE_middle=distances[:,:,4:8].mean(-1),ADE_last=distances[:,:,8:].mean(-1),FDE=q['FDE'],ADE=q['ADE'],J=q['J'],longitudinal_error=longitudinal,lateral_error=lateral,entry_heading_error=np.abs(wrapped(f['entry_angle']-gt['entry_angle']))*180/np.pi,internal_heading_error=np.abs(wrapped(f['internal_angle']-gt['internal_angle']))*180/np.pi,onset_time_diff=np.where(onsetok,abs(f['onset_frame']-gt['onset_frame'])*.4,np.nan),early_speed_diff=f['early_speed']-gt['early_speed'],late_speed_diff=f['late_speed']-gt['late_speed'],shape_distance=shape,path_length_diff=f['length']-gt['length'],phase_progress_diff=phase,speed_peak_step_diff=abs(speed.argmax(-1)-sg.argmax(-1)[:,None]),speed_trough_step_diff=abs(speed.argmin(-1)-sg.argmin(-1)[:,None]))
        save(HERE/f'E4_error/{split}/candidate_error_metrics.npz',sample_id=x['sample_id'],k_J=kj,k_A=ka,k_F=kf,k_mode=km,point_error=distances,**metrics)
        for group,mask in masks(x):
            for state,stateMask in [('success',q['native_good']),('failure',~q['native_good']),('refined_valid_failure',(~q['native_good'])&(assoc['reason']==4))]:
                base=mask&stateMask
                for ref,k in [('kJ',kj),('kA',ka),('kF',kf),('kMode',km)]:
                    active=base&(k>=0);idx=np.flatnonzero(active)
                    for metric,value in metrics.items():
                        vv=value[idx,k[idx]];finite_=np.isfinite(vv);v=vv[finite_]
                        rows.append(dict(population=split,group=group,state=state,reference=ref,metric=metric,N_all=int(base.sum()),N_reference=int(active.sum()),N_valid=int(finite_.sum()),mean=float(v.mean()) if len(v) else None,q25=float(np.quantile(v,.25)) if len(v) else None,median=float(np.median(v)) if len(v) else None,q75=float(np.quantile(v,.75)) if len(v) else None))
                for metric,value in metrics.items():
                    v=value[base].ravel();ok=np.isfinite(v);vv=v[ok]
                    rows.append(dict(population=split,group=group,state=state,reference='ALL20',metric=metric,N_all=int(base.sum()),N_reference=int(base.sum()*20),N_valid=int(ok.sum()),mean=float(vv.mean()) if len(vv) else None,q25=float(np.quantile(vv,.25)) if len(vv) else None,median=float(np.median(vv)) if len(vv) else None,q75=float(np.quantile(vv,.75)) if len(vv) else None))
        mark('E4_CONTINUOUS_ERRORS',population=split);g.check()
    table(HERE/'E4_error/mode_matched_error.csv',rows);resource(HERE/'E4_error/RESOURCE.json',g);finish(HERE/'E4_error')
if __name__=='__main__':main()
