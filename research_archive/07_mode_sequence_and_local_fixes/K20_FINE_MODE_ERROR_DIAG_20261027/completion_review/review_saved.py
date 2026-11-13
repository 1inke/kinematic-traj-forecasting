from pathlib import Path
import sys,math,csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from fine_features import variants,REASONS
from geometry_oracles import time_from_rates,motion,pooled
def independent_motion(p):
    vel=(p-np.concatenate((np.zeros(p.shape[:-2]+(1,2)),p[...,:-1,:]),-2))/.4
    acc=(vel[...,1:,:]-vel[...,:-1,:])/.4
    jerk=(acc[...,1:,:]-acc[...,:-1,:])/.4
    return dict(speed=np.sqrt((vel*vel).sum(-1)),acc=np.sqrt((acc*acc).sum(-1)),jerk=np.sqrt((jerk*jerk).sum(-1)))
def literal_quality(p,gt):
    d=np.sqrt(((p-gt[:,None])**2).sum(-1));a=d.sum(-1)/12;f=d[...,-1];j=np.maximum(a/.2,f/.4);return a,f,j,(a<=.2)&(f<=.4)
def scalar_winner_curve(p,a,b,valid):
    if not valid or (a==1 and b==1):return p.copy()
    z=np.vstack((np.zeros(2),p));seg=np.sqrt(((z[1:]-z[:-1])**2).sum(1));q=np.r_[0,np.cumsum(seg)/seg.sum()]
    c=(1-a*q[4]-b*(q[8]-q[4]))/(1-q[8]);out=[]
    assert .75<=a<=1.25 and .75<=b<=1.25 and .75-1e-12<=c<=1.25+1e-12
    for t in range(1,13):
        qt=a*q[t] if t<=4 else (a*q[4]+b*(q[t]-q[4]) if t<=8 else a*q[4]+b*(q[8]-q[4])+c*(q[t]-q[8]))
        out.append([np.interp(qt,q,z[:,dim]) for dim in range(2)])
    out=np.array(out);out[-1]=p[-1];return out
def main():
    g=guard(True);source=verify_sources()
    assert read(HERE/'execution_state.json')['status']=='SUCCESS_REVIEW_REQUIRED'
    assert read(HERE/'exit_status.json')['status']=='SUCCESS_REVIEW_REQUIRED'
    for part in ('E3_gt_free','E3_gt_audit','E4_error','E4_oracle','statistics'):verify_done(HERE/part)
    sourcebefore=source;slotcount=0;tuple_windows=0;qualitymax=0.;motionrows=[];populationrows=[]
    mainkey='a20_history3_s2'
    for split in SPLITS:
        x=xload(split);gt=yload(split)['gt'];p=x['native'];N=len(p)
        q=load(HERE/f'E3_gt_audit/{split}/native_quality.npz')
        qa=literal_quality(p,gt)
        for k,v in zip(('ADE','FDE','J','good'),qa):
            if v.dtype==bool:assert np.array_equal(q[k],v)
            else:assert np.max(abs(q[k]-v))<=2e-12
        for key,a,h,s in variants():
            f=load(HERE/f'E3_gt_free/{split}/{key}.npz');gg=load(HERE/f'E3_gt_audit/{split}/{key}/gt_features.npz');r=load(HERE/f'E3_gt_audit/{split}/{key}/presence.npz')
            for i in range(N):
                e=np.array([bool(gg['entry_valid'][i,0]) and int(v)==int(gg['entry'][i,0]) and int(v)!=3 for v in f['entry'][i]])
                o=np.array([bool(gg['onset_valid'][i,0]) and int(v)==int(gg['onset'][i,0]) and int(v)!=5 for v in f['onset'][i]])
                ss=np.array([tuple(int(f[k][i,j]) for k in ('start_trend','internal_trend','low_bits'))==tuple(int(gg[k][i,0]) for k in ('start_trend','internal_trend','low_bits')) for j in range(20)])
                for name,expected in [('entry_match',e),('onset_match',o),('speed_match',ss),('entry_onset_match',e&o),('full_match',e&o&ss)]:assert np.array_equal(r[name][i],expected)
                if not gg['entry_valid'][i,0] or gg['boundary'][i,0]:reason=0
                elif not gg['onset_valid'][i,0]:reason=5
                elif not e.any():reason=1
                elif not (e&o).any():reason=2
                elif not (e&o&ss).any():reason=3
                else:reason=4
                assert r['reason'][i]==reason;tuple_windows+=1
            if key==mainkey:
                # Independent direct angle formula on every candidate, no extractor call.
                ha=np.arctan2((x['past'][:,-1]-x['past'][:,-4])[:,1],(x['past'][:,-1]-x['past'][:,-4])[:,0])
                vec=p[:,:,3]-x['past'][:,-1,None];want=np.arctan2(np.sin(np.arctan2(vec[...,1],vec[...,0])-ha[:,None]),np.cos(np.arctan2(vec[...,1],vec[...,0])-ha[:,None]))
                assert np.max(abs(want[f['entry_valid']]-f['entry_angle'][f['entry_valid']]),initial=0)<2e-12
        parameters=load(HERE/f'E4_oracle/{split}/time_parameters.npz')
        timecurve=np.array([scalar_winner_curve(path,a,b,bool(v)) for path,a,b,v in zip(p.reshape(-1,12,2),parameters['rate_a'].ravel(),parameters['rate_b'].ravel(),parameters['time_feasible'].ravel())]).reshape(p.shape)
        assert np.array_equal(timecurve[:,:,-1],p[:,:,-1]);slotcount+=N*20
        for port in ('Native','END_unbounded','END_b02','END_b04','TIME_grid'):
            if port=='Native':out=p
            elif port=='TIME_grid':out=timecurve
            else:
                delta=gt[:,None,-1]-p[:,:,-1];bound={'END_unbounded':np.inf,'END_b02':.2,'END_b04':.4}[port]
                ok=np.sqrt((delta*delta).sum(-1))<=bound
                out=p*(1-np.arange(1,13)[None,None,:,None]/12)+np.arange(1,13)[None,None,:,None]/12*(p-p[:,:,-1,None,:]+gt[:,None,-1,None,:])
                out[:,:,-1]=gt[:,None,-1];out=np.where(ok[...,None,None],out,p)
            got=load(HERE/f'E4_oracle/{split}/{port}.npz');quality_=literal_quality(out,gt)
            for label,v in zip(('ADE','FDE','J'),quality_[:3]):
                error=np.max(abs(got[label]-v));qualitymax=max(qualitymax,float(error));assert error<2e-12
            assert np.array_equal(got['good'],quality_[3])
            independent=independent_motion(out)
            mm=np.stack([np.quantile(independent['speed'],.95,axis=-1),independent['acc'].mean(-1),np.quantile(independent['acc'],.95,axis=-1),independent['jerk'].mean(-1),np.quantile(independent['jerk'],.95,axis=-1)],-1)
            assert np.max(abs(mm-got['window_motion']))<2e-12
            if port=='TIME_grid':
                assert np.array_equal(got['FDE'],q['FDE'])
                assert not got['covered'][np.all(q['FDE']>.4,axis=1)].any()
            for pop in (('opt','train','val') if split=='opt' else (split,)):
                sel=load(OLD/'inputs/train_val.npz')[pop] if pop in ('train','val') else np.arange(N);xx={k:v[sel] for k,v in x.items()};basegood=q['native_good'][sel];covered=got['covered'][sel]
                for group,mask in masks(xx):
                    globalmask=np.zeros(N,bool);globalmask[sel[mask]]=True
                    raw=pooled(independent,globalmask)
                    motionrows.append(dict(population=pop,port=port,group=group,N=int(mask.sum()),speed_p95=raw[0],accel_mean=raw[1],accel_p95=raw[2],jerk_mean=raw[3],jerk_p95=raw[4]))
                    populationrows.append(dict(population=pop,port=port,group=group,N=int(mask.sum()),native_good=int(basegood[mask].sum()),oracle_good=int(covered[mask].sum()),rescue=int((mask&~basegood&covered).sum()),lost=int((mask&basegood&~covered).sum())))
            g.check()
    # Independently gather each actual sampled track, not count-matrix multiplication.
    cimax=0.;records=0;invalidexact=True
    for pop in ('opt','train','val','hold','inner'):
        draws=load(HERE/f'statistics/DRAWS_{pop}.npz')['draws']
        for variant in (mainkey,'oracle'):
            for group in GROUPS:
                s=load(HERE/f'statistics/sufficient/{pop}_{variant}_{group}.npz');num=s['numerator_by_track'];den=s['denominator_by_track'];values=[]
                for start in range(0,20000,128):
                    d=draws[start:start+128];a=num[d].sum(1);b=den[d].sum(1);values.append(np.divide(a,b,out=np.full(a.shape,np.nan),where=b>0))
                vals=np.concatenate(values)
                for k in range(len(s['metric'])):
                    v=vals[:,k];ok=np.isfinite(v);quant=np.quantile(v[ok],[.025,.975]) if ok.any() else np.array([np.nan,np.nan])
                    finite_=np.isfinite(quant)&np.isfinite(s['quantiles'][k])
                    if finite_.any():cimax=max(cimax,float(np.max(abs(quant[finite_]-s['quantiles'][k][finite_]))))
                    assert np.array_equal(np.isnan(quant),np.isnan(s['quantiles'][k]));assert np.max(abs(quant[finite_]-s['quantiles'][k][finite_]),initial=0)<2e-12
                records+=1
        g.check()
    table(HERE/'completion_review/MOTION_ALL5_POPULATIONS.csv',motionrows);table(HERE/'completion_review/ORACLE_ALL5_POPULATIONS.csv',populationrows)
    verify_sources();resource(HERE/'completion_review/RESOURCE.json',g)
    write(HERE/'completion_review/REVIEW.json',dict(status='PASS_COMPLETE_SAVED_CPU_REVIEW',at=now(),source=source,source_initial_final_SHA_EXACT=True,all_candidate_TIME_reconstructed=slotcount,all_END3_and_Native_full_quality_and_window5motion=True,quality_maxerror=qualitymax,all12_variants_tuple_decision_windows=tuple_windows,actual20000_bootstrap_gather_records=records,bootstrap_maxerror=cimax,bootstrap_scope='all5 populations x main/oracle x4groups; sensitivity production saved and source-locked, not claimed full second draw replay',whole_target_track=True,no_model_or_solver_or_optimizer_replay=True,no_new_predictions_saved=True,no_external_peer_review=True,no_delete=True))
    print('PASS REVIEW',slotcount,cimax,flush=True)
if __name__=='__main__':main()
