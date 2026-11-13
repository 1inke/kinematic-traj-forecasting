from common import *
from fine_features import *
def bootstrap_metrics(pop,variant,group,x,metrics,draw,count,trackix,rows):
    names=list(metrics)
    nums=np.stack([np.bincount(trackix,weights=np.where(d,v,0),minlength=count.shape[1]) for v,d in metrics.values()],axis=1)
    dens=np.stack([np.bincount(trackix,weights=d,minlength=count.shape[1]) for v,d in metrics.values()],axis=1)
    nn=count@nums;dd=count@dens;ratio=np.divide(nn,dd,out=np.full(nn.shape,np.nan),where=dd>0)
    for k,name in enumerate(names):
        valid=np.isfinite(ratio[:,k]);v=ratio[valid,k];quant=np.quantile(v,[.025,.975]) if len(v) else (np.nan,np.nan)
        rows.append(dict(population=pop,variant=variant,group=group,metric=name,numerator=float(nums[:,k].sum()),denominator=float(dens[:,k].sum()),estimate=finite(nums[:,k].sum()/dens[:,k].sum()) if dens[:,k].sum()>0 else None,CI95_low=finite(quant[0]),CI95_high=finite(quant[1]),invalid_draws=int((~valid).sum()),track_count=int(np.unique(x['track_key'][np.any(np.stack([d for v,d in metrics.values()]),axis=0)]).size)))
    save(HERE/f'statistics/sufficient/{pop}_{variant}_{group}.npz',metric=np.array(names),numerator_by_track=nums,denominator_by_track=dens,quantiles=np.array([[r['CI95_low'] if r['CI95_low'] is not None else np.nan,r['CI95_high'] if r['CI95_high'] is not None else np.nan] for r in rows[-len(names):]]))
def main():
    g=guard(True);rows=[];strata=[]
    for pop in ('opt','train','val','hold','inner'):
        split='opt' if pop in ('train','val') else pop;xx=xload(split);sel=load(OLD/'inputs/train_val.npz')[pop] if pop in ('train','val') else np.arange(len(xx['sample_id']));x={k:v[sel] for k,v in xx.items()};N=len(sel);tracks=np.unique(x['track_key']);idx=np.searchsorted(tracks,x['track_key'])
        rng=np.random.default_rng(20261027);draw=rng.integers(len(tracks),size=(20000,len(tracks)),dtype=np.int32);counts=np.array([np.bincount(v,minlength=len(tracks)) for v in draw],np.int16)
        save(HERE/f'statistics/DRAWS_{pop}.npz',tracks=tracks,draws=draw)
        q=load(HERE/f'E3_gt_audit/{split}/native_quality.npz');ng=q['native_good'][sel];fail=~ng;old=load(OLD/f'E2_oracle_coverage/{split}/coarse20/slot_removal_oracle.npz');oldpresent=old['GT_mode_present'][sel]
        for name,_,_,_ in variants():
            allf=load(HERE/f'E3_gt_free/{split}/{name}.npz');f={k:v[sel] for k,v in allf.items()};allr=load(HERE/f'E3_gt_audit/{split}/{name}/presence.npz');r={k:v[sel] for k,v in allr.items()}
            straight=f['L0']==2;turn=straight&np.isin(f['entry'],[1,2]);common=r['common_valid']
            for group,mask in masks(x):
                fixedfail=mask&fail;commonfail=fixedfail&common;ones=np.ones(N)
                metrics=dict(old_straight_entry_turn_candidate_rate=(turn.sum(1),mask&(straight.sum(1)>0)),old_straight_entry_turn_window_rate=(turn.any(1).astype(float),mask),history_invalid_fraction=((~f['history_heading_valid']).astype(float),mask),entry_undefined_candidate_fraction=((f['entry']==3).mean(1),mask),onset_undefined_candidate_fraction=((f['onset']==5).mean(1),mask))
                # Candidate rate denominator is candidate count, not windows.
                for reason,label in enumerate(REASONS):metrics[label]=((r['reason']==reason).astype(float),fixedfail)
                for label,key in [('entry','entry_match'),('onset','onset_match'),('speed','speed_match'),('entry_onset','entry_onset_match'),('refined','full_match')]:
                    metrics[label+'_present_common']=(r[key].any(1).astype(float),commonfail)
                    metrics[label+'_present_all_failures']=(r[key].any(1).astype(float),fixedfail)
                metrics['old_L0_present_common']=(oldpresent.astype(float),commonfail)
                # Use numerator/denominator on actual window data; candidate fraction special scale.
                value,den=metrics['old_straight_entry_turn_candidate_rate'];metrics['old_straight_entry_turn_candidate_rate']=(np.divide(value,straight.sum(1),out=np.zeros(N),where=straight.sum(1)>0),den)
                metrics['mean_old_straight_entry_turn_count']=(turn.sum(1),mask)
                bootstrap_metrics(pop,name,group,x,metrics,draw,counts,idx,rows)
                # Explicit candidate-weighted ratio in addition to equal-window rate.
                num=np.bincount(idx,weights=turn.sum(1)*mask,minlength=len(tracks));den=np.bincount(idx,weights=straight.sum(1)*mask,minlength=len(tracks));bd=counts@den;bn=counts@num;vals=np.divide(bn,bd,out=np.full(len(bd),np.nan),where=bd>0);vv=vals[np.isfinite(vals)]
                quant=np.quantile(vv,[.025,.975]) if len(vv) else (np.nan,np.nan)
                rows.append(dict(population=pop,variant=name,group=group,metric='candidate_pooled_turn_given_old_straight',numerator=float(num.sum()),denominator=float(den.sum()),estimate=float(num.sum()/den.sum()) if den.sum() else None,CI95_low=finite(quant[0]),CI95_high=finite(quant[1]),invalid_draws=int(np.isnan(vals).sum()),track_count=len(tracks)))
            if name=='a20_history3_s2':
                for scope,key in [('scene','scene'),('sequence','sequence'),('track','track_key')]:
                    for val in np.unique(x[key]):
                        for group,gm in masks(x):
                            mask=(x[key]==val)&gm;ff=mask&fail
                            if not mask.any():continue
                            rr=dict(population=pop,scope=scope,label=str(val),group=group,N=int(mask.sum()),native_good=int((mask&ng).sum()),failure=int(ff.sum()),old_straight_candidate_N=int(straight[mask].sum()),old_straight_entry_turn=int(turn[mask].sum()),window_any_entry_turn=int(turn.any(1)[mask].sum()))
                            rr.update({v:int((ff&(r['reason']==k)).sum()) for k,v in enumerate(REASONS)});strata.append(rr)
            g.check()
        for group,mask in masks(x):
            metrics={}
            ports={}
            for port in ('Native','END_unbounded','END_b02','END_b04','TIME_grid'):
                p=load(HERE/f'E4_oracle/{split}/{port}.npz');covered=p['covered'][sel];ports[port]=covered
                metrics[port+'_coverage']=(covered.astype(float),mask);metrics[port+'_rescue_of_failure']=(covered.astype(float),mask&fail);metrics[port+'_lost_of_nativegood']=((~covered).astype(float),mask&ng)
            metrics['END_b04_TIME_both']=((ports['END_b04']&ports['TIME_grid']).astype(float),mask&fail)
            metrics['END_b04_only']=((ports['END_b04']&~ports['TIME_grid']).astype(float),mask&fail)
            metrics['TIME_only']=((~ports['END_b04']&ports['TIME_grid']).astype(float),mask&fail)
            bootstrap_metrics(pop,'oracle',group,x,metrics,draw,counts,idx,rows)
        mark('STATISTICS_COMPLETE',population=pop,draws=20000);g.check()
    table(HERE/'statistics/groups.csv',rows);write(HERE/'statistics/bootstrap_ci.json',rows);table(HERE/'statistics/track_scene.csv',strata)
    resource(HERE/'statistics/RESOURCE.json',g);finish(HERE/'statistics',whole_target_track=True,descriptive_CI=True,no_seed_replication=True,no_invalid_redraw=True)
if __name__=='__main__':main()
