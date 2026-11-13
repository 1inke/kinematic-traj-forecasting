from common import *
from morphology import variants
def summaries():
    g=guard(True);allrows=[];strata=[];draw_receipts=[]
    for pop in ('opt','train','val','hold','inner'):
        split='opt' if pop in ('train','val') else pop;x0=load(HERE/f'inputs/X_{split}.npz');ixpop=load(HERE/'inputs/train_val.npz')[pop] if pop in ('train','val') else np.arange(len(x0['sample_id']));x={k:v[ixpop] for k,v in x0.items()};tracks=np.unique(x['track_key']);ix=np.searchsorted(tracks,x['track_key']);rng=np.random.default_rng(20261027);draw=rng.integers(len(tracks),size=(20000,len(tracks)),dtype=np.int32);count=np.zeros((20000,len(tracks)),np.int16)
        for i,d in enumerate(draw):count[i]=np.bincount(d,minlength=len(tracks))
        save(HERE/f'statistics/DRAWS_{pop}.npz',tracks=tracks,draws=draw);draw_receipts.append(dict(population=pop,tracks=len(tracks),draws=20000))
        for variant in variants():
            name=variant[0];m={k:v[ixpop] for k,v in load(HERE/f'E1_mode_distribution/{split}/{name}/modes.npz').items() if k!='slot'};r={k:v[ixpop] for k,v in load(HERE/f'E2_oracle_coverage/{split}/{name}/slot_removal_oracle.npz').items()};good=r['native_good'];fail=~good;defined=~r['undefined'];reason=r['failure_reason']
            metrics=dict(Pmax=(m['Pmax'],np.ones(len(good),bool)),Meff=(m['Meff'],np.ones(len(good),bool)),M=(m['M'],np.ones(len(good),bool)),same_pair=(m['same_pair'],np.ones(len(good),bool)),near_phi_pair=(m['near_phi_pair'],np.ones(len(good),bool)),low_fraction=(m['low'].mean(1),np.ones(len(good),bool)),undefined_fraction=((~m['turn_defined']&~m['low']).mean(1),np.ones(len(good),bool)),oracle_removable=(r['R'].sum(1),np.ones(len(good),bool)),strict_oracle_removable=(r['strictR'].sum(1),np.ones(len(good),bool)),structural_candidates=(m['structural'].sum(1),np.ones(len(good),bool)),native_good=(good.astype(float),np.ones(len(good),bool)),GT_present_given_failure=(r['GT_mode_present'].astype(float),fail&defined),GT_absent_failure_share=((reason==0).astype(float),fail),GT_present_failure_share=((reason==1).astype(float),fail),GT_undefined_failure_share=((reason==2).astype(float),fail),GT_boundary_failure_share=((reason==3).astype(float),fail))
            for group,mask in group_masks(x):
                row=dict(population=pop,variant=name,group=group,N=int(mask.sum()),tracks=len(np.unique(x['track_key'][mask])),native_good=int(good[mask].sum()),native_fail=int(fail[mask].sum()))
                for metric,(value,denom) in metrics.items():
                    active=mask&denom;num=np.bincount(ix,weights=value*active,minlength=len(tracks));den=np.bincount(ix,weights=active,minlength=len(tracks));ds=count@den;ns=count@num;v=np.divide(ns,ds,out=np.full(len(ds),np.nan),where=ds>0);ok=np.isfinite(v);quant=np.quantile(v[ok],[.025,.975]) if ok.any() else [None,None]
                    row[metric]=float(value[active].mean()) if active.any() else None;row[metric+'_num']=float((value*active).sum());row[metric+'_den']=int(active.sum());row[metric+'_CI95_low']=float(quant[0]) if ok.any() else None;row[metric+'_CI95_high']=float(quant[1]) if ok.any() else None;row[metric+'_invalid']=int((~ok).sum())
                for metric in ('Pmax','Meff','M','same_pair'):
                    v=m[metric][mask];row[metric+'_median']=float(np.median(v));row[metric+'_q05']=float(np.quantile(v,.05));row[metric+'_q95']=float(np.quantile(v,.95))
                if row['N']:
                    from scipy.stats import spearmanr
                    a=m['same_pair'][mask];b=r['R'].sum(1)[mask];row['Spearman_structural_same_vs_R']=float(spearmanr(a,b).statistic) if np.std(a)>0 and np.std(b)>0 else None
                allrows.append(row)
            # Scene/sequence/whole-track summaries, not pseudo-independent scene CI.
            for scope,key in (('scene','scene'),('sequence','sequence'),('track','track_key')):
                for label in np.unique(x[key]):
                    mask=x[key]==label
                    strata.append(dict(population=pop,variant=name,scope=scope,label=str(label),N=int(mask.sum()),tracks=len(np.unique(x['track_key'][mask])),native_good=int(good[mask].sum()),Pmax=float(m['Pmax'][mask].mean()),Meff=float(m['Meff'][mask].mean()),same_pair=float(m['same_pair'][mask].mean()),oracle_removable=float(r['R'][mask].sum(1).mean()),native_fail=int(fail[mask].sum()),absent_fail=int((fail&mask&(reason==0)).sum()),present_fail=int((fail&mask&(reason==1)).sum()),undefined_fail=int((fail&mask&(reason==2)).sum()),boundary_fail=int((fail&mask&(reason==3)).sum())))
            g.check()
        mark('WHOLE_TRACK_DESCRIPTIVE_BOOTSTRAP',population=pop,draws=20000);print('STATS',pop,flush=True)
    table(HERE/'statistics/group_statistics.csv',allrows);write(HERE/'statistics/GROUP_STATISTICS.json',allrows);table(HERE/'statistics/scene_sequence_track.csv',strata);write(HERE/'statistics/BOOTSTRAP_CONTRACT.json',dict(records=draw_receipts,unit='whole sequence|target track',seed=20261027,windowweighted=True,zero_denominator_not_redrawn=True,descriptive_no_significance_claim=True,no_native_seed_replication=True,scenes_not_independent_replications=True));resource(HERE/'statistics/RESOURCE.json',g);finish(HERE/'statistics');print('PASS STATS',flush=True)
if __name__=='__main__':summaries()
