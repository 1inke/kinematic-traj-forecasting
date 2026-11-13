from common import *
from morphology import features,categories,variants
from oracle import remove_metrics,scalar_check
def run():
    g=guard(True);cfg=read(HERE/'config/analysis_config.json');lk=read(HERE/'E1_LOCK.json');assert sha(HERE/'E1_mode_distribution/done.json')==lk['done_SHA'];verify_done(HERE/'E1_mode_distribution');assert sha(HERE/'config/analysis_config.json')==lk['config_SHA']
    out=HERE/'E2_oracle_coverage';summary=[];contributions=[]
    for split in SPLITS:
        x=load(HERE/f'inputs/X_{split}.npz');gt=load(HERE/f'inputs/GT_{split}.npz');assert np.array_equal(gt['sample_id'],x['sample_id']);a,f,j,good=quality(x['native'],gt['gt']);rem=remove_metrics(a,f,j,good);scalar_check(a[:128],f[:128],j[:128],good[:128],{k:v[:128] for k,v in rem.items()})
        save(out/split/'candidate_gt_metrics.npz',sample_id=x['sample_id'],slot=np.arange(20),ADE=a,FDE=f,J=j,good=good,**rem)
        for variant in variants():
            name,theta,mult,joint,cluster=variant;m=load(HERE/f'E1_mode_distribution/{split}/{name}/modes.npz');assert np.array_equal(m['sample_id'],x['sample_id']);R=m['structural']&rem['tol_gt_removable'];strict=m['structural']&rem['strict_gt_removable']
            if cluster is None:
                gf=features(gt['gt'][:,None],x['past'],cfg['slow_floor']*mult);gm,_=categories(gf,theta,cfg['slow_floor']*mult,joint);gm=gm[:,0]
                same=m['mode']==gm[:,None];present=same.any(1);hit=(same&good).any(1);cross=(~same&good).any(1)
                near_angle=gf['turn_defined'][:,0]&(abs(abs(np.rad2deg(gf['turn'][:,0]))-theta)<=1)
                near_speed=abs(gf['mean_speed'][:,0]-cfg['slow_floor']*mult)<=.05*cfg['slow_floor']*mult
                boundary=near_angle|near_speed;undef=gm==1
                phi=(m['phi']-np.asarray(cfg['phi_mean']))/np.asarray(cfg['phi_std']);yp=(gf['phi']-np.asarray(cfg['phi_mean']))/np.asarray(cfg['phi_std']);distance=np.linalg.norm(phi-yp,axis=-1);nearest=distance.argmin(1)
            else:gm=np.full(len(a),-1);same=np.zeros_like(good);present=np.zeros(len(a),bool);hit=present.copy();cross=present.copy();boundary=present.copy();undef=np.ones(len(a),bool);nearest=np.full(len(a),-1);distance=np.full_like(a,np.nan)
            reason=np.where(undef,2,np.where(boundary,3,np.where(present,1,0))).astype(np.int16)
            save(out/split/name/'slot_removal_oracle.npz',sample_id=x['sample_id'],R=R,strictR=strict,GT_mode=gm,GT_mode_present=present,GT_mode_count=same.sum(1),GT_mode_hit=hit,cross_mode_hit=cross,nearest_GT_slot=nearest,nearest_GT_distance=np.nanmin(distance,axis=1) if cluster is None else np.full(len(a),np.nan),boundary=boundary,undefined=undef,failure_reason=reason,good_count=good.sum(1),native_good=good.any(1))
            for i,sid in enumerate(x['sample_id']):
                summary.append(dict(population=split,variant=name,sample_id=int(sid),track=x['track_key'][i],scene=x['scene'][i],sequence=x['sequence'][i],group=int(x['group'][i]),native_good=bool(good[i].any()),J=float(j[i].min()),ADE=float(a[i].min()),FDE=float(f[i].min()),GT_mode=int(gm[i]),mode_present=bool(present[i]),mode_hit=bool(hit[i]),cross_mode_hit=bool(cross[i]),boundary=bool(boundary[i]),undefined=bool(undef[i]),failure_reason=int(reason[i]),good_slots=int(good[i].sum()),important_slots=int((~rem['tol_gt_removable'][i]).sum()),gt_removable=int(rem['tol_gt_removable'][i].sum()),structural=int(m['structural'][i].sum()),oracle_removable=int(R[i].sum()),strict_oracle_removable=int(strict[i].sum())))
            for label in range(11 if joint else (20 if cluster is not None else 5)):
                mode=m['mode']==label;has=mode.any(1);hm=(mode&good).any(1);best=np.min(np.where(mode,j,np.inf),axis=1)
                for group,mask in group_masks(x):
                    denomin=int(mask.sum());exists=int((has&mask).sum());hcount=int((hm&mask).sum());vals=best[has&mask]
                    contributions.append(dict(population=split,variant=name,mode=label,group=group,N=denomin,present_windows=exists,hit_windows=hcount,mean_slots=float(mode[mask].sum(1).mean()),hit_rate_all=hcount/denomin,hit_rate_given_present=hcount/exists if exists else None,J_q25=float(np.quantile(vals,.25)) if len(vals) else None,J_median=float(np.median(vals)) if len(vals) else None,J_q75=float(np.quantile(vals,.75)) if len(vals) else None,winJ_slots=int((rem['winJ']&mode&mask[:,None]).sum()),winA_slots=int((rem['winA']&mode&mask[:,None]).sum()),winF_slots=int((rem['winF']&mode&mask[:,None]).sum())))
            g.check()
        mark('E2_ALL_SLOTS_ORACLE',population=split,windows=len(a),arithmetic_deletions=20*len(a));print('E2',split,flush=True)
    table(out/'gt_mode_presence.csv',summary);table(out/'mode_contribution.csv',contributions);resource(out/'RESOURCE.json',g);finish(out,E1_lock_SHA=sha(HERE/'E1_LOCK.json'));print('PASS E2',flush=True)
if __name__=='__main__':run()
