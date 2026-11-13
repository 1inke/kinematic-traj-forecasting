from pathlib import Path
import sys
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from common import *
from morphology import assign,variants
from oracle import scalar_check
def main():
    g=guard(True);source=verify_sources();cfg=read(HERE/'config/analysis_config.json');verify_done(HERE/'E1_mode_distribution');verify_done(HERE/'E2_oracle_coverage');verify_done(HERE/'statistics');rows=read(HERE/'statistics/GROUP_STATISTICS.json');records=[];ci_max=0.;allslots=0;versions={}
    import scipy,matplotlib
    versions.update(numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__)
    for split in SPLITS:
        x=load(HERE/f'inputs/X_{split}.npz');gt=load(HERE/f'inputs/GT_{split}.npz');q=load(HERE/f'E2_oracle_coverage/{split}/candidate_gt_metrics.npz');assert np.array_equal(x['sample_id'],q['sample_id']);aa,ff,jj,gg=quality(x['native'],gt['gt'])
        for key,v in [('ADE',aa),('FDE',ff),('J',jj)]:assert np.max(abs(q[key]-v))<=2e-12
        assert np.array_equal(q['good'],gg)
        scalar_check(aa,ff,jj,gg,q);allslots+=20*len(aa)
        assert (q['strict_gt_removable'].sum(1)>=17).all()
        for variant in variants():
            name=variant[0];m=load(HERE/f'E1_mode_distribution/{split}/{name}/modes.npz');r=load(HERE/f'E2_oracle_coverage/{split}/{name}/slot_removal_oracle.npz')
            assert np.array_equal(m['sample_id'],x['sample_id']) and np.array_equal(r['sample_id'],x['sample_id'])
            assert np.array_equal(r['R'],m['structural']&q['tol_gt_removable']) and np.array_equal(r['strictR'],m['structural']&q['strict_gt_removable'])
            assert np.array_equal(m['structural'].sum(1),20-m['M'])
            for group,mask in group_masks(x):
                z=next(a for a in rows if a['population']==split and a['variant']==name and a['group']==group)
                assert z['N']==int(mask.sum()) and z['native_good']==int(gg[mask].any(1).sum())
                if variant[-1] is None:
                    fail=~gg.any(1)&mask
                    assert sum(int((fail&(r['failure_reason']==k)).sum()) for k in range(4))==int(fail.sum())
            records.append(dict(population=split,variant=name,n=len(aa),IDs_masks_R_strict_vs_tol_ALLPASS=True))
        g.check();print('REVIEW ALL_SLOTS',split,flush=True)
    # Saved20k whole-track draws: independent gather reductions, not producer count@sum.
    bootstrap_records=[]
    for pop in ('hold','inner'):
        x=load(HERE/f'inputs/X_{pop}.npz');bs=load(HERE/f'statistics/DRAWS_{pop}.npz');tracks=bs['tracks'];ix=np.searchsorted(tracks,x['track_key'])
        for name in ('coarse20','joint20','coarse15','coarse30','floor_half','floor_double','cluster_half','cluster_main','cluster_double'):
            m=load(HERE/f'E1_mode_distribution/{pop}/{name}/modes.npz');r=load(HERE/f'E2_oracle_coverage/{pop}/{name}/slot_removal_oracle.npz')
            for group,mask in group_masks(x):
                row=next(a for a in rows if a['population']==pop and a['variant']==name and a['group']==group)
                vals=np.stack([m['Pmax'],m['Meff'],m['same_pair'],r['R'].sum(1)],axis=1);num=np.array([vals[(ix==k)&mask].sum(0) for k in range(len(tracks))]);den=np.array([np.sum((ix==k)&mask) for k in range(len(tracks))]);draw=bs['draws'];ds=den[draw].sum(1);ns=np.empty((len(draw),4))
                for start in range(0,len(draw),128):ns[start:start+128]=num[draw[start:start+128]].sum(1)
                v=np.divide(ns,ds[:,None],out=np.full_like(ns,np.nan),where=ds[:,None]>0);ok=ds>0
                for k,metric in enumerate(('Pmax','Meff','same_pair','oracle_removable')):
                    ci=np.quantile(v[ok,k],[.025,.975]);expected=np.array([row[metric+'_CI95_low'],row[metric+'_CI95_high']]);error=float(abs(ci-expected).max());ci_max=max(ci_max,error);assert error<=2e-12 and int((~ok).sum())==row[metric+'_invalid']
                bootstrap_records.append(dict(population=pop,variant=name,group=group,all20000_reductions_checked=True))
            g.check()
        print('REVIEW 20K',pop,flush=True)
    # Actual invariance to irrelevant GT in dict: assign() sees only legal inputs.
    x=load(HERE/'inputs/X_opt.npz');sm={k:v[:128] for k,v in x.items()};a=assign({**sm,'GT_unused':np.zeros((128,12,2))},cfg,variants()[0]);b=assign({**sm,'GT_unused':np.full((128,12,2),999.)},cfg,variants()[0]);assert np.array_equal(a['mode'],b['mode']) and np.array_equal(a['Meff'],b['Meff'])
    assert verify_sources()==source and 'torch' not in sys.modules
    resource(BASE/'completion_review/RESOURCE.json',g);write(BASE/'completion_review/REVIEW.json',dict(status='PASS_CPU_COMPLETE_SAVED_REVIEW',source=source,versions=versions,all_single_deletions_scalar_checked=allslots,all_variants_populations=records,bootstrap_full20000_records=bootstrap_records,bootstrap_max_error=ci_max,mode_invariance_to_unused_GT_EXACT=True,E1_lock_SHA=sha(HERE/'E1_LOCK.json'),source_initial_final_SHA_EXACT=True,no_training_or_prediction_model=True,GT_oracle_not_safe_deletion_policy=True,no_external_review=True,no_delete=True))
    print('PASS FULL REVIEW',flush=True)
if __name__=='__main__':main()
