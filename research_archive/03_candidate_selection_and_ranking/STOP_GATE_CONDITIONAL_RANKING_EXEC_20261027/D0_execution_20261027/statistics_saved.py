from saved_score_audit import *
RATES={'top1_B':('best_B','B_nonempty'),'top5_B':('hitB5','B_nonempty'),'top10_B':('hitB10','B_nonempty'),'top1_rescue':('best_rescue','rescue_opportunity'),'top5_rescue':('hitRescue5','rescue_opportunity'),'top10_rescue':('hitRescue10','rescue_opportunity'),'blocked_good':('blocked_good','B_nonempty'),'blocked_rescue':('blocked_rescue','rescue_opportunity'),'internal_bad':('internal_bad','B_nonempty')}
def summarize_draws(x):
    valid=np.isfinite(x).all(1);m=np.full(len(x),np.nan);m[valid]=x[valid].mean(1)
    return dict(valid=int(valid.sum()),invalid=int((~valid).sum()),status='UNDERPOWERED' if (~valid).mean()>.05 else 'DESCRIPTIVE_ONLY',CI95=np.quantile(m[valid],[.025,.975]).tolist() if valid.any() else None)
def run(g):
    rng=np.random.default_rng(20261027);banks={};report=[]
    for split in SPLITS:
        example=load_npz(HERE/f'populations/{MODELS[0]}_seed42/{split}/PER_WINDOW_RANK_MARGIN.npz');tracks=np.unique(example['track_key']);ix=np.searchsorted(tracks,example['track_key'])
        draws=rng.integers(len(tracks),size=(20000,len(tracks)),dtype=np.int32);seeds=rng.integers(3,size=(20000,3),dtype=np.int32);counts=np.stack([np.bincount(x,minlength=len(tracks)) for x in draws])
        banks[split+'_tracks']=tracks;banks[split+'_draws']=draws;banks[split+'_paired_seed']=seeds
        for model in MODELS:
            allseed=[load_npz(HERE/f'populations/{model}_seed{s}/{split}/PER_WINDOW_RANK_MARGIN.npz') for s in SEEDS]
            assert all(np.array_equal(x['sample_id'],example['sample_id']) and np.array_equal(x['track_key'],example['track_key']) for x in allseed)
            for gi,group in enumerate(GROUPS):
                mask=np.ones(len(ix),bool) if gi==0 else example['group']==gi-1
                for metric,(num,den) in RATES.items():
                    tn=np.array([np.bincount(ix,weights=x[num]&x[den]&mask,minlength=len(tracks)) for x in allseed]);td=np.array([np.bincount(ix,weights=x[den]&mask,minlength=len(tracks)) for x in allseed])
                    # Production expands actual track draws; second implementation
                    # uses track multiplicity matrix, checking all20000 draws.
                    numerator=tn[:,draws].sum(-1).T;denom=td[:,draws].sum(-1).T
                    assert np.array_equal(numerator,counts@tn.T) and np.array_equal(denom,counts@td.T)
                    rates=np.full(numerator.shape,np.nan);np.divide(numerator,denom,out=rates,where=denom>0)
                    paired=rates[np.arange(20000)[:,None],seeds];fixed=summarize_draws(rates)
                    points=np.divide(tn.sum(1),td.sum(1),out=np.full(3,np.nan),where=td.sum(1)>0)
                    report.append(dict(model=model,split=split,group=group,metric=metric,points=[float(v) if np.isfinite(v) else None for v in points],denominators=td.sum(1).astype(int).tolist(),paired_seed=summarize_draws(paired),fixed_seed_track_only=fixed,all20000_independent_EXACT=True))
            g.check()
    save_npz(HERE/'BOOTSTRAP_DRAWS.npz',**banks);write_json(HERE/'BOOTSTRAP.json',dict(status='PASS_DESCRIPTIVE_ALL_ACTUAL_DRAWS_INDEPENDENT',records=report,invalid_no_redraw=True,no_confirmatory_D1_gate=True))
