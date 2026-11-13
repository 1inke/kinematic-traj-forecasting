"""Per-query means, fixed-library track bootstrap. No extra main tests."""
from core import *
METRICS=('regret','conflict','overlap','future_mean_distance')
def summarize_values(values,draws):
    a=np.asarray(values,np.float64);finite=np.isfinite(a);answers=np.full((len(draws),3),np.nan)
    for i,ix in enumerate(draws):
        sampled=a[:,ix];valid=np.isfinite(sampled);den=valid.sum(1);num=np.where(valid,sampled,0).sum(1)
        np.divide(num,den,out=answers[i],where=den>0)
        # Independently expand track multiplicity rather than treating pairs as replicates.
        counts=np.bincount(ix,minlength=a.shape[1]);den2=finite@counts;num2=np.where(finite,a,0)@counts
        other=np.full(3,np.nan);np.divide(num2,den2,out=other,where=den2>0)
        assert np.array_equal(np.isnan(other),np.isnan(answers[i]))
        assert np.max(abs(other[np.isfinite(other)]-answers[i,np.isfinite(other)]),initial=0)<=2e-12
    points=[];perseed=[]
    for k in range(3):
        v=a[k,finite[k]];points.append(float(v.mean()) if len(v) else None)
        ok=np.isfinite(answers[:,k]);perseed.append(dict(valid_draws=int(ok.sum()),invalid_draws=int((~ok).sum()),CI95=np.quantile(answers[ok,k],[.025,.975]).tolist() if ok.any() else None,n_query_tracks=int(finite[k].sum())))
    allvalid=np.isfinite(answers).all(1);combined=np.full(len(draws),np.nan);combined[allvalid]=answers[allvalid].mean(1)
    summary=dict(point_per_seed=points,seed_records=perseed,mean_fixed_three_seeds=float(np.mean(points)) if all(v is not None for v in points) else None,
        combined_valid=int(allvalid.sum()),combined_invalid=int((~allvalid).sum()),combined_CI95=np.quantile(combined[allvalid],[.025,.975]).tolist() if allvalid.any() else None)
    return summary,answers,combined
def main():
    g=guard();identity=load_npz(HERE/'prepared/identity_hold.npz');rows=identity['reference_rows'];ids=identity['sample_id'][rows];tracks=identity['track_key'][rows]
    with np.load(TWELVE/'prepared/observations_hold.npz') as z:assert np.array_equal(z['sample_id'],identity['sample_id'])
    labels=old_columns(42,'hold',('sample_id','group'));assert np.array_equal(labels['sample_id'],identity['sample_id']);group=labels['group'][rows]
    assert len(set(tracks))==len(tracks);rng=np.random.default_rng(20261026);draws=rng.integers(len(rows),size=(20000,len(rows)),dtype=np.int32)
    by={mode:{k:np.full((3,len(rows)),np.nan) for k in METRICS} for mode in ('primary','reverse')};support=[];pair_counts=[]
    for si,seed in enumerate(SEEDS):
        records=read_json(HERE/f'posthoc/seed{seed}/PAIRS.json');selection=read_json(HERE/f'selections/seed{seed}/SELECTION.json')
        sensitivity=read_json(HERE/f'posthoc/seed{seed}/ALIGNMENT_SENSITIVITY.json')
        for mode in ('primary','reverse'):
            supported=np.zeros(len(rows),bool);comparable=np.zeros(len(rows),bool)
            for qi,sid in enumerate(ids):
                pp=[r for r in records if r['mode']==mode and r['query_ID']==int(sid)];supported[qi]=bool(pp)
                for k in METRICS:
                    valid=[float(r[k]) for r in pp if r[k] is not None]
                    if valid:by[mode][k][si,qi]=float(np.mean(valid))
                comparable[qi]=any(r['both_positive'] for r in pp)
            cal=selection[mode]['calibration_valid'];overall=int(supported.sum());change=int((supported&(group==2)).sum())
            support.append(dict(seed=seed,mode=mode,calibration_valid=cal,overall_supported_tracks=overall,change_supported_tracks=change,
                overall_status='ESTIMABLE_DESCRIPTIVE_ONLY' if cal>=50 and overall>=50 else 'UNDERPOWERED',
                change_status='ESTIMABLE_DESCRIPTIVE_ONLY' if cal>=50 and change>=20 else 'UNDERPOWERED',
                comparable_both_positive_query_tracks=int(comparable.sum()),comparable_both_positive_change_tracks=int((comparable&(group==2)).sum()),
                tau=selection[mode]['tau'],all_queries=len(rows),unknown_or_unsupported_queries=int((~supported).sum())))
            pp=[r for r in records if r['mode']==mode]
            events={e:sum(r['event']==e for r in pp) for e in ('CONFLICT','COMMON_POSITIVE','BOTH_ORIGINAL_NO_OPPORTUNITY','ONE_ORIGINAL_NO_OPPORTUNITY','COMMON_DOMAIN_OPPORTUNITY_REMOVAL')}
            pair_counts.append(dict(seed=seed,mode=mode,pairs=len(pp),both_positive_pairs=sum(r['both_positive'] for r in pp),events=events,
                numerically_ambiguous_pairs=sum(any(r['numerically_ambiguous']) for r in pp),
                primary_pair_alignment_sensitive=sum(r['status']=='ALIGNMENT_SENSITIVE' for r in sensitivity) if mode=='primary' else None,
                primary_vs_reverse_selected_pair_symmetric_difference=len(set((r['query_ID'],r['library_ID']) for r in records if r['mode']=='primary')^set((r['query_ID'],r['library_ID']) for r in records if r['mode']=='reverse'))))
    results={};arrays=dict(track_draws=draws,query_ID=ids,query_track=tracks,group=group)
    for mode in by:
        for subgroup in ('overall','change'):
            for metric,values in by[mode].items():
                v=values.copy()
                if subgroup=='change':v[:,group!=2]=np.nan
                key=f'{mode}_{subgroup}_{metric}';r,a,b=summarize_values(v,draws);results[key]=r
                arrays['perquery_'+key]=v;arrays['draws_'+key]=a;arrays['combined_'+key]=b;g.check()
    out=HERE/'statistics';out.mkdir()
    save_npz(out/'BOOTSTRAP_DRAWS.npz',**arrays)
    write_json(out/'SUMMARY.json',dict(status='COMPLETE_DESCRIPTIVE_REVIEW_REQUIRED',support=support,pair_counts=pair_counts,records=results,
        fixed_library_only_not_library_uncertainty=True,query_mean_then_track=True,pairs_not_independent=True,family6_not_expanded=True,
        all_20000_actual_draws_independently_recomputed=True,no_invalid_redraw=True,not_Bayes_bound_or_entropy=True))
    resource(out/'RESOURCE.json',g);finish(out,dict(source=source(),draws=20000,rng=20261026,unit='fixed_query_tracks'))
    print('PASS query-track bootstrap independent all20000; support',support,flush=True)
if __name__=='__main__':main()
