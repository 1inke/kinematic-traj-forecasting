"""Paired all-inner-track and seed bootstrap; no stdlib statistics name shadowing."""
from core import *

def bootstrap(arrays,tracks,groups,draws=20000,rng_seed=20261023,save=None):
    unique,inv=np.unique(tracks,return_inverse=True);nt=len(unique);rng=np.random.default_rng(rng_seed)
    td=np.empty((draws,nt),np.int16);sd=np.empty((draws,3),np.int16)
    for k in range(draws):td[k]=rng.integers(nt,size=nt);sd[k]=rng.integers(3,size=3)
    defs=[('primary','change','R','P-A')]
    defs += [(f'{comp}:{group}:{metric}',group,metric,comp) for comp in ('P-A','P-native') for group in ('overall','change') for metric in ('coverage','minADE','minFDE','reference_success_lost_rate')]
    points=[];suff=[];group_tracks=[]
    for name,group,metric,comp in defs:
        mask=np.ones(len(groups),bool) if group=='overall' else groups==2
        ref='A_replay' if comp=='P-A' else 'native';num=[];den=[]
        for seed in range(3):
            p=arrays['P_relative'][seed];a=arrays[ref][seed]
            if metric=='R':values=np.maximum(p['Jmin']-arrays['native'][seed]['Jmin'],0)-np.maximum(a['Jmin']-arrays['native'][seed]['Jmin'],0);dm=mask
            elif metric=='reference_success_lost_rate':values=(a['coverage']&~p['coverage']).astype(float);dm=mask&a['coverage']
            else:values=p[metric].astype(float)-a[metric].astype(float);dm=mask
            nn=np.bincount(inv,weights=np.where(dm,values,0),minlength=nt);dd=np.bincount(inv,weights=dm,minlength=nt)
            num.append(nn);den.append(dd)
        num=np.array(num);den=np.array(den);suff.append((num,den));points.append(float(np.mean(num.sum(1)/den.sum(1))) if np.all(den.sum(1)>0) else None);group_tracks.append(len(np.unique(tracks[mask])))
    boot=np.full((draws,len(defs)),np.nan);fixed=boot.copy()
    for start in range(0,draws,128):
        stop=min(draws,start+128);counts=np.array([np.bincount(v,minlength=nt) for v in td[start:stop]])
        for j,(num,den) in enumerate(suff):
            nn=counts@num.T;dd=counts@den.T;rat=np.divide(nn,dd,out=np.full(nn.shape,np.nan),where=dd>0)
            fixed[start:stop,j]=rat.mean(1);boot[start:stop,j]=np.take_along_axis(rat,sd[start:stop],1).mean(1)
    # Second implementation: directly expand sampled track indices, no count-matrix path.
    review_max=0.
    for j,(num,den) in enumerate(suff):
        nn=num[:,td].sum(-1).T;dd=den[:,td].sum(-1).T;rat=np.divide(nn,dd,out=np.full(nn.shape,np.nan),where=dd>0)
        fr=rat.mean(1);br=np.array([rat[k,sd[k]].mean() for k in range(draws)])
        assert np.array_equal(np.isnan(br),np.isnan(boot[:,j])) and np.array_equal(np.isnan(fr),np.isnan(fixed[:,j]))
        for aa,bb in ((br,boot[:,j]),(fr,fixed[:,j])):
            good=np.isfinite(aa);delta=float(np.max(np.abs(aa[good]-bb[good]))) if good.any() else 0.;review_max=max(review_max,delta);assert delta<=1e-10
    results=[]
    for j,(name,group,metric,comp) in enumerate(defs):
        row=dict(name=name,group=group,metric=metric,contrast=comp,point=points[j],tracks=group_tracks[j])
        for tag,bb in [('paired_track_seed',boot[:,j]),('fixed_seed_track_only',fixed[:,j])]:
            valid=np.isfinite(bb);ok=group_tracks[j]>=20 and float((~valid).mean())<=.05
            row[tag]=dict(status='SUFFICIENT' if ok else 'INSUFFICIENT',valid=int(valid.sum()),invalid=int((~valid).sum()),CI95=np.quantile(bb[valid],[.025,.975]).tolist() if valid.any() else None,CI_family16=np.quantile(bb[valid],[.0015625,.9984375]).tolist() if j and valid.any() else None)
        results.append(row)
    if save:np.savez_compressed(save,track_population=unique,track_draws=td,seed_draws=sd,paired_values=boot,fixed_seed_values=fixed)
    return dict(draws=draws,rng_seed=rng_seed,secondary_family_count=16,reference_success_rate_definition='P loses this comparison reference success, denominator that reference good; self-reference lost=0. Native-success P-A difference is separately reported by mechanism guards.',results=results,independent_expanded_track_review_maxerror=review_max)

def finish(results):
    arrays={a:[] for a in (*ARMS,'native')};d=load_data()['inner'];tracks=d['track'];groups=d['groups']
    for s in SEEDS:
        for arm in ARMS:
            with np.load(HERE/f'runs/{arm}_seed{s}/predictions_inner.npz') as z:
                assert np.array_equal(z['sample_id'],d['ids']);ee=errors(z['paths'],z['gt'])
            arrays[arm].append(dict(Jmin=ee['J'].min(1),coverage=ee['coverage'],minADE=ee['ADE'].min(1),minFDE=ee['FDE'].min(1)))
        ee=errors(d['native'],d['gt']);arrays['native'].append(dict(Jmin=ee['J'].min(1),coverage=ee['coverage'],minADE=ee['ADE'].min(1),minFDE=ee['FDE'].min(1)))
    boot=bootstrap(arrays,tracks,groups,save=HERE/'BOOTSTRAP_DRAWS.npz');write_json(HERE/'BOOTSTRAP.json',boot)
    rr={a:[] for a in ARMS};rows=[];baseline=[]
    for s in SEEDS:
        i=SEEDS.index(s);mask=groups==2
        for arm in ARMS:rr[arm].append(float(np.maximum(arrays[arm][i]['Jmin'][mask]-arrays['native'][i]['Jmin'][mask],0).mean()))
        for group in ('overall','change'):
            a=results[f'A_replay:{s}']['metrics']['inner'][group];p=results[f'P_relative:{s}']['metrics']['inner'][group];gain=a['good']-a['native_good']
            checks=dict(control_gain_positive=gain>0,retain90pct_net_gain=gain>0 and p['good']-p['native_good']>=.9*gain,lost_no_worse=p['lost']<=a['lost'],ADE_no_worse_1pct=p['ADE']<=1.01*a['ADE'],FDE_no_worse_1pct=p['FDE']<=1.01*a['FDE'])
            rows.append(dict(seed=s,group=group,control=a,relative=p,checks=checks,passed=all(checks.values())))
        for split in ('FIT','inner'):
            nm=read_json(OLD/f'native_metrics_{split}.json')
            for group in ('overall','stable','transition','change'):
                p=results[f'P_relative:{s}']['metrics'][split][group];n=nm[group]
                checks=dict(coverage=p['coverage']>=.99*n['coverage'],ADE=p['ADE']<=1.01*n['ADE'],FDE=p['FDE']<=1.01*n['FDE'],**{k:p['motion'][k]<=1.01*n['motion'][k] for k in p['motion']})
                baseline.append(dict(seed=s,split=split,group=group,checks=checks,passed=all(checks.values())))
    primary=boot['results'][0];den=float(np.mean(rr['A_replay']));reduction=(den-float(np.mean(rr['P_relative'])))/den if den>0 else None
    mechanisms=dict(positive_regression_to_reduce=den>0,all_seed_R_lower=all(p<a for p,a in zip(rr['P_relative'],rr['A_replay'])),mean_R_reduction20pct=reduction is not None and reduction>=.2)
    for tag in ('paired_track_seed','fixed_seed_track_only'):mechanisms[tag]=primary[tag]['status']=='SUFFICIENT' and primary[tag]['CI95'][1]<0
    mechanisms['all_engineering_guards']=all(x['passed'] for x in rows)
    supported=all(mechanisms.values());ready=all(x['passed'] for x in baseline)
    status='BASELINE_READY_FOR_SEPARATE_COMPLEMENTARITY_DESIGN' if ready else 'MECHANISM_SUPPORTED_BASELINE_NOT_READY' if supported else 'COVERAGE_REGRESSION_TRADEOFF' if mechanisms['all_seed_R_lower'] and not all(x['checks']['retain90pct_net_gain'] for x in rows) else 'TRANSFER_UNRESOLVED'
    return dict(status=status,mechanism_supported=supported,checks=mechanisms,R=rr,relative_R_reduction=reduction,engineering_rows=rows,baseline_rows=baseline,baseline_ready=ready,primary=primary,automatic_next_training=False,major_status='NOT_ESTABLISHED')
