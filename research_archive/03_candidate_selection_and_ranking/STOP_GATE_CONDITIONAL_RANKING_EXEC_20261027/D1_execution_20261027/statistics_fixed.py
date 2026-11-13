"""Exactly two primary coverage comparisons; all actual draws rechecked."""
from core import *
def draw_summary(v,seed_draw):
    valid=np.isfinite(v).all(1);mean=np.full(len(v),np.nan);mean[valid]=v[valid].mean(1)
    chosen=v[np.arange(len(v))[:,None],seed_draw];ok=np.isfinite(chosen).all(1);a=np.full(len(v),np.nan);a[ok]=chosen[ok].mean(1)
    def ci(x,good):return dict(valid=int(good.sum()),invalid=int((~good).sum()),CI95=np.quantile(x[good],[.025,.975]).tolist() if good.any() else None,CI97_5=np.quantile(x[good],[.0125,.9875]).tolist() if good.any() else None)
    return dict(paired_seed=ci(a,ok),fixed_seed_sensitivity=ci(mean,valid),per_seed=[ci(v[:,i],np.isfinite(v[:,i])) for i in range(3)]),a,mean
def main():
    g=guard('CPU');records={};data={};gate_records=[]
    for seed in SEEDS:
        data[seed]={a:load_npz(HERE/f'populations/seed{seed}/hold/audit/{a}_sample_metrics.npz') for a in ARMS}
    base=data[42]['D'];tracks=np.unique(base['track_key']);ix=np.searchsorted(tracks,base['track_key']);change=base['group']==2
    for sd in data.values():
        for d in sd.values():assert np.array_equal(d['sample_id'],base['sample_id']) and np.array_equal(d['track_key'],base['track_key'])
    change_tracks=len(np.unique(base['track_key'][change]));rng=np.random.default_rng(20261027);draws=rng.integers(len(tracks),size=(20000,len(tracks)),dtype=np.int32);sd=rng.integers(3,size=(20000,3),dtype=np.int32);counts=np.stack([np.bincount(x,minlength=len(tracks)) for x in draws]);den=np.bincount(ix,weights=change,minlength=len(tracks));sden=den[draws].sum(1);assert np.array_equal(sden,counts@den)
    arrays=dict(track_draws=draws,paired_seed_draws=sd,track_keys=tracks,window_ID=base['sample_id'],window_track=base['track_key'],change=change)
    for key in ('D-S','D-native'):
        deltas=[]
        for seed in SEEDS:
            d=data[seed]['D'];ref=data[seed]['S']['good'] if key=='D-S' else d['native_good'];deltas.append(d['good'].astype(np.int64)-ref.astype(np.int64))
        num=np.array([np.bincount(ix,weights=x*change,minlength=len(tracks)) for x in deltas]);drawnum=num[:,draws].sum(-1).T;assert np.array_equal(drawnum,counts@num.T)
        values=np.full((20000,3),np.nan);np.divide(drawnum,sden[:,None],out=values,where=sden[:,None]>0);report,a,b=draw_summary(values,sd);points=np.array([x[change].mean() for x in deltas]);signal=bool((points>0).all() and points.mean()>=.02 and change_tracks>=20 and report['paired_seed']['invalid']<=1000 and report['fixed_seed_sensitivity']['invalid']<=1000 and report['paired_seed']['CI97_5'][0]>0 and report['fixed_seed_sensitivity']['CI97_5'][0]>0)
        records[key]=dict(points=points.tolist(),mean=float(points.mean()),change_windows=int(change.sum()),change_tracks=change_tracks,signal=signal,**report);arrays[key+'_values']=values;arrays[key+'_paired']=a;arrays[key+'_fixed']=b;g.check()
    safe=True;secondary_inner=True;major_perseed=True
    for seed in SEEDS:
        for split in ('hold','inner'):
            rows=read_json(HERE/f'populations/seed{seed}/{split}/audit/D_METRICS.json')
            for name,r in rows.items():
                if not r['n']:gate_records.append(dict(seed=seed,split=split,group=name,status='NA_EMPTY_GROUP'));safe=False;continue
                lost=None if r['native_good']==0 else bool(r['lost']/r['native_good']<=.01)
                protected=bool(r['ADE']<=1.01*r['native_ADE'] and r['FDE']<=1.01*r['native_FDE']) if name!='change' else None
                motion=bool(r['motion_pass']);gate_records.append(dict(seed=seed,split=split,group=name,n=r['n'],native_good=r['native_good'],lost=r['lost'],lost_pass=lost,protected_pass=protected,motion_pass=motion))
                safe=safe and lost is True and motion and (protected is not False)
            if split=='inner':secondary_inner=secondary_inner and rows['change']['good']>=rows['change']['native_good']
            else:major_perseed=major_perseed and rows['change']['ADE']<=.95*rows['change']['native_ADE'] and rows['change']['FDE']<=.95*rows['change']['native_FDE']
    out=HERE/'statistics';out.mkdir(exist_ok=True);save_npz(out/'BOOTSTRAP_DRAWS.npz',**arrays)
    report=dict(status='PASS_ALL_PRIMARY_ACTUAL_DRAWS_INDEPENDENT',source=source0(),records=records,D_protection_all_PASS=bool(safe),inner_change_nondecrease_allseed=bool(secondary_inner),signal=bool(all(x['signal'] for x in records.values()) and safe and secondary_inner),protection_records=gate_records,major_change_ADE_FDE_5percent_eachseed=bool(major_perseed),major_overall_status='NOT_ESTABLISHED_COST_AND_CORRELATION_REVIEW_UNRESOLVED',cost='Inherited397100_parameters_exceeds305000_and_CPU_parity_blocked_NOT_SOLVED',family=2,all20000_independent_track_count_EXACT=True,no_invalid_redraw=True,not_SOTA=True)
    write_json(out/'PRIMARY.json',report);resource(out/'RESOURCE.json',g);finish(out,dict(source=source0(),draws=20000,rng=20261027));print('PASS PRIMARY',report['signal'],flush=True);return report
if __name__=='__main__':main()
