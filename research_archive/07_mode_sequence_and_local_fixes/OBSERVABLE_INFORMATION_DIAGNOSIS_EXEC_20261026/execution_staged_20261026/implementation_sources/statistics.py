"""Shared whole-track/paired-seed draws; family6 including blocked/not-run tests."""
from base import *

def bootstrap(arrays,groups,tracks,draws=20000,rngseed=20261026):
    unique=np.unique(tracks);idx={t:i for i,t in enumerate(unique)};change=groups==2;den=np.bincount([idx[t] for t in tracks[change]],minlength=len(unique)).astype(np.float64)
    sums={name:np.stack([np.bincount([idx[t] for t in tracks[change]],weights=row[change],minlength=len(unique)) for row in values]) for name,values in arrays.items()}
    rng=np.random.default_rng(rngseed);trackdraw=np.empty((draws,len(unique)),np.int32);seeddraw=np.empty((draws,3),np.int8);answers={k:np.full(draws,np.nan) for k in sums};fixed={k:np.full(draws,np.nan) for k in sums}
    for i in range(draws):
        td=rng.integers(len(unique),size=len(unique));sd=rng.integers(3,size=3);trackdraw[i]=td;seeddraw[i]=sd;d=den[td].sum()
        if d:
            for k,v in sums.items():answers[k][i]=np.mean(v[sd][:,td].sum(1)/d);fixed[k][i]=np.mean(v[:,td].sum(1)/d)
    # Independent weighted expanded-track count arithmetic for EVERY saved draw.
    for i in range(draws):
        count=np.bincount(trackdraw[i],minlength=len(unique));d=float(count@den)
        for k,v in sums.items():
            if d:
                per=(v@count)/d
                assert abs(per[seeddraw[i]].mean()-answers[k][i])<=2e-12 and abs(per.mean()-fixed[k][i])<=2e-12
            else:assert np.isnan(answers[k][i]) and np.isnan(fixed[k][i])
    records={}
    for k in sums:
        v=answers[k];f=fixed[k];valid=np.isfinite(v);enough=int((den>0).sum())>=20 and valid.mean()>=.95
        records[k]=dict(point_per_seed=arrays[k][:,change].mean(1).tolist(),mean=float(arrays[k][:,change].mean()),valid=int(valid.sum()),invalid=int((~valid).sum()),change_tracks=int((den>0).sum()),status='ESTIMABLE' if enough else 'UNDERPOWERED',CI95=np.quantile(v[valid],[.025,.975]).tolist() if valid.any() else None,CI_family6=np.quantile(v[valid],[1/240,239/240]).tolist() if valid.any() else None,fixed_seed_CI_family6=np.quantile(f[valid],[1/240,239/240]).tolist() if valid.any() else None)
    return records,dict(track_draws=trackdraw,seed_draws=seeddraw,tracks=unique,**{f'values_{k}':v for k,v in answers.items()},**{f'fixed_{k}':v for k,v in fixed.items()})

def main():
    g=guard('AUDIT');out=HERE/'statistics';out.mkdir(exist_ok=True);c=dict(source=source(),family=6,draws=20000,rng=20261026)
    if (out/'done.json').exists():return completed(out,c)
    arrays={};native=[];group=None;tracks=None
    for arm in ('M_full','T_full','T_geom'):
        rows=[]
        for seed in SEEDS:
            with np.load(HERE/f'populations/{arm}_seed{seed}/hold/audit/sample_metrics.npz') as z:
                rows.append(z['good'].astype(np.float64))
                if group is None:group=z['group'];tracks=z['track_key'];ids=z['sample_id']
                else:assert np.array_equal(ids,z['sample_id']) and np.array_equal(tracks,z['track_key']) and np.array_equal(group,z['group'])
                if arm=='M_full':native.append(z['native_good'].astype(np.float64))
        arrays[arm]=np.stack(rows)
    native=np.stack(native);diff=dict(Tfull_minus_M=arrays['T_full']-arrays['M_full'],Tfull_minus_geom=arrays['T_full']-arrays['T_geom'],Tfull_minus_native=arrays['T_full']-native,Tgeom_minus_native=arrays['T_geom']-native,M_minus_native=arrays['M_full']-native)
    records,draw=bootstrap(diff,group,tracks);g.check();save_npz(out/'BOOTSTRAP_DRAWS.npz',**draw)
    for k,r in records.items():r['joint_positive_gate']=bool(r['status']=='ESTIMABLE' and min(r['point_per_seed'])>=0 and r['mean']>=.02 and r['CI_family6'][0]>0 and r['fixed_seed_CI_family6'][0]>0)
    signal=records['Tfull_minus_M']['joint_positive_gate'] and records['Tfull_minus_native']['joint_positive_gate']
    write_json(out/'PRIMARY_COMPARISONS.json',dict(records=records,family_fixed=6,primary_available=['Tfull_minus_M','Tfull_minus_geom'],Tfull_minus_perm='BLOCKED_NO_LEGAL_LAYER_SHIFT',step3_comparisons='NOT_AUTHORIZED_NOT_RUN',learning_scheme_signal=signal,observation_increment_signal='UNRESOLVED_NEGATIVE_CONTROL_BLOCKED',native_reference_not_independent3models=True,all_20000_draws_independently_recomputed=True,no_invalid_redraw=True));resource_report(out,g);finish(out,c)

if __name__=='__main__':
    try:main()
    except BaseException as e:fail(e);raise


