from core import *
def ci(values):
    ok=np.isfinite(values);return dict(valid=int(ok.sum()),invalid=int((~ok).sum()),CI95=np.quantile(values[ok],[.025,.975]).tolist() if ok.any() else None,CI97_5=np.quantile(values[ok],[.0125,.9875]).tolist() if ok.any() else None)
def main():
    g=guard(True);f=HERE/'statistics';f.mkdir(exist_ok=True);data={seed:{a:load(HERE/f'populations/seed{seed}/hold/audit/{a}_metrics.npz') for a in ('B0',)+ARMS} for seed in SEEDS};base=data[42]['B2'];track=np.unique(base['track_key']);ix=np.searchsorted(track,base['track_key']);change=base['group']==2;ct=len(np.unique(base['track_key'][change]));rng=np.random.default_rng(20261027);draw=rng.integers(len(track),size=(20000,len(track)),dtype=np.int32);seed_draw=rng.integers(3,size=(20000,3),dtype=np.int32);count=np.stack([np.bincount(row,minlength=len(track)) for row in draw]);den=np.bincount(ix,weights=change,minlength=len(track));drawden=den[draw].sum(1);assert np.array_equal(drawden,count@den);records={};arrays=dict(track_draws=draw,seed_draws=seed_draw,track_keys=track,sample_id=base['sample_id'])
    for ref in ('B1','B0'):
        delta=[]
        for seed in SEEDS:
            a=data[seed]['B2'];b=data[seed][ref];assert np.array_equal(a['sample_id'],base['sample_id']) and np.array_equal(a['track_key'],base['track_key']) and np.array_equal(b['sample_id'],base['sample_id']);delta.append(a['coverage'].astype(int)-b['coverage'].astype(int))
        nums=np.array([np.bincount(ix,weights=d*change,minlength=len(track)) for d in delta]);n=nums[:,draw].sum(-1).T;assert np.array_equal(n,count@nums.T);v=np.full((20000,3),np.nan);np.divide(n,drawden[:,None],out=v,where=drawden[:,None]>0);fixed=v.mean(1);paired=v[np.arange(20000)[:,None],seed_draw].mean(1);point=np.array([d[change].mean() for d in delta]);p=ci(paired);sens=ci(fixed);sig=bool((point>0).all() and point.mean()>=.02 and ct>=20 and p['invalid']<=1000 and sens['invalid']<=1000 and p['CI97_5'][0]>0 and sens['CI97_5'][0]>0);key='B2-'+ref;records[key]=dict(point=point.tolist(),mean=float(point.mean()),change_n=int(change.sum()),change_tracks=ct,paired=p,fixed_seed=sens,per_seed=[ci(v[:,i]) for i in range(3)],signal=sig);arrays[key+'_values']=v;g.check()
    protections=[];safe=True;inner=True;major=True
    for seed in SEEDS:
        for split in ('hold','inner'):
            rows=read_json(HERE/f'populations/seed{seed}/{split}/audit/B2_GROUPS.json')
            for name,r in rows.items():
                lp=r['native_good']>0 and r['lost']/r['native_good']<=.01;ep=(r['ADE']<=1.01*r['native_ADE'] and r['FDE']<=1.01*r['native_FDE']) if name!='change' else True;mp=r['motion_pass'];safe=safe and lp and ep and mp;protections.append(dict(seed=seed,split=split,group=name,n=r['n'],lost=r['lost'],native_good=r['native_good'],lost_pass=lp,protected_errors_pass=ep,motion_pass=mp))
            if split=='inner':inner=inner and rows['change']['good']>=rows['change']['native_good']
            else:major=major and rows['change']['ADE']<=.95*rows['change']['native_ADE'] and rows['change']['FDE']<=.95*rows['change']['native_FDE']
    save(f/'BOOTSTRAP_DRAWS.npz',**arrays);write_json(f/'PRIMARY.json',dict(status='PASS_ALL20000_DRAWS_INDEPENDENT',source=source(),comparisons=records,primary_family=2,protection_all_PASS=bool(safe),protection_records=protections,inner_change_eachseed_nondecrease=bool(inner),mechanism_signal=bool(all(r['signal'] for r in records.values()) and safe and inner),major_eachseed_ADE_FDE5percent=bool(major),major_overall='NOT_ESTABLISHED_INDEPENDENCE_AND_COST_REVIEW_REQUIRED',fixed_library_and_three_seed_exploration=True,no_invalid_redraw=True));resource(f/'RESOURCE.json',g);finish(f,dict(source=source(),draws=20000));print('PASS PRIMARY',flush=True)
if __name__=='__main__':main()
