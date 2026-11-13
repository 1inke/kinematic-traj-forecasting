"""Fixed family30 across both experiments; shared track and paired-seed draws."""
from pathlib import Path
import numpy as np
from joint_common import read,write,protocol,verify

def values(p):
    verify(read(p/'done.json')['files'])
    with np.load(p/'window_metrics.npz',allow_pickle=False) as f:
        first=[f['coverage'],f['ADE'].min(1),f['FDE'].min(1)]
    with np.load(p/'auxiliary.npz',allow_pickle=False) as f:
        return np.stack(first+[f[k] for k in ('kin_valid_coverage','kin_violation_rate','effective_good')],-1).astype(float),f['ids'],f['groups']

def compare(execution):
    root=Path(execution);native,ids,g=values(root/'populations/native/dev');blocks={}
    for family in ('existing','cvae'):
        for arm in ('accuracy_only','accuracy_dpp'):
            rr=[]
            for seed in (42,43,44):
                v,ii,gg=values(root/'populations'/f'{family}_{arm}_seed{seed}'/'dev')
                if not np.array_equal(ids,ii) or not np.array_equal(g,gg):raise ValueError('PAIRED_IDS')
                rr.append(v)
            blocks[(family,arm)]=np.stack(rr)
    ea=blocks['existing','accuracy_only'];ej=blocks['existing','accuracy_dpp']
    ca=blocks['cvae','accuracy_only'];cj=blocks['cvae','accuracy_dpp']
    v=np.stack([ej-ea,ej-native[None],cj-ca,cj-native[None],cj-ej])
    tracks=np.array(['|'.join(str(x).split('|')[:2]) for x in ids]);names=np.unique(tracks)
    ti=np.searchsorted(names,tracks);mask=g==2;count=np.bincount(ti,weights=mask,minlength=len(names))
    if len(np.unique(tracks[mask]))<20:
        write(root/'PRIMARY_COMPARISON.json',dict(status='UNDERPOWERED_CHANGE_TRACKS',family=30));return
    sums=np.zeros((5,3,len(names),6))
    for c in range(5):
        for s in range(3):
            for m in range(6):sums[c,s,:,m]=np.bincount(ti,weights=v[c,s,:,m]*mask,minlength=len(names))
    rng=np.random.default_rng(20261106);draw=np.full((20000,5,6),np.nan);fixed=np.full_like(draw,np.nan)
    tdall=np.empty((20000,len(names)),np.int32);sdall=np.empty((20000,3),np.int8);maxerr=0.
    for i in range(20000):
        td=rng.integers(len(names),size=len(names));sd=rng.integers(3,size=3);tdall[i]=td;sdall[i]=sd
        den=count[td].sum()
        if not den:continue
        res=sums[:,:,td].sum(2)/den;draw[i]=res[:,sd].mean(1);fixed[i]=res.mean(1)
        w=np.bincount(td,minlength=len(names));second=np.einsum('cstm,t->csm',sums,w)/np.dot(w,count)
        maxerr=max(maxerr,float(np.max(abs(second[:,sd].mean(1)-draw[i]))))
    if maxerr>2e-12:raise ValueError('BOOTSTRAP_SECOND_ARITHMETIC')
    np.savez_compressed(root/'BOOTSTRAP_DRAWS.npz',draw=draw,fixed=fixed,track=tdall,seed=sdall,names=names)
    rows=[];spec=protocol()['statistics']
    for c,label in enumerate(spec['comparisons']):
        for m,metric in enumerate(spec['metrics']):
            x=draw[:,c,m];xf=fixed[:,c,m];ok=np.isfinite(x);point=v[c][:,mask,m].mean(1)
            quant=lambda a,q:np.quantile(a[np.isfinite(a)],q).tolist() if np.isfinite(a).any() else None
            rows.append(dict(comparison=label,metric=metric,per_seed=point.tolist(),mean=float(point.mean()),
                ordinary95=quant(x,spec['ordinary']),family30=quant(x,spec['corrected']),
                fixedseed_family30=quant(xf,spec['corrected']),valid=int(ok.sum()),
                status='ESTIMABLE_EXPLORATORY' if ok.sum()>=19000 else 'UNDERPOWERED'))
    write(root/'PRIMARY_COMPARISON.json',dict(status='PASS_ALL20000_DRAWS_SECOND_ARITHMETIC',family=30,max_error=maxerr,rows=rows))

def protection(execution):
    r=Path(execution);rows=[]
    for split in ('fit','dev'):
        native=read(r/'populations/native'/split/'metrics.json')
        for family in ('existing','cvae'):
            for arm in ('accuracy_only','accuracy_dpp'):
                for seed in (42,43,44):
                    p=r/'populations'/f'{family}_{arm}_seed{seed}'/split
                    mm=read(p/'metrics.json');ss=read(p/'support.json')
                    for group in ('overall','stable','transition','change'):
                        a,b,s=mm[group],native[group],ss[group]
                        motion={k:a['motion'][k]<=1.01*b['motion'][k] for k in ('speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95')}
                        rows.append(dict(family=family,arm=arm,seed=seed,split=split,group=group,N=a['n'],good=a['good'],
                            native_good=b['good'],rescue=s['rescue'],lost=s['lost'],lost_rate=s['lost_rate'],
                            old_good_PASS=s['lost_rate'] is not None and s['lost_rate']<=.01,
                            protected_errors_PASS=None if group=='change' else a['minADE']<=1.01*b['minADE'] and a['minFDE']<=1.01*b['minFDE'],
                            five_motion=motion,all_five_PASS=all(motion.values()),
                            change5pp=None if group!='change' else 100*(a['good']-b['good'])>=5*a['n'],
                            change_ADEFDE5percent=None if group!='change' else a['minADE']<=.95*b['minADE'] and a['minFDE']<=.95*b['minFDE']))
    write(r/'PROTECTION_GATES.json',dict(status='ALL_ORIGINAL_LITERAL_GATES_DISCLOSED',rows=rows,
        automatic_baseline_upgrade=False,whole_model_independent_generalization_claim=False,
        cost_status='RAW_END_TO_END_NOT_IMPLEMENTED_NOT_PASS'))
