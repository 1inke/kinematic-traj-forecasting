"""Paired target-track x model-seed bootstrap, fixed family16; no stdlib shadow."""
import common as c
import numpy as np

def bootstrap(vectors,tracks,groups,draws=20000,independent=False):
    # vectors: A,B,native -> [3seeds,N,4metrics]; native replication is pairing,
    # NOT three independent native models.
    methods=('A','B','native');unique=np.unique(tracks);trackindex={t:i for i,t in enumerate(unique)};codes=np.array([trackindex[t] for t in tracks]);ntrack=len(unique)
    values={};tables={};counts={}
    for group,mask in [('overall',np.ones(len(tracks),bool)),('change',groups==2)]:
        counts[group]=np.bincount(codes[mask],minlength=ntrack).astype(float)
        for method in methods:
            a=vectors[method][:,mask,:]
            table=np.zeros((3,ntrack,4))
            for seed in range(3):
                for metric in range(4):table[seed,:,metric]=np.bincount(codes[mask],weights=a[seed,:,metric],minlength=ntrack)
            tables[method,group]=table
            values[method,group]=vectors[method][:,mask,:].mean(1).mean(0)
    rng=np.random.Generator(np.random.PCG64(20261019));result={};all_draws={}
    keys=[(comp,group,metric) for comp in ('B-A','B-native') for group in ('overall','change') for metric in range(4)]
    buffers={k:np.full(draws,np.nan) for k in keys}
    for start in range(0,draws,128):
        c.guard();n=min(128,draws-start)
        # Per-draw order: paired3 model seeds then all outer tracks.
        si=[];ti=[]
        for _ in range(n):si.append(rng.integers(0,3,3));ti.append(rng.integers(0,ntrack,ntrack))
        si=np.array(si);ti=np.array(ti)
        for group in ('overall','change'):
            denominator=counts[group][ti].sum(1);ok=denominator>0
            if independent:
                sampled={method:np.full((n,4),np.nan) for method in methods}
                for row in range(n):
                    if not ok[row]:continue
                    for method in methods:
                        seedmeans=[]
                        for seed in si[row]:seedmeans.append(sum((tables[method,group][seed,int(t)] for t in ti[row]),np.zeros(4))/denominator[row])
                        sampled[method][row]=np.mean(seedmeans,0)
            else:
                sampled={method:tables[method,group][si[:,:,None],ti[:,None,:],:].sum(2).mean(1)/np.where(ok,denominator,np.nan)[:,None] for method in methods}
            for comp,other in [('B-A','A'),('B-native','native')]:
                delta=sampled['B']-sampled[other]
                for metric in range(4):buffers[comp,group,metric][start:start+n]=delta[:,metric]
    metricnames=('coverage','minADE','minFDE','J')
    for comp,group,metric in keys:
        other=comp.split('-')[1];x=buffers[comp,group,metric];valid=np.isfinite(x);nt=int(np.sum(counts[group]>0));key=comp+'/'+group+'/'+metricnames[metric]
        result[key]=dict(point=float((values['B',group]-values[other,group])[metric]),valid_draws=int(valid.sum()),tracks=nt,status='SUFFICIENT_EXPLORATORY' if valid.sum()>=19000 and (group!='change' or nt>=20) else 'INSUFFICIENT',CI95=np.quantile(x[valid],[.025,.975]) if valid.any() else None,CI_family16=np.quantile(x[valid],[.0015625,.9984375]) if valid.any() else None)
        all_draws[key.replace('/','__')]=x
    # Equal track weighting is a separate descriptive sensitivity, not replacement.
    equal={}
    for group in ('overall','change'):
        mask=counts[group]>0
        for method in methods:equal[method+'/'+group]=(tables[method,group][:,mask]/counts[group][mask,None]).mean(1).mean(0)
    return dict(family=16,draws=draws,seed=20261019,quantiles=[.0015625,.9984375],metrics=result,equal_track_sensitivity=equal,not_independent_scene_inference=True),all_draws

def fixture():
    # Actual production draw algorithm vs independent track-loop on small data;
    # the full20000 draws and all16 endpoints are checked, zero groups not redrawn.
    rng=np.random.Generator(np.random.PCG64(89));tr=np.array(['x'+str(i//3) for i in range(24)]);g=np.array([i%3 for i in range(24)]);v={k:rng.normal(size=(3,24,4)) for k in ('A','B','native')}
    a,ad=bootstrap(v,tr,g);b,bd=bootstrap(v,tr,g,independent=True)
    for k in ad:assert np.allclose(ad[k],bd[k],atol=2e-12,rtol=0,equal_nan=True)
    return dict(status='PASS',draws=20000,family=16,max_error=max(float(np.nanmax(abs(ad[k]-bd[k]))) for k in ad),independent_track_loop=True)
