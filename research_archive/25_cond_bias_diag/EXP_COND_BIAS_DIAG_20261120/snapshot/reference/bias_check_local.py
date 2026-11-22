import numpy as np, collections, json, sys
B='C:/Users/linke/Documents/New project/qoder_work/EXP_DECEL_DIAG_EXEC_20261117/populations/C0_dev/B0_s{}/decel_window_metrics.npz'
rng=np.random.default_rng(20261120)
def boot(vals,tr,nb=2000):
    u,inv=np.unique(tr,return_inverse=True);s=np.bincount(inv,vals);c=np.bincount(inv)
    idx=rng.integers(0,len(u),(nb,len(u)));m=s[idx].sum(1)/c[idx].sum(1);return np.percentile(m,[2.5,97.5])
grid=np.round(np.arange(-0.5,0.5001,0.05),3)
out={}
for seed in (42,43,44,45,46):
    d=np.load(B.format(seed),allow_pickle=True); v=d['direction_valid']
    ids=d['ids'][v]; rec=np.array([i.split('|')[0] for i in ids]); tr=np.array(['|'.join(i.split('|')[:2]) for i in ids])
    P=d['candidate_prog_error_t12'][v]; L=d['candidate_lat_error_t12'][v]; w=d['candidate_probability'][v]
    man=d['maneuver'][v]; spd=d['observed_speed'][v]; hb=d['history_bin'][v]
    bi=d['best_index_t12'][v]; i=np.arange(len(bi)); bp=P[i,bi]; bl=L[i,bi]
    fde=np.sqrt(P**2+L**2).min(1)
    res=dict(n=int(v.sum()),tracks=len(set(tr)))
    # A: unconditional
    res['best_prog_mean']=(bp.mean(),*boot(bp,tr)); res['best_lat_mean']=(bl.mean(),*boot(bl,tr))
    wp=(w*P).sum(1); res['wmean_prog']=(wp.mean(),*boot(wp,tr))
    u=(P<0).mean(1); res['pit_prog_mean']=(u.mean(),*boot(u,tr))
    # B: conditional
    q=np.quantile(spd,[.2,.4,.6,.8]); sb=np.digitize(spd,q)
    cond={}
    for name,lab in [('speed_q',sb),('hist',hb),('rec',rec),('man',man)]:
        cond[name]={str(k):dict(n=int((lab==k).sum()),best_prog=float(bp[lab==k].mean()),wmean_prog=float(wp[lab==k].mean()),pit=float(u[lab==k].mean()),best_lat=float(bl[lab==k].mean())) for k in np.unique(lab)}
    res['cond']=cond
    # C: cross-fitted shift correction, grouped by track
    ut=np.unique(tr); perm=rng.permutation(len(ut)); fold={t:k%5 for k,t in zip(range(len(ut)),ut[perm])}; f=np.array([fold[t] for t in tr])
    feats={'K':np.array([f'{a}_{b}' for a,b in zip(sb,hb)]),'S':rec,'KS':np.array([f'{a}_{b}_{c}' for a,b,c in zip(sb,hb,rec)]),'GLOBAL':np.zeros(len(tr),int).astype(str)}
    corr={}
    for fn,cell in feats.items():
        for mode in ('prog','prog_lat'):
            new=np.empty(len(tr))
            for k in range(5):
                tr_m=f!=k; te=f==k
                for c in np.unique(cell[te]):
                    trc=tr_m&(cell==c); tec=te&(cell==c)
                    if trc.sum()<30: new[tec]=fde[tec]; continue
                    best=None
                    lat_grid=grid if mode=='prog_lat' else [0.0]
                    for a in grid:
                        for b in lat_grid:
                            val=np.sqrt((P[trc]-a)**2+(L[trc]-b)**2).min(1).mean()
                            if best is None or val<best[0]-1e-12: best=(val,a,b)
                    _,a,b=best
                    new[tec]=np.sqrt((P[tec]-a)**2+(L[tec]-b)**2).min(1)
            dlt=fde-new
            hard=np.isin(man,['turn','turn_and_speed'])
            noz=rec!='crowds_zara02'
            corr[f'{fn}_{mode}']=dict(rel_all=float(dlt.sum()/fde.sum()),ci_all=[float(x) for x in boot(dlt,tr)/fde.mean()],
                                      rel_hard=float(dlt[hard].sum()/fde[hard].sum()),rel_noz=float(dlt[noz].sum()/fde[noz].sum()))
    res['correction']=corr
    out[seed]=res
json.dump(out,open('C:/Users/linke/Documents/New project/qoder_work/EXP_COND_BIAS_DIAG_EXEC_20261120/reference/bias_check.json','w'),default=float,indent=1)
for s,r in out.items():
    print('seed',s,'n',r['n'],'tracks',r['tracks'])
    for k in ['best_prog_mean','best_lat_mean','wmean_prog','pit_prog_mean']: print('  ',k,np.round(r[k],3))
    for k,c in r['correction'].items(): print('   corr',k,{kk:(np.round(vv,4) if not isinstance(vv,list) else np.round(vv,4)) for kk,vv in c.items()})

