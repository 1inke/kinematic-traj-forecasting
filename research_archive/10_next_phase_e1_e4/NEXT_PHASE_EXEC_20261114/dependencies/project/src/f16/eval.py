"""Canonical NEW protocol evaluation. Historical evaluators are not patched."""
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
GROUPS=('overall','stable','transition','change')

def errors(paths,gt):
    p=np.asarray(paths,np.float64);y=np.asarray(gt,np.float64)
    if p.ndim!=4 or p.shape[1:]!=(20,12,2) or y.shape!=(len(p),12,2):raise ValueError('Expected N,K20,T12,XY2')
    if not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError('Nonfinite prediction or GT')
    e=np.linalg.norm(p-y[:,None],axis=-1);a=e.mean(-1);f=e[:,:,-1];j=np.maximum(a/.2,f/.4);good=(a<=.2)&(f<=.4);q=(a+.5*f).argmin(1);jw=j.argmin(1);row=np.arange(len(p))
    return dict(ADE=a,FDE=f,J=j,good=good,coverage=good.any(1),Q_winner=q,J_winner=jw,Q_ADE=a[row,q],Q_FDE=f[row,q],Q_segments=e[row,q].reshape(-1,3,4).mean(-1),J_segments=e[row,jw].reshape(-1,3,4).mean(-1))

def motion(paths):
    p=np.asarray(paths,np.float64);v=np.diff(p,axis=2,prepend=np.zeros_like(p[:,:,:1]))/.4;a=np.diff(v,axis=2)/.4;j=np.diff(a,axis=2)/.4
    s,aa,jj=[np.linalg.norm(x,axis=-1) for x in (v,a,j)]
    if (v.shape[2],a.shape[2],j.shape[2])!=(12,11,10):raise ValueError('Wrong physical axes')
    return dict(speed_p95=float(np.quantile(s,.95)),accel_mean=float(aa.mean()),accel_p95=float(np.quantile(aa,.95)),jerk_mean=float(jj.mean()),jerk_p95=float(np.quantile(jj,.95)))

def clusters(paths,good):
    answer=[]
    for p,g in zip(paths,good):
        d=np.linalg.norm(p[:,None]-p[None],axis=-1);cond=np.maximum(d.mean(-1)/.1,d[:,:,-1]/.2)[np.triu_indices(20,1)];label=fcluster(linkage(cond,method='complete'),1,criterion='distance');answer.append([len(np.unique(label)),len(np.unique(label[g]))])
    return np.asarray(answer,dtype=np.int32).reshape(-1,2)

def evaluate(paths,gt,groups,native,sample_ids,tracks,include_clusters=True):
    p=np.asarray(paths,np.float64);y=np.asarray(gt,np.float64);native=np.asarray(native,np.float64);groups=np.asarray(groups);ids=np.asarray(sample_ids);track=np.asarray(tracks)
    if not (len(p)==len(groups)==len(ids)==len(track)) or len(np.unique(ids))!=len(ids) or not set(np.unique(groups)).issubset({0,1,2}):raise ValueError('Identity/group contract')
    e=errors(p,y);ne=errors(native,y);cc=clusters(p,e['good']) if include_clusters else None;rows={}
    for i,name in enumerate(GROUPS):
        mask=np.ones(len(p),bool) if i==0 else groups==i-1;n=int(mask.sum())
        if not n:rows[name]={'n':0};continue
        g=e['coverage'];base=ne['coverage'];rescue=g&~base&mask;lost=~g&base&mask;den=int(base[mask].sum());mot=motion(p[mask]);nm=motion(native[mask]);drift=np.linalg.norm(p[mask]-native[mask],axis=-1)
        row=dict(n=n,tracks=len(np.unique(track[mask])),good=int(g[mask].sum()),native_good=den,coverage=float(g[mask].mean()),rescue=int(rescue.sum()),lost=int(lost.sum()),retained=int((g&base&mask).sum()),lost_rate_native_good=float(lost.sum()/den) if den else None,ADE=float(e['ADE'][mask].min(1).mean()),FDE=float(e['FDE'][mask].min(1).mean()),J=float(e['J'][mask].min(1).mean()),Q_winner_ADE=float(e['Q_ADE'][mask].mean()),Q_winner_FDE=float(e['Q_FDE'][mask].mean()),Q_segments=e['Q_segments'][mask].mean(0).tolist(),J_segments=e['J_segments'][mask].mean(0).tolist(),motion=mot,native_motion=nm,motion_pass=all(mot[k]<=1.01*nm[k] for k in mot),correction_mean=float(drift.mean()),correction_p95=float(np.quantile(drift,.95)),correction_max=float(drift.max()),rescue_IDs=ids[rescue].tolist(),lost_IDs=ids[lost].tolist())
        if cc is not None:row.update(distinct=float(cc[mask,0].mean()),effective_good=float(cc[mask,1].mean()))
        rows[name]=row
    return rows,e
