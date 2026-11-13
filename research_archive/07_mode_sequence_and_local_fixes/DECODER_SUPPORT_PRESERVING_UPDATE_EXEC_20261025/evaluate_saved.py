"""Finite saved-output evaluation and independent pdist/physics review, no training main."""
from task_adapter import *
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import pdist

def independently_review(path,metrics_path):
    with np.load(path,allow_pickle=False) as z:
        p=z['paths'].astype(float);y=z['gt'].astype(float);native=z['native'].astype(float);groups=z['groups'];ids=z['sample_id'];tracks=z['track']
    assert p.shape==(len(ids),20,12,2) and len(np.unique(ids))==len(ids) and np.isfinite(p).all()
    # Independent geometry and time-axis contractions; no evaluate/errors/motion calls.
    def geo(a):
        e=np.sqrt(((a-y[:,None])**2).sum(-1));ade=e.sum(2)/12;fde=e[:,:,11];jj=np.maximum(ade/.2,fde/.4);good=(ade<=.2)&(fde<=.4)
        return ade,fde,jj,good
    aa,ff,jj,good=geo(p);na,nf,nj,ng=geo(native);covered=good.any(1);nc=ng.any(1)
    cc=[]
    for pp,gg in zip(p,good):
        d=sum(pdist(pp[:,i,:]) for i in range(12))/12;dd=np.maximum(d/.1,pdist(pp[:,11,:])/.2)
        lab=fcluster(linkage(dd,method='complete'),1,criterion='distance');cc.append([len(set(lab)),len(set(lab[gg]))])
    cc=np.asarray(cc);metric=read_json(metrics_path);maxerr=0.;groups_out={}
    def ph(a):
        v=np.concatenate((a[:,:,:1],a[:,:,1:]-a[:,:,:-1]),2)/.4;ac=(v[:,:,1:]-v[:,:,:-1])/.4;jer=(ac[:,:,1:]-ac[:,:,:-1])/.4
        s,b,c=[np.sqrt((v*v).sum(-1)) for v in (v,ac,jer)]
        return dict(speed_p95=float(np.percentile(s,95)),accel_mean=float(b.mean()),accel_p95=float(np.percentile(b,95)),jerk_mean=float(c.mean()),jerk_p95=float(np.percentile(c,95)))
    for i,name in enumerate(('overall','stable','transition','change')):
        mask=np.ones(len(p),bool) if i==0 else groups==i-1;n=int(mask.sum());m=metric[name]
        assert m['n']==n
        if not n:continue
        discrete=dict(good=int(covered[mask].sum()),native_good=int(nc[mask].sum()),rescue=int((covered&~nc&mask).sum()),lost=int((~covered&nc&mask).sum()),retained=int((covered&nc&mask).sum()),tracks=len(np.unique(tracks[mask])))
        assert all(m[k]==v for k,v in discrete.items())
        numeric=dict(ADE=float(aa[mask].min(1).mean()),FDE=float(ff[mask].min(1).mean()),J=float(jj[mask].min(1).mean()),distinct=float(cc[mask,0].mean()),effective_good=float(cc[mask,1].mean()))
        for k,v in numeric.items():maxerr=max(maxerr,abs(v-m[k]));assert abs(v-m[k])<=2e-12
        pp=ph(p[mask]);nn=ph(native[mask])
        for target,ind in [('motion',pp),('native_motion',nn)]:
            for k,v in ind.items():maxerr=max(maxerr,abs(v-m[target][k]));assert abs(v-m[target][k])<=2e-12
        assert m['motion_pass']==all(pp[k]<=1.01*nn[k] for k in pp)
        assert m['rescue_IDs']==ids[covered&~nc&mask].tolist() and m['lost_IDs']==ids[~covered&nc&mask].tolist()
        groups_out[name]=discrete
    return dict(status='PASS_INDEPENDENT_SAVED_ALL_K20_CLUSTER_MOTION',rows=len(p),max_error=maxerr,groups=groups_out,no_model_forward=True)

def save_evaluate(m,d,folder,name):
    folder.mkdir(parents=True,exist_ok=True);before=frozen_state(m);p=predict(m,d['x'])
    path=folder/f'{name}_predictions.npz';np.savez_compressed(path,paths=p,native=d['native'],gt=d['gt'],sample_id=d['ids'],groups=d['groups'],track=d['track'])
    metrics,er=evaluate(p,d['gt'],d['groups'],d['native'],d['ids'],d['track']);write_json(folder/f'{name}_metrics.json',metrics)
    np.savez_compressed(folder/f'{name}_sample_metrics.npz',**er)
    review=independently_review(path,folder/f'{name}_metrics.json');write_json(folder/f'{name}_REVIEW.json',review)
    assert equal(before,frozen_state(m))
    return metrics,review
