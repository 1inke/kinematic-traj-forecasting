"""Production evaluator and second saved-data arithmetic/cluster/motion implementation."""
from common import *
from f16.eval import evaluate,errors,motion
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage,fcluster
GROUPS=('overall','stable','transition','change')

def junction(p,past):
    h=np.asarray(past,np.float64);j=np.linalg.norm(p[:,:,0]/.4-((h[:,-1]-h[:,-2])/.4)[:,None],axis=-1)/.4
    return dict(junction_mean=float(j.mean()),junction_p95=float(np.quantile(j,.95)))
def production(p,d):
    rows,e=evaluate(p,d['gt'],d['groups'],d['native'],d['ids'],d['track'])
    for k,name in enumerate(GROUPS):
        mask=np.ones(len(p),bool) if k==0 else d['groups']==k-1
        if mask.any():rows[name]['motion'].update(junction(np.asarray(p,float)[mask],d['past'][mask]));rows[name]['native_motion'].update(junction(d['native'][mask],d['past'][mask]))
    return rows,e
def independent(p,d,rows,er,guard=None):
    p=np.asarray(p,np.float64);y=d['gt'].astype(float);delta=p-y[:,None]
    distance=np.sqrt(np.sum(delta**2,axis=-1));ade=distance.sum(-1)/12;fde=distance[:,:,-1];j=np.maximum(ade/.2,fde/.4);good=(ade<=.2)&(fde<=.4)
    ae=np.sqrt(np.sum((d['native']-y[:,None])**2,-1));ng=(ae.mean(-1)<=.2)&(ae[:,:,-1]<=.4);base=ng.any(1);covered=good.any(1)
    maxerr=0.
    for key,x in [('ADE',ade),('FDE',fde),('J',j)]:
        e=float(np.max(abs(x-er[key])));maxerr=max(maxerr,e);assert e<=2e-12
    assert np.array_equal(good,er['good']) and np.array_equal(covered,er['coverage'])
    cc=[]
    for i,window in enumerate(p):
        dist=np.stack([pdist(window[:,t]) for t in range(12)],1);c=fcluster(linkage(np.maximum(dist.mean(1)/.1,dist[:,-1]/.2),method='complete'),1,criterion='distance')
        cc.append([len(np.unique(c)),len(np.unique(c[good[i]]))])
        if guard and i%256==0:guard.check()
    cc=np.array(cc)
    v=(p-np.concatenate([np.zeros_like(p[:,:,:1]),p[:,:,:-1]],axis=2))/.4
    a=(v[:,:,1:]-v[:,:,:-1])/.4
    jerk=((v[:,:,2:]-v[:,:,1:-1])-(v[:,:,1:-1]-v[:,:,:-2]))/(.4**2)
    vn,an,jn=[np.sqrt(np.sum(x*x,-1)) for x in (v,a,jerk)]
    past=np.asarray(d['past'],float);ju=np.sqrt(np.sum((p[:,:,0]/.4-((past[:,-1]-past[:,-2])/.4)[:,None])**2,-1))/.4
    for k,name in enumerate(GROUPS):
        mask=np.ones(len(p),bool) if k==0 else d['groups']==k-1
        r=rows[name];assert r['n']==int(mask.sum())
        if not mask.any():continue
        for key,x in [('good',covered&mask),('native_good',base&mask),('rescue',covered&~base&mask),('lost',~covered&base&mask)]:assert r[key]==int(x.sum()),(name,key)
        for key,x in [('ADE',ade.min(1)),('FDE',fde.min(1)),('J',j.min(1)),('distinct',cc[:,0]),('effective_good',cc[:,1])]:
            e=abs(float(x[mask].mean())-r[key]);maxerr=max(maxerr,e);assert e<=2e-12,(name,key,e)
        physics=dict(speed_p95=np.quantile(vn[mask],.95),accel_mean=an[mask].mean(),accel_p95=np.quantile(an[mask],.95),jerk_mean=jn[mask].mean(),jerk_p95=np.quantile(jn[mask],.95),junction_mean=ju[mask].mean(),junction_p95=np.quantile(ju[mask],.95))
        for key,x in physics.items():
            e=abs(float(x)-r['motion'][key]);maxerr=max(maxerr,e);assert e<=2e-12,(name,key,e)
        for typ in ('J','Q'):
            win=np.argmin(j if typ=='J' else ade+.5*fde,axis=1);seg=distance[np.arange(len(p)),win].reshape(-1,3,4).mean(-1)
            assert np.allclose(seg[mask].mean(0),r[typ+'_segments'],atol=2e-12,rtol=0)
    return dict(status='PASS_INDEPENDENT_ALL_SAVED_QUALITY_PDIST_CLUSTER_FIVE_MOTION_JUNCTION',N=len(p),K=20,max_error=maxerr)

def save_population(folder,groups,d,guard=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    npz(folder/'predictions.npz',paths=groups[:,0],all_group_paths=groups,sample_id=d['ids'],gt=d['gt'],native=d['native'],past=d['past'],groups=d['groups'],track=d['track'])
    saved=load_npz(folder/'predictions.npz');assert np.array_equal(saved['all_group_paths'],groups)
    metrics,er=production(saved['paths'],d);npz(folder/'sample_metrics.npz',**er);write(folder/'metrics.json',metrics)
    audit=independent(saved['paths'],d,metrics,load_npz(folder/'sample_metrics.npz'),guard);write(folder/'audit.json',audit)
    diagnostic={};cover=[];winners=[]
    for k in range(groups.shape[1]):
        e=errors(groups[:,k],d['gt']);cover.append(e['coverage']);winners.append(e['Q_winner'])
        diagnostic[str(k)]={name:dict(N=int(mask.sum()),good=int(e['coverage'][mask].sum()),ADE=float(e['ADE'][mask].min(1).mean()),FDE=float(e['FDE'][mask].min(1).mean())) for name,mask in [('overall',np.ones(len(groups),bool)),('change',d['groups']==2)]}
    cover=np.stack(cover,1);pairs=[]
    for a in range(groups.shape[1]):
        for b in range(a+1,groups.shape[1]):
            pa=groups[np.arange(len(groups)),a,winners[a]].astype(float);pb=groups[np.arange(len(groups)),b,winners[b]].astype(float)
            pairs.append(dict(a=a,b=b,selected_meanpath_distance=float(np.linalg.norm(pa-pb,axis=-1).mean()),both_good=int((cover[:,a]&cover[:,b]).sum()),union_good=int((cover[:,a]|cover[:,b]).sum())))
    write(folder/'auxiliary_diagnostic.json',dict(groups=diagnostic,pairs=pairs,union_GK_not_deployment=dict(K=20*groups.shape[1],good=int(cover.any(1).sum()),change_good=int(cover[d['groups']==2].any(1).sum()))))
    write(folder/'done.json',dict(status='SUCCESS',files={p.name:sha(p) for p in folder.iterdir() if p.is_file()},audit=audit['status']))
    return metrics
