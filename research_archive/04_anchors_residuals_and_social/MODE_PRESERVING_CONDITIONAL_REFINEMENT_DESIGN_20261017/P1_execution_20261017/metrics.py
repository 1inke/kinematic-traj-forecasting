import common as c
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
import objectives as o
c.extract(c.PARENT/'metrics.py',('evaluate',),globals())
legacy=evaluate
def support(p,n,ind=False):
    rows=[]
    for i in range(len(p)):
        if i%128==0:c.guard()
        if ind:
            mean=np.empty((20,20));end=np.empty((20,20))
            for j in range(20):
                e=np.sqrt(((n[i,j]-p[i])**2).sum(-1));mean[j]=e.mean(-1)/.1;end[j]=e[:,-1]/.2
        else:
            e=np.linalg.norm(n[i,:,None]-p[i,None,:],axis=-1);mean=e.mean(-1)/.1;end=e[...,-1]/.2
        d=np.maximum(mean,end);j=d.argmin(1);k=d.argmin(0)
        rows.append(np.stack((np.stack((mean[np.arange(20),j],end[np.arange(20),j],d[np.arange(20),j],j),-1),np.stack((mean[k,np.arange(20)],end[k,np.arange(20)],d[k,np.arange(20)],k),-1))))
    return np.stack(rows)
def labels(p,ind=False):
    out=[]
    for a in p:
        if ind:
            d=[]
            for j in range(19):
                x=np.sqrt(((a[j]-a[j+1:])**2).sum(-1));d.extend(np.maximum(x.mean(-1)/.1,x[:,-1]/.2))
        else:
            e=np.linalg.norm(a[:,None]-a[None,:],axis=-1);d=np.maximum(e.mean(-1)/.1,e[...,-1]/.2)[np.triu_indices(20,1)]
        out.append(fcluster(linkage(np.asarray(d),method='complete'),1,criterion='distance'))
    return np.asarray(out)
def evaluate(z,ind=False):
    met,aux=legacy(z,ind);p=z['paths'];n=z['native'];gt=z['gt'];cell=str(z.get('cell','A'))
    d=o.numpy_components(p,n,gt,cell);a0,f0,j0,g0,_=c.errors(n,gt,ind);owner=d['owner'];mask=np.ones_like(g0);mask[np.arange(len(p)),owner]=False
    aux.update(Q=d['Q'],Q0=d['Q0'],owner=owner,dynamic_owner=d['dynamic'],fixed_owner=d['fixed'],native_good_mask=g0,native_drift=d['drift'],support=support(p,n,ind),cluster_labels=labels(p,ind),native_cluster_labels=labels(n,ind))
    v=np.concatenate((p[:,:,:1],p[:,:,1:]-p[:,:,:-1]),axis=2)/.4;aa=np.diff(v,axis=2)/.4;jj=np.diff(aa,axis=2)/.4
    for key,x in (('speed',v),('acceleration',aa),('jerk',jj)):aux[key+'_magnitudes']=np.sqrt((x*x).sum(-1))
    for group,sel in c.masks(z['groups']):
        r=met[group]
        if not sel.any():continue
        no=mask&g0;bad=no&~aux['good'];repaired=aux['good'][np.arange(len(p)),owner]&~g0[np.arange(len(p)),owner]
        r.update(task=float(d['Q'][np.arange(len(p)),owner][sel].mean()),anchor=float((d['drift']*mask)[sel].sum()/(sel.sum()*19)),owner_disagreement=int((d['dynamic'][sel]!=d['fixed'][sel]).sum()),owner_ties=int(((d['Q'][sel]==d['Q'][sel].min(1)[:,None]).sum(1)>1).sum()),native_owner_ties=int(((d['Q0'][sel]==d['Q0'][sel].min(1)[:,None]).sum(1)>1).sum()),owner_counts=np.bincount(owner[sel],minlength=20),nonowner_native_good=int(no[sel].sum()),nonowner_good_lost=int(bad[sel].sum()),owner_repair_other_good_destroyed=int((repaired&bad.any(1)&sel).sum()),native_owner_final_Q=float(d['Q'][np.arange(len(p)),d['fixed']][sel].mean()),support_bidirectional_mean=aux['support'][sel,:,:,:3].mean((0,2)))
    aux['population_components']=np.array([d[k] for k in ('total','task','diversity','anchor')])
    return met,aux
def validate(z,met,aux,arm):
    if arm!='native20':
        # CPU recurrence independent of production einsum, use saved actual controls.
        u=z['controls'];tanh=np.tanh(z['raw']).astype(float);tanh/=np.maximum(1,np.sqrt((tanh*tanh).sum(-1,keepdims=True)));c.close(tanh,u,2e-7)
        v=np.zeros_like(u[:,:,0]);p=np.zeros_like(v);r=np.empty_like(u)
        for t in range(12):p=p+v+.2/144*u[:,:,t];v=v+2*.2/144*u[:,:,t];r[:,:,t]=p
        c.close(r,z['correction']);c.close(z['native']+r,z['paths'])
        assert np.all(np.linalg.norm(r,axis=-1)<=.2*(np.arange(1,13)/12)**2+2e-12)
    else:assert np.array_equal(z['paths'],z['native'])
    rows,aa=evaluate(z,True);c.compare(rows,met)
    for k in aa:
        if aa[k].dtype.kind in 'biu':assert np.array_equal(aa[k],aux[k]),k
        else:c.close(aa[k],aux[k])
    return dict(status='PASS',full_quality_clusters_motion_responsibility_support_independent=True)

