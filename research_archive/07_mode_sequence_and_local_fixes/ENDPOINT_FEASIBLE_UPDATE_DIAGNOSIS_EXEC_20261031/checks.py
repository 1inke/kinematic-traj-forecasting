"""GT-free frozen END lookup; full-population training-only acceptance.

The immutable bank is an exact arithmetic acceleration, not a changed guard.
Independent review reconstructs actions directly, without this bank.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import pdist
from contracts import space
from motion_guard import derivatives,candidate_metrics,pooled
from evaluate import sample_metrics

NAMES=('speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95','junction_mean','junction_p95')
GROUPS=('overall','stable','transition','change')

def quality(p,gt):
    e=np.linalg.norm(p-gt[:,None],axis=-1);a=e.mean(-1);f=e[...,-1]
    return dict(ADE=a,FDE=f,J=np.maximum(a/.2,f/.4),good=(a<=.2)&(f<=.4))

def junction(p,past):
    h=np.asarray(past,dtype=np.float64)
    return np.linalg.norm(p[...,0,:]/.4-((h[:,-1]-h[:,-2])/.4)[:,None],axis=-1)/.4

def groups(p,d):
    q=quality(p,d['gt']);jun=junction(p,d['x']['past']);rows=[]
    for k,name in enumerate(GROUPS):
        mask=np.ones(len(p),bool) if k==0 else d['x']['group']==k-1
        if not mask.any():rows.append(dict(group=name,N=0,status='NOT_ESTIMABLE'));continue
        co=q['good'][mask].any(1);nc=(d['nj'][mask]<=1).any(1)
        rows.append(dict(group=name,N=int(mask.sum()),good=int(co.sum()),rescue=int((co&~nc).sum()),lost=int((~co&nc).sum()),
            minADE=float(q['ADE'][mask].min(1).mean()),minFDE=float(q['FDE'][mask].min(1).mean()),minJ=float(q['J'][mask].min(1).mean()),
            **dict(zip(NAMES,pooled(p[mask]).tolist()+[float(jun[mask].mean()),float(np.quantile(jun[mask],.95))]))))
    return q,rows

def verdict(C,S,d,ref):
    qc=quality(C,d['gt']);qs,rows=groups(S,d);support=d['nj']<=1;nc=support.any(1)
    cfail=nc&~np.any(support&(qc['J']<=1),1);sfail=nc&~qs['good'].any(1)
    bits=dict(continuous_original_support=cfail.tolist(),guarded_old_success=sfail.tolist(),
        finite=[not(np.isfinite(C).all() and np.isfinite(S).all())],group_error=[],group_motion=[])
    for r,n in zip(rows,ref):
        if not r['N']:continue
        limit=1. if r['group']=='change' else 1.01
        bits['group_error'].extend([not(r[k]<=n[k]*limit) for k in ('minADE','minFDE')])
        bits['group_motion'].extend([not(r[k]<=n[k]*1.01) for k in NAMES])
    return dict(all_pass=not any(any(v) for v in bits.values()),bits=bits,groups=rows,
        C_original_support_minJ=np.where(nc,np.where(support,qc['J'],np.inf).min(1),0).tolist())

class Bank:
    def __init__(self,d):
        self.d=d;self.n=d['x']['native'];self.a=space('END');self.ix=np.arange(len(self.n))[:,None];self.kk=np.arange(20)[None]
        self.paths=self.n[:,:,None]+(np.arange(1,13)/12)[None,None,None,:,None]*self.a.params[None,None,:,None]
        self.paths[:,:,self.a.identity]=self.n
        # Same float64 operations and quantile axes as original scalar guard.
        base=candidate_metrics(self.n);pmet=candidate_metrics(self.paths)
        h=np.asarray(d['x']['past'],dtype=float);vobs=(h[:,-1]-h[:,-2])/.4
        jn=np.linalg.norm(self.n[:,:,0]/.4-vobs[:,None],axis=-1)/.4
        jp=np.linalg.norm(self.paths[:,:,:,0]/.4-vobs[:,None,None],axis=-1)/.4
        self.failed=np.concatenate((~(pmet<=base[:,:,None]*1.01),~(jp<=jn[:,:,None]*1.01)[...,None],(~np.isfinite(self.paths).all((-2,-1)))[...,None]),-1)
        self.failed[:,:,self.a.identity]=False
        assert np.array_equal(~self.failed.any(-1),d['safe'])
        self.ref=groups(self.n,d)[1]
    def decode(self,u):
        u=np.asarray(u)
        if u.shape!=self.n.shape[:2]+(2,) or not np.isfinite(u).all():raise ValueError('nonfinite/u shape')
        dist=((self.a.normalized[None,None]-u.astype(float)[:,:,None])**2).sum(-1)
        ids=self.a.priority[np.argmin(dist[:,:,self.a.priority],axis=-1)]
        Q=self.paths[self.ix,self.kk,ids].copy();f=self.failed[self.ix,self.kk,ids].copy()
        ex=np.where(f.any(-1),self.a.identity,ids)
        S=self.paths[self.ix,self.kk,ex].copy()
        C=self.n+.4*(np.arange(1,13)/12)[None,None,:,None]*u.astype(float)[:,:,None]
        return dict(u=u.copy(),C=C,Q=Q,S=S,raw_action_id=ids,executed_action_id=ex,failed=f)
    def evaluate(self,u):
        out=self.decode(u);return out,verdict(out['C'],out['S'],self.d,self.ref)

def scalar_paths(u,d):
    """Literal original scalar implementation, no production lookup."""
    from inference import predict_paths
    z=predict_paths(d['x']['past'],d['x']['native'],u,'END')
    return dict(u=u,C=d['x']['native']+.4*(np.arange(1,13)/12)[None,None,:,None]*u.astype(float)[:,:,None],
        Q=z['raw_paths'],S=z['guarded_paths'],raw_action_id=z['raw_action_id'],executed_action_id=z['executed_action_id'],failed=z['rejection_reason'])

def independent_paths(u,d):
    """Second direct reconstruction, no lookup bank or production failed bits."""
    action=space('END');n=d['x']['native'];h=np.asarray(d['x']['past'],float)
    dist=np.sum((np.asarray(u,float)[:,:,None,:]-action.normalized)**2,-1)
    ids=action.priority[np.argmin(dist[:,:,action.priority],-1)]
    Q=n+(np.arange(1,13)/12)[None,None,:,None]*action.params[ids][:,:,None,:]
    Q[ids==action.identity]=n[ids==action.identity]
    def cand(p):
        vv=(p-np.concatenate((np.zeros(p.shape[:-2]+(1,2)),p[...,:-1,:]),-2))/.4
        aa=(vv[...,1:,:]-vv[...,:-1,:])/.4
        jj=((vv[...,2:,:]-vv[...,1:-1,:])-(vv[...,1:-1,:]-vv[...,:-2,:]))/(.4**2)
        v,a,j=[np.sqrt(np.sum(x*x,-1)) for x in (vv,aa,jj)]
        jun=np.sqrt(np.sum((p[:,:,0]/.4-((h[:,-1]-h[:,-2])/.4)[:,None])**2,-1))/.4
        return np.stack((np.quantile(v,.95,axis=-1),a.mean(-1),np.quantile(a,.95,axis=-1),j.mean(-1),np.quantile(j,.95,axis=-1),jun),-1)
    failed=np.concatenate((~(cand(Q)<=cand(n)*1.01),(~np.isfinite(Q).all((-2,-1)))[...,None]),-1)
    failed[np.all(Q==n,(-2,-1))]=False
    ex=np.where(failed.any(-1),action.identity,ids);S=Q.copy();S[failed.any(-1)]=n[failed.any(-1)]
    return dict(u=u,C=n+.4*(np.arange(1,13)/12)[None,None,:,None]*np.asarray(u,float)[:,:,None],Q=Q,S=S,raw_action_id=ids,executed_action_id=ex,failed=failed)

def independent_groups(p,d):
    """Second implementation: explicit finite differences and scalar group tables."""
    p=np.asarray(p,float);h=np.asarray(d['x']['past'],float)
    e=np.sqrt(np.sum((p-d['gt'][:,None])**2,-1));a=e.mean(-1);f=e[:,:,-1];j=np.maximum(a/.2,f/.4);good=(a<=.2)&(f<=.4)
    v=(p-np.concatenate((np.zeros(p.shape[:-2]+(1,2)),p[...,:-1,:]),-2))/.4
    ac=(v[...,1:,:]-v[...,:-1,:])/.4
    je=(v[...,2:,:]-2*v[...,1:-1,:]+v[...,:-2,:])/(.4**2)
    # np.diff(n=2) uses successive differences, not algebraic reassociation.
    je=((v[...,2:,:]-v[...,1:-1,:])-(v[...,1:-1,:]-v[...,:-2,:]))/(.4**2)
    vn,an,jn=[np.sqrt(np.sum(x*x,-1)) for x in (v,ac,je)]
    jun=np.sqrt(np.sum((p[:,:,0]/.4-((h[:,-1]-h[:,-2])/.4)[:,None])**2,-1))/.4
    rows=[]
    for k,name in enumerate(GROUPS):
        m=np.ones(len(p),bool) if k==0 else d['x']['group']==k-1
        if not m.any():rows.append(dict(group=name,N=0,status='NOT_ESTIMABLE'));continue
        cv=good[m].any(1);nc=(d['nj'][m]<=1).any(1)
        rows.append(dict(group=name,N=int(m.sum()),good=int(cv.sum()),rescue=int((cv&~nc).sum()),lost=int((~cv&nc).sum()),
            minADE=float(a[m].min(1).mean()),minFDE=float(f[m].min(1).mean()),minJ=float(j[m].min(1).mean()),
            **dict(zip(NAMES,[float(np.quantile(vn[m],.95)),float(an[m].mean()),float(np.quantile(an[m],.95)),float(jn[m].mean()),float(np.quantile(jn[m],.95)),float(jun[m].mean()),float(np.quantile(jun[m],.95))]))))
    return dict(ADE=a,FDE=f,J=j,good=good),rows

def independent_verdict(C,S,d):
    qc,_=independent_groups(C,d);qs,rows=independent_groups(S,d);_,ref=independent_groups(d['x']['native'],d)
    nc=(d['nj']<=1).any(1);sup=d['nj']<=1
    bits=dict(continuous_original_support=(nc&~((qc['J']<=1)&sup).any(1)).tolist(),guarded_old_success=(nc&~qs['good'].any(1)).tolist(),finite=[not(np.isfinite(C).all() and np.isfinite(S).all())],group_error=[],group_motion=[])
    for r,n in zip(rows,ref):
        if not r['N']:continue
        bits['group_error'] += [r[x]>n[x]*(1 if r['group']=='change' else 1.01) for x in ('minADE','minFDE')]
        bits['group_motion'] += [r[x]>n[x]*1.01 for x in NAMES]
    return dict(all_pass=not any(any(v) for v in bits.values()),bits=bits,groups=rows)

def compare(a,b):
    if isinstance(a,dict):
        for k in b:compare(a[k],b[k])
    elif isinstance(a,list) and a and isinstance(a[0],dict):
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,(str,bool,int)) or a is None:assert a==b,(a,b)
    else:
        x,y=np.asarray(a),np.asarray(b)
        if x.dtype.kind in 'biuUS' or y.dtype.kind in 'biuUS':assert np.array_equal(x,y)
        else:assert np.allclose(x,y,atol=2e-12,rtol=0,equal_nan=True),float(np.max(np.abs(x-y)))

def full_metrics(p,d):
    z=sample_metrics(p,d['gt']);q,rows=groups(p,d)
    # Complete-link independent condensed-distance implementation.
    for i in range(len(p)):
        dd=np.stack([pdist(p[i,:,t]) for t in range(12)],1)
        lab=fcluster(linkage(np.maximum(dd.mean(1)/.1,dd[:,-1]/.2),method='complete'),1,criterion='distance')
        assert int(z['distinct'][i])==len(np.unique(lab))
        assert int(z['effective_good'][i])==len(np.unique(lab[q['good'][i]]))
    compare(dict(groups=rows),dict(groups=independent_groups(p,d)[1]))
    return z,rows
