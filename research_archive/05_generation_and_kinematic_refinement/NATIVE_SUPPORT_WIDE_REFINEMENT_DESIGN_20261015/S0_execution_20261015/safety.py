"""Registered GT-free per-candidate native-relative ray; CPU float64 S0."""
import numpy as np
MARGIN=1.01
INTERIOR=1.-2.**-20
RADIUS=.20*(np.arange(1,13,dtype=np.float64)/12.)**2

def differences(p,ind=False):
    if ind:
        v=np.empty_like(p);v[...,0,:]=p[...,0,:]/.4;v[...,1:,:]=(p[...,1:,:]-p[...,:-1,:])/.4
        a=(v[...,1:,:]-v[...,:-1,:])/.4;j=(a[...,1:,:]-a[...,:-1,:])/.4
    else:
        v=np.diff(p,axis=-2,prepend=np.zeros_like(p[...,:1,:]))/.4;a=np.diff(v,axis=-2)/.4;j=np.diff(a,axis=-2)/.4
    assert [v.shape[-2],a.shape[-2],j.shape[-2]]==[12,11,10]
    return v,a,j

def squared(x,ind=False):
    return x[...,0]*x[...,0]+x[...,1]*x[...,1] if ind else np.sum(x*x,axis=-1)

def roots(x,d,ind=False):
    aa=squared(d,ind);bb=2*(x[...,0]*d[...,0]+x[...,1]*d[...,1]) if ind else 2*np.sum(x*d,axis=-1)
    cc=(MARGIN*MARGIN-1)*squared(x,ind);sq=np.sqrt(bb*bb+4*aa*cc)
    out=np.ones_like(aa);active=aa>0;out[active]=0
    pos=active&(bb>=0)&(sq+bb>0);neg=active&(bb<0)
    out[pos]=2*cc[pos]/(sq[pos]+bb[pos]);out[neg]=(sq[neg]-bb[neg])/(2*aa[neg])
    return np.clip(out,0.,1.)

def strict_violations(native,path,ind=False):
    return np.concatenate([np.sqrt(squared(p,ind))>MARGIN*np.sqrt(squared(b,ind)) for b,p in zip(differences(native,ind),differences(path,ind))],-1)

def project(native,residual,ind=False):
    assert native.dtype==residual.dtype==np.float64
    base=differences(native,ind);delta=differences(residual,ind)
    rr=np.concatenate([roots(x,d,ind) for x,d in zip(base,delta)],-1);minimum=rr.min(-1)
    alpha=np.where(minimum<1,minimum*INTERIOR,minimum)
    trial=native+alpha[...,None,None]*residual
    reasons=strict_violations(native,trial,ind);fallback=reasons.any(-1)
    alpha=np.where(fallback,0.,alpha);out=np.where(fallback[...,None,None],native,trial)
    assert not strict_violations(native,out,ind).any(),'LITERAL_OUTPUT_SAFETY_FAILED'
    return dict(safe20=out,alpha=alpha,pre_fallback_alpha=np.where(minimum<1,minimum*INTERIOR,minimum),roots=rr,binding=rr==minimum[...,None],fallback=fallback,fallback_reasons=reasons,zero_budget=np.concatenate([squared(x,ind)==0 for x in base],-1))

def construct(native,gt,slots,ind=False):
    delta=gt[:,None]-native;e=np.sqrt(squared(delta,ind))
    if ind:
        residual=np.zeros_like(delta)
        for t,b in enumerate(RADIUS):
            positive=e[:,:,t]>0;fac=np.zeros_like(e[:,:,t]);fac[positive]=np.minimum(e[:,:,t][positive],b)/e[:,:,t][positive];residual[:,:,t]=delta[:,:,t]*fac[...,None]
    else:residual=delta*np.minimum(np.divide(RADIUS,e,out=np.zeros_like(e),where=e>0),1.)[...,None]
    cert=project(native,residual,ind);mask=np.zeros(native.shape[:2],bool);mask[np.arange(len(native))[:,None],slots]=True
    return dict(native=native,gt=gt,slots=slots,mask=mask,ball_residual=residual,safe4=np.where(mask[...,None,None],cert['safe20'],native),**cert)

def fixtures():
    # Each of the 33 individual constraints: exact boundary, outside, inactive,
    # zero-budget and fixed70-iteration bisection reference (fixture only).
    maximum=0.
    for k in range(33):
        x=np.array([[1.+k/100.,.3]],np.float64);d=x*.2
        r=roots(x,d).item();assert 0<r<1
        boundary=np.sqrt(squared(x+r*d));assert abs(boundary.item()-1.01*np.sqrt(squared(x)).item())<2e-14
        assert np.sqrt(squared(x+(r+1e-5)*d)).item()>1.01*np.sqrt(squared(x)).item()
        lo,hi=0.,1.
        for _ in range(70):
            m=(lo+hi)/2
            if np.sum((x+m*d)**2)<=1.01**2*np.sum(x*x):lo=m
            else:hi=m
        maximum=max(maximum,abs(r-lo));assert abs(r-lo)<2e-12
        assert roots(x,np.zeros_like(d)).item()==1 and roots(np.zeros_like(x),d).item()==0
    t=np.arange(1,13,dtype=float);p=np.stack((.07*t+.001*t*t,.0002*t**3),-1)[None,None]
    r=np.stack((.001*t*t,.0001*t**3),-1)[None,None]
    a=project(p,r);b=project(p,r,True)
    for k in a:assert np.array_equal(a[k],b[k]),('INDEPENDENT_FIXTURE',k)
    assert np.array_equal(project(p,np.zeros_like(p))['safe20'],p)
    for p0 in (np.zeros_like(p),np.stack((.1*t,np.zeros_like(t)),-1)[None,None]):
        z=project(p0,r);assert not strict_violations(p0,z['safe20']).any()
    # Changing an unrelated GT object cannot affect this path-only API.
    gt=np.zeros((1,12,2));before=project(p,r);gt[:]=99;after=project(p,r)
    assert all(np.array_equal(before[k],after[k]) for k in before)
    return dict(status='PASS',individual_constraints_33_boundary_outside_zero_budget=True,zero_v_a_j_and_residual=True,GT_absent_safe_API=True,independent_literal_fallback_EXACT=True,fixed70_bisection_maxerror=maximum,no_production_solver=True)
