"""Full isotropic and anisotropic score grids in the mean-path frame."""
import numpy as np

from adapt_math import errors, energy, hpd, ES_NAMES
from centered_scaling_ref import frames, weighted_mean, center_scale

ISO=np.arange(4,17)/10
AXIS=np.arange(5,17)/10
FIELDS=("minADE","minFDE","J","spread","HPD90","mean_bias","fast_steps",
        *(f"{name}_{part}" for name in ES_NAMES for part in ("ES","A","S")))


def from_paths(x,y,w,h,draws,center_w=None,scattering=None):
    f=errors(x,y)
    f["HPD90"]=hpd(x,y,h,draws)
    center=weighted_mean(x,w if center_w is None else center_w)
    f["mean_bias"]=np.linalg.norm(center[:,-1]-y[:,-1],axis=-1)
    speed=np.square(np.diff(x,axis=2,prepend=0)).sum(-1)
    f["fast_steps"]=(speed>(2.5*.4)**2).mean((1,2))
    e,a,s=energy(x,y,w,scattering)
    for i,name in enumerate(ES_NAMES):
        for part,v in (("ES",e),("A",a),("S",s)):
            f[f"{name}_{part}"]=v[:,i]
    return f


def basis(x,y,w,heading):
    m=weighted_mean(x,w)
    u=frames(x,w,heading)
    v=np.stack((-u[...,1],u[...,0]),axis=-1)
    q=x-m[:,None]
    truth=y-m
    l=(q*u[:,None]).sum(-1)
    a=(q*v[:,None]).sum(-1)
    return m,u,v,l,a,(truth*u).sum(-1),(truth*v).sum(-1)


def hpd_grid(m,u,v,l,a,t_l,t_a,h,draws,progress):
    """Reuse Gaussian factors for all 144 pairs, with the archived world draws."""
    uniform,noise=draws
    out=np.empty((len(l),len(AXIS),len(AXIS)),bool)
    k=l.shape[1]
    cdf=np.cumsum(np.full(k,1/k))
    cdf[-1]=1.
    for lo in range(0,len(l),16):
        sl=slice(lo,lo+16)
        e_l,e_a=l[sl,:,-1],a[sl,:,-1]
        n=len(e_l)
        row=np.arange(n)[:,None]
        slot=np.searchsorted(cdf,uniform[sl],side="right")
        width=h[sl,None,None]
        dist_l=(e_l[row,slot,None]-e_l[:,None])/width
        dist_a=(e_a[row,slot,None]-e_a[:,None])/width
        draw_l=(noise[sl]*u[sl,None,-1]).sum(-1)
        draw_a=(noise[sl]*v[sl,None,-1]).sum(-1)

        def kernels(dist,draw):
            x=dist[:,:,None,:]*AXIS[None,None,:,None]+draw[:,:,None,None]
            np.square(x,out=x)
            x *= -.5
            np.exp(x,out=x)
            return x

        left,right=kernels(dist_l,draw_l),kernels(dist_a,draw_a)
        density=left@right.swapaxes(-1,-2)
        t1=np.square((t_l[sl,None,None,-1]-e_l[:,None]*AXIS[None,:,None])/width)
        t2=np.square((t_a[sl,None,None,-1]-e_a[:,None]*AXIS[None,:,None])/width)
        target=np.exp(-.5*(t1[:,:,None]+t2[:,None])).sum(-1)
        out[sl]=(density>=target[:,None]).mean(1)<=.90
        if lo%1024==0:
            progress(lo,len(l))
    return out


def all_grids(x,y,w,heading,h,draws,progress):
    m,u,v,l,a,t_l,t_a=basis(x,y,w,heading)
    hp=hpd_grid(m,u,v,l,a,t_l,t_a,h,draws,lambda i,n:progress("HPD",i,n))
    # Isotropic 0.4 is the only point outside the anisotropic grid.
    iso04=from_paths(center_scale(x,w,.4),y,w,h,draws)
    i,j=np.triu_indices(x.shape[1],1)
    by_pair={}
    buffers=[[] for _ in range(4)]
    for lo in range(0,len(x),128):
        b=slice(lo,lo+128)
        pl=np.square(l[b,i]-l[b,j])
        pa=np.square(a[b,i]-a[b,j])
        for target,array in zip(buffers,(pl.sum(2),pa.sum(2),pl[:,:,-1].copy(),pa[:,:,-1].copy())):
            target.append(array)
    pl,pa,el,ea=(np.concatenate(z) for z in buffers)
    del buffers
    weights=w[:,i]*w[:,j]
    norm=x.shape[1]**2
    vl,va=(np.var(t[:,:,-1],axis=1) for t in (l,a))
    bias=np.linalg.norm(m[:,-1]-y[:,-1],axis=-1)
    dm=np.diff(m,axis=1,prepend=0)[:,None]
    dl=np.diff(l[...,None]*u[:,None],axis=2,prepend=0)
    da=np.diff(a[...,None]*v[:,None],axis=2,prepend=0)
    # Expand the squared step displacement as a quadratic in the two multipliers.
    vl2=np.square(dl).sum(-1); va2=np.square(da).sum(-1)
    v0=np.square(dm).sum(-1); vl0=2*(dl*dm).sum(-1); va0=2*(da*dm).sum(-1)
    v12=2*(dl*da).sum(-1)
    for ii,lon in enumerate(AXIS):
        dlon=np.square(lon*l-t_l[:,None])
        dist_l=lon**2*pl
        end_l=lon**2*el
        speed_l=v0+lon*vl0+lon**2*vl2
        for jj,lat in enumerate(AXIS):
            dist2=dlon+np.square(lat*a-t_a[:,None])
            d=np.sqrt(dist2)
            ade,fde=d.mean(2),d[:,:,-1]
            traj=np.sqrt(dist2.sum(2))
            t=np.sqrt(dist_l+lat**2*pa); e=np.sqrt(end_l+lat**2*ea)
            s=((t*weights).sum(1),t.sum(1)/norm,(e*weights).sum(1),e.sum(1)/norm)
            acc=((traj*w).sum(1),traj.mean(1),(fde*w).sum(1),fde.mean(1))
            f=dict(minADE=ade.min(1),minFDE=fde.min(1),J=((ade<=.2)&(fde<=.4)).any(1),
                   spread=np.sqrt(lon**2*vl+lat**2*va),HPD90=hp[:,ii,jj],mean_bias=bias,
                   fast_steps=(speed_l+lat*va0+lat**2*va2+lon*lat*v12>1.).mean((1,2)))
            for name,at,st in zip(ES_NAMES,acc,s):
                f[f"{name}_ES"],f[f"{name}_A"],f[f"{name}_S"]=at-st,at,st
            by_pair[float(lon),float(lat)]=f
        progress("OTHER_METRICS",ii+1,len(AXIS))
    # Use original-coordinate metrics for the identity, preserving exact legacy gates.
    base=from_paths(x,y,w,h,draws)
    np.testing.assert_array_equal(base["HPD90"],hp[:,5,5])
    by_pair[1.,1.]=base
    by_alpha={.4:iso04,**{float(z):by_pair[float(z),float(z)] for z in AXIS}}
    return by_alpha,by_pair
