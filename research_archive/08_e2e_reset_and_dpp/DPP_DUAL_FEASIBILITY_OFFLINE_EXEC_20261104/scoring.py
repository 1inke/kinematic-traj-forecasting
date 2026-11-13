"""NumPy64 stage-A only: no learned model, selector, or optimization."""
import numpy as np
from scipy.ndimage import distance_transform_edt
DT=.4

def states(past,paths):
    p=np.asarray(past,np.float64);y=np.asarray(paths,np.float64)
    if y.ndim!=4 or y.shape[1:]!=(20,12,2) or p.shape!=(len(y),8,2):raise ValueError('SHAPE')
    if not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('NONFINITE_PATH')
    hist=(p[:,-1]-p[:,-2])/DT
    v=np.diff(np.concatenate([np.broadcast_to(p[:,-1:,None,:].transpose(0,2,1,3),(len(y),20,1,2)),y],axis=2),axis=2)/DT
    a=np.diff(np.concatenate([np.broadcast_to(hist[:,None,None],(len(y),20,1,2)),v],axis=2),axis=2)/DT
    return np.linalg.norm(v,axis=-1),np.linalg.norm(a,axis=-1),np.linalg.norm(v[:,:,0]-hist[:,None],axis=-1)

def kinematic(past,paths,c):
    if any(not np.isfinite(c[k]) or c[k]<=0 for k in ('sigma_v','sigma_a','sigma_join')):raise ValueError('NONPOSITIVE_NORMALIZATION_SCALE')
    v,a,j=states(past,paths)
    pv=(np.maximum((v-c['vmax'])/c['sigma_v'],0)**2).mean(-1)
    pa=(np.maximum((a-c['amax'])/c['sigma_a'],0)**2).mean(-1)
    pj=np.maximum((j-c['joinmax'])/c['sigma_join'],0)**2
    penalty=pv+pa+pj
    logq=-penalty
    q=np.exp(logq)
    valid=(v<=c['vmax']).all(-1)&(a<=c['amax']).all(-1)&(j<=c['joinmax'])
    return dict(qkin=q,logqkin=logq,penalty_v=pv,penalty_a=pa,penalty_join=pj,kin_valid=valid)

def pair_squared(paths):
    flat=np.asarray(paths,np.float64).reshape(len(paths),20,24)
    return ((flat[:,:,None]-flat[:,None])**2).sum(-1)

def dpp(paths,q,k,epsilon):
    q=np.asarray(q,np.float64)
    if q.shape!=(len(paths),20) or not np.isfinite(q).all() or (q<0).any() or (q>1).any() or k<=0 or not 0<epsilon<1:raise ValueError('DPP_INPUT')
    S=np.exp(-k*pair_squared(paths));qt=epsilon+(1-epsilon)*q;r=np.sqrt(qt)
    L=r[:,:,None]*S*r[:,None]
    I=np.broadcast_to(np.eye(20),L.shape)
    inverse=np.linalg.solve(I+L,I)
    marginal=1-np.diagonal(inverse,axis1=1,axis2=2)
    ec=marginal.sum(-1)
    eigen=np.linalg.eigvalsh(L)
    err=float(np.max(abs(ec-(eigen/(1+eigen)).sum(-1))))
    if eigen.min()<-2e-12 or err>2e-12:raise ValueError('DPP_PSD_OR_SPECTRAL_AUDIT')
    if float(np.max(abs(np.diagonal(L,axis1=1,axis2=2)-qt)))>2e-12:raise ValueError('QUALITY_SQUARED_NOT_Q')
    return dict(ec=ec,marginal=marginal,min_eigen=float(eigen.min()),solve_spectral_error=err)

def asd_fsd(paths,valid):
    dist=np.sqrt(pair_squared(paths));end=np.linalg.norm(paths[:,:,None,-1]-paths[:,None,:,-1],axis=-1)
    pair=valid[:,:,None]&valid[:,None]&np.triu(np.ones((20,20),bool),1)
    n=pair.sum((1,2));a=np.full(len(paths),np.nan);f=a.copy()
    np.divide((dist*pair).sum((1,2)),n,out=a,where=n>0)
    np.divide((end*pair).sum((1,2)),n,out=f,where=n>0)
    return a,f,n

def sdf_from_mask(free,known,metres_per_pixel):
    if np.shape(free)!=np.shape(known) or np.ndim(free)!=2 or metres_per_pixel<=0:raise ValueError('MAP_INPUT')
    free=np.asarray(free,bool)
    if not np.asarray(known,bool).all():raise ValueError('SDF_UNKNOWN_NEEDS_CERTIFIED_DISTANCE_FIELD_NOT_FREE_OBSTACLE_GUESS')
    if free.all() or not free.any():raise ValueError('SDF_NEEDS_OBSERVED_FREE_AND_OBSTACLE_BOUNDARY')
    return (distance_transform_edt(free)-distance_transform_edt(~free))*metres_per_pixel

def scene_score(p0,path,distance_metres,known,world_to_grid,spacing=.05,margin=.1,tau=.1):
    """Certified metric SDF required in production. Unknown/outside -> not assessed, never q=1."""
    if spacing<=0 or tau<=0 or np.shape(distance_metres)!=np.shape(known) or np.shape(world_to_grid)!=(3,3):raise ValueError('SCENE_CONFIG')
    if not np.isfinite(np.asarray(path)).all() or not np.isfinite(np.asarray(p0)).all() or not np.isfinite(world_to_grid).all():raise ValueError('NONFINITE_SCENE_COORDINATE')
    nodes=np.vstack([p0,path]);points=[]
    for a,b in zip(nodes[:-1],nodes[1:]):
        count=max(1,int(np.ceil(np.linalg.norm(b-a)/spacing)))
        points.extend(a+(b-a)*t for t in np.arange(1,count+1)/count)
    pts=np.asarray(points);h=np.c_[pts,np.ones(len(pts))]@np.asarray(world_to_grid,np.float64).T
    finite=np.isfinite(h).all(1)&(h[:,2]!=0);xy=np.zeros((len(pts),2));xy[finite]=h[finite,:2]/h[finite,2,None]
    col=np.floor(xy[:,0]).astype(int);row=np.floor(xy[:,1]).astype(int)
    H,W=known.shape;inside=finite&(row>=0)&(col>=0)&(row<H-1)&(col<W-1)
    ok=np.zeros(len(pts),bool);values=np.zeros(len(pts))
    for i in np.flatnonzero(inside):
        r,c=row[i],col[i]
        if not np.asarray(known[r:r+2,c:c+2]).all():continue
        fx,fy=xy[i]-[c,r];z=distance_metres[r:r+2,c:c+2]
        if not np.isfinite(z).all():raise ValueError('KNOWN_SDF_NONFINITE')
        values[i]=(1-fy)*((1-fx)*z[0,0]+fx*z[0,1])+fy*((1-fx)*z[1,0]+fx*z[1,1]);ok[i]=True
    if not ok.any():return dict(qscene=None,known_fraction=0.,violation=None,status='UNKNOWN_NOT_ASSESSED')
    penalty=np.logaddexp(0,(margin-values[ok])/tau).mean()
    return dict(qscene=float(np.exp(-penalty)),known_fraction=float(ok.mean()),violation=bool((values[ok]<0).any()),status='PARTIAL_KNOWN' if not ok.all() else 'ASSESSED')

def original_radius_quality(z,radius,omega=1.):
    z=np.asarray(z,np.float64)
    if not np.isfinite(z).all() or radius<=0 or omega<=0:raise ValueError('LATENT_RADIUS_INPUT')
    return omega*np.exp(np.minimum(radius**2-(z*z).sum(-1),0))

def turning(past,paths):
    v=np.asarray(past,np.float64)[:,-1]-np.asarray(past,np.float64)[:,-2]
    end=paths[:,:,-1]-past[:,-1,None]
    norm=np.linalg.norm(v,axis=-1)[:,None]*np.linalg.norm(end,axis=-1)
    angle=np.full(norm.shape,np.nan)
    cosine=np.zeros(norm.shape)
    np.divide((end*v[:,None]).sum(-1),norm,out=cosine,where=norm>0)
    angle[norm>0]=np.degrees(np.arccos(np.clip(cosine[norm>0],-1,1)))
    return angle
