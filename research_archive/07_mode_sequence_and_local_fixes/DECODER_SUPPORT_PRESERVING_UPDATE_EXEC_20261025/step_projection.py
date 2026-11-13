"""Float64 active-set closest projection onto zero, one or two halfspaces."""
import itertools
import numpy as np

def project(d0,rows):
    d0=np.asarray(d0,dtype=np.float64);rows=np.asarray(rows,dtype=np.float64).reshape(-1,len(d0))
    assert len(rows)<=2 and np.isfinite(d0).all() and np.isfinite(rows).all()
    norms=np.linalg.norm(rows,axis=1);keep=norms!=0;G=rows[keep]/norms[keep,None]
    tol=1e-10*max(1.,float(np.linalg.norm(d0)));candidates=[]
    for size in range(len(G)+1):
        for active in itertools.combinations(range(len(G)),size):
            A=G[list(active)];lam=np.linalg.pinv(A@A.T,rcond=1e-12)@(A@d0) if size else np.empty(0)
            d=d0-A.T@lam if size else d0.copy();residual=G@d
            station=d-d0+(A.T@lam if size else 0)
            if (not size or np.min(lam)>=-tol) and (not len(G) or np.max(residual)<=tol) and np.linalg.norm(station)<=tol and (not size or np.max(np.abs(A@d))<=tol):
                candidates.append((float(np.dot(d-d0,d-d0)),active,lam,d,residual))
    if not candidates:raise ArithmeticError('NO_CERTIFIED_CLOSEST_PROJECTION')
    dist,active,lam,d,res=min(candidates,key=lambda v:v[0])
    assert np.linalg.norm(d)<=np.linalg.norm(d0)+tol,'PROJECTION_EXPANDS_NORM'
    return d,dict(G=G,nonzero_rows=keep,row_norms=norms,active=list(active),multipliers=lam,raw_Gd0=G@d0,raw_GdP=res,tolerance=tol,objective_distance_squared=dist,stationarity_norm=float(np.linalg.norm(d-d0+(G[list(active)].T@lam if len(active) else 0))))

def cast_certificate(theta0,ideal,actual,G,tol):
    e=actual-ideal;bound=float(np.linalg.norm(e));ideal_res=G@ideal;actual_res=G@actual
    assert np.isfinite(actual).all()
    assert not len(G) or np.max(np.abs(actual_res-ideal_res))<=bound+tol
    return dict(cast_error_norm=bound,ideal_Gd=ideal_res,actual_Gd=actual_res,raw_positive_actual=bool(len(G) and np.max(actual_res)>0),actual_minus_ideal_dot_bound=bound,solver_tolerance=tol)

def fixtures():
    cases=[([2.,3.],[]),([2.,3.],[[1,0]]),([2.,3.],[[1,0],[0,1]]),([-2.,3.],[[1,0],[0,1]]),([2.,3.],[[1,0],[1,0]]),([2.,3.],[[1,0],[-1,0]]),([0.,0.],[[1,0],[0,1]]),([-2.,-3.],[[1,0],[0,1]]),([2.,3.],[[0,0],[0,1]]),([2.,3.],[[1,0],[1,1e-5]])]
    output=[]
    for d,rows in cases:
        d=np.asarray(d);r=np.asarray(rows).reshape(-1,2);p,c=project(d,r)
        # Independent 2-D geometry: feasible input, projections on each line, origin.
        nonzero=r[np.linalg.norm(r,axis=1)>0];can=[np.zeros(2),d.copy()]
        for row in nonzero:can.append(d-row*np.dot(row,d)/np.dot(row,row))
        feasible=[v for v in can if not len(nonzero) or max(nonzero@v)<=1e-10*max(1,np.linalg.norm(d))]
        independent=min(feasible,key=lambda v:np.linalg.norm(v-d)**2)
        assert np.max(np.abs(p-independent))<=2e-9
        pp,_=project(p,r);assert np.max(np.abs(p-pp))<=2e-9
        ratio=np.linalg.norm(p)/np.linalg.norm(d) if np.linalg.norm(d) else 0
        assert abs(np.linalg.norm(ratio*d)-np.linalg.norm(p))<2e-12
        output.append(dict(d0=d.tolist(),rows=r.tolist(),projected=p.tolist(),active=c['active']))
    return output
