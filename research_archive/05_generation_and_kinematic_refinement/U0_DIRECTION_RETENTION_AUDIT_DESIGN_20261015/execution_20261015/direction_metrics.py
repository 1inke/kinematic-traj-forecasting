import numpy as np
TOL=1e-10
def quality(p,gt):
    e=p-gt[:,None];d=np.sqrt(np.sum(e*e,axis=-1));a=d.mean(-1);f=d[:,:,-1];j=np.maximum(a/.2,f/.4)
    return a,f,j,(a<=.2)&(f<=.4)

def candidate(p,q,gt,independent=False):
    if independent:
        cells=[candidate(p[:,k:k+1],q[:,k:k+1],gt,False) for k in range(p.shape[1])]
        return {key:np.concatenate([z[key] for z in cells],axis=1) for key in cells[0]}
    a,f,j,g=quality(p,gt);a1,f1,j1,g1=quality(q,gt);e=p-gt[:,None];r=q-p
    en=np.sqrt((e*e).sum(-1));rn=np.sqrt((r*r).sum(-1));dot=(e*r).sum(-1)
    point=np.divide(dot,en,out=rn.copy(),where=en>0);da=point.mean(-1);df=point[:,:,-1]
    ac=a/.2;fc=f/.4;dj=np.where(ac>fc,da/.2,np.where(fc>ac,df/.4,np.maximum(da/.2,df/.4)))
    near=np.abs(ac-fc)<=TOL;uncertain=near|(np.abs(dj)<=TOL);delta=j1-j
    kind=np.zeros(j.shape,np.int8);kind[(~uncertain)&(dj>TOL)]=1;kind[(~uncertain)&(dj < -TOL)&(delta>0)]=2;kind[(~uncertain)&(delta<0)]=3
    # Positive initial slope of a convex function cannot give actual improvement.
    assert not np.any((~uncertain)&(dj>TOL)&(delta < -2e-12))
    cross=2*dot.mean(-1);step=(r*r).sum(-1).mean(-1);dmse=((q-gt[:,None])**2).sum(-1).mean(-1)-(e*e).sum(-1).mean(-1)
    endpoint_cross=2*dot[:,:,-1];endpoint_step=(r[:,:,-1]**2).sum(-1);endpoint_delta=((q[:,:,-1]-gt[:,None,-1])**2).sum(-1)-(e[:,:,-1]**2).sum(-1)
    enorm=np.sqrt((e*e).sum((-1,-2)));rnorm=np.sqrt((r*r).sum((-1,-2)));valid=(enorm>0)&(rnorm>0)
    cosine=np.divide(dot.sum(-1),enorm*rnorm,out=np.full(j.shape,np.nan),where=valid)
    en1=np.sqrt(((q-gt[:,None])**2).sum(-1))
    return dict(A0=a,F0=f,J0=j,good0=g,A1=a1,F1=f1,J1=j1,good1=g1,dpoint=point,dA=da,dF=df,dJ=dj,near_tie=near,uncertain=uncertain,kind=kind,deltaJ=delta,deltaA=a1-a,deltaF=f1-f,mse_cross=cross,mse_step=step,deltaMSE=dmse,endpoint_cross=endpoint_cross,endpoint_step=endpoint_step,endpoint_deltaMSE=endpoint_delta,path_cosine=cosine,cosine_defined=valid,zero_error_points=(en==0).sum(-1),tiny_nonzero_error_points=((en>0)&(en<=TOL)).sum(-1),correction_mean=rn.mean(-1),correction_endpoint=rn[:,:,-1],correction_max=rn.max(-1),segments0=en.reshape(*en.shape[:2],3,4).mean(-1),segments1=en1.reshape(*en1.shape[:2],3,4).mean(-1))

def check_identities(z):
    import common as c
    return max(c.close(z['deltaMSE'],z['mse_cross']+z['mse_step']),c.close(z['endpoint_deltaMSE'],z['endpoint_cross']+z['endpoint_step']))
