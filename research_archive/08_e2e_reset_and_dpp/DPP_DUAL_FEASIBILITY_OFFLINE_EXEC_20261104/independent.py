"""Second arithmetic implementation for preflight and saved-array review."""
import numpy as np

def score_loop(past,paths,c,return_details=False):
    q=np.empty((len(paths),20));valid=np.zeros_like(q,bool)
    pv_all=q.copy();pa_all=q.copy();pj_all=q.copy();log_all=q.copy()
    for b in range(len(paths)):
        vh=(past[b,-1].astype(np.float64)-past[b,-2])/0.4
        for k in range(20):
            vv=[];aa=[];previous=past[b,-1].astype(np.float64);lastv=vh
            for pt in paths[b,k].astype(np.float64):
                v=(pt-previous)/.4;vv.append(float(np.sqrt(np.dot(v,v))))
                a=(v-lastv)/.4;aa.append(float(np.sqrt(np.dot(a,a))))
                previous=pt;lastv=v
            join=float(np.linalg.norm((paths[b,k,0]-past[b,-1])/.4-vh))
            pv=sum(max(0,(x-c['vmax'])/c['sigma_v'])**2 for x in vv)/12
            pa=sum(max(0,(x-c['amax'])/c['sigma_a'])**2 for x in aa)/12
            pj=max(0,(join-c['joinmax'])/c['sigma_join'])**2
            penalty=pv+pa+pj
            pv_all[b,k]=pv;pa_all[b,k]=pa;pj_all[b,k]=pj;log_all[b,k]=-penalty
            q[b,k]=np.exp(-penalty)
            valid[b,k]=max(vv)<=c['vmax'] and max(aa)<=c['amax'] and join<=c['joinmax']
    if return_details:return q,valid,dict(penalty_v=pv_all,penalty_a=pa_all,penalty_join=pj_all,logqkin=log_all)
    return q,valid

def spectral_ec_loop(paths,q,k,eps,return_marginal=False):
    out=[];marginal=[]
    for y,quality in zip(paths,q):
        S=np.empty((20,20))
        for i in range(20):
            for j in range(20):
                distance=float(sum(float((a-b)**2) for a,b in zip(y[i].ravel(),y[j].ravel())))
                S[i,j]=np.exp(-k*distance)
        r=np.sqrt(eps+(1-eps)*quality)
        values,vectors=np.linalg.eigh(S*r[:,None]*r[None])
        out.append(float(sum(v/(1+v) for v in values)))
        if return_marginal:marginal.append((vectors**2)@(values/(1+values)))
    if return_marginal:return np.array(out),np.array(marginal)
    return np.array(out)

def quality_loop(paths,future):
    a=np.empty((len(paths),20));f=a.copy()
    for n in range(len(paths)):
        for k in range(20):
            d=[float(np.linalg.norm(paths[n,k,t].astype(np.float64)-future[n,t].astype(np.float64))) for t in range(12)]
            a[n,k]=sum(d)/12;f[n,k]=d[-1]
    return a,f
