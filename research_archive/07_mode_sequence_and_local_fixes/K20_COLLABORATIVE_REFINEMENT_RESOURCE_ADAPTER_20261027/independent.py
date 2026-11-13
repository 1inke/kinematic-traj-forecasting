from core import *
def numpy_path_gradient(p,y,n,sc):
    # Independent analytic derivative in path coordinates, no Torch calls.
    N,K,T,_=p.shape;diff=p-y[:,None];dist=np.linalg.norm(diff,axis=-1);unit=np.divide(diff,dist[...,None],out=np.zeros_like(diff),where=dist[...,None]>0);a=dist.mean(-1)/.2;f=dist[:,:,-1]/.4
    nd=np.linalg.norm(n-y[:,None],axis=-1);nj=np.maximum(nd.mean(-1)/.2,nd[:,:,-1]/.4).min(1);inner=np.stack((a,f),axis=-1)/.1;em=np.exp(inner-inner.max(-1,keepdims=True));wa=em/em.sum(-1,keepdims=True);soft=.1*(inner.max(-1)+np.log(em.sum(-1)));ex=np.exp((-soft/.1)-(-soft/.1).max(1,keepdims=True));wk=ex/ex.sum(1,keepdims=True)
    ga=(1+(nj>1))[:,None]/N*wk*wa[:,:,0]/.2;gf=(1+(nj>1))[:,None]/N*wk*wa[:,:,1]/.4
    literal=np.maximum(a,f);win=literal.argmin(1);newj=literal[np.arange(N),win];active=(nj<=1)&(newj>nj)
    for i in np.flatnonzero(active):
        k=win[i]
        if a[i,k]>=f[i,k]:ga[i,k]+=1/(N*.2)
        else:gf[i,k]+=1/(N*.4)
    grad=unit*ga[:,:,None,None]/T;grad[:,:,-1]+=unit[:,:,-1]*gf[:,:,None]
    def levels(z):
        v=np.diff(z,axis=2,prepend=np.zeros_like(z[:,:,:1]))/.4;a=np.diff(v,axis=2)/.4;j=np.diff(a,axis=2)/.4;return [v,a,j]
    lv=levels(p);ln=levels(n);new=np.stack([np.linalg.norm(x,axis=-1).mean((1,2)) for x in lv],axis=-1);old=np.stack([np.linalg.norm(x,axis=-1).mean((1,2)) for x in ln],axis=-1);factors=.1*2*np.maximum((new-old)/sc,0)/sc/(N*3);gg=[]
    for i,x in enumerate(lv):
        norm=np.linalg.norm(x,axis=-1,keepdims=True);u=np.divide(x,norm,out=np.zeros_like(x),where=norm>0);gg.append(u*factors[:,None,None,i,None]/(K*x.shape[2]))
    for i in (2,1):gg[i-1][:,:,1:]+=gg[i]/.4;gg[i-1][:,:,:-1]-=gg[i]/.4
    grad+=gg[0]/.4;grad[:,:,:-1]-=gg[0][:,:,1:]/.4;grad+=.01*2*(p-n)/(N*K*T*.2**2);return grad
def reconstruct(native,anchors,gate,basis,bound):
    r=np.zeros_like(native)
    for t in range(12):
        for k in range(3):r[:,:,t]+=.2/bound*basis[t,k]*anchors[:,:,k]*gate
    return native+r
