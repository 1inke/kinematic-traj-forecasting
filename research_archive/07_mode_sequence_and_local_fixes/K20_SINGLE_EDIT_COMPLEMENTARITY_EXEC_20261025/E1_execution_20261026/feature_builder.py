"""GT-free, fixed324 channels, no slot-ID embedding or labels."""
import numpy as np
def build(past,tokens,native,kin,action_motion):
    p=np.asarray(past,np.float64);n=np.asarray(native,np.float64);k=np.asarray(kin,np.float64)
    t=np.asarray(tokens,np.float32)
    assert p.shape==(8,2) and t.shape==(8,96) and n.shape==k.shape==(20,12,2) and action_motion.shape==(401,5)
    v=np.diff(p,axis=0)/.4;a=np.diff(v,axis=0)/.4
    history=np.r_[p.ravel(),v.ravel(),a.ravel()]
    context=np.r_[t[-1],t.mean(axis=0)].astype(np.float64)
    ep=n[:,-1];centered=ep-ep.mean(0)
    covariance=np.array([(centered[:,0]**2).mean(),(centered[:,0]*centered[:,1]).mean(),(centered[:,1]**2).mean()])
    shared=np.r_[history,context,n.mean(0).ravel(),covariance,action_motion[0]]
    assert shared.shape==(266,)
    out=np.zeros((401,324),np.float64);out[:,:266]=shared;out[0,-1]=1
    distance=np.linalg.norm(k[:,None]-n[None],axis=-1)
    D=np.maximum(distance.mean(-1)/.1,distance[:,:,-1]/.2)
    for r in range(20):
        sl=slice(1+20*r,1+20*(r+1))
        out[sl,266:290]=n[r].ravel();out[sl,290:314]=k.reshape(20,24)
        out[sl,314:316]=k[:,-1]-n[r,-1];out[sl,316:321]=action_motion[sl]-action_motion[0]
        out[sl,321]=D[:,r];out[sl,322]=np.delete(D,r,axis=1).min(1)
    if not np.isfinite(out).all():raise ValueError('Nonfinite observable features')
    return out
class WindowMoments:
    """Equal window weights, eligible-with-STOP equal within each window."""
    def __init__(self):self.n=0;self.mean=np.zeros(323,np.float64);self.M2=np.zeros(323,np.float64)
    def add(self,x,eligible):
        values=np.asarray(x,np.float64)[eligible,:323]
        if not len(values):raise ValueError('STOP must always be eligible')
        mu=values.mean(0);variance=((values-mu)**2).mean(0)
        self.n+=1;delta=mu-self.mean;self.mean+=delta/self.n
        self.M2+=variance+delta*delta*((self.n-1)/self.n)
    def finish(self):
        if not self.n:raise ValueError('No training windows')
        variance=self.M2/self.n
        if (variance<0).any():raise ValueError('Stable variance unexpectedly negative')
        std=np.sqrt(variance);std[std<1e-6]=1
        return self.mean.copy(),std
