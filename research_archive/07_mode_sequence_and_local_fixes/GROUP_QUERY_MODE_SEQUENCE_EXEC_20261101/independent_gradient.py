"""Full NumPy chain rule, no torch.autograd and no production loss call."""
import numpy as np
from scipy.special import erf
from objectives import numpy_loss_gradient

def gelu(x):return .5*x*(1+erf(x/np.sqrt(2)))
def dgelu(x):return .5*(1+erf(x/np.sqrt(2)))+x*np.exp(-x*x/2)/np.sqrt(2*np.pi)
def check(m,features,gt):
    st={k:v.detach().cpu().numpy().astype(float) for k,v in m.state_dict().items()}
    grads={k:np.zeros_like(v) for k,v in st.items() if k not in ('mean','std')};x=(features-st['mean'])/st['std'];cache=[]
    def layer(x,prefix):return x@st[prefix+'.weight'].T+st[prefix+'.bias']
    t0=layer(x,'trunk.0');t1=gelu(t0);t2=layer(t1,'trunk.2');h=gelu(t2);ys=[]
    qs=['queries']+[f'aux.{i}' for i in range(len(m.aux))]
    for name in qs:
        c=np.concatenate((np.broadcast_to(h[:,None],(len(x),20,128)),np.broadcast_to(st[name][None],(len(x),20,16))),-1)
        a=layer(c,'fuse.0');b=gelu(a);z=layer(b,'fuse.2');w=gelu(z);out=layer(w,'head').reshape(len(x),20,12,2)
        ys.append(out);cache.append((c,a,b,z,w))
    paths=np.stack(ys,1);value,dp,faces=numpy_loss_gradient(paths,gt);dh=np.zeros_like(h)
    def backward(inp,dout,prefix):
        ii=inp.reshape(-1,inp.shape[-1]);dd=dout.reshape(-1,dout.shape[-1]);grads[prefix+'.weight']+=dd.T@ii;grads[prefix+'.bias']+=dd.sum(0)
        return dout@st[prefix+'.weight']
    for k,name in enumerate(qs):
        c,a,b,z,w=cache[k];dw=backward(w,dp[:,k].reshape(len(x),20,24),'head');dz=dw*dgelu(z);db=backward(b,dz,'fuse.2');da=db*dgelu(a);dc=backward(c,da,'fuse.0')
        dh+=dc[:,:,:128].sum(1);grads[name]+=dc[:,:,128:].sum(0)
    dt2=dh*dgelu(t2);dt1=backward(t1,dt2,'trunk.2');dt0=dt1*dgelu(t0);backward(x,dt0,'trunk.0')
    return value,paths,grads,faces
