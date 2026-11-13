"""Independent NumPy EC objective derivatives and current decoder reconstruction."""
import numpy as np
from scipy.special import erf

def loss_and_path_gradient(past,paths,cal,arm):
    y=np.asarray(paths,np.float64);p=np.asarray(past,np.float64);n=len(y)
    previous=np.concatenate([np.broadcast_to(p[:,-1,None,None],(n,20,1,2)),y[:,:,:-1]],2)
    v=(y-previous)/.4;vh=(p[:,-1]-p[:,-2])/.4
    a=(v-np.concatenate([np.broadcast_to(vh[:,None,None],(n,20,1,2)),v[:,:,:-1]],2))/.4
    join=v[:,:,0]-vh[:,None];vs=np.sqrt((v*v).sum(-1));ac=np.sqrt((a*a).sum(-1));js=np.sqrt((join*join).sum(-1))
    rv=np.maximum((vs-cal['vmax'])/cal['sigma_v'],0);ra=np.maximum((ac-cal['amax'])/cal['sigma_a'],0);rj=np.maximum((js-cal['joinmax'])/cal['sigma_join'],0)
    P=(rv*rv).mean(-1)+(ra*ra).mean(-1)+rj*rj;q=np.exp(-P)
    quality=np.ones_like(q) if arm=='uniform' else cal['epsilon_q']+(1-cal['epsilon_q'])*q
    r=np.sqrt(quality);flat=y.reshape(n,20,24);delta=flat[:,:,None]-flat[:,None]
    S=np.exp(-cal['k']*(delta*delta).sum(-1));L=r[:,:,None]*S*r[:,None]
    eye=np.broadcast_to(np.eye(20),L.shape);inverse=np.linalg.solve(eye+L,eye)
    ec=20-np.trace(inverse,axis1=1,axis2=2);G=inverse@inverse;W=G*L
    grad=(-4*cal['k']*(W[:,:,:,None]*delta).sum(2)).reshape(y.shape)
    if arm=='kin':
        coefficient=-(W.sum(-1)*(1-cal['epsilon_q'])*q/quality)
        safev=np.where(vs>0,vs,1);safea=np.where(ac>0,ac,1);safej=np.where(js>0,js,1)
        gv=2*rv[...,None]/(cal['sigma_v']*12)*v/safev[...,None]
        ga=2*ra[...,None]/(cal['sigma_a']*12)*a/safea[...,None]
        gj=2*rj[...,None]/cal['sigma_join']*join/safej[...,None]
        gv+=ga/.4;gv[:,:,:-1]-=ga[:,:,1:]/.4;gv[:,:,0]+=gj
        gy=gv/.4;gy[:,:,:-1]-=gv[:,:,1:]/.4
        grad+=coefficient[...,None,None]*gy
    return float(-ec.mean()),-grad/n,dict(v_active=rv>0,a_active=ra>0,j_active=rj>0,qkin=q,EC=ec)

def decoder_np(control,cache,indices,weights):
    z=np.asarray(control,np.float64);c=cache['context'][indices,0].astype(np.float64)
    x=c[:,None]+weights['mode_embedding'][None]+z
    mean=x.mean(-1,keepdims=True);var=((x-mean)**2).mean(-1,keepdims=True)
    x=(x-mean)/np.sqrt(var+1e-5)*weights['mode_mlp.0.weight']+weights['mode_mlp.0.bias']
    x=x@weights['mode_mlp.1.weight'].T+weights['mode_mlp.1.bias'];x=.5*x*(1+erf(x/np.sqrt(2)))
    scale=np.logaddexp(0,c@weights['proposal_scale.weight'].T+weights['proposal_scale.bias'])+.25
    delta=.75*np.tanh(x@weights['proposal_delta.weight'].T+weights['proposal_delta.bias']).reshape(len(z),20,3,2)
    anchors=scale[:,None,None]*(weights['anchor_templates'][None]+delta)
    relative=np.einsum('tp,bkpc->bktc',weights['interpolation'],anchors)
    local=cache['past_loc'][indices,0,-1,None,None].astype(np.float64)+relative+cache['center'][indices,0,None].astype(np.float64)
    return np.einsum('bkti,bji->bktj',local,cache['matrix'][indices].astype(np.float64))+cache['origin'][indices,None,None].astype(np.float64)
