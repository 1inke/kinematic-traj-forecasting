"""Independent functional architecture and NumPy64 geometry/coordinator/scorer values."""
import math
import numpy as np
import torch
from torch.nn import functional as F


def geom(a,p,xp):
    a1,a2,a3=a[:,:,0],a[:,:,1],a[:,:,2]
    short=p[:,-1]-p[:,-2]; long=(p[:,-1]-p[:,-4])/3
    acceleration=short-long
    u1=a1/4; u2=(a2-a1)/4; u3=(a3-a2)/4
    if xp is np:
        deviation=np.sqrt(((u1-(short+.5*acceleration)[:,None])**2).sum(-1,keepdims=True)+1e-6)
        direction=u3/(np.sqrt((u3*u3).sum(-1,keepdims=True)+1e-4)+.05)
        alignment=(direction*acceleration[:,None]).sum(-1,keepdims=True)
        return np.concatenate([a1,a2,a3,u2-u1,u3-u2,deviation,alignment],-1)
    deviation=torch.sqrt(((u1-(short+.5*acceleration)[:,None])**2).sum(-1,keepdim=True)+1e-6)
    direction=u3/(torch.sqrt((u3*u3).sum(-1,keepdim=True)+1e-4)+.05)
    alignment=(direction*acceleration[:,None]).sum(-1,keepdim=True)
    return torch.cat([a1,a2,a3,u2-u1,u3-u2,deviation,alignment],-1)


def execute(params,z,c,phi,kind,differential=False):
    numpy=isinstance(z,np.ndarray)
    def linear(x,n):return x@params[n+'.weight'].T+(params[n+'.bias'] if n+'.bias' in params else 0)
    def ln(x,n):
        if not numpy:return F.layer_norm(x,(x.shape[-1],),params[n+'.weight'],params[n+'.bias'],1e-5)
        return (x-x.mean(-1,keepdims=True))/np.sqrt(((x-x.mean(-1,keepdims=True))**2).mean(-1,keepdims=True)+1e-5)*params[n+'.weight']+params[n+'.bias']
    def soft(x):
        if not numpy:return torch.softmax(x,-1)
        e=np.exp(x-x.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)
    if kind=='score':
        anchors,past=c,phi;g=geom(anchors,past,np if numpy else torch)
        cat=np.concatenate if numpy else torch.cat
        joined=cat([z,g],-1);a=linear(ln(joined,'norm'),'proj')
        if numpy:
            import scipy.special
            s=.5*a*(1+scipy.special.erf(a/np.sqrt(2)))
            rel=s-s.mean(1,keepdims=True);end=anchors[:,:,-1]
            dist=((end[:,:,None]-end[:,None])**2).sum(-1)
            density=np.exp(-dist/.25).mean(-1,keepdims=True);norm=np.sqrt((g[:,:,6:8]**2).sum(-1,keepdims=True))
        else:
            s=F.gelu(a);rel=s-s.mean(1,keepdim=True);end=anchors[:,:,-1]
            dist=((end[:,:,None]-end[:,None])**2).sum(-1)
            density=torch.exp(-dist/.25).mean(-1,keepdim=True);norm=torch.sqrt((g[:,:,6:8]**2).sum(-1,keepdim=True))
        return linear(cat([s,rel,density,norm],-1),'out_logit').squeeze(-1)
    cat=np.concatenate if numpy else torch.cat
    zz=z+(linear(phi,'geo_proj') if 'geo_proj.weight' in params else 0)
    qin=ln(zz-zz.mean(1,keepdims=True) if numpy else zz-zz.mean(1,keepdim=True),'qk_norm')
    vin=ln(zz,'v_norm');b,k,_=zz.shape
    def heads(x):
        x=x.reshape(b,k,4,4)
        return x.transpose(0,2,1,3) if numpy else x.permute(0,2,1,3)
    def trans(x):return x.swapaxes(-1,-2) if numpy else x.transpose(-1,-2)
    w=soft(heads(linear(qin,'q1'))@trans(heads(linear(qin,'k1')))/2)
    if differential:w=w-.5*soft(heads(linear(qin,'q2'))@trans(heads(linear(qin,'k2')))/2)
    o=w@heads(linear(vin,'value'))
    o=(o.transpose(0,2,1,3) if numpy else o.permute(0,2,1,3)).reshape(b,k,16)
    ce=np.broadcast_to(c[:,None],(b,k,64)) if numpy else c[:,None].expand(-1,k,-1)
    gate=linear(cat([qin,ce],-1),'gate');gate=1/(1+np.exp(-gate)) if numpy else torch.sigmoid(gate)
    return z+linear(gate*ln(o,'onorm'),'out_delta')


def check_component(module,z,c,phi,kind,differential=False):
    import copy
    m=copy.deepcopy(module).cpu().double()
    # Scorer zero last layer is only changed in this explicit smooth numerical fixture, not production init.
    if kind=='score':
        with torch.no_grad():m.out_logit.weight.copy_(torch.linspace(-.09,.11,m.out_logit.weight.numel()).reshape_as(m.out_logit.weight))
    inputs=[x.detach().cpu().double() for x in (z,c,phi)]
    z,c,phi=inputs;p=dict(m.named_parameters())
    actual=m(z,c,phi);reference=execute(p,z,c,phi,kind,differential)
    weight=torch.linspace(-.3,.7,actual.numel()).reshape_as(actual)
    ga=torch.autograd.grad((actual*weight).sum()/actual.numel(),tuple(p.values()),retain_graph=True)
    gr=torch.autograd.grad((reference*weight).sum()/actual.numel(),tuple(p.values()))
    value_error=float(np.max(np.abs(actual.detach().numpy()-execute({k:v.detach().numpy() for k,v in p.items()},z.numpy(),c.numpy(),phi.numpy(),kind,differential))))
    gradient_error=max(float((a-b).abs().max()) for a,b in zip(ga,gr))
    if value_error>1e-12 or gradient_error>1e-10:raise ValueError('GEO_INDEPENDENT_CHAIN')
    fd=[]
    for direction in range(2):
        vectors=[torch.sin(torch.arange(v.numel(),dtype=torch.float64)+1+direction).reshape_as(v) for v in p.values()]
        norm=sum((v*v).sum() for v in vectors).sqrt();vectors=[v/norm for v in vectors]
        target=sum((g*v).sum() for g,v in zip(ga,vectors)).item()
        initial={k:v.detach().clone() for k,v in p.items()}
        for eps in (1e-4,1e-5):
            values=[]
            for sign in (-1,1):
                with torch.no_grad():
                    for (k,v),d in zip(p.items(),vectors):v.copy_(initial[k]+sign*eps*d)
                    values.append(float((m(z,c,phi)*weight).sum()/actual.numel()))
            error=abs((values[1]-values[0])/(2*eps)-target)
            if error>1e-6+1e-3*abs(target):raise ValueError('GEO_PARAMETER_FD')
            fd.append(dict(direction=direction,eps=eps,error=error))
        with torch.no_grad():
            for k,v in p.items():v.copy_(initial[k])
    order=torch.arange(19,-1,-1)
    with torch.no_grad():
        perm=m(z[:,order],c[:,order] if kind=='score' else c,phi if kind=='score' else phi[:,order])
    perm_error=float((perm-actual.detach()[:,order]).abs().max())
    if perm_error>1e-6:raise ValueError('GEO_PERMUTATION')
    return dict(numpy64_value_error=value_error,functional_allparameter_gradient_error=gradient_error,fixed_two_epsilon_FD=fd,permutation_error=perm_error)

