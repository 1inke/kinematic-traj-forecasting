"""Twelve fixed output-space probes; never training or deployable inference."""
import numpy as np
from contracts import ALPHAS
from checks import quality

def probe(u0,u1,d,bank):
    native_good=(d['nj']<=1).any(1);support=d['nj']<=1
    chosen=np.zeros(128,float);u=np.empty_like(u0);pending=np.ones(128,bool);trace=[]
    for alpha in ALPHAS+(0.,):
        # Actual output-space interpolation dtype is float32, saved before geometry.
        trial=u0+alpha*(u1-u0);out=bank.decode(trial)
        cq=quality(out['C'],d['gt']);sq=quality(out['S'],d['gt'])
        local=(~native_good)|(((support&(cq['J']<=1)).any(1))&sq['good'].any(1))
        local &= np.isfinite(trial).all((1,2))&np.isfinite(out['S']).all((1,2,3))
        take=pending&local;u[take]=trial[take];chosen[take]=alpha;pending[take]=False
        trace.append(dict(alpha=alpha,local_pass=local.tolist(),first_selected=take.tolist()))
    if pending.any():raise AssertionError('current alpha0 lacks local support')
    out,v=bank.evaluate(u)
    return out,v,chosen,trace
