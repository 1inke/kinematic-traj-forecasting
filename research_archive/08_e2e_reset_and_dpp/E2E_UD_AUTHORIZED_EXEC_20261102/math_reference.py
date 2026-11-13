"""Independent functional Torch chain + NumPy64 branch values, not module reuse."""
import numpy as np
import torch
from torch.nn import functional as F


def torch_reference(arm,p,x,context=None):
    def lin(value,name):return F.linear(value,p[name+'.weight'],p.get(name+'.bias'))
    def norm(value,name):return F.layer_norm(value,(p[name+'.weight'].numel(),),p[name+'.weight'],p[name+'.bias'],1e-5)
    if arm.startswith('U'):
        first=torch.tanh(lin(x[:,0],'init'));state=first
        for i in range(1,8):
            s=x[:,i]
            if arm=='U-GRU':
                gi=F.linear(s,p['cell.weight_ih'],p['cell.bias_ih'])
                gh=F.linear(state,p['cell.weight_hh'],p['cell.bias_hh'])
                ir,iz,inn=gi.chunk(3,-1);hr,hz,hn=gh.chunk(3,-1)
                reset=torch.sigmoid(ir+hr);keep=torch.sigmoid(iz+hz)
                candidate=torch.tanh(inn+reset*hn)
                state=(1-keep)*candidate+keep*state
            else:
                innovation=lin(s,'u')-lin(state,'predict')
                joined=torch.cat([s,innovation],-1)
                state=torch.sigmoid(lin(joined,'alpha'))*state+torch.sigmoid(lin(joined,'beta'))*torch.tanh(lin(innovation,'error'))
        return lin(norm(state-first,'norm'),'out')
    normalized=norm(x,'norm');n,k,_=x.shape
    def heads(value):return value.reshape(n,k,4,8).permute(0,2,1,3)
    q,k1,v=[heads(lin(normalized,name)) for name in ('q1','k1','value')]
    a=torch.softmax(torch.matmul(q,k1.transpose(-1,-2))/np.sqrt(8),-1)
    if arm=='D-DiffGate':
        q2,k2=[heads(lin(normalized,name)) for name in ('q2','k2')]
        a=a-.5*torch.softmax(torch.matmul(q2,k2.transpose(-1,-2))/np.sqrt(8),-1)
    value=torch.matmul(a,v).permute(0,2,1,3).reshape(n,20,32)
    gate=torch.sigmoid(lin(torch.cat([normalized,context[:,None].expand(-1,20,-1)],-1),'gate'))
    return x+lin(gate*norm(value,'onorm'),'out')


def numpy_reference(arm,p,x,context=None):
    def lin(value,name):return value@p[name+'.weight'].T+p.get(name+'.bias',0)
    def norm(value,name):
        centered=value-value.mean(-1,keepdims=True)
        return centered/np.sqrt((centered*centered).mean(-1,keepdims=True)+1e-5)*p[name+'.weight']+p[name+'.bias']
    def sigmoid(value):return 1/(1+np.exp(-value))
    def softmax(value):
        a=np.exp(value-value.max(-1,keepdims=True));return a/a.sum(-1,keepdims=True)
    if arm.startswith('U'):
        first=np.tanh(lin(x[:,0],'init'));state=first
        for i in range(1,8):
            s=x[:,i]
            if arm=='U-GRU':
                ir,iz,inn=np.split(s@p['cell.weight_ih'].T+p['cell.bias_ih'],3,-1)
                hr,hz,hn=np.split(state@p['cell.weight_hh'].T+p['cell.bias_hh'],3,-1)
                keep=sigmoid(iz+hz);candidate=np.tanh(inn+sigmoid(ir+hr)*hn)
                state=(1-keep)*candidate+keep*state
            else:
                innovation=lin(s,'u')-lin(state,'predict');joined=np.concatenate([s,innovation],-1)
                state=sigmoid(lin(joined,'alpha'))*state+sigmoid(lin(joined,'beta'))*np.tanh(lin(innovation,'error'))
        return lin(norm(state-first,'norm'),'out')
    normalized=norm(x,'norm');n,k,_=x.shape
    def heads(value):return value.reshape(n,k,4,8).transpose(0,2,1,3)
    q,k1,v=[heads(lin(normalized,name)) for name in ('q1','k1','value')]
    a=softmax(q@k1.swapaxes(-1,-2)/np.sqrt(8))
    if arm=='D-DiffGate':
        q2,k2=[heads(lin(normalized,name)) for name in ('q2','k2')]
        a=a-.5*softmax(q2@k2.swapaxes(-1,-2)/np.sqrt(8))
    value=(a@v).transpose(0,2,1,3).reshape(n,20,32)
    gate=sigmoid(lin(np.concatenate([normalized,np.broadcast_to(context[:,None],normalized.shape)],-1),'gate'))
    return x+lin(gate*norm(value,'onorm'),'out')
