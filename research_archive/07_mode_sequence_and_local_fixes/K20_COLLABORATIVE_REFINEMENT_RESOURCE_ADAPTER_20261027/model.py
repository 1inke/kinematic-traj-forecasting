from core import *
import torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
class Refiner(nn.Module):
    def __init__(self,stats,arm):
        super().__init__();assert arm in ARMS;self.arm=arm
        for k in ('mean','std','motion_scales','basis','basis_l1_bound'):self.register_buffer(k,torch.from_numpy(np.asarray(stats[k]).copy()))
        self.enc=nn.Linear(263,32);self.qkv=nn.Linear(32,96);self.out=nn.Linear(32,32);self.norm1=nn.LayerNorm(32);self.ff=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,32));self.norm2=nn.LayerNorm(32);self.offset=nn.Linear(32,6);self.gate=nn.Linear(32,1)
        nn.init.zeros_(self.offset.weight);nn.init.zeros_(self.offset.bias);nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)
    def forward(self,raw,native):
        h=F.gelu(self.enc(((raw.double()-self.mean)/self.std).to(raw.dtype)));q,k,v=self.qkv(h).reshape(-1,20,3,2,16).unbind(2);q,k,v=[x.transpose(1,2) for x in (q,k,v)];s=(q@k.transpose(-1,-2))/4
        if self.arm=='B1':s=s.masked_fill(~torch.eye(20,device=s.device,dtype=torch.bool),-torch.inf)
        w=s.softmax(-1);z=self.norm1(h+self.out((w@v).transpose(1,2).reshape(-1,20,32)));z=self.norm2(z+self.ff(z))
        anchors=torch.tanh(self.offset(z).reshape(-1,20,3,2)).double();anchors=anchors/torch.linalg.vector_norm(anchors,dim=-1,keepdim=True).clamp_min(1);gate=self.gate(z).sigmoid().double();res=.2/self.basis_l1_bound*torch.einsum('tr,bkrd->bktd',self.basis,anchors)*gate[...,None]
        return native.double()+res,dict(residual=res,gate=gate,anchors=anchors,attention=w)
def candidate_errors(paths,gt):
    d=torch.linalg.vector_norm(paths-gt[:,None].double(),dim=-1);a=d.mean(-1)/.2;f=d[:,:,-1]/.4;j=torch.where(a>=f,a,f);return a,f,j.min(1).values
def mean_motion(p):
    v=torch.diff(p,dim=2,prepend=torch.zeros_like(p[:,:,:1]))/.4;a=torch.diff(v,dim=2)/.4;j=torch.diff(a,dim=2)/.4
    return torch.stack([torch.linalg.vector_norm(x,dim=-1).mean((1,2)) for x in (v,a,j)],dim=-1)
def objective(paths,gt,native,scales):
    a,f,j=candidate_errors(paths,gt);_,_,nj=candidate_errors(native,gt);soft=.1*torch.logsumexp(torch.stack((a,f),dim=-1)/.1,dim=-1);soft=-.1*torch.logsumexp(-soft/.1,dim=1)
    termset=((1+(nj>1).double())*soft).mean();keep=((nj<=1).double()*F.relu(j-nj.detach())).mean();motion=F.relu((mean_motion(paths)-mean_motion(native).detach())/scales).square().mean();small=((paths-native).square().sum(-1)).mean()/.2**2
    return termset+keep+.1*motion+.01*small,dict(set=termset,keep=keep,motion=motion,small=small)
def numpy_objective(p,y,n,sc):
    def err(z):
        d=np.linalg.norm(z-y[:,None],axis=-1);a=d.mean(-1)/.2;f=d[:,:,-1]/.4;return a,f,np.maximum(a,f).min(1)
    def lse(z,axis):
        m=z.max(axis=axis,keepdims=True);return np.squeeze(m,axis)+np.log(np.exp(z-m).sum(axis))
    def mm(z):
        v=np.diff(z,axis=2,prepend=np.zeros_like(z[:,:,:1]))/.4;a=np.diff(v,axis=2)/.4;j=np.diff(a,axis=2)/.4;return np.stack([np.linalg.norm(x,axis=-1).mean((1,2)) for x in (v,a,j)],axis=-1)
    a,f,j=err(p);_,_,nj=err(n);soft=.1*lse(np.stack((a,f),axis=-1)/.1,-1);soft=-.1*lse(-soft/.1,1);terms=dict(set=float(((1+(nj>1))*soft).mean()),keep=float(((nj<=1)*np.maximum(j-nj,0)).mean()),motion=float((np.maximum((mm(p)-mm(n))/sc,0)**2).mean()),small=float(((p-n)**2).sum(-1).mean()/.2**2));return terms['set']+terms['keep']+.1*terms['motion']+.01*terms['small'],terms
