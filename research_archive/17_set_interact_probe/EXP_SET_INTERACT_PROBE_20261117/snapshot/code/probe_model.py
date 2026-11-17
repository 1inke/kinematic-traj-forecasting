import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS={'A1':('self',False),'A2':('full',False),'A3':('sparse',False),'A4':('self',True),'A5':('sparse',True),'A6':('full',True)}
class Refiner(nn.Module):
    def __init__(self,arm):
        super().__init__();self.arm=arm;self.kind,self.reallocate=ARMS[arm];self.heads=4;self.hd=6
        self.inp=nn.Sequential(nn.Linear(77,24),nn.LayerNorm(24))
        self.q=nn.Linear(24,24,bias=False);self.k=nn.Linear(24,24,bias=False);self.v=nn.Linear(24,24,bias=False);self.o=nn.Linear(24,24,bias=False)
        self.n1=nn.LayerNorm(24);self.ffn=nn.Sequential(nn.Linear(24,48),nn.GELU(),nn.Linear(48,24));self.n2=nn.LayerNorm(24)
        self.bias=nn.Sequential(nn.Linear(4,16),nn.GELU(),nn.Linear(16,4))
        self.anchor_out=nn.Linear(24,6);self.score_out=nn.Linear(24,1)
        for layer in (self.bias[-1],self.anchor_out,self.score_out):nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
    def relations(self,relative,base_logits):
        end=relative[:,:,-1];delta=end[:,None]-end[:,:,None];dist=torch.linalg.vector_norm(delta,dim=-1)
        features=torch.cat([delta,dist[...,None],(base_logits[:,None]-base_logits[:,:,None])[...,None]],dim=-1)
        eye=torch.eye(20,dtype=torch.bool,device=end.device)[None]
        closest=dist.masked_fill(eye,float('inf')).argsort(dim=-1,stable=True)[...,:4]
        graph=eye.expand(len(end),-1,-1).clone();graph.scatter_(-1,closest,True)
        return features,graph
    def forward(self,mode,relative,scale,base_logits):
        n,k=relative.shape[:2];rel=relative/scale[:,None,None]
        previous=torch.cat([torch.zeros_like(rel[:,:,:1]),rel[:,:,:-1]],dim=2)
        segments=rel-previous;endnorm=torch.linalg.vector_norm(rel[:,:,-1],dim=-1,keepdim=True)
        x=self.inp(torch.cat([mode,rel.flatten(-2),segments.flatten(-2),endnorm],dim=-1))
        h=lambda t:t.reshape(n,k,4,6).transpose(1,2)
        attention=h(self.q(x))@h(self.k(x)).transpose(-1,-2)/math.sqrt(6)
        if self.kind=='sparse':
            features,graph=self.relations(relative,base_logits)
            attention=attention+self.bias(features).permute(0,3,1,2)
            attention=attention.masked_fill(~graph[:,None],float('-inf'))
        elif self.kind=='self':
            attention=attention.masked_fill(~torch.eye(k,dtype=torch.bool,device=x.device),float('-inf'))
        attention=attention.softmax(-1)
        y=(attention@h(self.v(x))).transpose(1,2).reshape(n,k,24)
        x=self.n1(x+self.o(y));x=self.n2(x+self.ffn(x))
        shift=.30*scale[:,None,None]*torch.tanh(self.anchor_out(x)).reshape(n,k,3,2)
        return relative+shift,self.score_out(x).squeeze(-1)

def score_head():return nn.Sequential(nn.LayerNorm(70),nn.Linear(70,64),nn.GELU(),nn.Linear(64,1))
def make_basis():
    w=torch.zeros(12,3)
    for seg in range(3):
        for j in range(4):
            w[4*seg+j,seg]=(j+1)/4
            if seg:w[4*seg+j,seg-1]=1-(j+1)/4
    return w
def predict(refiner,head,cache,index=None):
    get=lambda k:cache[k] if index is None else cache[k][index]
    mode,rel,scale,z=[get(k) for k in ('mode','relative','scale','logits')]
    new,dz=refiner(mode,rel,scale,z)
    logits=head(torch.cat([mode,new.flatten(-2)],dim=-1)).squeeze(-1)+dz
    paths=torch.einsum('tp,nkpc->nktc',make_basis().to(new.device),new)
    return paths,logits,new-rel

def baseline_loss(paths,y,logits,criterion):
    pred=paths[:,None];target=y[:,None];z=logits[:,None]
    base,detail=criterion(pred,target,z)
    distances=torch.linalg.vector_norm(pred-target.unsqueeze(-3),dim=-1)
    quality=distances.mean(-1)+.25*distances[...,-1]
    responsibility=torch.softmax(-quality.detach()/.10,-1)
    coverage=(responsibility*quality).sum(-1).mean()
    return base+.20*coverage,detail

def allocation(paths,y,logits):
    with torch.no_grad():
        dist=torch.linalg.vector_norm(paths-y[:,None],dim=-1);quality=dist.mean(-1)+.25*dist[:,:,-1];winner=quality.argmin(-1)
        row=torch.arange(len(paths),device=paths.device);gate=dist[row,winner,-1]>.4
        end=paths[:,:,-1];pair=torch.linalg.vector_norm(end[:,None]-end[:,:,None],dim=-1)
        higher=logits[:,None]>logits[:,:,None]
        redundant_distance=pair.masked_fill(~higher,float('inf')).min(-1).values
        redundant_distance[row,winner]=float('inf');r=redundant_distance.argmin(-1)
        active=gate&torch.isfinite(redundant_distance[row,r])
    return winner,r,active,gate
def extra_loss(paths,y,logits):
    winner,r,active,gate=allocation(paths,y,logits);row=torch.arange(len(paths),device=paths.device)
    per_window=F.smooth_l1_loss(paths[row,r],y,reduction='none').mean(dim=(-1,-2))
    value=.5*(per_window*active).mean()
    return value,dict(winner=winner,r=r,active=active,gate=gate)
