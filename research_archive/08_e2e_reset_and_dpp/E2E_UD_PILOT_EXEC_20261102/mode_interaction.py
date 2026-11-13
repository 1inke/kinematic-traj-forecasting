"""Single/differential softmax, same normalisation/gate and zero output."""
import math
import torch
from torch import nn


class GatedModeAttention(nn.Module):
    def __init__(self,dim=96,proj_dim=32,heads=4,differential=False,lambda_fixed=.5):
        super().__init__()
        if proj_dim%heads:raise ValueError('HEAD_DIMENSION')
        self.heads=heads;self.width=proj_dim//heads;self.differential=differential
        self.lambda_fixed=float(lambda_fixed)
        self.norm=nn.LayerNorm(dim)
        self.q1=nn.Linear(dim,proj_dim,bias=False)
        self.k1=nn.Linear(dim,proj_dim,bias=False)
        self.value=nn.Linear(dim,proj_dim,bias=False)
        if differential:
            self.q2=nn.Linear(dim,proj_dim,bias=False);self.k2=nn.Linear(dim,proj_dim,bias=False)
        self.gate=nn.Linear(dim*2,proj_dim)
        self.onorm=nn.LayerNorm(proj_dim)
        self.out=nn.Linear(proj_dim,dim,bias=False)
        for name,layer in self.named_modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.out.weight)
        self.diagnostics=None

    def forward(self,z,context):
        normalized=self.norm(z);n,k,_=z.shape
        def heads(x):return x.reshape(n,k,self.heads,self.width).transpose(1,2)
        q1,k1,v=[heads(layer(normalized)) for layer in (self.q1,self.k1,self.value)]
        a1=torch.softmax(q1@k1.transpose(-1,-2)/math.sqrt(self.width),-1)
        weights=a1
        if self.differential:
            q2,k2=[heads(layer(normalized)) for layer in (self.q2,self.k2)]
            a2=torch.softmax(q2@k2.transpose(-1,-2)/math.sqrt(self.width),-1)
            weights=a1-self.lambda_fixed*a2
        o=(weights@v).transpose(1,2).reshape(n,k,-1)
        gate=torch.sigmoid(self.gate(torch.cat([normalized,context[:,None].expand(-1,k,-1)],-1)))
        update=self.out(gate*self.onorm(o))
        self.diagnostics={'negative_fraction':(weights<0).float().mean().detach(),
            'attention_weights':weights.detach(),'gate_mean':gate.mean().detach(),
            'update_norm':torch.linalg.vector_norm(update,dim=-1).detach()}
        return z+update


class DifferentialGatedModeAttention(GatedModeAttention):
    def __init__(self,dim=96,proj_dim=32,heads=4,lambda_fixed=.5):
        super().__init__(dim,proj_dim,heads,True,lambda_fixed)
