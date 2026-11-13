"""Design equations, dimension parameterized; not production-authorized yet."""
import torch
from torch import nn


def compute_past_kinematic_state_6d(past,frequency=2.5):
    if past.ndim!=3 or past.shape[1:]!=(8,2):raise ValueError('PAST8_XY_ONLY')
    difference=(past[:,1:]-past[:,:-1])*frequency
    velocity=torch.cat([difference[:,:1],difference],1)
    acceleration=(velocity[:,2:]-velocity[:,1:-1])*frequency
    acceleration=torch.cat([acceleration[:,:1],acceleration[:,:1],acceleration],1)
    speed=torch.linalg.vector_norm(velocity,dim=-1,keepdim=True)
    fallback=torch.zeros_like(velocity);fallback[...,0]=1
    heading=torch.where(speed>1e-3,velocity/speed.clamp_min(1e-3),fallback)
    return torch.cat([velocity/1.5,acceleration/2.,heading],-1)


class GRUStateUpdate(nn.Module):
    def __init__(self,hidden_dim=32,out_dim=96):
        super().__init__()
        self.init=nn.Linear(6,hidden_dim,bias=False)
        self.cell=nn.GRUCell(6,hidden_dim)
        self.norm=nn.LayerNorm(hidden_dim)
        self.out=nn.Linear(hidden_dim,out_dim,bias=False)
        nn.init.xavier_normal_(self.init.weight)
        for name,p in self.cell.named_parameters():
            if 'weight' in name:nn.init.xavier_normal_(p)
            else:nn.init.zeros_(p)
        nn.init.zeros_(self.out.weight)
        self.diagnostics=None

    def forward(self,states):
        m1=torch.tanh(self.init(states[:,0]));m=m1
        changes=[]
        for t in range(1,8):
            previous=m;m=self.cell(states[:,t],m)
            changes.append(torch.linalg.vector_norm(m-previous,dim=-1))
        self.diagnostics={'state_change':torch.stack(changes,1).detach(),
                          'endpoint_state_change':torch.linalg.vector_norm(m-m1,dim=-1).detach()}
        return self.out(self.norm(m-m1))


class GatedDeltaStateUpdate(nn.Module):
    def __init__(self,hidden_dim=32,out_dim=96):
        super().__init__()
        self.init=nn.Linear(6,hidden_dim,bias=False)
        self.u=nn.Linear(6,hidden_dim,bias=False)
        self.predict=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.error=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.alpha=nn.Linear(6+hidden_dim,hidden_dim)
        self.beta=nn.Linear(6+hidden_dim,hidden_dim)
        self.norm=nn.LayerNorm(hidden_dim)
        self.out=nn.Linear(hidden_dim,out_dim,bias=False)
        for layer in (self.init,self.u,self.predict,self.error,self.alpha,self.beta):nn.init.xavier_normal_(layer.weight)
        nn.init.constant_(self.alpha.bias,1.)
        nn.init.constant_(self.beta.bias,-.5)
        nn.init.zeros_(self.out.weight)
        self.diagnostics=None

    def forward(self,states):
        m1=torch.tanh(self.init(states[:,0]));m=m1
        en,be,al=[],[],[]
        for t in range(1,8):
            e=self.u(states[:,t])-self.predict(m)
            joined=torch.cat([states[:,t],e],-1)
            a=torch.sigmoid(self.alpha(joined));b=torch.sigmoid(self.beta(joined))
            m=a*m+b*torch.tanh(self.error(e))
            en.append(torch.linalg.vector_norm(e,dim=-1));be.append(b.mean(-1));al.append(a.mean(-1))
        self.diagnostics={'innovation_norm':torch.stack(en,1).detach(),
            'beta':torch.stack(be,1).detach(),'alpha':torch.stack(al,1).detach(),
            'endpoint_state_change':torch.linalg.vector_norm(m-m1,dim=-1).detach()}
        return self.out(self.norm(m-m1))
