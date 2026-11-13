"""Q only: preserve original group0 construction and Bx20 kernels."""
import torch
from torch import nn
from common import old

class GroupContextK20(old.ContextK20):
    def __init__(self,mean,std,groups=1,aux_seed=2026110142):
        if groups not in (1,3):raise ValueError('Only registered Q groups')
        super().__init__(mean,std)
        g=torch.Generator(device='cpu').manual_seed(aux_seed)
        self.aux=nn.ParameterList([nn.Parameter(torch.randn(20,16,generator=g)*.02) for _ in range(groups-1)])
        assert sum(p.numel() for p in self.parameters())==82008+320*(groups-1)
    def one(self,features,q):
        h=self.trunk((features-self.mean)/self.std);h=h[:,None].expand(-1,20,-1)
        z=self.fuse(torch.cat((h,q[None].expand(len(h),-1,-1)),dim=-1))
        return self.head(z).reshape(-1,20,12,2)
    def forward(self,features,train_groups=True):
        # Main path identical to archived ContextK20. No GT/ID/group/logit API.
        if not train_groups:return self.one(features,self.queries)
        # Shared trunk computed once as specified; same Bx20 decoder each group.
        h=self.trunk((features-self.mean)/self.std);hh=h[:,None].expand(-1,20,-1)
        outputs=[]
        for q in [self.queries,*self.aux]:
            z=self.fuse(torch.cat((hh,q[None].expand(len(h),-1,-1)),dim=-1))
            outputs.append(self.head(z).reshape(-1,20,12,2))
        return torch.stack(outputs,1)
    def export(self):
        m=old.ContextK20(self.mean.detach().cpu(),self.std.detach().cpu())
        m.load_state_dict({k:v for k,v in self.state_dict().items() if not k.startswith('aux.')});return m
