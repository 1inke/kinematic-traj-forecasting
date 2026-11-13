"""B-line building blocks only. A native/KIN adapter with proven FIT ownership
is REQUIRED before this can become a runnable scientific experiment.
"""
import torch
from torch import nn

class SoftHybrid(nn.Module):
    def __init__(self,native_head,kin_head,query_dim,context_dim):
        super().__init__();self.native_head=native_head;self.kin_head=kin_head;self.gate=nn.Linear(query_dim+context_dim,1)
        nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)
    def forward(self,queries,observed_context,*,hard_diagnostic=False):
        if queries.ndim!=3 or queries.shape[1]!=20:raise ValueError('K20 queries required')
        h=observed_context[:,None].expand(-1,20,-1);x=torch.cat((queries,h),-1)
        native=self.native_head(queries,observed_context);kin=self.kin_head(queries,observed_context)
        if native.shape!=kin.shape or native.shape[1:]!=(20,12,2):raise ValueError('Both heads must output K20 x12 x2')
        alpha=torch.sigmoid(self.gate(x))
        mix=(alpha>=.5).to(alpha.dtype) if hard_diagnostic else alpha
        p=mix[:,:,:,None]*native+(1-mix[:,:,:,None])*kin
        return dict(paths=p,native_head=native,kin_head=kin,alpha=alpha)

def set_loss(paths,gt,alpha,*,temperature,coverage_weight,entropy_weight,margin,softmin_definition):
    if temperature<=0 or softmin_definition not in ('logsumexp','logmeanexp'):raise ValueError('Explicit positive temperature and softmin convention required')
    d=torch.linalg.vector_norm(paths-gt[:,None],dim=-1);ade=d.mean(-1);fde=d[:,:,-1];q=ade+.5*fde
    weights=torch.softmax(-q/temperature,dim=1);wta=(weights*q).sum(1).mean()
    j=torch.maximum(ade/.2,fde/.4);soft=-temperature*torch.logsumexp(-j/temperature,dim=1)
    if softmin_definition=='logmeanexp':soft=soft+temperature*torch.log(torch.tensor(paths.shape[1],dtype=paths.dtype,device=paths.device))
    coverage=torch.relu(soft-1+margin).mean()
    # Entropy of batch/slot usage, NOT individual-alpha entropy forcing all to.5.
    mean=alpha.mean().clamp(1e-7,1-1e-7);negative_entropy=mean*mean.log()+(1-mean)*(1-mean).log()
    return wta+coverage_weight*coverage+entropy_weight*negative_entropy,dict(softWTA=wta,coverage=coverage,negative_entropy=negative_entropy)

def fit_teacher_chamfer(student,teacher,good,sample_ids,fit_ids):
    if not set(sample_ids).issubset(set(fit_ids)):raise ValueError('Teacher supervision must be FIT-only')
    terms=[]
    for p,t,g in zip(student,teacher,good):
        if bool(g.any()):terms.append(torch.linalg.vector_norm(t[g,None]-p[None],dim=-1).mean(-1).min(1).values.mean())
    return torch.stack(terms).mean() if terms else student.sum()*0
