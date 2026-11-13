"""Past-only generation APIs. CVAE posterior is deliberately separate."""
import torch
from torch import nn
from joint_common import legacy, seed_all

class ExistingDSF(nn.Module):
    def __init__(self,seed,device='cpu'):
        super().__init__()
        self.adapter,_=legacy().load_current(device)
        self.sampler=legacy().new_sampler(seed,device)
    def generate(self,bundle):
        z=self.sampler(bundle['sampler_context'])
        y,_=legacy().decode_logits(self.adapter,z,bundle)
        return y,dict(control=z)
    def frozen_state(self): return self.adapter.model.state_dict()

class ConditionalVAE(nn.Module):
    def __init__(self,seed,latent=16,hidden=128):
        super().__init__();seed_all(seed);self.latent=latent
        self.prior=nn.Sequential(nn.Linear(64,hidden),nn.ReLU(),nn.Linear(hidden,2*latent))
        self.posterior=nn.Sequential(nn.Linear(88,hidden),nn.ReLU(),nn.Linear(hidden,2*latent))
        self.decoder=nn.Sequential(nn.Linear(64+latent,hidden),nn.ReLU(),
                                   nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,24))
        self.sampler=nn.Sequential(nn.Linear(64,hidden),nn.ReLU(),nn.Linear(hidden,20*latent))
        nn.init.zeros_(self.sampler[-1].weight);nn.init.zeros_(self.sampler[-1].bias)
        g=torch.Generator().manual_seed(731000+seed)
        self.register_buffer('base_noise',torch.randn(20,latent,generator=g))
    @staticmethod
    def gaussian(x):
        mu,logvar=x.chunk(2,-1)
        return mu,logvar.clamp(-8.,8.)
    def decode_local(self,c,z):
        expanded=c[:,None].expand(-1,z.shape[1],-1)
        return self.decoder(torch.cat([expanded,z],-1)).reshape(len(c),z.shape[1],12,2)
    @staticmethod
    def to_world(y,bundle):
        return torch.einsum('bktd,bcd->bktc',y,bundle['matrix'])+bundle['origin'][:,None,None]
    @staticmethod
    def to_local(y,bundle):
        return torch.einsum('btd,bdc->btc',y-bundle['origin'][:,None],bundle['matrix'])
    def generate(self,bundle):
        c=bundle['sampler_context'];mu,lv=self.gaussian(self.prior(c))
        eps=self.base_noise[None]+self.sampler(c).reshape(len(c),20,self.latent)
        z=mu[:,None]+torch.exp(.5*lv)[:,None]*eps
        local=self.decode_local(c,z)
        return self.to_world(local,bundle),dict(z=z,epsilon=eps,prior_mu=mu,prior_logvar=lv)
    def posterior_training_terms(self,bundle,future,*,sigma=.2):
        c=bundle['sampler_context'];target=self.to_local(future,bundle)
        qm,qlv=self.gaussian(self.posterior(torch.cat([c,target.flatten(1)],-1)))
        pm,plv=self.gaussian(self.prior(c))
        z=qm+torch.exp(.5*qlv)*torch.randn_like(qm)
        reconstruction=self.decode_local(c,z[:,None])[:,0]
        # Gaussian NLL up to an additive fixed constant, NOT called ADE/FDE.
        recon=.5*((reconstruction-target)/sigma).square().mean((1,2))
        kl=.5*(plv-qlv+(qlv.exp()+(qm-pm).square())/plv.exp()-1).sum(-1)
        return recon.mean(),kl.mean()
    def frozen_state(self): return {}

def make(family,seed,device='cpu'):
    seed_all(seed)
    if family=='existing': return ExistingDSF(seed,device).to(device)
    if family=='cvae': return ConditionalVAE(seed).to(device)
    raise ValueError('UNKNOWN_FAMILY')

def parameter_inventory(model):
    total=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    posterior=sum(p.numel() for p in model.posterior.parameters()) if isinstance(model,ConditionalVAE) else 0
    is_cvae=isinstance(model,ConditionalVAE)
    return dict(stored_parameters=total,trainable_parameters=trainable,
                inference_module_parameters=total-posterior,
                posterior_training_only_parameters=posterior,
                old305000_target_PASS=None if is_cvae else total<=305000,
                frozen_observation_encoder_not_counted_for_cached_CVAE=is_cvae,
                whole_model_count='PENDING_MATCHED_F16_CONTEXT_SCOPE' if is_cvae else total)
