"""Gaussian-latent DSF adapter. No F16/h3 embeddings are treated as latents.

A real generator must implement encode_context(past)->[B,C] and
decode(z,context,p0)->[B,T,2] in WORLD METRES. No future argument exists.
"""
import torch
from torch import nn
from interface_common import read,check_files,sha

class FrozenGaussianGenerator(nn.Module):
    def __init__(self,generator,contract):
        super().__init__();self.generator=generator;self.contract=dict(contract)
        if contract.get('latent_prior')!='standard_normal' or contract.get('output_unit')!='world_metres' or contract.get('dt_seconds')!=.4 or contract.get('T')!=12:raise ValueError('GENERATOR_SEMANTICS')
        if not isinstance(contract.get('latent_dim'),int) or contract['latent_dim']<1:raise ValueError('LATENT_DIM')
        if not isinstance(contract.get('context_dim'),int) or contract['context_dim']<1:raise ValueError('CONTEXT_DIM')
        if not all(callable(getattr(generator,n,None)) for n in ('encode_context','decode')):raise ValueError('GENERATOR_API')
        generator.eval()
        for parameter in generator.parameters():parameter.requires_grad_(False);parameter.grad=None
    def train(self,mode=True):
        super().train(False);self.generator.eval();return self
    def context(self,past):
        if past.ndim!=3 or past.shape[1:]!=(8,2) or not torch.isfinite(past).all():raise ValueError('PAST_ONLY_INPUT')
        value=self.generator.encode_context(past)
        if value.shape!=(len(past),self.contract['context_dim']) or not torch.isfinite(value).all():raise ValueError('CONTEXT_OUTPUT')
        return value
    def decode(self,z,context,p0):
        if z.ndim!=3 or z.shape[1:]!=(20,self.contract['latent_dim']) or not torch.isfinite(z).all():raise ValueError('LATENT_SHAPE')
        n=len(z);flat=z.reshape(n*20,-1);c=context[:,None,:].expand(-1,20,-1).reshape(n*20,-1);p=p0[:,None,:].expand(-1,20,-1).reshape(n*20,2)
        # Do NOT use no_grad(): decoder parameters are fixed but z needs gradients.
        y=self.generator.decode(flat,c,p)
        if y.shape!=(n*20,12,2) or not torch.isfinite(y).all():raise ValueError('DECODER_OUTPUT')
        return y.reshape(n,20,12,2)

class DSFSampler(nn.Module):
    def __init__(self,context_dim,latent_dim):
        super().__init__();self.latent_dim=latent_dim
        self.net=nn.Sequential(nn.Linear(context_dim,128),nn.ReLU(),nn.Linear(128,20*latent_dim))
    def forward(self,context):return self.net(context).reshape(len(context),20,self.latent_dim)

class DSFPipeline(nn.Module):
    def __init__(self,generator,sampler):super().__init__();self.generator=generator;self.sampler=sampler
    def forward(self,past):
        context=self.generator.context(past);z=self.sampler(context)
        return dict(paths=self.generator.decode(z,context,past[:,-1]),z=z)

def load_generator(contract_json,factory):
    c=read(contract_json)
    if c.get('status')!='PRETRAINED_GENERATOR_ADMITTED' or not c.get('fit_only_training_receipt') or not c.get('evaluation_receipt'):raise ValueError('UNTRAINED_OR_UNAUDITED_GENERATOR')
    check_files(c['evidence_files'])
    for key in ('fit_only_training_receipt','evaluation_receipt','architecture_source','normalization_receipt'):
        if c[key] not in c['evidence_files']:raise ValueError('UNBOUND_GENERATOR_EVIDENCE '+key)
    if sha(c['checkpoint'])!=c['checkpoint_sha256']:raise ValueError('GENERATOR_CHECKPOINT_SHA')
    model=factory(c) # caller supplies the reviewed architecture, never arbitrary remote imports.
    state=torch.load(c['checkpoint'],map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    return FrozenGaussianGenerator(model,c)

def kinematic_torch(past,paths,c):
    if past.shape!=(len(paths),8,2) or paths.shape[1:]!=(20,12,2) or not torch.isfinite(past).all() or not torch.isfinite(paths).all():raise ValueError('KINEMATIC_INPUT_CONTRACT')
    import math
    if any(not math.isfinite(c[key]) or c[key]<=0 for key in ('sigma_v','sigma_a','sigma_join')):raise ValueError('KINEMATIC_NORMALIZATION_SCALE')
    hist=(past[:,-1]-past[:,-2])/.4
    positions=torch.cat([past[:,-1,None,None].expand(-1,20,1,-1),paths],dim=2)
    v=(positions[:,:,1:]-positions[:,:,:-1])/.4
    a=(torch.cat([hist[:,None,None].expand(-1,20,1,-1),v],2)[:,:,1:]-torch.cat([hist[:,None,None].expand(-1,20,1,-1),v],2)[:,:,:-1])/.4
    penalty=(torch.relu((v.norm(dim=-1)-c['vmax'])/c['sigma_v'])**2).mean(-1)+(torch.relu((a.norm(dim=-1)-c['amax'])/c['sigma_a'])**2).mean(-1)+torch.relu(((v[:,:,0]-hist[:,None]).norm(dim=-1)-c['joinmax'])/c['sigma_join'])**2
    return torch.exp(-penalty)

def expected_cardinality(paths,q,k,epsilon=1e-6):
    if paths.shape[1:]!=(20,12,2) or q.shape!=paths.shape[:2] or not torch.isfinite(q).all() or (q<0).any() or (q>1).any() or k<=0 or not 0<epsilon<1:raise ValueError('DPP_CONTRACT')
    d=((paths[:,:,None]-paths[:,None])**2).sum((-1,-2));r=torch.sqrt(epsilon+(1-epsilon)*q)
    L=torch.exp(-k*d)*r[:,:,None]*r[:,None];I=torch.eye(20,device=paths.device,dtype=paths.dtype).expand_as(L)
    return (1-torch.linalg.solve(I+L,I).diagonal(dim1=-2,dim2=-1)).sum(-1)

def original_radius_weight(z,radius,omega=1.):
    if radius<=0 or omega<=0:raise ValueError('RADIUS')
    # This is r, NOT q. The original paper multiplier is retained unchanged.
    return omega*torch.exp(torch.minimum(torch.zeros_like(z[...,0]),radius**2-(z*z).sum(-1)))

def quality_weights(arm,z,qkin=None,qscene=None,radius=None,omega=1.,epsilon=1e-6):
    if arm=='B0_original':return original_radius_weight(z,radius,omega)
    if arm=='B4_uniform':return torch.ones(z.shape[:2],dtype=z.dtype,device=z.device)
    if arm not in ('B1_scene','B2_kin','B3_joint'):raise ValueError('UNKNOWN_ARM')
    if arm in ('B1_scene','B3_joint'):
        if qscene is None or qscene.shape!=z.shape[:2] or not torch.isfinite(qscene).all():raise ValueError('REAL_SCENE_SCORE_REQUIRED_NO_DEFAULT')
    if arm in ('B2_kin','B3_joint'):
        if qkin is None or qkin.shape!=z.shape[:2] or not torch.isfinite(qkin).all():raise ValueError('KIN_SCORE_REQUIRED')
    q=qscene if arm=='B1_scene' else qkin if arm=='B2_kin' else qscene*qkin
    if (q<0).any() or (q>1).any() or not 0<epsilon<1:raise ValueError('QUALITY_RANGE')
    return torch.sqrt(epsilon+(1-epsilon)*q)

def ec_from_weights(paths,r,k):
    if paths.shape[1:]!=(20,12,2) or r.shape!=paths.shape[:2] or not torch.isfinite(paths).all() or not torch.isfinite(r).all() or (r<0).any() or k<=0:raise ValueError('DPP_WEIGHT_CONTRACT')
    d=((paths[:,:,None]-paths[:,None])**2).sum((-1,-2));L=torch.exp(-k*d)*r[:,:,None]*r[:,None]
    I=torch.eye(20,device=paths.device,dtype=paths.dtype).expand_as(L)
    return (1-torch.linalg.solve(I+L,I).diagonal(dim1=-2,dim2=-1)).sum(-1)

def scene_torch(p0,paths,map_arrays,spacing=.05,margin=.1,tau=.1):
    """Piecewise differentiable counterpart of Stage A scene_score.

    Adaptive subdivision counts and known masks are nondifferentiable admission
    decisions. Values/coordinates within each admitted region remain autograd.
    Unknown returns NaN and an explicit coverage denominator, not a free path.
    """
    import numpy as np
    import torch.nn.functional as F
    if paths.shape!=(20,12,2) or p0.shape!=(2,) or spacing<=0 or tau<=0:raise ValueError('SCENE_TORCH_SHAPE')
    dtype,device=paths.dtype,paths.device
    G=torch.as_tensor(map_arrays['world_to_grid'],dtype=dtype,device=device)
    known=torch.as_tensor(map_arrays['known'],dtype=torch.bool,device=device)
    sdf=torch.as_tensor(np.nan_to_num(map_arrays['distance_metres'],nan=0.),dtype=dtype,device=device)[None,None]
    height,width=known.shape;qualities=[];fractions=[]
    for path in paths:
        nodes=torch.cat([p0[None],path],0);points=[]
        for a,b in zip(nodes[:-1],nodes[1:]):
            count=max(1,int(torch.ceil((b-a).norm().detach()/spacing).item()))
            if count>100000:raise ValueError('UNBOUNDED_SEGMENT_SAMPLING')
            t=torch.arange(1,count+1,device=device,dtype=dtype)/count;points.append(a+(b-a)*t[:,None])
        pts=torch.cat(points);h=torch.cat([pts,torch.ones_like(pts[:,:1])],1)@G.T
        finite=torch.isfinite(h).all(1)&(h[:,2]!=0)
        xy=h[:,:2]/torch.where(finite,h[:,2],torch.ones_like(h[:,2]))[:,None]
        ix=torch.floor(xy[:,0]).long();iy=torch.floor(xy[:,1]).long()
        inside=finite&(ix>=0)&(iy>=0)&(ix<width-1)&(iy<height-1)
        safe_x=ix.clamp(0,width-2);safe_y=iy.clamp(0,height-2)
        ok=inside&known[safe_y,safe_x]&known[safe_y+1,safe_x]&known[safe_y,safe_x+1]&known[safe_y+1,safe_x+1]
        fractions.append(ok.to(dtype).mean())
        if not ok.any():qualities.append(torch.full((),float('nan'),dtype=dtype,device=device));continue
        selected=xy[ok]
        grid=torch.stack([2*selected[:,0]/(width-1)-1,2*selected[:,1]/(height-1)-1],-1)[None,None]
        distance=F.grid_sample(sdf,grid,mode='bilinear',align_corners=True,padding_mode='zeros').flatten()
        qualities.append(torch.exp(-F.softplus((margin-distance)/tau).mean()))
    return torch.stack(qualities),torch.stack(fractions)
