"""Same-slot accuracy + bounded EC term; CVAE auxiliary likelihood/KL explicit."""
import torch

def same_slot_accuracy(paths,future):
    if paths.shape[1:]!=(20,12,2) or future.shape!=(len(paths),12,2): raise ValueError('ACCURACY_SHAPE')
    distance=torch.linalg.vector_norm(paths.double()-future[:,None].double(),dim=-1)
    ade=distance.mean(-1);fde=distance[:,:,-1]
    # min(dim) uses FIRST physical tie, unlike amax distributed tie gradients.
    joint=ade+.5*fde;value,winner=joint.min(1)
    return value.mean(),dict(ADE=ade,FDE=fde,winner=winner)

def quality(past,paths,c):
    h=(past[:,-1]-past[:,-2])/.4
    p=torch.cat([past[:,-1,None,None].expand(-1,20,1,-1),paths],2)
    v=(p[:,:,1:]-p[:,:,:-1])/.4
    vv=torch.cat([h[:,None,None].expand(-1,20,1,-1),v],2)
    a=(vv[:,:,1:]-vv[:,:,:-1])/.4
    pv=torch.relu((v.norm(dim=-1)-c['vmax'])/c['sigma_v']).square().mean(-1)
    pa=torch.relu((a.norm(dim=-1)-c['amax'])/c['sigma_a']).square().mean(-1)
    pj=torch.relu(((v[:,:,0]-h[:,None]).norm(dim=-1)-c['joinmax'])/c['sigma_join']).square()
    return torch.exp(-pv-pa-pj)

def expected_cardinality(paths,q,cal):
    if q.shape!=paths.shape[:2] or not torch.isfinite(q).all() or ((q<0)|(q>1)).any(): raise ValueError('QUALITY_RANGE')
    squared=(paths[:,:,None]-paths[:,None]).square().sum((-1,-2))
    r=torch.sqrt(cal['epsilon_q']+(1-cal['epsilon_q'])*q)
    L=torch.exp(-cal['k']*squared)*r[:,:,None]*r[:,None]
    eye=torch.eye(20,device=paths.device,dtype=paths.dtype).expand_as(L)
    return (eye-torch.linalg.solve(eye+L,eye)).diagonal(dim1=-2,dim2=-1).sum(-1)

def loss(model,family,arm,bundle,future,cal,spec):
    if arm not in ('accuracy_only','accuracy_dpp'): raise ValueError('ARM')
    paths,aux=model.generate(bundle) # No future argument on prediction API.
    accuracy,per=same_slot_accuracy(paths,future)
    q=quality(bundle['raw_past'][:,0].double(),paths.double(),cal)
    ec=expected_cardinality(paths.double(),q,cal)
    coefficient=spec['dpp']['lambda'] if arm=='accuracy_dpp' else 0.
    recon=accuracy.new_zeros(());kl=accuracy.new_zeros(())
    if family=='cvae':
        recon,kl=model.posterior_training_terms(bundle,future,sigma=spec['cvae']['reconstruction_sigma_metres'])
    total=accuracy-coefficient*ec.mean()/20+recon+spec['cvae']['beta_KL']*kl
    return total,dict(accuracy=accuracy,EC=ec.mean(),qkin=q.mean(),reconstruction=recon,KL=kl,
                     paths=paths,winner=per['winner'],aux=aux)
