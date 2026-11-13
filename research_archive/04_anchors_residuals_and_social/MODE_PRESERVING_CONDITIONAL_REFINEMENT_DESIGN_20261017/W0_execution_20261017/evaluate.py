"""Actual float32 U0 controls, independent saved arithmetic, literal scientific gates."""
import common as c
import numpy as np
K=np.asarray([[.20/144*(2*(t-q)+1) if q<=t else 0. for q in range(12)] for t in range(12)])
BOUND=.20*(np.arange(1,13)/12)**2
def reconstruct(native,primal):
    import torch
    primal=np.asarray(primal,float);assert primal.shape==(20,12,2)
    assert np.isfinite(primal).all() and (np.abs(primal)<1).all()
    raw=np.arctanh(primal).astype(np.float32)
    with torch.no_grad():
        v=torch.tanh(torch.as_tensor(raw,device='cuda')).double()
        controls=(v/torch.clamp(torch.linalg.vector_norm(v,dim=-1,keepdim=True),min=1))[None]
        correction=torch.einsum('tj,bkjd->bktd',torch.as_tensor(K,device='cuda'),controls)
        paths=torch.as_tensor(native.astype(np.float32),device='cuda').double()[None]+correction
    return dict(raw=raw,controls=controls.cpu().numpy()[0],correction=correction.cpu().numpy()[0],paths=paths.cpu().numpy()[0])
def independent_paths(native,controls):
    r=np.zeros_like(controls,dtype=float)
    for t in range(12):
        for q in range(t+1):r[:,t]+=.20/144*(2*(t-q)+1)*controls[:,q]
    return native.astype(np.float32).astype(float)+r
def certificate(native,gt,slot,z,ind=False):
    paths=z['paths'];a,f,j,g,_=c.errors(paths[None],gt[None],ind);nm=c.motion(native[None],ind);pm=c.motion(paths[None],ind)
    reasons=[]
    if not g[0,slot]:reasons.append('TARGET_SAME_SLOT_QUALITY')
    if not (np.linalg.norm(z['controls'],axis=-1)<=1).all():reasons.append('CONTROL_DISK')
    if not (np.linalg.norm(z['correction'],axis=-1)<=BOUND).all():reasons.append('POSITION_BOUND')
    for k in nm:
        if pm[k]>1.01*nm[k]:reasons.append(k)
    vel=np.diff(paths,axis=1,prepend=np.zeros_like(paths[:,:1]))/.4;acc=np.diff(vel,axis=1)/.4
    nv=np.diff(native,axis=1,prepend=np.zeros_like(native[:,:1]))/.4;na=np.diff(nv,axis=1)/.4
    return dict(pass_literal=not reasons,reasons=reasons,slot=int(slot),ADE=float(a[0,slot]),FDE=float(f[0,slot]),J=float(j[0,slot]),quality_margin_ADE=.2-float(a[0,slot]),quality_margin_FDE=.4-float(f[0,slot]),
      motion=pm,native_motion=nm,all20_candidate_delta_accel_mean=np.linalg.norm(acc,axis=-1).mean(1)-np.linalg.norm(na,axis=-1).mean(1),
      modified_slots=int((np.linalg.norm(paths-native,axis=-1).max(1)>0).sum()),nonzero=bool(np.any(paths!=native)))
def primal_residual(native,gt,slot,u,gamma=0.):
    u=np.asarray(u,float);paths=native+np.einsum('tq,kqd->ktd',K,u)
    a,f,j,g,_=c.errors(paths[None],gt[None]);nm=c.motion(native[None]);pm=c.motion(paths[None])
    return dict(gamma=float(gamma),gamma_low_excess=float(-gamma),gamma_high_excess=float(gamma-1),max_internal_radius_excess=float(np.linalg.norm(u,axis=-1).max()-(1-2**-20)),margin_ADE_excess=float(a[0,slot]-.2*(1-gamma)),margin_FDE_excess=float(f[0,slot]-.4*(1-gamma)),finite=bool(np.isfinite(u).all()),max_control_norm_minus1=float(np.linalg.norm(u,axis=-1).max()-1),max_position_excess=float((np.linalg.norm(paths-native,axis=-1)-BOUND).max()),target_ADE_excess=float(a[0,slot]-.2),target_FDE_excess=float(f[0,slot]-.4),accel_mean_excess=pm['accel_mean']-1.01*nm['accel_mean'],jerk_mean_excess=pm['jerk_mean']-1.01*nm['jerk_mean'])
def portfolio(z,paths,ind=False):
    native=z['native'];a,f,j,g,segments=c.errors(paths,z['gt'],ind);ng=c.errors(native,z['gt'],ind)[3].any(1);good=g.any(1);cluster=c.clusters(paths,g,ind);rows={}
    for group,mask in c.masks(z['groups']):
        pm=c.motion(paths[mask],ind);nm=c.motion(native[mask],ind)
        rows[group]=dict(n=int(mask.sum()),native_good=int(ng[mask].sum()),good=int(good[mask].sum()),rescue=int((good&~ng&mask).sum()),lost=int((~good&ng&mask).sum()),coverage=float(good[mask].mean()),minADE=float(a[mask].min(1).mean()),minFDE=float(f[mask].min(1).mean()),minJ=float(j[mask].min(1).mean()),effective_good=float(cluster[mask,1].mean()),distinct=float(cluster[mask,0].mean()),singleJslot_segments=segments[mask].mean(0),motion=pm,native_motion=nm,all5motion_pass=all(pm[k]<=1.01*nm[k] for k in pm),rescued_ids=z['sample_id'][good&~ng&mask],lost_ids=z['sample_id'][~good&ng&mask])
    return rows,dict(ADE=a,FDE=f,J=j,good=g,clusters=cluster,segments=segments,sample_id=z['sample_id'])

