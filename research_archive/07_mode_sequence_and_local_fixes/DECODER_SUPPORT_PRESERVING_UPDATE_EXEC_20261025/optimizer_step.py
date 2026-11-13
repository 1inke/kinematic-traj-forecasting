"""Exact task AdamW proposal followed by U/N/P full-parameter update arbitration."""
from task_adapter import *
from step_projection import project,cast_certificate

def flat(ps):return torch.cat([p.detach().reshape(-1).double() for p in ps]).cpu().numpy()

def assign_flat(ps,value):
    i=0
    with torch.no_grad():
        for p in ps:
            n=p.numel();p.copy_(torch.as_tensor(value[i:i+n],device=p.device,dtype=p.dtype).reshape(p.shape));i+=n
    assert i==len(value)

def step(m,o,x,y,group,arm,measure_after=False,save_vectors=False):
    assert arm in ARMS
    m.train();ps=tuple(m.parameters());theta0=flat(ps)
    p=m(x);value,info=task(p,y);jm,ji=quality(p,y)
    rows=[];R=[];counts=[];gradient_norms=[]
    for k in (0,1):
        mask=torch.as_tensor(group==k,device=p.device);counts.append(int(mask.sum()))
        if mask.any():
            r=jm[mask].mean();gg=torch.autograd.grad(r,ps,retain_graph=True,allow_unused=False)
            row=torch.cat([v.detach().reshape(-1).double() for v in gg]).cpu().numpy()
            assert np.isfinite(row).all();R.append(float(r.detach()));gradient_norms.append(float(np.linalg.norm(row)));rows.append(row)
        else:R.append(None);gradient_norms.append(0.);rows.append(np.zeros_like(theta0))
    o.zero_grad(set_to_none=True);value.backward()
    assert torch.isfinite(value) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps)
    gn=torch.nn.utils.clip_grad_norm_(ps,5.);assert torch.isfinite(gn)
    o.step();adam=flat(ps);d0=adam-theta0
    dP,cert=project(d0,rows);n0=float(np.linalg.norm(d0));nP=float(np.linalg.norm(dP));ratio=nP/n0 if n0 else 0.
    dN=ratio*d0
    ideal=d0 if arm=='U' else dN if arm=='N' else dP
    if arm!='U':assign_flat(ps,theta0+ideal)
    actual=flat(ps)-theta0
    cc=cast_certificate(theta0,ideal,actual,cert['G'],cert['tolerance'])
    if arm=='P':assert not len(cert['G']) or max(cert['G']@actual)<=cc['cast_error_norm']+cert['tolerance']
    rec=dict(arm=arm,n=len(x),counts=counts,R_before=R,task_before=float(value.detach()),task_component=float(info['task'].detach()),diversity=float(info['div'].detach()),gradient_norms=gradient_norms,preclip_task_grad_norm=float(gn),d0_norm=n0,dP_norm=nP,d_actual_norm=float(np.linalg.norm(actual)),norm_ratio=ratio,exact_zero_P=bool(nP==0),raw_Gd0=cert['raw_Gd0'],raw_GdP=cert['raw_GdP'],active=cert['active'],multipliers=cert['multipliers'],stationarity_norm=cert['stationarity_norm'],solver_tolerance=cert['tolerance'],cast=cc,winner_counts=torch.bincount(info['winner'],minlength=20).cpu().numpy(),Jslot_counts=torch.bincount(ji['slot'],minlength=20).cpu().numpy(),J_extra_ties=int((ji['j_ties']-1).sum()),J_component_ties=int(ji['component_ties'].sum()),G_rows=len(cert['G']),raw_cosines=[float(v/n0) if n0 else None for v in cert['raw_Gd0']],delta_angle_cosine=float(np.dot(d0,dP)/(n0*nP)) if n0 and nP else None)
    if measure_after:
        # No stochastic layers. Preserve all RNG around diagnostic forward explicitly.
        g=torch.Generator();rng=capture_rng(g)
        with torch.no_grad():
            pp=m(x);jafter,_=quality(pp,y);aft=[]
            for k in (0,1):
                mask=torch.as_tensor(group==k,device=pp.device);aft.append(float(jafter[mask].mean()) if mask.any() else None)
            rec.update(R_after=aft,task_after=float(task(pp,y)[0]),h=float(sum(max(a-b,0) for a,b in zip(aft,R) if b is not None)))
        restore_rng(rng,g)
    vectors=dict(theta0=theta0,d0=d0,dP=dP,dN=dN,rows=np.asarray(rows),G=cert['G'],d_actual=actual) if save_vectors else None
    return rec,vectors
