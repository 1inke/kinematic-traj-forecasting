"""Synthetic unit tests only. No old data, real-model training, OS task or launch.

Uses tiny fixture optimizer updates solely to test zero-initialized derivatives.
This is NOT an actual FIT GPU preflight/resume/source/physics audit.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
from models import ContextK20,objective,temperature
sys.path.insert(0,str(Path(__file__).resolve().parent/'direction_A_time_basis'))
sys.path.insert(0,str(Path(__file__).resolve().parent/'direction_B_scene_prior'))
from basis import knot3,projector,project,cv,svd4,screen_gate
from prior import Record,EndpointPrior,bucket,probability,modes,heading

def run_tests():
    torch.set_num_threads(2);rng=np.random.default_rng(123)
    psi=knot3();p=projector(psi)
    d2=np.diff(np.vstack((np.zeros((1,3)),psi.T)),n=2,axis=0)
    assert np.flatnonzero(np.abs(d2).sum(1)).tolist()==[3,7]
    assert psi[0,0]==.25 and not np.all(psi[:,0]==0)
    past=rng.normal(size=(9,8,2));paths=rng.normal(size=(9,20,12,2))
    projected=project(paths,past,psi)
    np.testing.assert_allclose(project(projected,past,psi),projected,atol=2e-12,rtol=0)
    sv,svvalues=svd4(rng.normal(size=(9,12,2)),past)
    np.testing.assert_allclose(sv@sv.T,np.eye(4),atol=2e-12,rtol=0)
    assert (sv[:,-1]>0).all()
    assert not screen_gate(.97,[180]*3,[190]*3,166,[.7]*3)['passed']
    assert not screen_gate(1,[180]*3,[160]*3,166,[.7]*3)['passed']
    # Detached aWTA derivative is exactly nonnegative w/B, not softmax-through-Q.
    q=torch.tensor([[0.,.2,1.],[.4,.5,2.]],dtype=torch.double,requires_grad=True)
    w=torch.softmax(-q/.2,1).detach();value=(q*w).sum(1).mean()
    grad=torch.autograd.grad(value,q)[0]
    assert not w.requires_grad and torch.equal(grad,w/2) and (grad>=0).all()
    assert temperature(1)==.2 and abs(temperature(592)-.02)<1e-16
    # Full coordinate derivative against finite differences, away from faces.
    pred=torch.tensor(rng.normal(size=(2,20,12,2)),dtype=torch.double,requires_grad=True)
    gt=torch.tensor(rng.normal(size=(2,12,2)),dtype=torch.double)
    for arm in ('C1_hard','C1_awta'):
        val,di=objective(pred,gt,arm,137);grad=torch.autograd.grad(val,pred)[0]
        # aWTA's weights must remain frozen in FD, per stop-gradient contract.
        fixed=di['weights'].clone()
        def fixed_value(x):
            e=torch.linalg.vector_norm(x-gt[:,None],dim=-1);qq=e.mean(-1)+.5*e[:,:,-1]
            pair=torch.triu_indices(20,20,1);end=x[:,:,-1]
            div=torch.relu(.3-torch.linalg.vector_norm(end[:,pair[0]]-end[:,pair[1]],dim=-1)).mean()
            return (fixed*qq).sum(1).mean()+.05*div
        ix=(0,3,5,1)
        for eps in (1e-4,1e-5):
            plus=pred.detach().clone();minus=plus.clone();plus[ix]+=eps;minus[ix]-=eps
            fd=(fixed_value(plus)-fixed_value(minus))/(2*eps)
            torch.testing.assert_close(fd,grad[ix],atol=2e-9,rtol=0)
    means=np.zeros(210);stds=np.ones(210);x=torch.tensor(rng.normal(size=(3,210)),dtype=torch.float32)
    for arm in ('C1_hard','C1_awta','C2_isomlp','C2_setmix','A_cv_free','A_cv_basis'):
        torch.manual_seed(42);m=ContextK20(means,stds,arm,p)
        assert m(x).shape==(3,20,12,2)
        assert sum(t.numel() for t in m.parameters())==(98552 if arm.startswith('C2') else 82008)
        if arm.startswith('A_'):
            expected=x[:,16:18,None]*torch.arange(1,13)[None,None]*.4
            assert torch.equal(m(x),expected.transpose(1,2)[:,None].expand(-1,20,-1,-1))
    # Jacobian isolation/communication is tested at the hidden-slot interface.
    for arm in ('C2_isomlp','C2_setmix'):
        torch.manual_seed(42);m=ContextK20(means,stds,arm).double()
        torch.nn.init.normal_(m.mix2.weight,std=.1)
        h=torch.tensor(rng.normal(size=(1,20,128)),requires_grad=True,dtype=torch.double)
        out=m.communicate(h)[0,0].sum();jac=torch.autograd.grad(out,h)[0]
        if arm=='C2_isomlp': assert torch.count_nonzero(jac[0,1:])==0
        else: assert torch.count_nonzero(jac[0,1:])>0
        torch.nn.init.normal_(m.head.weight,std=.1)
        xx=x.double()
        jq=torch.autograd.grad(m(xx)[0,0].sum(),m.queries)[0]
        if arm=='C2_isomlp':assert torch.count_nonzero(jq[1:])==0
        else:assert torch.count_nonzero(jq[1:])>0
        # Full-network double parameter FD, no active-face change allowed.
        yy=y.double() if 'y' in locals() else torch.tensor(rng.normal(size=(3,12,2)),dtype=torch.double)
        val,info=objective(m(xx),yy,arm,1)
        gp=torch.autograd.grad(val,m.mix1.weight)[0][0,0]
        original=m.mix1.weight[0,0].item();winners=info['winner'].clone()
        for eps in (1e-4,1e-5):
            with torch.no_grad():m.mix1.weight[0,0]=original+eps
            vp,ip=objective(m(xx),yy,arm,1)
            with torch.no_grad():m.mix1.weight[0,0]=original-eps
            vm,im=objective(m(xx),yy,arm,1)
            assert torch.equal(winners,ip['winner']) and torch.equal(winners,im['winner'])
            torch.testing.assert_close((vp-vm)/(2*eps),gp,atol=2e-9,rtol=0)
            with torch.no_grad():m.mix1.weight[0,0]=original
        perm=torch.randperm(20)
        torch.testing.assert_close(m.communicate(h[:,perm]),m.communicate(h)[:,perm],atol=2e-12,rtol=0)
    h=torch.tensor([[[1.],[1.]]],requires_grad=True);torch.amax(h,dim=1).sum().backward()
    assert torch.equal(h.grad,torch.full_like(h,.5))
    # Phased gradient activation: head first, W2 next, then W1 becomes connected.
    m=ContextK20(means,stds,'C2_setmix');opt=torch.optim.AdamW(m.parameters(),lr=.001)
    found={k:False for k in ('mix1.weight','mix1.bias','mix2.weight','mix2.bias')}
    y=torch.tensor(rng.normal(size=(3,12,2)),dtype=torch.float32)
    for step in range(1,5):
        opt.zero_grad(set_to_none=True);loss,_=objective(m(x),y,'C2_setmix',step);loss.backward()
        for k,v in m.named_parameters():
            if k in found:found[k]|=bool(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.norm()>0)
        opt.step()
    assert all(found.values()),found
    assert bucket([6,6])==255 and bucket([6.001,0])==256 and bucket([-6,-6])==0
    np.testing.assert_allclose(probability(np.zeros(257)).sum(),1,atol=2e-12)
    assert modes(np.zeros(257)).shape==(4,2)
    assert heading(np.zeros((8,2)))==-1
    recs=[Record(i,'s::'+str(i),'s','hotel',i,i+12,(i%3,0),0,(.1,.2),True) for i in range(24)]
    recs.append(Record(24,'s::24','s','hotel',-1,-1,(0,0),0,(7.,0.),False))
    bank=EndpointPrior(recs,range(25))
    api=dict(sample_id=99,track='s::0',sequence='s',scene='hotel',t0=30,cell=(0,0),heading=0,verified_time=True)
    for arm in ('M0_motion_iso','PRIOR_static','SHUF_static','PRIOR_causal','SHUF_causal'):
        a=bank.query(arm=arm,**api)
        assert 's::0' not in a['source_tracks'] and np.isclose(a['probability'].sum(),1)
        if arm.endswith('causal') and not a['cold_start']: assert max(a['source_end_indices'])<=30
    assert bank.query(arm='PRIOR_causal',**dict(api,verified_time=False))['status']=='time_unverifiable'
    cold=bank.query(arm='PRIOR_causal',**dict(api,t0=0))
    assert cold['cold_start'] and cold['support']==0 and cold['level']=='M0_nonsequence'
    assert np.array_equal(bank.modes(bank.query(arm='M0_motion_iso',**api)),modes(bank.query(arm='M0_motion_iso',**api)['counts']))
    from common import require_split,owned_output
    for split in ('outer','original_CAL','cal_v11','test'):
        try:require_split(split)
        except ValueError:pass
        else:raise AssertionError('Forbidden split accepted')
    try:owned_output(Path(__file__).resolve().parents[2]/'old_results','direction_A_time_basis')
    except ValueError:pass
    else:raise AssertionError('Old/unowned output accepted')
    return dict(status='PASS_SYNTHETIC_ONLY',test_groups=18,actual_GPU_preflight=False,production_started=False)

if __name__=='__main__':
    import json
    print(json.dumps(run_tests(),ensure_ascii=False))
