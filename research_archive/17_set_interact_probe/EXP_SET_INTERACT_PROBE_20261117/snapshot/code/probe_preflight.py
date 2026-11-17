import copy,hashlib,importlib.util,sys,time
import numpy as np
import torch
from probe_io import ROOT,OLD,read,write,save_torch,seal,now
from probe_train import Trainer,initialize,cpu_tree
from probe_model import predict,baseline_loss,make_basis
def technical_preflight(data,head,criterion,device):
    path=ROOT/'preflight/REAL_CACHE.json'
    if path.exists():
        from probe_io import verify
        verify(read(path)['evidence_files']);return
    fit=data['fit']['tensors'];cache={k:v[:65].clone() for k,v in fit.items()};checks=[]
    for arm in ('A1','A2','A3','A4','A5','A6'):
        model=initialize(arm,0,device)
        with torch.no_grad():
            p,z,shift=predict(model,head,cache)
            expected=torch.einsum('tp,nkpc->nktc',make_basis().to(device),cache['relative'])
            torch.testing.assert_close(p,expected,atol=0,rtol=0);torch.testing.assert_close(z,cache['logits'],atol=1e-4,rtol=1e-5)
            checks.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),max_init_logit_difference=float((z-cache['logits']).abs().max())))
    assert len(set(v['parameters'] for v in checks))==1
    # Compare original B0 loss value and derivatives against its independent NumPy implementation.
    spec=importlib.util.spec_from_file_location('old_independent_loss',OLD/'loss_independent.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    p=torch.einsum('tp,nkpc->nktc',make_basis().to(device),cache['relative'][:3]).detach().cpu().double().requires_grad_();z=cache['logits'][:3].detach().cpu().double().requires_grad_();y=cache['y'][:3].detach().cpu().double();crit=copy.deepcopy(criterion).cpu().double()
    value,_=baseline_loss(p,y,z,crit);value.backward();independent,gp,gz,_=module.loss_grad(p.detach().numpy()[:,None],y.numpy()[:,None],z.detach().numpy()[:,None],crit.regression.weights.numpy())
    np.testing.assert_allclose(float(value),independent,atol=1e-9,rtol=1e-9);np.testing.assert_allclose(p.grad.numpy(),gp[:,0],atol=1e-8,rtol=1e-8);np.testing.assert_allclose(z.grad.numpy(),gz[:,0],atol=1e-8,rtol=1e-8)
    contract={'technical_fixture':True};a=Trainer('A5',101,cache,head,criterion,device,epochs=2,batch=32);a.epoch_step();state=a.snapshot(contract);a.epoch_step()
    state_path=ROOT/'preflight/resume_fixture.pt';save_torch(state_path,state)
    b=Trainer('A5',101,cache,head,criterion,device,epochs=2,batch=32);b.restore(torch.load(state_path,map_location='cpu',weights_only=False),contract);b.epoch_step()
    for k,v in a.model.state_dict().items():assert torch.equal(v,b.model.state_dict()[k]),k
    assert a.history[-1]['loss']==b.history[-1]['loss'] and a.history[-1]['order_hash']==b.history[-1]['order_hash']
    # A brief real-fit throughput measurement, not a dev-driven parameter search.
    small={k:v[:256] for k,v in fit.items()};bench=Trainer('A5',102,small,head,criterion,device,epochs=20,batch=256);times=[]
    for _ in range(5):
        row=bench.epoch_step();times.append(row['seconds'])
    write(path,dict(status='PASS',at=now(),device=device,zero_init_and_parameter_counts=checks,base_loss_value_and_gradient='MATCH_INDEPENDENT_NUMPY',epoch_resume='EXACT_PARAMETERS_LOSS_ORDER',probe_fixture_not_a_research_result=True,seconds_per_256_sample_update_median=float(np.median(times[1:])),evidence_files=seal([state_path])))
