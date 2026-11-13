"""Actual frozen checkpoint interface test on 8 hash-fixed FIT observations.

No generator/DSF training, no held GT, no claim of scientific effectiveness.
"""
import time,traceback
import numpy as np
import torch
from interface_common import *
from current_h3_adapter import load_current,BASE
from dsf_interface import ec_from_weights,kinematic_torch,DSFSampler,quality_weights

def main():
    torch.set_num_threads(1);started=time.perf_counter();device='cuda' if torch.cuda.is_available() else 'cpu'
    out=ROOT/'current_h3_check';out.mkdir(exist_ok=False)
    try:
        import importlib.util
        spec=importlib.util.spec_from_file_location('setup_guard',ROOT.parent/'K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py');rg=importlib.util.module_from_spec(spec);spec.loader.exec_module(rg)
        guard=rg.Guard(3*1024**3);admission=guard.check(mode='startup',force=True)
        adapter,evidence=load_current(device)
        with np.load(BASE/'C0/fit.npz',allow_pickle=False) as f:
            ids=f['ids'];import hashlib
            ix=np.array(sorted(range(len(ids)),key=lambda i:hashlib.sha256(('dpp-current-latent-fit8-v1|'+str(ids[i])).encode()).hexdigest())[:8])
            past=torch.as_tensor(f['past'][ix],device=device);valid=torch.as_tensor(f['num_valid'][ix],device=device);selected_ids=ids[ix]
        before={k:v.detach().cpu().clone() for k,v in adapter.model.state_dict().items()}
        bundle=adapter.encode(past,valid);z=torch.zeros(8,20,64,device=device,requires_grad=True)
        y=adapter.decode(z,bundle);native=bundle['native']
        difference=float((y-native).abs().max().detach());identical=bool(torch.equal(y.detach(),native))
        if not identical:raise ValueError('ZERO_CONTROL_NATIVE_PATH_NOT_BITWISE '+str(difference))
        c=read(STAGE_A/'CALIBRATION.json');q=kinematic_torch(past[:,0].double(),y.double(),c)
        ec=ec_from_weights(y.double(),torch.sqrt(c['epsilon_q']+(1-c['epsilon_q'])*q),c['k'])
        gradient=torch.autograd.grad(-ec.mean(),z)[0]
        if not torch.isfinite(gradient).all() or not gradient.abs().any():raise ValueError('NO_ACTUAL_LATENT_DPP_GRADIENT')
        delta=.01*torch.sin(torch.arange(z.numel(),device=device,dtype=z.dtype).reshape_as(z)+1)
        moved=adapter.decode(delta,bundle)
        movement=float((moved-y).abs().max().detach())
        if movement<=0:raise ValueError('GENERATOR_IGNORES_CONTROL')
        sampler=DSFSampler(64,64).to(device)
        torch.nn.init.zeros_(sampler.net[-1].weight);torch.nn.init.zeros_(sampler.net[-1].bias)
        scores=adapter.decode(sampler(bundle['sampler_context']),bundle)
        loss=-ec_from_weights(scores.double(),torch.ones(8,20,device=device,dtype=torch.double),c['k']).mean();loss.backward()
        if not sampler.net[-1].weight.grad.abs().any():raise ValueError('NO_SAMPLER_GRADIENT')
        if any(p.grad is not None for p in adapter.model.parameters()):raise ValueError('OLD_WEIGHT_GRADIENT')
        for k,v in before.items():
            if not torch.equal(v,adapter.model.state_dict()[k].cpu()):raise ValueError('OLD_WEIGHT_BUFFER_MUTATION '+k)
        guard.check(force=True)
        np.savez_compressed(out/'actual_fit8.npz',ids=selected_ids,native=native.cpu().numpy(),zero_control=y.detach().cpu().numpy(),perturbed=moved.detach().cpu().numpy(),latent_gradient=gradient.detach().cpu().numpy())
        write(out/'RESULT.json',dict(status='PASS_ACTUAL_FROZEN_CURRENT_GENERATOR_LATENT_INTERFACE_NOT_TRAINING',device=device,N=8,K=20,latent_dim=64,context_dim=64,zero_control_native_bitwise=identical,max_identity_error=difference,fixed_perturbation_max_path_change_m=movement,kin_DPP_control_gradient_abs_sum=float(gradient.abs().sum()),all_generator_weights_bool_buffers_exact=True,old_weight_gradients_none=True,sampler_last_gradient_positive=True,sampler_upstream_step0_zero_expected=True,sampler_parameters=sum(p.numel() for p in sampler.parameters()),frozen_generator_parameters=sum(p.numel() for p in adapter.model.parameters()),no_gaussian_prior_or_original_radius_claim=True,slot_specific_templates_retained=True,no_scientific_training_or_CVAE_pretraining=True,checkpoint=evidence,resource=admission,final_snapshot=rg.snapshot(),observations=guard.observations,elapsed_seconds=time.perf_counter()-started,files=seal([out/'actual_fit8.npz'])))
        print('ACTUAL_CURRENT_GENERATOR_INTERFACE_PASS',identical,movement,float(gradient.abs().sum()),flush=True)
    except BaseException as e:
        write(out/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),no_training=True));raise

if __name__=='__main__':main()
