"""Actual explicit CurrentDSFPipeline assembly regression, no optimizer."""
import torch,numpy as np
from interface_common import *
from current_h3_adapter import load_current,CurrentDSFPipeline,BASE
from dsf_interface import ec_from_weights

def main():
    torch.set_num_threads(1);device='cuda' if torch.cuda.is_available() else 'cpu'
    model,evidence=load_current(device);pipeline=CurrentDSFPipeline(model).to(device)
    with np.load(BASE/'C0/fit.npz',allow_pickle=False) as f:
        past=torch.as_tensor(f['past'][:8],device=device);n=torch.as_tensor(f['num_valid'][:8],device=device)
    old={k:v.detach().cpu().clone() for k,v in model.model.state_dict().items()}
    result=pipeline(past,n)
    if not torch.equal(result['paths'].detach(),result['native']):raise ValueError('PIPELINE_ZERO_NOT_EXACT')
    c=read(STAGE_A/'CALIBRATION.json');loss=-ec_from_weights(result['paths'].double(),torch.ones(8,20,dtype=torch.double,device=device),c['k']).mean();loss.backward()
    if pipeline.sampler.net[-1].weight.grad is None or not pipeline.sampler.net[-1].weight.grad.abs().any():raise ValueError('PIPELINE_GRADIENT')
    for k,v in old.items():
        if not torch.equal(v,model.model.state_dict()[k].cpu()):raise ValueError('PIPELINE_CHANGED_OLD '+k)
    if any(p.grad is not None for p in model.model.parameters()):raise ValueError('OLD_MODEL_GRAD')
    write(ROOT/'CURRENT_PIPELINE_CHECK.json',dict(status='PASS_ACTUAL_CURRENT_PIPELINE_ASSEMBLY_NOT_SCIENTIFIC_TRAINING',N=8,device=device,original_path_bitwise=True,new_sampler_gradient_positive=True,old_parameters_and_bool_buffers_exact=True,old_parameter_gradients_none=True,original_generator_parameters=sum(p.numel() for p in model.model.parameters()),new_DSF_parameters=sum(p.numel() for p in pipeline.sampler.parameters()),no_CVAE_needed_for_this_interface=True,latent_prior='none_deterministic_slot_query_offset',evidence=evidence,source_files=seal([ROOT/'current_h3_adapter.py',ROOT/'check_current_pipeline.py',ROOT/'dsf_interface.py'])))
    print('CURRENT_PIPELINE_ASSEMBLY_PASS')

if __name__=='__main__':main()
