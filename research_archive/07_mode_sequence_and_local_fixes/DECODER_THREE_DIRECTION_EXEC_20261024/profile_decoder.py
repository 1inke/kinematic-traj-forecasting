"""Opt-in actual CUDA decoder profile. Never labels Linear MACs full-model MAC.

C2's full-model requirement must additionally be supplied by an encoder+decoder
pipeline measurement artifact. This utility cannot prove that requirement.
"""
import argparse
from pathlib import Path
import sys,time
sys.dont_write_bytecode=True
import numpy as np
import torch
from common import *
from training import make_model,PACKAGES
from f16.resource import Guard

def profile(package,execution,estimate):
    if not torch.cuda.is_available():raise RuntimeError('Actual GPU required')
    guard=Guard(estimate);guard.check('startup')
    results={}
    for arm in PACKAGES[package]:
        guard.check()
        model=make_model(arm,42,execution).eval();mac=[0];hooks=[]
        for layer in model.modules():
            if isinstance(layer,torch.nn.Linear):
                hooks.append(layer.register_forward_hook(lambda owner,args,out:mac.__setitem__(0,mac[0]+out.numel()*owner.in_features)))
        with torch.inference_mode():model(torch.zeros(1,210,device='cuda'))
        for h in hooks:h.remove()
        timings={}
        for batch in (1,256):
            x=torch.zeros(batch,210,device='cuda');torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for _ in range(8):model(x)
                torch.cuda.synchronize();start=time.perf_counter()
                for _ in range(64):model(x)
                torch.cuda.synchronize()
            timings[str(batch)]=dict(seconds_per_batch=(time.perf_counter()-start)/64,
                GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        results[arm]=dict(trainable_parameters=sum(p.numel() for p in model.parameters()),
            decoder_Linear_MAC_per_window=mac[0],latency=timings,
            includes_encoder=False,includes_data_IO=False,full_model_MAC_available=False)
        del model;torch.cuda.empty_cache()
    guard.check()
    result=dict(status='MEASURED_DECODER_ONLY',results=results,source_digest=verify_sources(execution/'source_manifest.json'),
        resource_estimate_bytes=estimate,resource_observations=guard.observations)
    write_json(execution/'DECODER_PROFILE.json',result)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('package',choices=PACKAGES);ap.add_argument('--execution',type=Path,required=True)
    ap.add_argument('--estimated-peak-gib',type=float,required=True)
    a=ap.parse_args()
    a.execution=owned_output(a.execution,'direction_A_time_basis' if a.package=='A' else 'direction_C_set_mechanisms')
    profile(a.package,a.execution,int(a.estimated_peak_gib*2**30))
