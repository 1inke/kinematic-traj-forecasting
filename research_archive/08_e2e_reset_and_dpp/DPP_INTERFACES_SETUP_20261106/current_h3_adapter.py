"""Human-authorized generalized DSF interface for existing frozen F16+h3.

Offsets of the 20 learned 64D mode queries are continuous CONTROL LATENTS,
NOT Gaussian random variables. Fixed slot-specific anchor templates remain.
No h3 Gaussian prior, likelihood, original chi-square radius or CVAE is claimed.
"""
import os,sys,importlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from interface_common import ROOT,read,check_files,sha

BASE=ROOT.parent/'E2E_MOTION_RESEARCH_RESET_EXEC_20261102'

def baseline_api():
    # Import only reviewed definitions, never a training launcher. Keep any
    # optional library caches in the NEW directory rather than historical runs.
    os.environ.setdefault('TORCH_HOME',str(ROOT/'runtime_cache/torch'))
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'runtime_cache/matplotlib'))
    sys.path.insert(0,str(BASE))
    existing=sys.modules.get('common')
    if existing is not None and Path(existing.__file__).resolve()!=BASE/'common.py':raise RuntimeError('BASELINE_COMMON_NAMESPACE_COLLISION')
    return importlib.import_module('baseline_factory')

class CurrentH3LatentAdapter(nn.Module):
    latent_dim=64
    context_dim=64
    latent_prior='none_deterministic_slot_query_offset'
    def __init__(self,model,api):
        super().__init__();self.model=model;self.api=api
        d=model.decoder
        if d.num_modes!=20 or d.pred_len!=12 or d.anchor_steps!=(4,8,12) or d.use_refiner or d.mode_embedding.shape!=(20,64):raise ValueError('EXPECTED_EXISTING_H3_CONTRACT')
        model.eval()
        for p in model.parameters():p.requires_grad_(False);p.grad=None
    def train(self,mode=True):super().train(False);self.model.eval();return self
    def encode(self,raw_past,num_valid):
        if raw_past.ndim!=4 or raw_past.shape[1:]!=(9,8,2) or num_valid.shape!=(len(raw_past),):raise ValueError('RAW_OBSERVED_NEIGHBORS_ONLY')
        captured=[]
        handle=self.model.decoder.register_forward_pre_hook(lambda module,args:captured.append(args))
        try:
            with torch.no_grad():
                out,logits,_,transform=self.api.prediction(self.model,raw_past,num_valid,None,False)
                native=self.api.world_paths(out,transform)
        finally:handle.remove()
        if len(captured)!=1:raise ValueError('DECODER_CAPTURE_COUNT')
        tokens,past_loc=captured[0]
        d=self.model.decoder
        with torch.no_grad():
            context=d.context(torch.cat([tokens[:,:,-1],tokens.mean(2),d.motion_features(past_loc)],-1))
        return dict(context=context.detach(),sampler_context=context[:,0].detach(),past_loc=past_loc.detach(),transform=tuple(x.detach() for x in transform),native=native.detach())
    def decode(self,offset,bundle):
        if offset.shape!=(len(bundle['context']),20,64) or not torch.isfinite(offset).all():raise ValueError('CONTROL_LATENT_SHAPE')
        d=self.model.decoder;context=bundle['context'];past_loc=bundle['past_loc']
        # Neighbor controls remain exactly zero. Preserve native B,A,K operation
        # shapes to test zero-offset identity against original forward bytes.
        full_offset=torch.cat([offset[:,None],offset.new_zeros((len(offset),8,20,64))],1)
        mode=d.mode_mlp(context[:,:,None]+d.mode_embedding[None,None]+full_offset)
        scale=F.softplus(d.proposal_scale(context))+.25
        delta=.75*torch.tanh(d.proposal_delta(mode)).reshape(*mode.shape[:-1],3,2)
        relative_anchors=scale[:,:,None,None]*(d.anchor_templates[None,None]+delta)
        coarse_relative=torch.einsum('tp,bakpc->baktc',d.interpolation,relative_anchors)
        coarse=past_loc[:,:,None,-1:,:]+coarse_relative
        return self.api.world_paths(coarse[:,:1],bundle['transform'])

def load_current(device='cpu'):
    lock=read(BASE/'B0_FINAL_LOCK.json');check_files(lock['files'])
    check_files(read(BASE/'source_manifest.json')['files'])
    api=baseline_api();model,_=api.make_model(device,42)
    file=BASE/'B0_seed42/final.pt'
    # This exact LOCAL checkpoint is already SHA-bound to the scientific lock.
    # It contains NumPy/Python RNG in addition to weights; never use this for
    # untrusted downloaded checkpoints. No remote checkpoint is loaded.
    state=torch.load(file,map_location='cpu',weights_only=False)
    if state['epoch']!=8 or state['updates']!=2720:raise ValueError('B0_FINAL_EPOCH_UPDATE_CONTRACT')
    model.load_state_dict(state['ema'],strict=True)
    return CurrentH3LatentAdapter(model,api),dict(checkpoint=str(file),checkpoint_sha256=sha(file),selection='fixed_epoch8_EMA',latent_prior=CurrentH3LatentAdapter.latent_prior,no_generator_weight_training=True)

class CurrentDSFPipeline(nn.Module):
    def __init__(self,adapter):
        super().__init__();from dsf_interface import DSFSampler
        self.generator=adapter;self.sampler=DSFSampler(64,64)
        nn.init.zeros_(self.sampler.net[-1].weight);nn.init.zeros_(self.sampler.net[-1].bias)
    def forward(self,raw_past,num_valid):
        bundle=self.generator.encode(raw_past,num_valid)
        control=self.sampler(bundle['sampler_context'])
        return dict(paths=self.generator.decode(control,bundle),control_latent=control,native=bundle['native'])
