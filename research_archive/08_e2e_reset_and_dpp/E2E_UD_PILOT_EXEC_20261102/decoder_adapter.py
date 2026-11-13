"""Diagnostic bridge to ACTUAL h3 context64. No old file edits or head changes."""
from ud_common import *
from baseline_factory import torch
from torch import nn
from torch.nn import functional as F
from hierarchical_decoder_models import HierarchicalLearnedAnchorDecoder
from motion_state_update import GRUStateUpdate,GatedDeltaStateUpdate,compute_past_kinematic_state_6d
from mode_interaction import GatedModeAttention,DifferentialGatedModeAttention


class UDDecoder(HierarchicalLearnedAnchorDecoder):
    def __init__(self,base,arm):
        nn.Module.__init__(self)
        if base.use_refiner:raise ValueError('ONLY_ORIGINAL_H3_NO_REFINER')
        for name,child in base._modules.items():self.add_module(name,child)
        for name,param in base._parameters.items():self.register_parameter(name,param)
        for name,buffer in base._buffers.items():self.register_buffer(name,buffer)
        for name in ('num_modes','pred_len','anchor_steps','anchor_count','use_refiner','top_m'):
            setattr(self,name,getattr(base,name))
        self.arm=arm
        dim=base.mode_embedding.shape[-1]
        if arm=='U-GRU':self.update=GRUStateUpdate(out_dim=dim)
        elif arm=='U-Delta':self.update=GatedDeltaStateUpdate(out_dim=dim)
        elif arm=='D-Gate':self.interaction=GatedModeAttention(dim=dim)
        elif arm=='D-DiffGate':self.interaction=DifferentialGatedModeAttention(dim=dim)
        else:raise ValueError('UNKNOWN_ARM')
        self.last_anchor_proposals=None;self.last_coarse_trajectories=None
        self.last_selected_indices=None;self._training_target=None

    def forward(self,encoded_tokens,past_loc):
        recent=encoded_tokens[:,:,-1];pooled=encoded_tokens.mean(2)
        context=self.context(torch.cat([recent,pooled,self.motion_features(past_loc)],-1))
        if hasattr(self,'update'):
            delta=self.update(compute_past_kinematic_state_6d(past_loc[:,0]))
            context=context+torch.cat([delta[:,None],torch.zeros_like(context[:,1:])],1)
        mode_input=context[:,:,None]+self.mode_embedding[None,None]
        if hasattr(self,'interaction'):
            b,a,k,d=mode_input.shape
            mode_input=self.interaction(mode_input.reshape(b*a,k,d),context.reshape(b*a,d)).reshape(b,a,k,d)
        mode=self.mode_mlp(mode_input)
        scale=F.softplus(self.proposal_scale(context))+.25
        delta=.75*torch.tanh(self.proposal_delta(mode)).reshape(*mode.shape[:-1],self.anchor_count,2)
        relative=scale[:,:,None,None]*(self.anchor_templates[None,None]+delta)
        coarse_relative=torch.einsum('tp,bakpc->baktc',self.interpolation,relative)
        coarse=past_loc[:,:,None,-1:]+coarse_relative
        logits=self.score_head(torch.cat([mode,relative.flatten(-2)],-1)).squeeze(-1)
        self.last_anchor_proposals=(past_loc[:,:,None,None,-1,:]+relative).detach()
        self.last_coarse_trajectories=coarse.detach()
        self.last_selected_indices=None
        return coarse,logits
