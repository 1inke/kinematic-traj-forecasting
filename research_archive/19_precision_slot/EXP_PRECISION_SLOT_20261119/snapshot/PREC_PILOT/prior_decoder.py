"""Single CV mode with the original h3 residual and optional original self refinement."""
import torch
from torch import nn
from torch.nn import functional as F

CV_SLOT = 19


class PriorDecoder(nn.Module):
    def __init__(self, base, with_self):
        super().__init__()
        assert base.num_modes == 20 and tuple(base.anchor_steps) == (4, 8, 12)
        assert not base.use_refiner and (not with_self or base.basis_variant == "linear")
        # Preserve the parameter names and registration order of the matched control.
        for name, child in base._modules.items():
            self.add_module(name, child)
        for name, param in base._parameters.items():
            self.register_parameter(name, param)
        for name, value in base._buffers.items():
            self.register_buffer(name, value, persistent=name not in base._non_persistent_buffers_set)
        self.motion_features = base.motion_features
        self.with_self = with_self
        self.last_pass1_relative = None
        self.last_anchor_proposals = None
        self.last_coarse_trajectories = None
        self.last_selected_indices = None

    def set_training_target(self, target):
        pass

    def forward(self, encoded_tokens, past_loc):
        recent, pooled = encoded_tokens[:, :, -1], encoded_tokens.mean(dim=2)
        context = self.context(torch.cat((recent, pooled, self.motion_features(past_loc)), dim=-1))
        mode = self.mode_mlp(context[:, :, None] + self.mode_embedding[None, None])
        scale = F.softplus(self.proposal_scale(context)) + 0.25
        delta = .75 * torch.tanh(self.proposal_delta(mode)).reshape(*mode.shape[:-1], 3, 2)
        learned = scale[:, :, None, None] * (self.anchor_templates[None, None] + delta)
        displacement = past_loc[:, :, -1] - past_loc[:, :, -2]
        steps = displacement.new_tensor([4., 8., 12.])
        cv = displacement[:, :, None] * steps[None, None, :, None]
        adapted = cv + scale[:, :, None] * delta[:, :, CV_SLOT]
        relative = torch.cat((learned[:, :, :CV_SLOT], adapted[:, :, None]), dim=2)
        if self.with_self:
            b, a, k = mode.shape[:3]
            refined, dlogit = self.refiner(mode.reshape(b*a,k,-1), relative.reshape(b*a,k,3,2),
                                           scale.reshape(b*a,2))
            refined = refined.reshape(b,a,k,3,2)
            dlogit = dlogit.reshape(b,a,k)
            basis = self.curve_W
        else:
            refined = relative
            dlogit = None
            basis = self.interpolation
        xy = torch.einsum('tp,bakpc->baktc',basis,refined) + past_loc[:, :, None, -1:, :]
        logits = self.score_head(torch.cat((mode, refined.flatten(-2)),dim=-1)).squeeze(-1)
        if dlogit is not None:
            logits = logits + dlogit
        self.last_pass1_relative = relative.detach()
        self.last_anchor_proposals = (past_loc[:, :, None, None, -1, :] + refined).detach()
        self.last_coarse_trajectories = xy.detach()
        return xy, logits
