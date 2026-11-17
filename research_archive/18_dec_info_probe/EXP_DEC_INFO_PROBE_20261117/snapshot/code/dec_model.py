"""Frozen-feature candidate refiners; all forecast inputs exclude future labels."""
import math
import torch
from torch import nn
from dec_io import INPUTS
from probe_model import Refiner as PriorRefiner, make_basis, predict as prior_predict


def risk_geometry(paths, pos, vel, valid):
    """Vectorized discrete closest approach, Q_j(t)-P_k(t), earliest tie."""
    times = torch.arange(1, 13, device=paths.device, dtype=paths.dtype) * .4
    extrap = pos[:, :, None] + vel[:, :, None] * times[None, None, :, None]
    diff = extrap[:, None] - paths[:, :, None]
    squared, step = diff.square().sum(-1).min(-1)
    dca = squared.sqrt()
    ttc = times[step]
    closest = diff.gather(-2, step[..., None, None].expand(-1, -1, -1, 1, 2)).squeeze(-2)
    d0 = torch.linalg.vector_norm(pos, dim=-1)
    risk = torch.exp(-ttc / 2.5) * torch.exp(-dca / 1.25) + .10 * torch.exp(-d0[:, None] / 3.)
    eligible = valid[:, None] & (dca < 3.)
    rank = risk.masked_fill(~eligible, -torch.inf).argsort(dim=-1, descending=True, stable=True)
    idx = rank[..., :3]
    keep = eligible.gather(-1, idx)
    features = torch.cat((dca[..., None], (ttc / 4.8)[..., None],
                         d0[:, None, :, None].expand_as(dca[..., None]), closest), dim=-1)
    return idx, keep, features, dca


def route(arm, base_paths, pos, vel, valid, self_vel):
    n, k = base_paths.shape[:2]
    if arm == 'C3':
        idx, keep, features, _ = risk_geometry(base_paths, pos, vel, valid)
        return idx, keep, features.gather(2, idx[..., None].expand(-1, -1, -1, 5))
    distance = torch.linalg.vector_norm(pos, dim=-1)
    raw = torch.cat((pos, vel - self_vel[:, None], distance[..., None]), dim=-1)
    if arm == 'C2':
        idx = torch.arange(pos.shape[1], device=pos.device)[None].expand(n, -1)
    elif arm == 'C4':
        idx = distance.masked_fill(~valid, torch.inf).argsort(dim=-1, stable=True)[:, :3]
    else:
        raise ValueError(arm)
    rows = torch.arange(n, device=pos.device)[:, None]
    return (idx[:, None].expand(-1, k, -1), valid[rows, idx][:, None].expand(-1, k, -1),
            raw[rows, idx][:, None].expand(-1, k, -1, -1))


class InfoRefiner(PriorRefiner):
    def __init__(self, arm):
        super().__init__('A1')
        self.arm = arm
        if arm == 'A1L':
            old = self.ffn
            width = 220
            self.ffn = nn.Sequential(nn.Linear(24, width), nn.GELU(), nn.Linear(width, 24))
            with torch.no_grad():
                self.ffn[0].weight[:48].copy_(old[0].weight)
                self.ffn[0].bias[:48].copy_(old[0].bias)
                self.ffn[2].weight[:, :48].copy_(old[2].weight)
                self.ffn[2].bias.copy_(old[2].bias)
        elif arm.startswith('C'):
            self.cross_q = nn.Linear(24, 24, bias=False)
            self.cross_k = nn.Linear(96, 24, bias=False)
            self.cross_v = nn.Linear(96, 24, bias=False)
            self.cross_out = nn.Linear(24, 24, bias=False)
            self.cross_n1 = nn.LayerNorm(24)
            self.cross_ffn = nn.Sequential(nn.Linear(24, 48), nn.GELU(), nn.Linear(48, 24))
            self.cross_n2 = nn.LayerNorm(24)
            self.time_bias = nn.Parameter(torch.zeros(8, 4))
            self.relation_bias = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 4))
            nn.init.zeros_(self.relation_bias[-1].weight)
            nn.init.zeros_(self.relation_bias[-1].bias)
        elif arm != 'A1':
            raise ValueError(arm)
        self.register_buffer('basis', make_basis(), persistent=False)

    def stem(self, mode, relative, scale):
        n, k = relative.shape[:2]
        rel = relative / scale[:, None, None]
        previous = torch.cat((torch.zeros_like(rel[:, :, :1]), rel[:, :, :-1]), dim=2)
        x = self.inp(torch.cat((mode, rel.flatten(-2), (rel - previous).flatten(-2),
                               torch.linalg.vector_norm(rel[:, :, -1], dim=-1, keepdim=True)), dim=-1))
        h = lambda t: t.reshape(n, k, 4, 6).transpose(1, 2)
        att = h(self.q(x)) @ h(self.k(x)).transpose(-1, -2) / math.sqrt(6)
        att = att.masked_fill(~torch.eye(k, dtype=torch.bool, device=x.device), -torch.inf)
        value = (att.softmax(-1) @ h(self.v(x))).transpose(1, 2).reshape(n, k, 24)
        x = self.n1(x + self.o(value))
        return self.n2(x + self.ffn(x))

    def forward(self, mode, relative, scale, logits, tokens_self=None, tokens_nbr=None,
                nbr_pos=None, nbr_vel=None, nbr_valid=None, self_vel=None):
        if not self.arm.startswith('C'):
            return super().forward(mode, relative, scale, logits)
        x = self.stem(mode, relative, scale)
        n, k = x.shape[:2]
        q = self.cross_q(x).reshape(n, k, 4, 6)
        sk = self.cross_k(tokens_self.float()).reshape(n, 8, 4, 6)
        sv = self.cross_v(tokens_self.float()).reshape(n, 8, 4, 6)
        a = torch.einsum('nkhd,nshd->nkhs', q, sk) / math.sqrt(6) + self.time_bias.T
        if self.arm == 'C1':
            value = torch.einsum('nkhs,nshd->nkhd', a.softmax(-1), sv)
        else:
            # C2/C4 only need window geometry; construct pass-1 paths only in C3.
            base = (torch.einsum('tp,nkpc->nktc', self.basis, relative)
                    if self.arm == 'C3' else relative)
            idx, keep, rel = route(self.arm, base, nbr_pos, nbr_vel, nbr_valid, self_vel)
            nk = self.cross_k(tokens_nbr.float()).reshape(n, 8, 4, 6)
            nv = self.cross_v(tokens_nbr.float()).reshape(n, 8, 4, 6)
            rows = torch.arange(n, device=x.device)
            if self.arm == 'C3':
                nk = nk[rows[:, None, None], idx]
                nv = nv[rows[:, None, None], idx]
                b = torch.einsum('nkhd,nkmhd->nkhm', q, nk) / math.sqrt(6)
            else:
                nk = nk[rows[:, None], idx[:, 0]]
                nv = nv[rows[:, None], idx[:, 0]]
                b = torch.einsum('nkhd,nmhd->nkhm', q, nk) / math.sqrt(6)
            b = b + self.relation_bias(rel).transpose(-1, -2)
            weights = torch.cat((a, b.masked_fill(~keep[:, :, None], -torch.inf)), dim=-1).softmax(-1)
            value = torch.einsum('nkhs,nshd->nkhd', weights[..., :8], sv)
            if self.arm == 'C3':
                value = value + torch.einsum('nkhm,nkmhd->nkhd', weights[..., 8:], nv)
            else:
                value = value + torch.einsum('nkhm,nmhd->nkhd', weights[..., 8:], nv)
        x = self.cross_n1(x + self.cross_out(value.reshape(n, k, 24)))
        x = self.cross_n2(x + self.cross_ffn(x))
        shift = .30 * scale[:, None, None] * torch.tanh(self.anchor_out(x)).reshape(n, k, 3, 2)
        return relative + shift, self.score_out(x).squeeze(-1)


def batch_inputs(model, cache, index):
    fields = INPUTS[:4] if not model.arm.startswith('C') else INPUTS[:5] if model.arm == 'C1' else INPUTS
    return tuple(cache[k] if index is None else cache[k][index] for k in fields)


def predict(model, head, cache, index=None):
    if not model.arm.startswith('C'):
        return prior_predict(model, head, cache, index)
    inp = batch_inputs(model, cache, index)
    mode, relative, scale, base_logits = inp[:4]
    new, dz = model(*inp)
    out_logits = head(torch.cat((mode, new.flatten(-2)), dim=-1)).squeeze(-1) + dz
    return torch.einsum('tp,nkpc->nktc', model.basis, new), out_logits, new - relative
