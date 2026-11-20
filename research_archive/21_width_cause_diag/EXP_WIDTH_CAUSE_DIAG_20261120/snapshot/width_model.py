"""Frozen h3 replay and exact per-window decomposition of the archived B0 loss."""

import types

import torch
from torch.nn import functional as F

from baseline_factory import ModeQueryLoss, loss_value, make_model, prediction, world_paths

TERMS = ("reg", "cls", "endpoint", "diversity", "short", "coverage")
WEIGHTS = (1., .5, .25, .05, .2, .2)


def outputs(decoder, mode, relative, last):
    path = last + torch.einsum("tp,bakpc->baktc", decoder.interpolation, relative)
    logits = decoder.score_head(torch.cat((mode, relative.flatten(-2)), dim=-1)).squeeze(-1)
    return path, logits


def h3_forward_diag(decoder, tokens, past, scale_mult=None, cand_mult=None, *, track=False):
    assert not decoder.use_refiner and tuple(decoder.anchor_steps) == (4, 8, 12)
    recent, pooled = tokens[:, :, -1], tokens.mean(dim=2)
    context = decoder.context(torch.cat((recent, pooled, decoder.motion_features(past)), dim=-1))
    mode = decoder.mode_mlp(context[:, :, None] + decoder.mode_embedding[None, None])
    scale = F.softplus(decoder.proposal_scale(context)) + .25
    delta = .75 * torch.tanh(decoder.proposal_delta(mode)).reshape(*mode.shape[:-1], decoder.anchor_count, 2)
    templates = decoder.anchor_templates.detach()
    if track:
        scale = scale.detach().requires_grad_()
        delta = delta.detach().requires_grad_()
        templates = templates.requires_grad_()
    if scale_mult is not None:
        scale = scale * scale_mult[:, None, :]
    relative = scale[:, :, None, None] * (templates[None, None] + delta)
    if cand_mult is not None:
        relative = relative * cand_mult[:, None, :, None, None]
    if track:
        for tensor in (scale, delta, relative):
            tensor.retain_grad()
    last = past[:, :, None, -1:, :]
    paths, logits = outputs(decoder, mode, relative, last)
    return paths, logits, (mode, scale, delta, templates, relative, last)


def model_for(path, device, seed):
    torch.manual_seed(seed)
    model, cfg = make_model(device, seed)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["ema"], strict=True)
    criterion = ModeQueryLoss(cfg).to(device)
    if "loss_buffers" in state:
        criterion.load_state_dict(state["loss_buffers"], strict=True)
    assert (criterion.winner_fde_weight, criterion.classification_weight, criterion.endpoint_weight,
            criterion.diversity_weight, criterion.short_horizon_weight) == (.25, .5, .25, .05, .2)
    assert criterion.short_horizon_steps == 2 and criterion.soft_classification
    assert criterion.diversity_margin == .4 and criterion.soft_temperature == .5
    assert criterion.hard_classification_weight == .25
    return model.eval().requires_grad_(False), criterion


def tensors(data, index, device):
    return [torch.as_tensor(data[key][index], device=device) for key in ("past", "num_valid", "future")]


def replay(model, data, index, device, track=False, check_original=False):
    decoder = model.decoder
    original = decoder.forward
    captured = {}

    def forward(self, tokens, past):
        path, logits, captured["aux"] = h3_forward_diag(self, tokens, past, track=track)
        if check_original:
            with torch.no_grad():
                real = original(tokens, past)
            errors = [float((a.detach()-b).abs().max()) for a, b in zip((path, logits), real)]
            assert max(errors) <= 1e-6, errors
            captured["original_forward_max_abs"] = errors
        return path, logits

    decoder.forward = types.MethodType(forward, decoder)
    try:
        paths, logits, target, transform = prediction(model, *tensors(data, index, device))
    finally:
        decoder.forward = original
    return paths, logits, target, transform, captured


def rescale(decoder, aux, scale_mult=None, cand_mult=None):
    mode, scale, delta, template, relative, last = [x.detach() for x in aux]
    if scale_mult is not None:
        relative = (scale[:, :, None, None] * scale_mult) * (template[None, None] + delta)
    if cand_mult is not None:
        # Only focal candidates enter the metrics.
        relative = relative[:, :1] * cand_mult[:, None, :, None, None]
        return outputs(decoder, mode[:, :1], relative, last[:, :1])
    return outputs(decoder, mode, relative, last)


def per_window_loss(p, y, z, criterion):
    dist = torch.linalg.vector_norm(p - y.unsqueeze(-3), dim=-1)
    q = dist.mean(-1) + criterion.winner_fde_weight * dist[..., -1]
    gap = q.detach() - q.detach().amin(-1, keepdim=True)
    winner = q.detach().argmin(-1)
    pwin = p.gather(2, winner[..., None, None, None].expand(-1, -1, 1, p.shape[-2], 2)).squeeze(2)
    errors = F.smooth_l1_loss(pwin, y, reduction="none")[:, 0]
    reg = (errors.sum(-1) * criterion.regression.weights).mean(-1)
    endpoint = errors[:, -1].mean(-1)
    short = errors[:, :criterion.short_horizon_steps].mean((1, 2))
    target = torch.softmax(-gap / (criterion.soft_temperature * gap.mean(-1, keepdim=True).clamp_min(.05)), -1)
    logp = torch.log_softmax(z, dim=-1)
    cls = (-(target * logp).sum(-1) -
           criterion.hard_classification_weight * logp.gather(-1, winner[..., None]).squeeze(-1))[:, 0]
    xy = p[..., -1, :]
    d = torch.cdist(xy, xy)[:, 0]
    upper = torch.triu_indices(d.shape[-1], d.shape[-1], 1, device=p.device)
    pairs = d[:, upper[0], upper[1]]
    diversity = F.relu(criterion.diversity_margin - pairs).mean(-1)
    r = torch.softmax(-q.detach() / .1, dim=-1)
    coverage = (r * q).sum(-1)[:, 0]
    eff = torch.exp(-(r * torch.log(r.clamp_min(torch.finfo(r.dtype).tiny))).sum(-1))[:, 0]
    return torch.stack((reg, cls, endpoint, diversity, short, coverage), dim=1), winner[:, 0], eff, pairs


def checked_terms(p, y, z, criterion, audit):
    terms, winner, eff, pairs = per_window_loss(p, y, z, criterion)
    with torch.no_grad():
        real, detail = loss_value(p, y, z, criterion)
        expected = torch.stack([detail[k] for k in ("regression", "classification", "endpoint",
                                                     "diversity", "short_horizon")])
        average = terms.mean(0)
        torch.testing.assert_close(average[:5], expected, atol=2e-7, rtol=1e-6)
        assert torch.equal(winner, detail["winner"][:, 0])
        obtained = (average * p.new_tensor(WEIGHTS)).sum()
        error = float((obtained-real).abs() / real.abs().clamp_min(1e-12))
        assert error <= 1e-6, error
        audit["loss_max_relative_error"] = max(audit.get("loss_max_relative_error", 0.), error)
        audit["loss_batches"] = audit.get("loss_batches", 0) + 1
    return terms, winner, eff, pairs


def gradients(terms, winner, eff, aux, selected, audit):
    _, scale, delta, templates, relative, _ = aux
    T_centered = templates - templates.mean(0)
    T_unit = T_centered / torch.linalg.vector_norm(T_centered.flatten(1), dim=-1).clamp_min(1e-12)[:, None, None]
    grad_s, grad_winner, grad_norm, grad_templates = [], [], [], []
    idx = torch.arange(len(winner), device=winner.device)
    weight = terms.new_tensor(WEIGHTS)
    weighted = terms * weight
    part = list(weighted.unbind(1)) + [weighted.sum(1)]
    for j, each in enumerate(part):
        G, direct_scale, direct_template = torch.autograd.grad(
            each[selected].sum(), (relative, scale, templates), retain_graph=j < len(part)-1)
        G = G[:, 0]
        per_axis = G * (delta[:, 0].detach() + templates.detach()[None])
        torch.testing.assert_close(direct_scale[:, 0], per_axis.sum((1, 2)), atol=2e-6, rtol=2e-5)
        per_T = G * scale[:, 0, None, None].detach()
        torch.testing.assert_close(direct_template, per_T.sum(0), atol=2e-6, rtol=2e-5)
        audit["scale_chain_max_abs"] = max(audit.get("scale_chain_max_abs", 0.),
                                           float((direct_scale[:, 0]-per_axis.sum((1, 2))).abs().max()))
        audit["template_chain_max_abs"] = max(audit.get("template_chain_max_abs", 0.),
                                              float((direct_template-per_T.sum(0)).abs().max()))
        grad_s.append(direct_scale[:, 0].sum(1))
        grad_winner.append(per_axis[idx, winner].sum((1, 2)))
        grad_norm.append(torch.linalg.vector_norm(G.flatten(2), dim=2))
        grad_templates.append((per_T * T_unit.detach()).sum((1, 2, 3)))
    normalized = delta[:, 0].detach() / .75
    sat = normalized.abs() > .95
    inward = normalized * templates.detach() < 0
    outward = normalized * templates.detach() > 0
    return dict(S=torch.stack(grad_s,1), winner_S=torch.stack(grad_winner,1),
                anchor_grad_norm=torch.stack(grad_norm,1), template_radial=torch.stack(grad_templates,1),
                effective_candidates=eff, winner=winner, saturated=sat.sum((2,3)),
                inward_saturated=(sat & inward).sum((2,3)), outward_saturated=(sat & outward).sum((2,3)))
