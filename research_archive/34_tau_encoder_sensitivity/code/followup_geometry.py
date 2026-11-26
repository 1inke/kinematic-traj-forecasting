"""Selection rules shared by the EXP-TAU C0 follow-ups."""
import torch


def matched_distance(distance, candidates, budget):
    """Choose the closest valid neighbors using the paired T2 row's degree."""
    order = distance.masked_fill(~candidates, torch.inf).argsort(dim=-1, stable=True)
    selected = torch.arange(order.shape[-1], device=order.device) < budget
    return torch.zeros_like(candidates).scatter(-1, order, selected)


def learned_top_p(scores, allowed, p):
    """Keep the smallest non-self prefix reaching p of the learned mass."""
    eye = torch.eye(allowed.shape[-1], device=allowed.device, dtype=torch.bool)
    valid = allowed & ~eye
    with torch.no_grad():
        pair = scores.detach().float().logsumexp(-1).mean(1)
        pair = pair.masked_fill(~valid, -1.0e4)
        weights = pair.softmax(-1).masked_fill(~valid, 0.0)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1.0e-8)
        mass, order = weights.sort(descending=True, stable=True)
        before = mass.cumsum(-1) - mass
        keep = (before < p) & valid.gather(-1, order)
        return torch.zeros_like(valid).scatter(-1, order, keep) | (allowed & eye)
