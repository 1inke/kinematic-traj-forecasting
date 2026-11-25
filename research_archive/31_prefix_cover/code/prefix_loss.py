"""EXP-PREFIX-COVER: nested prefix-coverage loss added on top of the unchanged B0 loss.

Total loss  L = core.loss_value(pred, target, logits, criterion)            # exact B0 (archive 26 baseline_factory.py:83)
              + LAMBDA * mean_{k in KS} [ regression(x_{j*_k}) + endpoint_weight * smoothL1(endpoint(x_{j*_k})) ]
j*_k = argmin over the first k entries of a GT-free, detached order pi of q_j = ADE_j + 0.25 * FDE_j.
pi depends on the arm:  N-cov  greedy facility-location order, weights softmax(logits)  (= archive 27 order_numpy.risk_order)
                        N-conf descending logits (stable)                              (= archive 27 order_numpy.conf_order)
                        N-rand random permutation from a private generator (does not touch the global RNG stream)
                        N-win  no order; every term uses the global winner (extra-winner-weight control)
                        A      LAMBDA term absent (exact B0)
NOT RUN IN THE AUTHORING ENVIRONMENT (no torch there). PF-0..PF-4 must pass locally before any formal run.
"""
import torch
from torch.nn import functional as F

KS = (1, 3, 5)
LAMBDA = 0.5
ARMS = ("A", "N-cov", "N-conf", "N-rand", "N-win")
RAND_SEED_OFFSET = 7919


@torch.no_grad()
def coverage_order(end: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """end (B,K,2), w (B,K) summing to 1. Step r picks argmin_s sum_j w_j*min(best_j, D[s,j]); ties -> lowest index.
    Same rule as archive 27 order_numpy.risk_order (PF-1 requires 100% identical orders)."""
    end, w = end.double(), w.double()
    B, K, _ = end.shape
    rows = torch.arange(B, device=end.device)
    D = torch.cdist(end, end)
    best = torch.full((B, K), float("inf"), dtype=D.dtype, device=D.device)
    taken = torch.zeros(B, K, dtype=torch.bool, device=D.device)
    tie = torch.arange(K, device=D.device, dtype=D.dtype) * 1e-12
    out = []
    for _ in range(K):
        obj = (w[:, None, :] * torch.minimum(best[:, None, :], D)).sum(-1) + tie
        obj = obj.masked_fill(taken, float("inf"))
        c = obj.argmin(1)
        out.append(c)
        taken[rows, c] = True
        best = torch.minimum(best, D[rows, c])
    return torch.stack(out, 1)


@torch.no_grad()
def arm_order(arm, pred, logits, generator):
    end = pred[:, 0, :, -1, :].detach()
    z = logits[:, 0].detach().double()
    if arm == "N-cov":
        return coverage_order(end, torch.softmax(z, -1))
    if arm == "N-conf":
        return torch.argsort(-z, dim=-1, stable=True)
    if arm == "N-rand":
        return torch.argsort(torch.rand(z.shape, generator=generator, device=z.device), dim=-1)
    raise ValueError(arm)


def prefix_loss(arm, pred, target, logits, criterion, generator=None):
    """pred (B,1,K,T,2), target (B,1,T,2), logits (B,1,K). Returns (loss, detail)."""
    if arm == "A":
        return pred.new_zeros(()), {}
    dist = torch.linalg.vector_norm(pred - target.unsqueeze(-3), dim=-1)[:, 0]          # (B,K,T)
    q = (dist.mean(-1) + criterion.winner_fde_weight * dist[..., -1]).detach()          # (B,K)
    winner = q.argmin(-1)
    if arm == "N-win":
        chosen = [winner for _ in KS]
    else:
        order = arm_order(arm, pred, logits, generator)
        qo = q.gather(1, order)
        chosen = [order.gather(1, qo[:, :k].argmin(-1, keepdim=True)).squeeze(1) for k in KS]
    terms, detail = [], {}
    for k, j in zip(KS, chosen):
        index = j[:, None, None, None, None].expand(-1, 1, 1, *pred.shape[-2:])
        sel = pred.gather(2, index).squeeze(2)                                            # (B,1,T,2)
        terms.append(criterion.regression(sel, target)
                     + criterion.endpoint_weight * F.smooth_l1_loss(sel[..., -1, :], target[..., -1, :]))
        detail[f"distinct{k}"] = float((j != winner).float().mean())                     # MC-1 input
    return LAMBDA * torch.stack(terms).mean(), detail


def make_generator(device, seed):
    g = torch.Generator(device=device)
    g.manual_seed(seed + RAND_SEED_OFFSET)
    return g
