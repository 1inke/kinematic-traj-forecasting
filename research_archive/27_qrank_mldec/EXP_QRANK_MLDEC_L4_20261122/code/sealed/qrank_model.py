"""Reference (NOT run here: no torch in the design workspace) for EXP-QRANK-ML.
Extends archive 26 code/sealed/advisor_model.py. Reuses CrossLayer, zero, set_valid unchanged.
Everything numeric below is the pre-registered value; preflight PF-* must pass before Stage 1.

Arms are described by a tuple (order, attn, layers, emta):
  order in {"none","conf","rand","qual","risk","riskp"}; attn in {"full","causal"};
  emta in {None,"E0","QFL","E1","Ewide"}.
"""
import torch
from torch import nn
from torch.nn import functional as F

from advisor_model import CrossLayer, zero, set_valid   # archive 26, unchanged

ARMS = {
    "D3-full":   ("none",  "full",   3, None),
    "O-conf":    ("conf",  "causal", 3, None),
    "O-rand":    ("rand",  "causal", 3, None),
    "O-qual":    ("qual",  "causal", 3, None),
    "O-risk":    ("risk",  "causal", 3, None),
    "O-riskp":   ("riskp", "causal", 3, None),
    "O-conf+E0": ("conf",  "causal", 3, "E0"),
    "O-risk+E0": ("risk",  "causal", 3, "E0"),
    "O-conf+QFL": ("conf", "causal", 3, "QFL"),
    "O-risk+QFL": ("risk", "causal", 3, "QFL"),
    # reserve (P2)
    "O-risk+E1": ("risk",  "causal", 3, "E1"),
    "O-risk+Ewide": ("risk", "causal", 3, "Ewide"),
    "O-qual-q":  ("qual",  "causal", 3, None),        # Q-hat = q-hat only (see qhat(..., only_q=True))
    "O-risk-full": ("risk", "full",  3, None),        # order + rank embedding but no causal mask
    "D2-full": ("none", "full", 2, None), "O-risk-L2": ("risk", "causal", 2, None),
    "D4-full": ("none", "full", 4, None), "O-risk-L4": ("risk", "causal", 4, None),
}
FORMAL = ("B0","D3-full","O-conf","O-rand","O-qual","O-risk",
          "O-conf+E0","O-risk+E0","O-conf+QFL","O-risk+QFL","O-riskp")

DT, T_FUT = .4, 12
THETA0 = .5            # E0 / quality-head match threshold at the final step (m), linear in t
TEMP_Q = .5            # weights for O-risk: softmax(-Q_hat / 0.5)  (same temperature as B0 soft cls target)
KIN_W = .10            # weight of the kinematic sub-score (m per m/s^2)
SHIFT = .30            # per-layer bounded update: SHIFT * scale * tanh(.)


# ---------------------------------------------------------------- GT-free sub-scores
def observed_speed(past):                       # past (B,8,2) model frame = metres (rotation+translation only)
    return torch.diff(past, dim=1).norm(dim=-1).mean(1) / DT


def kinematic_reference(past):
    v = torch.diff(past, dim=1) / DT
    return (torch.diff(v, dim=1) / DT).norm(dim=-1).amax(1) + 1.0          # (B,) m/s^2


def kinematic_excess(paths, past, ref):         # paths (B,K,12,2)
    full = torch.cat((past[:, None, -2:].expand(-1, paths.size(1), -1, -1), paths), 2)
    acc = torch.diff(torch.diff(full, dim=2) / DT, dim=2) / DT               # (B,K,12,2)
    return F.relu(acc.norm(dim=-1) - ref[:, None, None]).mean(-1)            # (B,K)


def qhat(qraw, paths, past, ref, use_kin=True, only_q=False):
    """Loss-like quality score, lower = better, metres. Only used to ORDER candidates."""
    ade = (qraw[..., 0].exp() - .05).clamp_min(0)
    fde = (qraw[..., 1].exp() - .05).clamp_min(0)
    q = ade + .25 * fde                                   # predicted B0 quality q = ADE + 0.25 FDE
    if only_q:
        return q
    q = q + THETA0 * (1 - torch.sigmoid(qraw[..., 2]))    # predicted EMTA-miss penalty
    if use_kin:
        q = q + KIN_W * kinematic_excess(paths, past, ref)
    return q


# ---------------------------------------------------------------- orderings (all under no_grad)
def risk_order(end, w):
    """Greedy facility location over all K; identical to order_numpy.risk_order (PF-R1)."""
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


class MLDecoder(nn.Module):
    """Layer 0 = sealed h3 (identical to B0). Layers 1..L: reorder -> self-attn (causal/full)
    -> cross-attn to memory -> bounded anchor update -> residual score update.
    Every arm builds the SAME modules in the SAME order (same init per seed); unused ones stay idle."""

    def __init__(self, h3, arm):
        super().__init__()
        self.base, self.arm = h3, arm
        self.order_kind, attn, self.L, self.emta = ARMS[arm]
        self.only_q = arm == "O-qual-q"
        self.use_kin = True
        self.valid, self.outs = None, None
        self.memory_proj = nn.Linear((h3.context[0].normalized_shape[0] - 7) // 2, 64)
        self.layers = nn.ModuleList([CrossLayer(attn) for _ in range(self.L)])
        self.geometric = nn.ModuleList([nn.Linear(6, 64) for _ in range(self.L)])
        self.rank = nn.Parameter(torch.empty(self.L, 20, 64)); nn.init.normal_(self.rank, std=.02)
        self.offset = nn.ModuleList([zero(nn.Linear(64, 6)) for _ in range(self.L)])
        self.score = nn.ModuleList([nn.Sequential(nn.LayerNorm(70), nn.Linear(70, 64), nn.GELU(),
                                                  zero(nn.Linear(64, 1))) for _ in range(self.L)])
        self.quality = nn.ModuleList([nn.Sequential(nn.LayerNorm(70), nn.Linear(70, 64), nn.GELU(),
                                                    nn.Linear(64, 3)) for _ in range(self.L + 1)])
        # O-rand uses its OWN generators so the global RNG (augmentation) stays identical to the other arms:
        self.train_generator = None         # set once: torch.Generator(device).manual_seed(seed + 1000)
        self.eval_generator = None          # reset to manual_seed(12345) before each evaluation

    # same as archive 26
    def memory(self, encoded):
        own, agents = encoded[:, 0], encoded[:, :, -1]
        n = encoded.size(0)
        pad = torch.arange(agents.shape[1], device=encoded.device)[None] >= self.valid.reshape(n, 1)
        padding = torch.cat((torch.zeros_like(own[:, :, 0], dtype=torch.bool), pad), 1)
        return self.memory_proj(torch.cat((own, agents), 1)), padding

    def proposal(self, h, scale):
        b = self.base
        delta = .75 * torch.tanh(b.proposal_delta(h)).reshape(h.size(0), h.size(1), 3, 2)
        return scale[:, None, None] * (b.anchor_templates[None] + delta)

    def trajectory(self, anchors, last):
        return last[:, None, None] + torch.einsum("tp,bkpc->bktc", self.base.interpolation, anchors)

    @torch.no_grad()
    def make_order(self, paths, logit, qraw, past, ref):
        k = self.order_kind
        if k == "none":
            return None
        if k == "conf":
            return torch.argsort(logit, dim=1, descending=True, stable=True)
        if k == "rand":
            g = self.train_generator if self.training else self.eval_generator
            return torch.rand(logit.shape, device=logit.device, generator=g).argsort(1)
        if k == "riskp":
            return risk_order(paths[:, :, -1], torch.softmax(logit.double(), 1))
        Q = qhat(qraw, paths, past, ref, use_kin=self.use_kin, only_q=self.only_q)
        if k == "qual":
            return torch.argsort(Q, dim=1, stable=True)
        return risk_order(paths[:, :, -1], torch.softmax(-Q.double() / TEMP_Q, 1))   # "risk"

    def forward(self, encoded, past):
        b = self.base
        enc, foc = encoded[:, 0], past[:, :1]
        c = b.context(torch.cat((enc[:, -1], enc.mean(1), b.motion_features(foc)[:, 0]), -1))
        q = b.mode_mlp(c[:, None] + b.mode_embedding[None])
        scale = F.softplus(b.proposal_scale(c)) + .25
        last, p0 = foc[:, 0, -1], foc[:, 0]
        ref = kinematic_reference(p0)
        a = self.proposal(q, scale)
        logit = b.score_head(torch.cat((q, a.flatten(2)), -1)).squeeze(-1)
        memory, padding = self.memory(encoded)
        h, order, outs = q, None, []
        for l in range(self.L + 1):
            paths = self.trajectory(a, last)
            qraw = self.quality[l](torch.cat((h, (a / scale[:, None, None]).flatten(2)), -1).detach())
            outs.append(dict(paths=paths, logits=logit, qraw=qraw, order=order, anchors=a))
            if l == self.L:
                break
            order = self.make_order(paths, logit, qraw, p0, ref)
            x = h + self.geometric[l]((a / scale[:, None, None]).flatten(2))
            if order is not None:
                x = x.gather(1, order[..., None].expand_as(x)) + self.rank[l][None]
            y = self.layers[l](x, memory, padding)
            if order is not None:
                y = y.gather(1, order.argsort(1)[..., None].expand_as(y))
            a = a + SHIFT * scale[:, None, None] * torch.tanh(self.offset[l](y)).reshape(*y.shape[:2], 3, 2)
            logit = logit + self.score[l](torch.cat((y, a.flatten(2)), -1)).squeeze(-1)
            h = y
        self.outs, self.past, self.ref = outs, p0, ref
        return outs[-1]["paths"][:, None], outs[-1]["logits"][:, None]


def attach(model, arm):
    if arm != "B0":
        device = next(model.parameters()).device
        with torch.random.fork_rng(devices=[]):
            model.decoder = MLDecoder(model.decoder.cpu(), arm)
        model.decoder.to(device)
        model.register_forward_pre_hook(set_valid, with_kwargs=True)


# ---------------------------------------------------------------- losses
def emta_indices(dist, order, threshold):
    """dist (B,K,T); order (B,K); threshold (B,) final-step metres, linear in t (archive 26 rule)."""
    steps = torch.arange(1, dist.size(-1) + 1, device=dist.device, dtype=dist.dtype) / dist.size(-1)
    match = (dist <= threshold[:, None, None] * steps).all(-1)
    first = match.gather(1, order).to(torch.int64).argmax(1)
    fallback = (dist.mean(2) + .25 * dist[:, :, -1]).argmin(1)
    return torch.where(match.any(1), order.gather(1, first[:, None])[:, 0], fallback), match


def emta_threshold(kind, past):
    if kind in ("E0", "QFL"):
        return torch.full((past.size(0),), THETA0, device=past.device, dtype=past.dtype)
    if kind == "Ewide":
        return torch.full((past.size(0),), 1.0, device=past.device, dtype=past.dtype)
    return (.10 * observed_speed(past) * DT * T_FUT).clamp(.5, 1.5)          # E1


def confidence_loss(logits, winner, q, use_qfl):
    binary = F.one_hot(winner, logits.shape[-1]).to(logits.dtype)
    if use_qfl:
        # QFL adapts the GFL continuous quality target to the selected trajectory.
        # The error target is detached and shares the frozen B0 quality scale.
        target = binary * torch.exp(-q.detach() / TEMP_Q)
        raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return ((logits.sigmoid() - target).square() * raw).mean()
    raw = F.binary_cross_entropy_with_logits(logits, binary, reduction="none")
    return ((1 - torch.exp(-raw)).square() * raw * (.25 * binary + .75 * (1 - binary))).mean()


def emta_layer_loss(pred, target, logits, order, threshold, criterion, use_qfl=False):
    """Exactly archive-26 S4 except: threshold per window, order = this layer's input order.
    pred (B,1,K,T,2), target (B,1,T,2), logits (B,1,K)."""
    dist = torch.linalg.vector_norm(pred - target.unsqueeze(-3), dim=-1)
    q = dist.mean(-1) + .25 * dist[..., -1]
    with torch.no_grad():
        winner, match = emta_indices(dist[:, 0], order, threshold)
    winner = winner[:, None]
    index = winner[..., None, None, None].expand(*winner.shape, 1, pred.shape[-2], 2)
    select = pred.gather(-3, index).squeeze(-3)
    regression = criterion.regression(select, target)
    endpoint = F.smooth_l1_loss(select[..., -1, :], target[..., -1, :])
    s = criterion.short_horizon_steps
    early = F.smooth_l1_loss(select[..., :s, :], target[..., :s, :])
    focal = confidence_loss(logits, winner, q, use_qfl)
    last = pred[..., -1, :]
    distances = torch.cdist(last, last)
    upper = torch.ones_like(distances, dtype=torch.bool).triu(1)
    diversity = F.relu(criterion.diversity_margin - distances[upper]).mean()
    cover = (torch.softmax(-q.detach() / .10, dim=-1) * q).sum(-1).mean()
    loss = (regression + criterion.classification_weight * focal + criterion.endpoint_weight * endpoint +
            criterion.short_horizon_weight * early + criterion.diversity_weight * diversity + .20 * cover)
    return loss, match, regression.detach(), focal.detach()


def quality_loss(o, target):
    """Targets from this layer's own (detached) candidates; inputs of the head are detached in forward,
    so this loss only trains the quality head (PF-Q1 checks trunk gradients are exactly zero)."""
    with torch.no_grad():
        dist = torch.linalg.vector_norm(o["paths"] - target, dim=-1)           # (B,K,T); target (B,1,T,2)
        ade, fde = dist.mean(-1), dist[..., -1]
        steps = torch.arange(1, dist.size(-1) + 1, device=dist.device, dtype=dist.dtype) / dist.size(-1)
        match = (dist <= THETA0 * steps).all(-1).float()
    r = o["qraw"]
    return (F.smooth_l1_loss(r[..., 0], torch.log(ade + .05), beta=.1) +
            F.smooth_l1_loss(r[..., 1], torch.log(fde + .05), beta=.1) +
            F.binary_cross_entropy_with_logits(r[..., 2], match))


def ml_loss(core, model, arm, target, criterion, supervise_quality=True):
    """Deep supervision: every layer weight 1.0. Layer 0 always B0 loss. Layers >=1: B0 loss, or EMTA.
    Rank hinge (archive 26 S-family) on the final logits in ALL ML arms. Quality head loss weight 0.5."""
    dec = model.decoder
    kind = ARMS[arm][3]
    layers, matches = [], None
    total = 0.
    for l, o in enumerate(dec.outs):
        p, z = o["paths"][:, None], o["logits"][:, None]
        if l == 0 or kind is None:
            lv, detail = core.loss_value(p, target, z, criterion)
            reg, cls = detail["regression"], detail["classification"]
        else:
            lv, matches, reg, cls = emta_layer_loss(
                p, target, z, o["order"], emta_threshold(kind, dec.past), criterion, kind == "QFL")
        quality = quality_loss(o, target) if supervise_quality else lv.new_zeros(())
        total = total + lv + .5 * quality
        layers.append(torch.stack(tuple(torch.as_tensor(val, device=lv.device) for val in
                                        (lv.detach(), reg, cls, quality.detach()))))
    o = dec.outs[-1]
    dist = torch.linalg.vector_norm(o["paths"][:, None] - target.unsqueeze(-3), dim=-1)
    q = dist.mean(-1) + .25 * dist[..., -1]
    logits = o["logits"][:, None]
    pair = q.unsqueeze(-1) + .20 < q.unsqueeze(-2)
    penalty = F.relu(1 - (logits.unsqueeze(-1) - logits.unsqueeze(-2)))
    hinge = .10 * (penalty * pair).sum() / pair.sum().clamp_min(1)
    return total + hinge, matches, torch.stack(layers), hinge.detach()
