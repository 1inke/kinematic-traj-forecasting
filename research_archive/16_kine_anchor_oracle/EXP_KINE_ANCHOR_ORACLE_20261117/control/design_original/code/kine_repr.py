"""Template families for EXP-KINE-ANCHOR-ORACLE (NumPy only, no training of any network).

Every family uses the SAME parametric shape as the B0 h3 decoder: three key points at future
steps 4/8/12 in the focal heading frame, linearly interpolated to 12 steps (B0 interpolation
matrix). The families differ only in the coordinate system the 20 templates live in:

  P      absolute positions (metres)                        ~ B0 templates without a scale
  P_AUG  as P, but clustered on speed-augmented futures      ~ "why not just augment"
  S      positions divided by v_eff*DT  (speed normalised)  ~ A without maneuver coordinates
  K      per segment: speed ratio rho_s and direction phi_s ~ A (maneuver templates)

v_eff = max(v_rec, V0): a fixed prior floor so stationary people do not blow up the ratio.
"""
from __future__ import annotations

import numpy as np

from kine_common import DT, K

KEY_STEPS = (4, 8, 12)
KEY_INDEX = np.array(KEY_STEPS) - 1
SEG_LEN = 4
V0 = 0.30            # m/s, prior (slow walking); sensitivity 0.15 / 0.50 reported, never selected
RHO_EPS = 0.05       # log(rho + eps); also the "no direction" threshold
LAMBDA = 1.0         # weight of phi (rad) against log speed ratio; sensitivity 0.5 / 2 reported
AUG_RANGE = (0.5, 2.0)   # identical to EXP-SPEED-AUG
AUG_PROB = 0.5
AUG_COPIES = 4
ORACLE_SCALES = np.geomspace(0.25, 4.0, 17)
FAMILIES = ("P", "P_AUG", "S", "K")


def interpolation_weights(pred_len=12, anchor_steps=KEY_STEPS):
    """Copied from hierarchical_decoder_models.interpolation_weights."""
    w = np.zeros((pred_len, len(anchor_steps)))
    prev = 0
    for seg, step in enumerate(anchor_steps):
        for t in range(prev + 1, step + 1):
            a = (t - prev) / (step - prev)
            w[t - 1, seg] = a
            if seg:
                w[t - 1, seg - 1] = 1.0 - a
        prev = step
    if anchor_steps[-1] < pred_len:
        w[anchor_steps[-1]:, -1] = 1.0
    return w


W_INTERP = interpolation_weights()


def keypoints(future_local):
    return np.asarray(future_local, np.float64)[:, KEY_INDEX]          # [N,3,2]


def v_eff(v_rec, v0=V0):
    return np.maximum(np.asarray(v_rec, np.float64), v0)


# ----------------------------------------------------------------------------- K coordinates
def to_maneuver(kp, ve, lam=LAMBDA):
    """kp [N,3,2] metres -> feature [N,6] = (log(rho_s+eps) x3, lam*phi_s x3)."""
    prev = np.concatenate([np.zeros_like(kp[:, :1]), kp[:, :-1]], axis=1)
    u = (kp - prev) / (SEG_LEN * DT * ve[:, None, None])                 # normalised mean velocity
    rho = np.linalg.norm(u, axis=-1)
    phi = np.arctan2(u[..., 1], u[..., 0])
    for s in range(3):                                                  # no direction when ~still
        still = rho[:, s] < RHO_EPS
        phi[still, s] = phi[still, s - 1] if s else 0.0
    return np.concatenate([np.log(rho + RHO_EPS), lam * phi], axis=1)


def from_maneuver(feat, ve, lam=LAMBDA):
    """feat [M,6] (templates) and ve [N] -> keypoints [N,M,3,2]."""
    rho = np.maximum(np.exp(feat[:, :3]) - RHO_EPS, 0.0)
    phi = feat[:, 3:] / lam
    step = rho[..., None] * np.stack([np.cos(phi), np.sin(phi)], axis=-1)   # [M,3,2] per unit
    seg = step[None] * (SEG_LEN * DT * np.asarray(ve, np.float64))[:, None, None, None]
    return np.cumsum(seg, axis=2)


# ----------------------------------------------------------------------------- features
def features(family, kp, ve, lam=LAMBDA):
    if family in ("P", "P_AUG"):
        return kp.reshape(len(kp), -1)
    if family == "S":
        return (kp / (DT * ve[:, None, None])).reshape(len(kp), -1)
    if family == "K":
        return to_maneuver(kp, ve, lam)
    raise ValueError(family)


def candidates(family, templates, ve, lam=LAMBDA):
    """templates [20,6] in the family's space -> candidate key points [N,20,3,2] (metres)."""
    n = len(ve)
    if family in ("P", "P_AUG"):
        return np.broadcast_to(templates.reshape(1, -1, 3, 2), (n, len(templates), 3, 2))
    if family == "S":
        return templates.reshape(1, -1, 3, 2) * (DT * ve)[:, None, None, None]
    if family == "K":
        return from_maneuver(templates, ve, lam)
    raise ValueError(family)


def augment_keypoints(kp, rng):
    """EXP-SPEED-AUG on the future only (templates never see the past): with p=0.5 scale the
    whole future about the last observed point by U(0.5,2). AUG_COPIES independent copies."""
    out = []
    for _ in range(AUG_COPIES):
        s = rng.uniform(*AUG_RANGE, size=len(kp))
        s = np.where(rng.random(len(kp)) < AUG_PROB, s, 1.0)
        out.append(kp * s[:, None, None])
    return np.concatenate(out, axis=0)


# ----------------------------------------------------------------------------- k-means
def kmeans(x, k=K, seed=0, iters=200):
    """Deterministic k-means++ + Lloyd. Empty clusters are re-seeded with the farthest point."""
    x = np.asarray(x, np.float64)
    if len(x) < k:
        raise ValueError("TOO_FEW_POINTS_FOR_KMEANS")
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))]]
    d2 = ((x - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(len(x), 1 / len(x))
        centers.append(x[rng.choice(len(x), p=p)])
        d2 = np.minimum(d2, ((x - centers[-1]) ** 2).sum(1))
    c = np.array(centers)
    assign = None
    for _ in range(iters):
        dist = _sqdist(x, c)
        new = dist.argmin(1)
        if assign is not None and np.array_equal(new, assign):
            break
        assign = new
        for j in range(k):
            members = assign == j
            if members.any():
                c[j] = x[members].mean(0)
            else:
                far = dist.min(1).argmax()
                c[j] = x[far]
                assign[far] = j
    return c


def _sqdist(x, c, chunk=65536):
    out = np.empty((len(x), len(c)))
    for i in range(0, len(x), chunk):
        xi = x[i:i + chunk]
        out[i:i + chunk] = (xi ** 2).sum(1)[:, None] - 2 * xi @ c.T + (c ** 2).sum(1)[None]
    return np.maximum(out, 0)


def fit_templates(family, kp_fit, ve_fit, seed, lam=LAMBDA):
    if family == "P_AUG":
        kp_fit = augment_keypoints(kp_fit, np.random.default_rng(10_000 + seed))
        return kmeans(features("P", kp_fit, None), seed=seed)
    return kmeans(features(family, kp_fit, ve_fit, lam), seed=seed)


# ----------------------------------------------------------------------------- metrics
def best_of(cand_kp, future_local, chunk=4096):
    """cand_kp [N,M,3,2], future [N,12,2] -> minADE, minFDE (independent minima, metres)."""
    n = len(future_local)
    ade, fde = np.empty(n), np.empty(n)
    for i in range(0, n, chunk):
        path = np.einsum("tp,nmpc->nmtc", W_INTERP, cand_kp[i:i + chunk])
        err = np.linalg.norm(path - future_local[i:i + chunk, None], axis=-1)   # [n,M,12]
        ade[i:i + chunk] = err.mean(-1).min(-1)
        fde[i:i + chunk] = err[..., -1].min(-1)
    return ade, fde


def oracle_scale_P(templates, future_local):
    """REFERENCE ONLY (uses the ground truth): per window pick the isotropic scale of the P
    templates that minimises best-of-20 ADE = the ceiling of B0's learned scale mechanism."""
    n = len(future_local)
    best_ade, best_fde = np.full(n, np.inf), np.full(n, np.inf)
    base = templates.reshape(1, -1, 3, 2)
    for s in ORACLE_SCALES:
        a, f = best_of(np.broadcast_to(base * s, (n,) + base.shape[1:]), future_local)
        better = a < best_ade
        best_ade = np.where(better, a, best_ade)
        best_fde = np.where(better, f, best_fde)
    return best_ade, best_fde


def template_table(family, templates, lam=LAMBDA):
    """Human-readable K templates: speed ratio and direction (deg) per segment."""
    if family != "K":
        return None
    rho = np.maximum(np.exp(templates[:, :3]) - RHO_EPS, 0.0)
    phi = np.degrees(templates[:, 3:] / lam)
    order = np.argsort(rho[:, 2])
    return [dict(rho=[round(float(v), 3) for v in rho[j]], phi_deg=[round(float(v), 1) for v in phi[j]])
            for j in order]
