"""Past-conditioned candidate ranker: score(past, path)."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DT = 0.4


def build_cond_features(past: np.ndarray, paths: np.ndarray, family: np.ndarray) -> np.ndarray:
    """past (B,8,2), paths (B,K,12,2), family (B,K) -> (B,K,F)."""
    B, K, T, _ = paths.shape
    past = past.astype(np.float64)
    paths = paths.astype(np.float64)
    # past velocity / last disp
    v0 = (past[:, -1] - past[:, -2]) / DT  # B,2
    past_speed = np.linalg.norm(v0, axis=-1, keepdims=True)  # B,1
    past_len = np.linalg.norm(np.diff(past, axis=1), axis=-1).sum(-1, keepdims=True)

    ep = paths[:, :, -1, :]
    ep_n = np.linalg.norm(ep, axis=-1, keepdims=True)
    # alignment of endpoint with v0
    v0n = v0 / (np.linalg.norm(v0, axis=-1, keepdims=True) + 1e-6)
    ep_u = ep / (ep_n + 1e-6)
    align = (ep_u * v0n[:, None, :]).sum(-1, keepdims=True)

    v = np.diff(paths, axis=2, prepend=np.zeros_like(paths[:, :, :1])) / DT
    speed = np.linalg.norm(v, axis=-1)
    a = np.diff(v, axis=2) / DT
    accel = np.linalg.norm(a, axis=-1)
    # first-step velocity vs v0
    v1 = v[:, :, 0, :]
    v1_align = (v1 * v0n[:, None, :]).sum(-1, keepdims=True) / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-6)
    dv0 = np.linalg.norm(v1 - v0[:, None, :], axis=-1, keepdims=True)

    # lateral endpoint (perp to v0)
    perp = np.stack([-v0n[:, 1], v0n[:, 0]], axis=-1)
    lat = (ep * perp[:, None, :]).sum(-1, keepdims=True)
    along = (ep * v0n[:, None, :]).sum(-1, keepdims=True)

    # consensus: dist to median endpoint of same family
    fam = family
    # rough: dist to mean of all 40 later — here per-tensor mean
    mean_ep = ep.mean(1, keepdims=True)
    to_mean = np.linalg.norm(ep - mean_ep, axis=-1, keepdims=True)

    plen = np.linalg.norm(np.diff(paths, axis=2), axis=-1).sum(-1, keepdims=True)
    fam_f = fam.astype(np.float64)[..., None]

    # broadcast past stats
    ps = np.broadcast_to(past_speed[:, None, :], (B, K, 1)).copy()
    pl = np.broadcast_to(past_len[:, None, :], (B, K, 1)).copy()

    # cross-family min dist if we have both — caller can append; here within-set min to others
    d = np.linalg.norm(ep[:, :, None, :] - ep[:, None, :, :], axis=-1)
    eye = np.eye(K, dtype=bool)
    d[:, eye] = np.inf
    nn = d.min(axis=-1, keepdims=True)

    return np.concatenate([
        ep, ep_n, align, lat, along,
        speed.mean(-1, keepdims=True), speed.max(-1, keepdims=True), speed[:, :, -1:],
        accel.mean(-1, keepdims=True), accel.max(-1, keepdims=True),
        v1_align, dv0, plen, to_mean, nn, fam_f, ps, pl,
        np.broadcast_to(v0[:, None, :], (B, K, 2)).copy(),
    ], axis=-1)


class CondRanker(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_cond_ranker(X, y, steps=2500, lr=2e-3, batch=8192, seed=0):
    rng = np.random.default_rng(seed)
    mu = X.mean(0); sig = X.std(0) + 1e-6
    Xs = ((X - mu) / sig).astype(np.float32)
    ys = y.astype(np.float32)
    pos = np.where(ys >= 0.5)[0]
    neg = np.where(ys < 0.5)[0]
    # also include soft mid
    net = CondRanker(Xs.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    net.train()
    for t in range(steps):
        n_pos = batch // 2
        ip = rng.choice(pos, size=min(n_pos, len(pos)), replace=len(pos) < n_pos)
        inn = rng.choice(neg, size=min(batch - len(ip), len(neg)), replace=True)
        ix = np.concatenate([ip, inn])
        xb = torch.from_numpy(Xs[ix]).to(DEVICE)
        yb = torch.from_numpy(ys[ix]).to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(net(xb), yb)
        loss.backward()
        opt.step()
    net.eval()
    return dict(
        type="cond_mlp",
        state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
        mu=mu, sig=sig, d_in=Xs.shape[1],
    )


@torch.no_grad()
def score_cond(feat, model):
    Xs = ((feat - model["mu"]) / model["sig"]).astype(np.float32)
    net = CondRanker(model["d_in"]).to(DEVICE)
    net.load_state_dict(model["state_dict"])
    net.eval()
    outs = []
    for i in range(0, len(Xs), 65536):
        xb = torch.from_numpy(Xs[i:i+65536]).to(DEVICE)
        outs.append(torch.sigmoid(net(xb)).cpu().numpy())
    return np.concatenate(outs, 0)




def train_pairwise_ranker(X_sample, y_good, steps=2000, lr=2e-3, pairs_per_step=8192, seed=0):
    """Vectorized pairwise hinge within sample. X (N,40,F), y (N,40)."""
    rng = np.random.default_rng(seed)
    N, M, F = X_sample.shape
    flat = X_sample.reshape(-1, F)
    mu = flat.mean(0); sig = flat.std(0) + 1e-6
    Xs = ((X_sample - mu) / sig).astype(np.float32)
    Y = (y_good > 0.5)
    net = CondRanker(F, hidden=128).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    # build pos/neg index lists
    pos_lists, neg_lists, both = [], [], []
    for i in range(N):
        p = np.where(Y[i])[0]
        n = np.where(~Y[i])[0]
        pos_lists.append(p); neg_lists.append(n)
        if len(p) and len(n):
            both.append(i)
    both = np.asarray(both, dtype=np.int64)
    net.train()
    for step in range(steps):
        b_ix = rng.choice(both, size=pairs_per_step, replace=True)
        pi = np.empty(pairs_per_step, dtype=np.int64)
        ni = np.empty(pairs_per_step, dtype=np.int64)
        for j, i in enumerate(b_ix):
            pi[j] = rng.choice(pos_lists[i])
            ni[j] = rng.choice(neg_lists[i])
        xp = torch.from_numpy(Xs[b_ix, pi]).to(DEVICE)
        xn = torch.from_numpy(Xs[b_ix, ni]).to(DEVICE)
        loss = torch.clamp(1.0 - (net(xp) - net(xn)), min=0).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return dict(
        type="cond_mlp",
        state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
        mu=mu, sig=sig, d_in=F,
    )


def train_cond_bce(X, y, steps=3000, lr=2e-3, batch=8192, seed=0):
    """Balanced BCE on flattened features (fast, strong baseline)."""
    rng = np.random.default_rng(seed)
    mu = X.mean(0); sig = X.std(0) + 1e-6
    Xs = ((X - mu) / sig).astype(np.float32)
    ys = y.astype(np.float32)
    pos = np.where(ys >= 0.5)[0]
    neg = np.where(ys < 0.5)[0]
    net = CondRanker(Xs.shape[1], hidden=128).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    net.train()
    for tstep in range(steps):
        ip = rng.choice(pos, size=batch // 2, replace=True)
        inn = rng.choice(neg, size=batch // 2, replace=True)
        ix = np.concatenate([ip, inn])
        xb = torch.from_numpy(Xs[ix]).to(DEVICE)
        yb = torch.from_numpy(ys[ix]).to(DEVICE)
        opt.zero_grad(); loss_fn(net(xb), yb).backward(); opt.step()
    net.eval()
    return dict(
        type="cond_mlp",
        state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
        mu=mu, sig=sig, d_in=Xs.shape[1],
    )
