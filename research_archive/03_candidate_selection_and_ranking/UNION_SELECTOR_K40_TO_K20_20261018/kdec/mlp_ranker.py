"""Tiny MLP ranker trained on FIT soft quality labels."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TinyRanker(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_ranker(X, y, steps=1500, lr=3e-3, batch=8192, seed=0):
    rng = np.random.default_rng(seed)
    mu = X.mean(0)
    sig = X.std(0) + 1e-6
    Xs = ((X - mu) / sig).astype(np.float32)
    ys = y.astype(np.float32)
    pos = np.where(ys >= 0.5)[0]
    neg = np.where(ys < 0.5)[0]
    d_in = Xs.shape[1]
    net = TinyRanker(d_in).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    net.train()
    for t in range(steps):
        n_pos = batch // 2
        n_neg = batch - n_pos
        ip = rng.choice(pos, size=min(n_pos, len(pos)), replace=len(pos) < n_pos)
        inn = rng.choice(neg, size=min(n_neg, len(neg)), replace=len(neg) < n_neg)
        ix = np.concatenate([ip, inn])
        xb = torch.from_numpy(Xs[ix]).to(DEVICE)
        yb = torch.from_numpy(ys[ix]).to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(net(xb), yb)
        loss.backward()
        opt.step()
    net.eval()
    return dict(
        state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
        mu=mu, sig=sig, d_in=d_in,
    )


@torch.no_grad()
def score_mlp(feat, model):
    Xs = ((feat - model["mu"]) / model["sig"]).astype(np.float32)
    net = TinyRanker(model["d_in"]).to(DEVICE)
    net.load_state_dict(model["state_dict"])
    net.eval()
    out = []
    for i in range(0, len(Xs), 65536):
        xb = torch.from_numpy(Xs[i:i + 65536]).to(DEVICE)
        out.append(torch.sigmoid(net(xb)).cpu().numpy())
    return np.concatenate(out, 0)
