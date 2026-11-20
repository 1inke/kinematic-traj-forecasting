"""Reference implementation of centered spread scaling for EXP-ADAPT-WIDTH-CENTERED.
paths: [N,K,T,2] relative to last observed point (as in archive 23 adapt_data); w: [N,K] probabilities."""
import numpy as np

def weighted_mean(paths, w):
    return np.einsum("nk,nktc->ntc", w, paths)

def center_scale(paths, w, alpha):
    """alpha: scalar or [N]. Keeps the probability-weighted mean path fixed; scales deviations."""
    a = np.broadcast_to(np.asarray(alpha, float), (paths.shape[0],))[:, None, None, None]
    m = weighted_mean(paths, w)[:, None]
    return m + a * (paths - m)

def frames(paths, w, heading, min_step=0.05):
    """Per-step along-track unit vector u[N,T,2] of the weighted mean path (central differences,
    m_0 = 0 = last observed point). Steps whose difference norm < min_step m fall back to the
    window direction: mean endpoint if |m_T| >= 0.5 m, else the observation heading (B0 rule)."""
    m = weighted_mean(paths, w)
    z = np.concatenate([np.zeros_like(m[:, :1]), m], 1)            # [N,T+1,2]
    d = np.empty_like(m)
    d[:, :-1] = z[:, 2:] - z[:, :-2]
    d[:, -1] = z[:, -1] - z[:, -2]
    end = m[:, -1]
    fallback = np.where((np.linalg.norm(end, axis=-1) >= 0.5)[:, None], end, heading)
    fallback = fallback / np.linalg.norm(fallback, axis=-1, keepdims=True)
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    u = np.where(n >= min_step, d / np.maximum(n, 1e-12), fallback[:, None])
    return u

def center_scale_aniso(paths, w, a_lon, a_lat, heading):
    m = weighted_mean(paths, w)[:, None]
    u = frames(paths, w, heading)[:, None]                          # [N,1,T,2]
    v = np.stack([-u[..., 1], u[..., 0]], -1)
    dev = paths - m
    lon = (dev * u).sum(-1, keepdims=True); lat = (dev * v).sum(-1, keepdims=True)
    a_lon = np.broadcast_to(np.asarray(a_lon, float), (paths.shape[0],))[:, None, None, None]
    a_lat = np.broadcast_to(np.asarray(a_lat, float), (paths.shape[0],))[:, None, None, None]
    return m + a_lon * lon * u + a_lat * lat * v

def manipulation_checks(paths, w, heading, grid=(0.4, 0.7, 1.0, 1.3)):
    out = {}
    base_pair = np.linalg.norm(paths[:, :, None] - paths[:, None], axis=-1)
    m0 = weighted_mean(paths, w)
    for a in grid:
        x = center_scale(paths, w, a)
        out[f"mean_shift_max_{a}"] = float(np.abs(weighted_mean(x, w) - m0).max())
        pair = np.linalg.norm(x[:, :, None] - x[:, None], axis=-1)
        out[f"pair_ratio_err_{a}"] = float(np.abs(pair - a * base_pair).max())
        y = center_scale_aniso(paths, w, a, a, heading)
        out[f"aniso_equal_vs_iso_{a}"] = float(np.abs(y - x).max())
    out["identity"] = float(np.abs(center_scale(paths, w, 1.0) - paths).max())
    return out

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, K, T = 400, 20, 12
    t = np.arange(1, T + 1) * 0.4
    speed = rng.uniform(0.0, 1.6, N)                                  # mix of stationary and moving
    base = np.stack([speed[:, None] * t, np.zeros((N, T))], -1)       # straight forward mean motion
    jitter = rng.normal(0, 0.15, (N, K, 1, 2)) * (t / t[-1])[None, None, :, None] * 4
    paths = base[:, None] + jitter
    w = rng.dirichlet(np.ones(K), N)
    truth = base + rng.normal(0, 0.15, (N, 1, 2)) * (t / t[-1])[None, :, None] * 4
    heading = np.tile([1.0, 0.0], (N, 1))
    print(manipulation_checks(paths, w, heading))
    def minfde(x): return np.linalg.norm(x[:, :, -1] - truth[:, None, -1], axis=-1).min(1)
    moving = speed > 0.8
    for a in (0.9, 1.0, 1.1):
        print(a, "uncentered", minfde(paths * a)[moving].mean().round(3),
              "centered", minfde(center_scale(paths, w, a))[moving].mean().round(3))
