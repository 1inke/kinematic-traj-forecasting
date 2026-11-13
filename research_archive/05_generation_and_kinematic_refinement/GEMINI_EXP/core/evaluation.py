import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

GROUPS = ('overall', 'stable', 'transition', 'change')

def masks(groups):
    """Returns boolean masks for overall, stable, transition, change."""
    return [(name, np.ones(len(groups), bool) if j == 0 else groups == j - 1) for j, name in enumerate(GROUPS)]

def compute_errors(p, gt):
    """
    p: (N, K, 12, 2)
    gt: (N, 12, 2)
    Returns:
      a: ADE (N, K)
      f: FDE (N, K)
      j: J metric (N, K) = max(ADE / 0.2, FDE / 0.4)
      g: good mask (N, K) = (ADE <= 0.2) & (FDE <= 0.4)
      one: segments error (N, 3) for the winner
    """
    d = p.astype(np.float64) - gt[:, None].astype(np.float64)
    e = np.linalg.norm(d, axis=-1)
    a = e.mean(-1)
    f = e[:, :, -1]
    j = np.maximum(a / 0.2, f / 0.4)
    g = (a <= 0.2) & (f <= 0.4)
    ix = j.argmin(1)
    one = e[np.arange(len(p)), ix].reshape(-1, 3, 4).mean(-1)
    return a, f, j, g, one

def compute_motion(p):
    """
    5 pooled motion metrics:
    dt = 0.4s
    v[0] = p[0] / 0.4 (or diff prepend zeros)
    a = diff(v) / 0.4
    jerk = diff(a) / 0.4
    """
    p = p.astype(np.float64)
    v = np.diff(p, axis=2, prepend=np.zeros_like(p[:, :, :1])) / 0.4
    a = np.diff(v, axis=2) / 0.4
    j = np.diff(a, axis=2) / 0.4
    s = np.linalg.norm(v, axis=-1)
    aa = np.linalg.norm(a, axis=-1)
    jj = np.linalg.norm(j, axis=-1)
    return dict(
        speed_p95=float(np.quantile(s, 0.95)),
        accel_mean=float(aa.mean()),
        accel_p95=float(np.quantile(aa, 0.95)),
        jerk_mean=float(jj.mean()),
        jerk_p95=float(np.quantile(jj, 0.95))
    )

def compute_clusters(p, g):
    """
    Computes complete-linkage clustering per window.
    Distance between candidates: D = max(meanpath / 0.1, endpoint / 0.2)
    Cut threshold = 1.0
    Returns:
      distinct: number of clusters (N,)
      effective_good: number of clusters containing at least 1 good candidate (N,)
    """
    out = []
    for paths, good in zip(p, g):
        K = len(paths)
        if K < 2:
            out.append((1, int(good.any())))
            continue
        diff = paths[:, None] - paths[None, :]
        dist_t = np.linalg.norm(diff, axis=-1)
        mean_d = dist_t.mean(-1) / 0.1
        end_d = dist_t[:, :, -1] / 0.2
        d_mat = np.maximum(mean_d, end_d)
        condensed = d_mat[np.triu_indices(K, 1)]
        cl = fcluster(linkage(condensed, method='complete'), 1.0, criterion='distance')
        distinct = len(np.unique(cl))
        eff_good = len(np.unique(cl[good])) if good.any() else 0
        out.append((distinct, eff_good))
    return np.asarray(out)

def evaluate_metrics(z):
    """
    Evaluates complete metrics across 4 groups.
    z is a dict containing:
      paths: (N, 20, 12, 2)
      native: (N, 20, 12, 2)
      gt: (N, 12, 2)
      groups: (N,) [0: stable, 1: transition, 2: change]
      track: (N,)
      sample_id: (N,)
    """
    p = z['paths']
    native = z['native']
    gt = z['gt']

    a, f, j, g, early = compute_errors(p, gt)
    na, nf, nj, ng, _ = compute_errors(native, gt)
    cc = compute_clusters(p, g)
    gg = g.any(1)
    basegood = ng.any(1)

    qw = (a + 0.5 * f).argmin(1)
    jw = j.argmin(1)
    nw = (na + 0.5 * nf).argmin(1)

    error = np.linalg.norm(p - gt[:, None], axis=-1)
    q_seg = error[np.arange(len(p)), qw].reshape(-1, 3, 4).mean(-1)

    aux = dict(
        ADE=a,
        FDE=f,
        J=j,
        good=g,
        clusters=cc,
        vectors4=np.stack((gg, a.min(1), f.min(1), j.min(1)), 1),
        native_vectors4=np.stack((basegood, na.min(1), nf.min(1), nj.min(1)), 1),
        Q_winner=qw,
        J_winner=jw,
        native_Q_winner=nw,
        Q_segments=q_seg,
        J_segments=early
    )

    rows = {}
    for name, m in masks(z['groups']):
        if not m.any():
            rows[name] = dict(n=0)
            continue
        mot = compute_motion(p[m])
        base = compute_motion(native[m])
        den = int((basegood & m).sum())
        drift = np.linalg.norm(p[m] - native[m], axis=-1)

        motion_pass = all(mot[k] <= 1.01 * base[k] for k in mot)

        rows[name] = dict(
            n=int(m.sum()),
            tracks=len(set(z['track'][m])),
            good=int((gg & m).sum()),
            native_good=den,
            coverage=float(gg[m].mean()),
            rescue=int((gg & ~basegood & m).sum()),
            lost=int((~gg & basegood & m).sum()),
            retained=int((gg & basegood & m).sum()),
            lost_rate_native_good=float((~gg & basegood & m).sum() / den) if den else 0.0,
            ADE=float(a[m].min(1).mean()),
            FDE=float(f[m].min(1).mean()),
            J=float(j[m].min(1).mean()),
            native_ADE=float(na[m].min(1).mean()),
            native_FDE=float(nf[m].min(1).mean()),
            native_J=float(nj[m].min(1).mean()),
            effective_good=float(cc[m, 1].mean()),
            distinct=float(cc[m, 0].mean()),
            motion=mot,
            native_motion=base,
            motion_pass=bool(motion_pass),
            correction_mean=float(drift.mean()),
            correction_p95=float(np.quantile(drift, 0.95)),
            correction_max=float(drift.max()),
            Q_winner_switch_rate=float((qw[m] != nw[m]).mean()),
            rescue_IDs=z['sample_id'][gg & ~basegood & m].tolist(),
            lost_IDs=z['sample_id'][~gg & basegood & m].tolist()
        )

    return rows, aux
