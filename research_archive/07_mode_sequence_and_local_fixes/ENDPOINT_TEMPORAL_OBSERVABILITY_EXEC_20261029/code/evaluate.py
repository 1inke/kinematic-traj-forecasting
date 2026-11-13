"""Saved path metrics: same-slot quality, original complete-link, pooled physics."""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from motion_guard import pooled, junction, inspect, MOTION_NAMES
from build_teacher import samepath_quality


def clusters(paths):
    delta = np.linalg.norm(paths[:, None]-paths[None, :], axis=-1)
    distance = np.maximum(delta.mean(-1)/.10, delta[..., -1]/.20)
    np.fill_diagonal(distance, 0.)
    return fcluster(linkage(squareform(distance, checks=True), method='complete'),
                    t=1., criterion='distance')


def sample_metrics(paths, gt):
    p = np.asarray(paths, dtype=np.float64)
    a, f, j, good = samepath_quality(p, gt[:, None])
    winner = j.argmin(1)
    errors = np.linalg.norm(p[np.arange(len(p)), winner]-gt, axis=-1)
    distinct, effective = [], []
    for i in range(len(p)):
        labels = clusters(p[i])
        distinct.append(len(np.unique(labels)))
        effective.append(len(np.unique(labels[good[i]])))
    return dict(ADE=a, FDE=f, J=j, good=good, covered=good.any(1),
                minADE=a.min(1), minFDE=f.min(1), minJ=j.min(1), winner=winner,
                early_ADE=errors[:, :4].mean(1), middle_ADE=errors[:, 4:8].mean(1),
                late_ADE=errors[:, 8:].mean(1), distinct=np.array(distinct),
                effective_good=np.array(effective),
                coverage_sensitive_03_05=((a <= .3) & (f <= .5)).any(1))


def evaluate_population(x, gt, paths, check=lambda: None):
    p, n = np.asarray(paths, dtype=np.float64), np.asarray(x['native'], dtype=np.float64)
    if p.shape != n.shape or not np.isfinite(p).all():
        raise ValueError('invalid evaluation paths')
    check()
    sample, base = sample_metrics(p, gt), sample_metrics(n, gt)
    checks = np.empty(p.shape[:2], bool)
    jnew, jold = [], []
    for i in range(len(p)):
        check()
        jnew.append(junction(x['past'][i], p[i]))
        jold.append(junction(x['past'][i], n[i]))
        for k in range(20):
            checks[i, k] = inspect(x['past'][i], n[i, k], p[i, k])[0]
    jnew, jold = np.array(jnew), np.array(jold)
    rows = []
    for group, mask in [('overall', np.ones(len(p), bool))]+[
            (name, x['group'] == i) for i, name in enumerate(('stable', 'transition', 'change'))]:
        count = int(mask.sum())
        if not count:
            rows.append(dict(group=group, N=0, status='NA_EMPTY'))
            continue
        b, c = base['covered'][mask], sample['covered'][mask]
        pm, nm = pooled(p[mask]), pooled(n[mask])
        rows.append(dict(group=group, N=count,
            native_good=int(b.sum()), good=int(c.sum()), rescue=int((~b & c).sum()),
            lost=int((b & ~c).sum()), retained=int((b & c).sum()),
            lost_fraction=float((b & ~c).sum()/b.sum()) if b.any() else None,
            coverage_gain=float(c.mean()-b.mean()),
            **{key: float(sample[key][mask].mean()) for key in
               ('minADE', 'minFDE', 'minJ', 'distinct', 'effective_good',
                'early_ADE', 'middle_ADE', 'late_ADE')},
            **{'native_'+key: float(base[key][mask].mean()) for key in ('minADE', 'minFDE', 'minJ')},
            motion=dict(zip(MOTION_NAMES, map(float, pm))),
            native_motion=dict(zip(MOTION_NAMES, map(float, nm))),
            motion_literal_pass=bool(np.all(pm <= nm*1.01)),
            junction_mean=float(jnew[mask].mean()), junction_p95=float(np.quantile(jnew[mask], .95)),
            native_junction_mean=float(jold[mask].mean()), native_junction_p95=float(np.quantile(jold[mask], .95)),
            junction_literal_pass=bool(jnew[mask].mean() <= jold[mask].mean()*1.01 and
                np.quantile(jnew[mask], .95) <= np.quantile(jold[mask], .95)*1.01),
            all20_candidate_pass_windows=int(checks[mask].all(1).sum()),
            bootstrap_frame_track_count=int(len(np.unique(x['track_key']))),
            group_track_count=int(len(np.unique(x['track_key'][mask])))))
    sample.update(native_covered=base['covered'], candidate_pass=checks)
    return sample, rows
