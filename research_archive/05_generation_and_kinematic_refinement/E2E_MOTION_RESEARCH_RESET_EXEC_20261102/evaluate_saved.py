"""World-coordinate same-slot quality and causal motion; no historical native model."""
import sys
import time
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from common import *
from causal_dataset import motion
sys.path.insert(0, str(ROOT / 'src'))
from f16.eval import errors, clusters

GROUPS = ('overall', 'stable', 'transition', 'change')


def pooled(m):
    return dict(speed_p95=float(np.quantile(m['speed'], .95)),
        accel_mean=float(m['acceleration'].mean()), accel_p95=float(np.quantile(m['acceleration'], .95)),
        jerk_mean=float(m['jerk'].mean()), jerk_p95=float(np.quantile(m['jerk'], .95)),
        junction_accel_mean=float(m['junction_acceleration'].mean()),
        junction_accel_p95=float(np.quantile(m['junction_acceleration'], .95)),
        junction_jerk_mean=float(m['junction_jerk'].mean()),
        junction_jerk_p95=float(np.quantile(m['junction_jerk'], .95)))


def evaluate(paths, logits, data, guard=None):
    paths = np.asarray(paths, np.float64)
    n = len(paths)
    if paths.shape != (n, 20, 12, 2) or logits.shape != (n, 20):
        raise ValueError('BAD_OUTPUT_SHAPE')
    if len(data['ids']) != n or len(np.unique(data['ids'])) != n or not np.isfinite(paths).all() or not np.isfinite(logits).all():
        raise ValueError('BAD_OUTPUT_ID_OR_NONFINITE')
    y = data['future'].astype(np.float64)
    e = errors(paths, y)
    cc = clusters(paths, e['good'])
    # Independent second condensed-distance implementation, all K20 and all windows.
    independent_cc = []
    for i, (p, good) in enumerate(zip(paths, e['good'])):
        d = np.stack([pdist(p[:, t]) for t in range(12)], -1)
        c = fcluster(linkage(np.maximum(d.mean(1) / .10, d[:, -1] / .20), method='complete'), 1, criterion='distance')
        independent_cc.append((len(np.unique(c)), len(np.unique(c[good]))))
        if guard and i % 256 == 0:
            guard.check()
    if not np.array_equal(cc, independent_cc):
        raise ValueError('CLUSTER_IMPLEMENTATION_MISMATCH')
    # Second direct error arithmetic, no shared quality helper.
    dd = np.sqrt(np.sum((paths - y[:, None])**2, axis=-1))
    max_error = max(float(np.max(np.abs(dd.sum(-1) / 12 - e['ADE']))),
                    float(np.max(np.abs(dd[..., -1] - e['FDE']))))
    if max_error > 2e-12 or not np.array_equal((dd.sum(-1) / 12 <= .2) & (dd[..., -1] <= .4), e['good']):
        raise ValueError('QUALITY_INDEPENDENT_MISMATCH')
    past = data['past'][:, 0].astype(np.float64)
    m = motion(np.broadcast_to(past[:, None], (n, 20, 8, 2)), paths)
    # Independent stencil, including true observed last position (not world origin).
    v = (paths - np.concatenate([np.broadcast_to(past[:, None, -1:, :], (n, 20, 1, 2)), paths[:, :, :-1]], axis=2)) / .4
    a = (v[:, :, 1:] - v[:, :, :-1]) / .4
    j = (v[:, :, 2:] - 2 * v[:, :, 1:-1] + v[:, :, :-2]) / .16
    for key, value in [('speed', v), ('acceleration', a), ('jerk', j)]:
        error = float(np.max(np.abs(np.sqrt(np.sum(value**2, -1)) - m[key])))
        max_error = max(max_error, error)
        if error > 2e-11:
            raise ValueError('MOTION_INDEPENDENT_STENCIL_MISMATCH')
    reference = read_json(HERE / 'C0/EVALUATION_CONTRACT.json')['tail_thresholds']
    top = logits.argmax(1)
    row = np.arange(n)
    result = {}
    tracks = np.asarray(['|'.join(x.split('|')[:2]) for x in data['ids']])
    for k, name in enumerate(GROUPS):
        mask = np.ones(n, bool) if k == 0 else data['group'] == k - 1
        count = int(mask.sum())
        if not count:
            result[name] = {'n': 0}
            continue
        mm = {key: value[mask] for key, value in m.items()}
        tails = {}
        for port, chosen in [('allK20', None), ('score_top1_not_probability', top), ('GT_oracle_J_slot_not_deployable', e['J_winner'])]:
            tails[port] = {}
            for key, value in m.items():
                x = value[mask] if chosen is None else value[row, chosen][mask]
                threshold = reference[key]
                excess = np.maximum(x - threshold, 0)
                positive = excess[excess > 0]
                tails[port][key] = dict(denominator=int(x.size), threshold=threshold,
                    exceedance_count=int((x > threshold).sum()), exceedance_rate=float((x > threshold).mean()),
                    positive_excess_p95=float(np.quantile(positive, .95)) if len(positive) else 0.,
                    relative_inference='NOT_ESTIMABLE_ZERO_REFERENCE' if threshold == 0 else 'DESCRIPTIVE_PILOT_ONLY')
        result[name] = dict(n=count, target_tracks=len(np.unique(tracks[mask])),
            good=int(e['coverage'][mask].sum()), coverage=float(e['coverage'][mask].mean()),
            minADE=float(e['ADE'][mask].min(1).mean()), minFDE=float(e['FDE'][mask].min(1).mean()),
            minJ=float(e['J'][mask].min(1).mean()), distinct=float(cc[mask, 0].mean()),
            effective_good=float(cc[mask, 1].mean()), motion=pooled(mm), tail_risk=tails,
            J_segments=e['J_segments'][mask].mean(0).tolist(), Q_segments=e['Q_segments'][mask].mean(0).tolist(),
            score_top1_ADE=float(e['ADE'][row, top][mask].mean()), score_top1_FDE=float(e['FDE'][row, top][mask].mean()),
            LEGACY_SHADOW={'old_native_comparator': 'NOT_COMPARABLE_NO_MATCHED_OUTPUT_ON_NEW_IDS',
                           'B0_self_support_loss': 0, 'self_comparison_is_not_learned_retention': True,
                           'old_major': 'NOT_ASSESSED_NOT_UPGRADED'})
    return result, e, {'status': 'PASS_ALL_WINDOWS_INDEPENDENT_QUALITY_PDIST_FULLK20_MOTION',
                       'windows': n, 'max_arithmetic_error': max_error}


def save_population(folder, paths, logits, data, guard=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder / 'predictions.npz', paths=np.asarray(paths, np.float64),
                        logits=logits, ids=data['ids'], future=data['future'], past=data['past'][:, 0], groups=data['group'])
    with np.load(folder / 'predictions.npz') as saved:
        if not np.array_equal(saved['ids'], data['ids']) or not np.array_equal(saved['paths'], paths):
            raise ValueError('SAVED_OUTPUT_NOT_EXACT')
        rows, per_window, audit = evaluate(saved['paths'], saved['logits'], data, guard)
    np.savez_compressed(folder / 'window_metrics.npz', **per_window)
    write_json(folder / 'metrics.json', rows)
    write_json(folder / 'audit.json', audit)
    worst = np.argsort(-per_window['J'].min(1), kind='stable')[:50]
    write_csv(folder / 'WORST_CASES.csv', [dict(id=str(data['ids'][i]), group=int(data['group'][i]),
        minJ=float(per_window['J'][i].min()), minADE=float(per_window['ADE'][i].min()),
        minFDE=float(per_window['FDE'][i].min()), J_slot=int(per_window['J_winner'][i]),
        Q_slot=int(per_window['Q_winner'][i])) for i in worst])
    write_json(folder / 'done.json', {'status': 'SUCCESS', 'files': seal(folder.iterdir()),
                                    'source': sha(HERE / 'source_manifest.json') if (HERE / 'source_manifest.json').exists() else 'PREFLIGHT_UNSEALED'})
    return rows
