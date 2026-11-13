"""Independent scalar geometry for saved-path checking (not external peer review)."""
import numpy as np
from action_geometry import space


def scalar_apply(path, branch, action_id):
    """Does not call production interpolation/geometry/guard."""
    p = np.asarray(path, dtype=np.float64)
    s = space(branch)
    k = int(action_id)
    if k == s.identity:
        return p.copy()
    if k < 0 or k >= len(s.params):
        raise ValueError('bad saved action index')
    if branch == 'END':
        return np.array([p[t] + ((t+1)/12)*s.params[k] for t in range(12)])
    z = np.vstack((np.zeros((1, 2)), p))
    lengths = np.array([np.linalg.norm(z[t+1]-z[t]) for t in range(12)])
    if np.any(lengths <= 0):
        raise ValueError('nonidentity TIME on nonmonotone path')
    q = np.cumsum(lengths)/lengths.sum()
    q[-1] = 1.
    knots = np.r_[0., q]
    if np.any(np.diff(knots) <= 0):
        raise ValueError('indistinguishable knots')
    a, b = s.params[k]
    w1, w2, w3 = q[3], q[7]-q[3], 1-q[7]
    c = (1-a*w1-b*w2)/w3
    if not .75 <= c <= 1.25:
        raise ValueError('invalid third rate')
    result = []
    for t in range(12):
        if t == 11:
            result.append(p[-1].copy())
            continue
        target = a*q[t] if t < 4 else (a*w1+b*(q[t]-w1) if t < 8
                  else a*w1+b*w2+c*(q[t]-q[7]))
        seg = min(int(np.searchsorted(knots, target, side='right'))-1, 11)
        seg = max(seg, 0)
        fraction = (target-knots[seg])/(knots[seg+1]-knots[seg])
        result.append(z[seg]+fraction*(z[seg+1]-z[seg]))
    return np.array(result)


def scalar_metrics(path, past):
    p, h = np.asarray(path, float), np.asarray(past, float)
    velocity = np.array([(p[t]-(p[t-1] if t else np.zeros(2)))/.4 for t in range(12)])
    acceleration = np.array([(velocity[t+1]-velocity[t])/.4 for t in range(11)])
    # Match the registered direct second difference, not a different association.
    jerk = np.array([((velocity[t+2]-velocity[t+1])-(velocity[t+1]-velocity[t]))/(.4**2)
                     for t in range(10)])
    v, a, j = [np.linalg.norm(z, axis=-1) for z in (velocity, acceleration, jerk)]
    return np.array([np.quantile(v, .95), a.mean(), np.quantile(a, .95),
                     j.mean(), np.quantile(j, .95),
                     np.linalg.norm(velocity[0]-(h[-1]-h[-2])/.4)/.4])


def scalar_decode(path, prediction, branch):
    """Independent legal-set and nearest-action selection, including exact ties."""
    s = space(branch)
    u = np.asarray(prediction, float)
    if u.shape != (2,) or not np.isfinite(u).all():
        raise ValueError('invalid saved continuous parameters')
    valid = np.ones(len(s.params), bool)
    if branch == 'TIME':
        z = np.vstack((np.zeros((1, 2)), np.asarray(path, float)))
        lengths = np.array([np.linalg.norm(z[i+1]-z[i]) for i in range(12)])
        valid[:] = False
        if np.all(lengths > 0):
            q = np.cumsum(lengths)/lengths.sum()
            q[-1] = 1.
            if np.all(np.diff(np.r_[0., q]) > 0):
                for i, (a, b) in enumerate(s.params):
                    c = (1-a*q[3]-b*(q[7]-q[3]))/(1-q[7])
                    valid[i] = np.isfinite(c) and .75 <= c <= 1.25
        valid[s.identity] = True
    distance = np.sum((s.normalized-u)**2, axis=1)
    return min(np.flatnonzero(valid), key=lambda i: (
        distance[i], i != s.identity, np.sum(s.normalized[i]**2), i))


def check_saved(x, saved, branch, arithmetic_atol=2e-12, check=lambda: None):
    """Arithmetic tolerance never changes scientific safety decisions."""
    if not np.array_equal(x['sample_id'], saved['sample_id']):
        raise ValueError('audit IDs/order mismatch')
    n = len(x['sample_id'])
    for key, shape in (('raw_paths', (n, 20, 12, 2)), ('guarded_paths', (n, 20, 12, 2)),
                       ('continuous_param', (n, 20, 2)), ('raw_action_id', (n, 20)),
                       ('executed_action_id', (n, 20)), ('rejection_reason', (n, 20, 7))):
        if saved[key].shape != shape or not np.isfinite(saved[key]).all():
            raise ValueError('invalid saved shape/finite contract: '+key)
    if any(saved[k].dtype.kind not in 'iu' for k in ('raw_action_id', 'executed_action_id')):
        raise ValueError('saved action IDs must be integers')
    if saved['rejection_reason'].dtype.kind != 'b':
        raise ValueError('saved reasons must be boolean')
    actions = space(branch)
    maximum = 0.
    for i in range(len(x['sample_id'])):
        check()
        for k in range(20):
            n, h = x['native'][i, k], x['past'][i]
            rid, eid = int(saved['raw_action_id'][i, k]), int(saved['executed_action_id'][i, k])
            if scalar_decode(n, saved['continuous_param'][i, k], branch) != rid:
                raise AssertionError('saved raw action is not the registered nearest action')
            raw = scalar_apply(n, branch, rid)
            safe = scalar_apply(n, branch, eid)
            error = max(float(np.max(np.abs(raw-saved['raw_paths'][i, k]))),
                        float(np.max(np.abs(safe-saved['guarded_paths'][i, k]))))
            maximum = max(maximum, error)
            if error > arithmetic_atol:
                raise AssertionError('saved geometry differs from independent scalar replay')
            # Guard comparisons use stored production path, not reinterpolated last bits.
            sp = saved['raw_paths'][i, k]
            failure = np.r_[~(scalar_metrics(sp, h) <= scalar_metrics(n, h)*1.01), False]
            if np.array_equal(sp, n):
                failure[:] = False
            passed = not failure.any()
            if eid != (rid if passed else actions.identity):
                raise AssertionError('literal guard decision differs; retain boundary evidence')
            if not np.array_equal(saved['rejection_reason'][i, k], failure):
                raise AssertionError('per-metric rejection record inconsistent')
    return dict(status='PASS_SCALAR_GEOMETRY_AND_GUARD_ONLY', max_error=maximum,
                windows=len(x['sample_id']), no_model_replay=True,
                full_quality_clusters_bootstrap_audit=False,
                note='Not a complete production acceptance receipt.')
