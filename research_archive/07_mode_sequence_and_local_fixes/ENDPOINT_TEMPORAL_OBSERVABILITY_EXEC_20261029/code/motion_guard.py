"""GT-free, literal 1.01 motion + history junction filter."""
import numpy as np

MOTION_NAMES = ('speed_p95', 'accel_mean', 'accel_p95', 'jerk_mean', 'jerk_p95')
FAILURE_NAMES = MOTION_NAMES + ('junction', 'nonfinite')


def derivatives(paths):
    p = np.asarray(paths, dtype=np.float64)
    zero = np.zeros(p.shape[:-2] + (1, 2))
    v = np.diff(np.concatenate((zero, p), axis=-2), axis=-2) / .4
    a = np.diff(v, axis=-2) / .4
    j = np.diff(v, n=2, axis=-2) / (.4**2)
    return tuple(np.linalg.norm(x, axis=-1) for x in (v, a, j))


def candidate_metrics(paths):
    v, a, j = derivatives(paths)
    return np.stack((np.quantile(v, .95, axis=-1), a.mean(-1),
                     np.quantile(a, .95, axis=-1), j.mean(-1),
                     np.quantile(j, .95, axis=-1)), axis=-1)


def junction(past, paths):
    h = np.asarray(past, dtype=np.float64)
    vobs = (h[-1]-h[-2]) / .4
    return np.linalg.norm(np.asarray(paths)[..., 0, :] / .4-vobs, axis=-1) / .4


def inspect(past, native, proposals):
    """One observed window, one native slot, any leading proposal dimensions."""
    h, n, p = (np.asarray(x, dtype=np.float64) for x in (past, native, proposals))
    if h.shape != (8, 2) or n.shape != (12, 2) or p.shape[-2:] != (12, 2):
        raise ValueError('motion input shapes')
    if not np.isfinite(h).all() or not np.isfinite(n).all():
        raise ValueError('nonfinite original input')
    finite = np.isfinite(p).all(axis=(-2, -1))
    with np.errstate(invalid='ignore', over='ignore'):
        fail_motion = ~(candidate_metrics(p) <= candidate_metrics(n)*1.01)
        fail_junction = ~(junction(h, p) <= junction(h, n)*1.01)
    failed = np.concatenate((fail_motion, fail_junction[..., None],
                             (~finite)[..., None]), axis=-1)
    identity = np.all(p == n, axis=(-2, -1))
    failed[identity] = False
    return ~failed.any(axis=-1), failed


def guarded(past, native, proposal):
    passed, reasons = inspect(past, native, proposal)
    return (np.array(proposal if passed else native, dtype=np.float64, copy=True),
            bool(passed), reasons)


def pooled(paths):
    v, a, j = derivatives(paths)
    if not v.size:
        raise ValueError('empty motion population')
    return np.array([np.quantile(v, .95), a.mean(), np.quantile(a, .95),
                     j.mean(), np.quantile(j, .95)])
