"""GT-free decoding, guard and fixed P3 composition."""
import numpy as np
from action_geometry import space, decode, apply_action
from motion_guard import guarded


def predict_paths(past, native, parameters, branch, check=lambda: None):
    p, u = np.asarray(native, dtype=np.float64), np.asarray(parameters)
    if u.shape != p.shape[:2]+(2,) or p.shape[1:] != (20, 12, 2):
        raise ValueError('prediction shapes')
    a = space(branch)
    raw, safe = np.empty_like(p), np.empty_like(p)
    raw_id = np.empty(p.shape[:2], np.int32)
    executed = np.empty_like(raw_id)
    reasons = np.empty(p.shape[:2]+(7,), bool)
    for i in range(len(p)):
        check()
        for k in range(20):
            j = decode(p[i, k], u[i, k], a)
            raw[i, k] = apply_action(p[i, k], a, j)
            safe[i, k], passed, reasons[i, k] = guarded(past[i], p[i, k], raw[i, k])
            raw_id[i, k], executed[i, k] = j, j if passed else a.identity
    return dict(continuous_param=u, raw_action_id=raw_id, executed_action_id=executed,
                rejection_reason=reasons, raw_paths=raw, guarded_paths=safe)


def combine(past, native, endpoint_ids, time_ids, check=lambda: None):
    """Inputs are EXECUTED IDs (identity after rejection), not unguarded IDs."""
    p = np.asarray(native, dtype=np.float64)
    if np.shape(endpoint_ids) != p.shape[:2] or np.shape(time_ids) != p.shape[:2]:
        raise ValueError('P3 IDs mismatch')
    e, t = space('END'), space('TIME')
    raw, safe = np.empty_like(p), np.empty_like(p)
    reasons = np.empty(p.shape[:2]+(7,), bool)
    for i in range(len(p)):
        check()
        for k in range(20):
            ep, tp = int(endpoint_ids[i, k]), int(time_ids[i, k])
            # Revalidate each executed input to prevent composing unguarded records.
            pe, pt = apply_action(p[i, k], e, ep), apply_action(p[i, k], t, tp)
            if not guarded(past[i], p[i, k], pe)[1] or not guarded(past[i], p[i, k], pt)[1]:
                raise ValueError('P3 requires individually guarded actions')
            proposal = pt if ep == e.identity else pt+(np.arange(1, 13)/12)[:, None]*e.params[ep]
            raw[i, k] = proposal
            safe[i, k], _, reasons[i, k] = guarded(past[i], p[i, k], proposal)
    return dict(raw_paths=raw, guarded_paths=safe, rejection_reason=reasons)
