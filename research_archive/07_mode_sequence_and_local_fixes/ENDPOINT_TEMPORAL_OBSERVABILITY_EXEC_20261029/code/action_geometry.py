"""Frozen END49/TIME441 geometry. No GT, model, filesystem or training access."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ActionSpace:
    branch: str
    params: np.ndarray
    normalized: np.ndarray
    identity: int
    priority: np.ndarray


def space(branch):
    if branch == 'END':
        p = np.array([(i / 10, j / 10) for i in range(-4, 5)
                      for j in range(-4, 5) if i*i+j*j <= 16], dtype=np.float64)
        u = p / .4
        identity = int(np.flatnonzero(np.all(p == 0, axis=1))[0])
    elif branch == 'TIME':
        rates = .75 + .025 * np.arange(21, dtype=np.float64)
        p = np.array([(a, b) for a in rates for b in rates])
        u = (p - 1) / .25
        identity = int(np.flatnonzero(np.all(p == 1, axis=1))[0])
    else:
        raise ValueError('branch must be END or TIME')
    ids = np.arange(len(p))
    order = np.lexsort((ids, (u*u).sum(1), ids != identity))
    for a in (p, u, order):
        a.flags.writeable = False
    return ActionSpace(branch, p, u, identity, order)


def checked_path(path):
    p = np.asarray(path, dtype=np.float64)
    if p.shape != (12, 2) or not np.isfinite(p).all():
        raise ValueError('expected finite [12,2] path')
    return p


def progress(path):
    p = checked_path(path)
    z = np.vstack((np.zeros((1, 2)), p))
    lengths = np.linalg.norm(np.diff(z, axis=0), axis=1)
    total = lengths.sum()
    if total <= 0 or not np.all(lengths > 0):
        return z, None
    q = np.cumsum(lengths) / total
    q[-1] = 1.
    # Floating-point indistinguishable knots cannot be interpolated strictly.
    if not np.all(np.diff(np.r_[0., q]) > 0):
        return z, None
    return z, q


def legal_actions(path, actions):
    p = checked_path(path)
    if actions.branch == 'END':
        return np.ones(len(actions.params), dtype=bool)
    _, q = progress(p)
    valid = np.zeros(len(actions.params), dtype=bool)
    if q is not None:
        w1, w2, w3 = q[3], q[7]-q[3], 1-q[7]
        c = (1-actions.params[:, 0]*w1-actions.params[:, 1]*w2) / w3
        valid = np.isfinite(c) & (c >= .75) & (c <= 1.25)
    valid[actions.identity] = True
    return valid


def apply_action(path, actions, action_id):
    p = checked_path(path)
    k = int(action_id)
    if k < 0 or k >= len(actions.params):
        raise ValueError('action index out of range')
    if k == actions.identity:
        return p.copy()
    if not legal_actions(p, actions)[k]:
        raise ValueError('geometrically invalid action')
    if actions.branch == 'END':
        return p + (np.arange(1, 13) / 12)[:, None] * actions.params[k]
    z, q = progress(p)
    a, b = actions.params[k]
    w1, w2, w3 = q[3], q[7]-q[3], 1-q[7]
    c = (1-a*w1-b*w2) / w3
    qq = np.r_[a*q[:4], a*w1+b*(q[4:8]-w1),
               a*w1+b*w2+c*(q[8:]-q[7])]
    qq[-1] = 1.
    out = np.stack([np.interp(qq, np.r_[0., q], z[:, d]) for d in range(2)], -1)
    out[-1] = p[-1]
    return out


def enumerate_actions(path, actions):
    valid = legal_actions(path, actions)
    # Invalid rows retain Native but MUST remain masked in oracle selection.
    p = checked_path(path)
    proposals = np.repeat(p[None], len(valid), axis=0)
    ids = np.flatnonzero(valid & (np.arange(len(valid)) != actions.identity))
    if actions.branch == 'END':
        proposals[ids] = p[None] + (np.arange(1, 13)/12)[None, :, None] * actions.params[ids, None, :]
    elif len(ids):
        z, q = progress(p)
        a, b = actions.params[ids, 0:1], actions.params[ids, 1:2]
        w1, w2, w3 = q[3], q[7]-q[3], 1-q[7]
        c = (1-a*w1-b*w2)/w3
        qq = np.concatenate((a*q[:4], a*w1+b*(q[4:8]-w1),
                             a*w1+b*w2+c*(q[8:]-q[7])), axis=1)
        qq[:, -1] = 1.
        proposals[ids] = np.stack([np.interp(qq.ravel(), np.r_[0., q], z[:, d]).reshape(len(ids), 12)
                                   for d in range(2)], axis=-1)
        proposals[ids, -1] = p[-1]
    return proposals, valid


def choose(values, mask, actions):
    values, mask = np.asarray(values), np.asarray(mask, dtype=bool)
    if values.shape != (len(actions.params),) or mask.shape != values.shape:
        raise ValueError('selection shape mismatch')
    if not mask.any() or not np.isfinite(values[mask]).all():
        raise ValueError('empty/nonfinite admissible set')
    # Exact ties only. No epsilon changes the scientific action selected.
    best = values[mask].min()
    candidates = mask & (values == best)
    return int(next(k for k in actions.priority if candidates[k]))


def decode(path, normalized_prediction, actions):
    u = np.asarray(normalized_prediction, dtype=np.float64)
    if u.shape != (2,) or not np.isfinite(u).all():
        raise ValueError('invalid predicted parameters')
    return choose(((actions.normalized-u)**2).sum(1),
                  legal_actions(path, actions), actions)
