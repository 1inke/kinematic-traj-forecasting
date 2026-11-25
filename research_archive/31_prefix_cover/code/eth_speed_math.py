"""NumPy-only transforms. No future labels in the scale-selection interface."""
from __future__ import annotations

import numpy as np

DT = .4
SPEED_NAMES = ('low', 'middle', 'high')
MANEUVERS = ('stationary', 'start', 'stop', 'turn_and_speed', 'turn',
             'acceleration', 'deceleration', 'steady')


def observed_speed(past, dt=DT):
    p = np.asarray(past, np.float64)
    if p.ndim == 4:
        p = p[:, 0]
    if p.shape[1:] != (8, 2) or not np.isfinite(p).all() or dt <= 0:
        raise ValueError('BAD_OBSERVATIONS')
    return np.linalg.norm(np.diff(p, axis=1), axis=-1).mean(1) / dt


def fit_profile(past):
    v = observed_speed(past)
    p50, p90 = np.quantile(v, [.5, .9])
    if p90 <= .1:
        raise ValueError('TRAINING_SPEED_RANGE_TOO_SMALL')
    return dict(n=len(v), p50=float(p50), p90=float(p90),
                p95=float(np.quantile(v, .95)), maximum=float(v.max()))


def speed_bins(past, profile):
    # Exactly one group per window; edges are fitted on the 95% TRAIN subset.
    return np.searchsorted([profile['p50'], profile['p90']], observed_speed(past), side='right')


def scales(past, profile):
    # Shrink only observations above TRAIN p90; never estimate ETH TEST quantiles.
    v = observed_speed(past)
    return np.clip(profile['p90'] / np.maximum(v, profile['p90']), .5, 1.)


def transform_past(past, num_valid, scale, kind='motion'):
    p = np.asarray(past)
    n = np.asarray(num_valid)
    s = np.asarray(scale, np.float64)
    if p.shape[1:] != (9, 8, 2) or n.shape != (len(p),) or s.shape != (len(p),):
        raise ValueError('TRANSFORM_SHAPE')
    if not np.isfinite(p).all() or not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError('TRANSFORM_NONFINITE_OR_SCALE')
    if ((n < 1) | (n > 9)).any():
        raise ValueError('NUM_VALID_RANGE')
    if kind not in ('motion', 'scene'):
        raise ValueError('UNKNOWN_TRANSFORM')
    out = p.copy()
    # Leave scale=1 windows bitwise untouched (including padding).
    chosen = np.flatnonzero(s != 1)
    for i in chosen:
        anchor = p[i, :n[i], -1:] if kind == 'motion' else p[i, :1, -1:]
        out[i, :n[i]] = anchor + s[i] * (p[i, :n[i]] - anchor)
    return out


def inverse_paths(paths, original_past, scale):
    result = np.asarray(paths, np.float64).copy()
    scale = np.asarray(scale, np.float64)
    if result.shape != (len(original_past), 20, 12, 2) or (scale <= 0).any():
        raise ValueError('INVERSE_SHAPE_OR_SCALE')
    idx = np.flatnonzero(scale != 1)
    anchor = np.asarray(original_past, np.float64)[idx, 0, -1, None, None, :]
    result[idx] = anchor + (result[idx] - anchor) / scale[idx, None, None, None]
    if not np.isfinite(result).all():
        raise ValueError('NONFINITE_INVERTED_PATHS')
    return result


def maneuver_labels(past, future):
    """Evaluation only. Labels use future; never call from inference or scaling."""
    p = np.asarray(past, np.float64)[:, 0]
    f = np.asarray(future, np.float64)
    obs = observed_speed(p)
    v = np.diff(np.concatenate([p[:, -1:], f], axis=1), axis=1) / DT
    fut = np.linalg.norm(v, axis=-1).mean(1)
    a = (p[:, -1] - p[:, -4]) / (3 * DT)
    b = (f[:, -1] - p[:, -1]) / (12 * DT)
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    angle = np.degrees(np.arccos(np.clip((a*b).sum(1) / np.maximum(norm, 1e-12), -1, 1)))
    angle[norm <= 1e-8] = 0
    ratio = fut / np.maximum(obs, .1)
    turn = angle >= 30
    acc, dec = ratio >= 1.25, ratio <= .8
    labels = np.full(len(p), 'steady', dtype='<U20')
    labels[dec] = 'deceleration'; labels[acc] = 'acceleration'
    labels[turn] = 'turn'; labels[turn & (acc | dec)] = 'turn_and_speed'
    labels[(obs >= .1) & (fut < .1)] = 'stop'
    labels[(obs < .1) & (fut >= .1)] = 'start'
    labels[(obs < .1) & (fut < .1)] = 'stationary'
    return labels


def gain(base, treatment):
    return (base - treatment) / base if base is not None and treatment is not None and base > 0 else None


def block(metrics, mask):
    count = int(mask.sum())
    if not count:
        return dict(n=0, small_group=True, minADE=None, minFDE=None, coverage=None)
    out = {k: float(np.mean(metrics[k][mask])) for k in
           ('minADE', 'minFDE', 'coverage', 'top1_ADE', 'top1_FDE', 'top1_good')}
    out.update(n=count, small_group=count < 20,
               ADE_p95=float(np.quantile(metrics['minADE'][mask], .95)),
               FDE_p95=float(np.quantile(metrics['minFDE'][mask], .95)))
    return out


def stratify(metrics, past, future, profile, groups):
    bins = speed_bins(past, profile)
    labels = maneuver_labels(past, future)
    all_rows = np.ones(len(past), bool)
    return dict(overall=block(metrics, all_rows),
                speed={name: block(metrics, bins == i) for i, name in enumerate(SPEED_NAMES)},
                maneuver={name: block(metrics, labels == name) for name in MANEUVERS},
                original_group={name: block(metrics, groups == i)
                                for i, name in enumerate(('stable', 'transition', 'change'))},
                speed_by_maneuver={name: {m: block(metrics, (bins == i) & (labels == m))
                                         for m in MANEUVERS}
                                   for i, name in enumerate(SPEED_NAMES)})


def diagnostic_decision(results, audit):
    """Fixed exploratory triage, not a significance test or causal verdict."""
    e = results['eth']; b = e['cases']['identity']; p = e['cases']['high_motion']
    r = e['cases']['reverse_motion']
    ade = gain(b['overall']['minADE'], p['overall']['minADE'])
    fde = gain(b['overall']['minFDE'], p['overall']['minFDE'])
    reverse = gain(b['overall']['minADE'], r['overall']['minADE'])
    h = gain(b['speed']['high']['minADE'], p['speed']['high']['minADE'])
    hotel = results['hotel']['cases']
    hotel_gain = gain(hotel['identity']['overall']['minADE'], hotel['high_motion']['overall']['minADE'])
    checks = dict(pipeline_audit_passed=bool(audit['pipeline_consistent']),
                  baseline_replay_passed=bool(e['baseline_replay_passed']),
                  low_speed_identity_passed=bool(e['low_speed_identity_passed']),
                  high_speed_n_at_least_30=b['speed']['high']['n'] >= 30,
                  overall_ADE_gain_at_least_5pct=ade is not None and ade >= .05,
                  overall_FDE_gain_at_least_2pct=fde is not None and fde >= .02,
                  high_ADE_gain_at_least_10pct=h is not None and h >= .10,
                  forward_beats_reverse_by_2pp=ade is not None and reverse is not None and ade-reverse >= .02)
    return dict(status='ELIGIBLE_EXPLORATORY' if all(checks.values()) else 'DO_NOT_TRAIN_YET',
                checks=checks, ETH_ADE_gain=ade, ETH_FDE_gain=fde, ETH_high_ADE_gain=h,
                reverse_ADE_gain=reverse, HOTEL_ADE_gain=hotel_gain,
                response_scope='also_seen_in_HOTEL' if hotel_gain is not None and hotel_gain >= .05
                               else 'ETH_stronger_or_HOTEL_weak',
                causal_conclusion='NOT_ESTABLISHED', training_started=False,
                note='Eligibility is a fixed cost-control heuristic; it does not prove speed is the cause.')
