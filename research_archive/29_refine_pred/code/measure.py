"""Compute past-only covariates, then measure frozen decoder outputs."""

import numpy as np


def vector_norm(x):
    return np.linalg.norm(x, axis=-1)


def observed(past, speed_fn):
    p = np.asarray(past, dtype=np.float64)
    steps = np.diff(p, axis=1)
    length = np.hypot(steps[..., 0], steps[..., 1])
    speeds = length / .4
    early = steps[:, :4].mean(axis=1)
    recent = steps[:, 4:].mean(axis=1)
    denom = vector_norm(early) * vector_norm(recent)
    cosine = (early * recent).sum(axis=1) / np.maximum(denom, 1e-12)
    valid = vector_norm(recent) > .03
    turn = np.where((vector_norm(early) > .03) & valid,
                    np.degrees(np.arccos(cosine.clip(-1, 1))), 0.)
    net = steps.sum(axis=1)
    efficiency = vector_norm(net) / np.maximum(length.sum(axis=1), 1e-12)
    direction = recent / np.maximum(vector_norm(recent)[:, None], 1e-12)
    avg = speeds.mean(axis=1)
    similarity = (steps[:, -1] * net).sum(axis=1) / np.maximum(length[:, -1] * vector_norm(net), 1e-12)
    steady = ((speeds.max(axis=1) > .1) & (avg >= .1) & (np.abs(speeds - avg[:, None]).max(axis=1) <= .25 * avg)
              & (efficiency >= .98) & (similarity >= np.cos(np.deg2rad(15))))
    return dict(X1=speed_fn(p), X2=speeds[:, -3:].mean(axis=1) / np.maximum(speeds[:, :4].mean(axis=1), .1),
                X3=turn, X4=efficiency, valid=valid, observed_steady=steady), direction


def geometry(paths, target, direction, logits, *, compare_order, ordering):
    endpoints = paths[:, :, -1]
    offset = endpoints - target[:, None, -1]
    differences = paths - target[:, None]
    distances = vector_norm(differences)
    fde = distances[:, :, -1]
    best = fde.argmin(axis=1)
    top1 = logits.argmax(axis=1)
    indices = np.arange(len(paths))
    logits64 = logits.astype(np.float64)
    logit_delta = logits64 - logits64.max(axis=1, keepdims=True)
    norm = np.exp(logit_delta).sum(axis=1, keepdims=True)
    log_prob = logit_delta - np.log(norm)
    prob = np.exp(log_prob)
    if compare_order:
        conf, risk = ordering
        conf_fde = np.take_along_axis(fde, conf(logits), axis=1).min(axis=1)
        risk_fde = np.take_along_axis(fde, risk(endpoints, prob), axis=1).min(axis=1)
        np.testing.assert_array_equal(conf_fde, risk_fde)
        np.testing.assert_array_equal(conf_fde, fde[indices, best])
    cross = np.column_stack((-direction[:, 1], direction[:, 0]))
    centered = endpoints - endpoints.mean(axis=1, keepdims=True)
    return dict(minFDE=fde[indices, best], minADE=distances.mean(axis=2).min(axis=1), Top1_FDE=fde[indices, top1],
                brier_minFDE=fde[indices, best] + (1 - prob[indices, best]) ** 2,
                spread_long=np.einsum("nki,ni->nk", centered, direction).std(axis=1),
                spread_lat=np.einsum("nki,ni->nk", centered, cross).std(axis=1),
                entropy=-(prob * log_prob).sum(axis=1), best_slot=best,
                best_error_long=np.einsum("ni,ni->n", offset[indices, best], direction),
                best_error_lat=np.einsum("ni,ni->n", offset[indices, best], cross))
