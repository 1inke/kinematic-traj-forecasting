"""Whole-track paired-seed bootstrap; six-comparison family never shrinks."""
import numpy as np


def make_draws(track_key, draws=20000, seed=20261028):
    tracks = np.unique(track_key)
    if not len(tracks):
        raise ValueError('empty frame')
    rng = np.random.default_rng(seed)
    return dict(tracks=tracks,
                track_draw=rng.integers(len(tracks), size=(draws, len(tracks)), dtype=np.int32),
                seed_draw=rng.integers(3, size=(draws, 3), dtype=np.int8))


def paired_ci(difference, track_key, group_mask, draw_table, family=6):
    """difference [3,N], identical indices must be reused for all comparisons."""
    d = np.asarray(difference, dtype=np.float64)
    mask = np.asarray(group_mask, dtype=bool)
    if d.shape != (3, len(track_key)) or mask.shape != (len(track_key),):
        raise ValueError('bootstrap shape mismatch')
    if not np.isfinite(d).all() or family not in (2, 6):
        raise ValueError('nonfinite difference or invalid fixed family')
    tracks, inverse = np.unique(track_key, return_inverse=True)
    if not np.array_equal(tracks, draw_table['tracks']):
        raise ValueError('draw frame mismatch')
    sums = np.stack([np.bincount(inverse[mask], weights=row[mask], minlength=len(tracks)) for row in d])
    counts = np.bincount(inverse[mask], minlength=len(tracks))
    paired = np.full(len(draw_table['track_draw']), np.nan)
    fixed = paired.copy()
    for i, idx in enumerate(draw_table['track_draw']):
        den = counts[idx].sum()
        if den:
            by_seed = sums[:, idx].sum(1)/den
            paired[i] = by_seed[draw_table['seed_draw'][i]].mean()
            fixed[i] = by_seed.mean()
    valid = np.isfinite(paired)
    # Unit fixtures may use fewer draws, but cannot produce a production-ready CI.
    enough = len(paired) == 20000 and valid.sum() >= 19000
    tail = .05/(2*family)
    def intervals(v):
        v = v[np.isfinite(v)]
        return None if not len(v) else dict(
            descriptive95=np.quantile(v, [.025, .975]).tolist(),
            simultaneous=np.quantile(v, [tail, 1-tail]).tolist())
    report = dict(status='ESTIMABLE' if enough else 'INSUFFICIENT_PRODUCTION_DRAWS',
        valid=int(valid.sum()), invalid=int((~valid).sum()), family=family,
        frame_tracks=len(tracks), group_tracks=int(np.count_nonzero(counts)),
        per_seed=d[:, mask].mean(1).tolist() if mask.any() else None,
        paired=intervals(paired), fixed_seed_sensitivity=intervals(fixed))
    return report, dict(paired=paired, fixed_seed=fixed)
