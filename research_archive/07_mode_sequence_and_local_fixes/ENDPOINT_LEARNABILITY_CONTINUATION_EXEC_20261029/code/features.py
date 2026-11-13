"""Train-only 40D statistics. No GT or metadata features."""
import numpy as np


def raw_features(past, native):
    h, p = np.asarray(past, dtype=np.float64), np.asarray(native, dtype=np.float64)
    if h.shape != (len(p), 8, 2) or p.shape[1:] != (20, 12, 2):
        raise ValueError('feature shapes')
    history = np.broadcast_to(h.reshape(-1, 1, 16), (len(p), 20, 16))
    return np.concatenate((history, p.reshape(-1, 20, 24)), -1)


def fit_statistics(past, native, sample_id):
    x = raw_features(past, native)
    mean, std = x.mean(axis=(0, 1)), x.std(axis=(0, 1))
    std[std < 1e-6] = 1.
    return dict(mean=mean, std=std, train_sample_id=np.array(sample_id, copy=True))


def transform(past, native, stats, arm):
    if arm not in ('G', 'H'):
        raise ValueError('arm must be G or H')
    mean, std = np.asarray(stats['mean']), np.asarray(stats['std'])
    if mean.shape != (40,) or std.shape != (40,) or not (std > 0).all():
        raise ValueError('invalid statistics')
    x = (raw_features(past, native)-mean)/std
    if arm == 'G':
        x[..., :16] = 0.
    if not np.isfinite(x).all():
        raise ValueError('nonfinite normalized feature')
    return x.astype(np.float32)
