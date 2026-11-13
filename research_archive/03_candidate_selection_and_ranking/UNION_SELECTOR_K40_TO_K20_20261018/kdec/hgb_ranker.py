"""HistGradientBoosting candidate scorer."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def train_hgb(X, y, seed=0):
    yb = (y >= 0.5).astype(np.int32)
    # sample balance
    rng = np.random.default_rng(seed)
    pos = np.where(yb == 1)[0]
    neg = np.where(yb == 0)[0]
    n = min(len(pos) * 3, len(neg), 250_000)
    take_pos = rng.choice(pos, size=min(len(pos), n // 2 + len(pos) // 2), replace=False)
    take_neg = rng.choice(neg, size=min(len(neg), len(take_pos) * 2), replace=False)
    ix = np.concatenate([take_pos, take_neg])
    rng.shuffle(ix)
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        min_samples_leaf=40,
        l2_regularization=0.1,
        random_state=seed,
    )
    clf.fit(X[ix], yb[ix])
    return dict(type="hgb", clf=clf)


def score_hgb(feat, model):
    proba = model["clf"].predict_proba(feat)[:, 1]
    return proba
