"""锁定超参的 HGB；声明 soft→good 退化。"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

HGB_PARAMS = dict(
    max_depth=6,
    learning_rate=0.08,
    max_iter=200,
    min_samples_leaf=40,
    l2_regularization=0.1,
)


def train_hgb(X, y_soft, seed=0):
    """y_soft 会被阈值化为 good（>=0.5）；与原实现一致。"""
    yb = (y_soft >= 0.5).astype(np.int32)
    rng = np.random.default_rng(seed)
    pos = np.where(yb == 1)[0]
    neg = np.where(yb == 0)[0]
    n = min(len(pos) * 3, len(neg), 250_000)
    take_pos = rng.choice(pos, size=min(len(pos), n // 2 + len(pos) // 2), replace=False)
    take_neg = rng.choice(neg, size=min(len(neg), len(take_pos) * 2), replace=False)
    ix = np.concatenate([take_pos, take_neg])
    rng.shuffle(ix)
    clf = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=seed)
    clf.fit(X[ix], yb[ix])
    return dict(
        type="hgb",
        clf=clf,
        params=HGB_PARAMS,
        label_note="soft_max(exp(-max(ADE/0.25,FDE/0.5)),good) thresholds to binary good at 0.5",
        n_train=int(len(ix)),
        n_pos=int((yb[ix] == 1).sum()),
        n_neg=int((yb[ix] == 0).sum()),
        seed=int(seed),
    )


def score_hgb(feat, model):
    return model["clf"].predict_proba(feat)[:, 1]


def save_hgb(model, path):
    joblib.dump(model, path)


def load_hgb(path):
    return joblib.load(path)
