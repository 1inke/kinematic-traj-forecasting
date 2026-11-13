"""Fit 5 HGB regressors per arm. Identical rows/order; only targets differ."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from .config import HGB_PARAMS, OUTPUT_NAMES, FEATURE_DIM


def make_regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=int(seed), **HGB_PARAMS)


def fit_arm(X: np.ndarray, Y: np.ndarray, seed: int, arm: str, out_dir: Path) -> dict:
    """
    X (N,113), Y (N,5) columns = OUTPUT_NAMES.
    Saves joblib per output + meta.
    """
    assert X.shape[1] == FEATURE_DIM
    assert Y.shape[1] == len(OUTPUT_NAMES)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(arm=arm, seed=seed, n_rows=int(X.shape[0]), params=HGB_PARAMS, outputs={})
    models = {}
    for j, name in enumerate(OUTPUT_NAMES):
        y = Y[:, j]
        info = dict(
            mean=float(y.mean()),
            std=float(y.std()),
            min=float(y.min()),
            max=float(y.max()),
            n_pos=(int((y > 0).sum()) if name != "L" else int((y > 0).sum())),
            n_zero=int((y == 0).sum()),
            n_unique=int(len(np.unique(y))),
            is_constant=bool(len(np.unique(y)) <= 1),
        )
        reg = make_regressor(seed)
        reg.fit(X, y)
        path = out_dir / f"{arm}_{name}_seed{seed}.joblib"
        joblib.dump(reg, path)
        models[name] = reg
        meta["outputs"][name] = {**info, "path": str(path)}
        print(f"  fitted {arm}/{name} seed{seed} const={info['is_constant']} mean={info['mean']:.6g}", flush=True)
    (out_dir / f"{arm}_meta_seed{seed}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return dict(models=models, meta=meta)


def load_arm(out_dir: Path, arm: str, seed: int) -> dict:
    models = {}
    for name in OUTPUT_NAMES:
        path = out_dir / f"{arm}_{name}_seed{seed}.joblib"
        models[name] = joblib.load(path)
    return models


def predict_arm(models: dict, X: np.ndarray) -> np.ndarray:
    """Return (N,5) in OUTPUT_NAMES order."""
    cols = [models[n].predict(X) for n in OUTPUT_NAMES]
    return np.stack(cols, axis=1)
