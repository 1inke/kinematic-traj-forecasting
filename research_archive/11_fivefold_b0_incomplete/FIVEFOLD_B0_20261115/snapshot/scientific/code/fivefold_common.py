"""Shared constants and write-once helpers for the five-fold B0 study (numpy only)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
OBS, PRED, MAX_AGENTS, K = 8, 12, 9, 20
DT = 0.4
VAL_FRACTION, VAL_MODE = 0.05, "scene_tail"      # identical to EXP-156
BUDGETS = (64, 128)
BUDGET_SEED = 42
FINAL_SEEDS = (42, 43, 44)
# E* rule (fixed before any run): 128 if the equal-weight mean relative val minADE gain
# 64->128 is >= 1% and the equal-weight mean val minFDE does not get worse; else 64.
BUDGET_MIN_GAIN = 0.01
BUDGET_FLAG_GAIN = 0.03   # >=3% gain is flagged "possibly not saturated at 128" (no 256 run here)


def sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write_json_once(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fold_files(data_dir, fold):
    d = Path(data_dir)
    return {k: d / f"{fold}_{k}.npy" for k in ("data_train", "num_train", "data_test", "num_test")}
