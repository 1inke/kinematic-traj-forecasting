"""JSON helpers: strip large arrays, keep summaries."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items() if not str(k).startswith("_raw")}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.size > 2000:
            return dict(_type="ndarray", shape=list(obj.shape), dtype=str(obj.dtype),
                        sha=None, omitted=True)
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def eval_public(ev: dict) -> dict:
    out = {}
    for k, v in ev.items():
        if k.startswith("_"):
            if k == "_classes":
                out[k] = {ck: (cv.tolist() if isinstance(cv, np.ndarray) else cv) for ck, cv in v.items()}
            continue
        out[k] = jsonable(v)
    return out


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")
