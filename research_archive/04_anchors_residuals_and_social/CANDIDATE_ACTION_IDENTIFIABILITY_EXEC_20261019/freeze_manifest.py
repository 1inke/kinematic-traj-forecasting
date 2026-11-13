"""Freeze manifest and source lock for Q0 Action Identifiability Diagnostic.
Strict provenance tracking: records input file SHAs, environment metadata, and protocol.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import scipy
import sklearn

EXEC_DIR = Path(__file__).resolve().parent
ROOT = EXEC_DIR.parent

INPUT_FILES = [
    ROOT / "REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz",
    ROOT / "F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz",
    ROOT / "GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz",
    ROOT / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz",
    ROOT / "COMPLEMENTARY_POOL_VALIDATION_20261018/results/v1/FIT_CAL_SPLIT.json",
]

for s in (42, 43, 44):
    INPUT_FILES.append(ROOT / f"UNION_SELECTOR_K40_TO_K20_20261018/results/cache_pred/kin_FIT_seed{s}.npy")
    INPUT_FILES.append(ROOT / f"CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018/k1_exec/seed{s}/train_data/rows_seed{s}.npz")
    for tgt in ("dA", "dF", "dJ", "dC", "L"):
        INPUT_FILES.append(ROOT / f"CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018/k1_exec/seed{s}/models/S_{tgt}_seed{s}.joblib")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    manifest = {}
    for p in INPUT_FILES:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input asset: {p}")
        manifest[str(p.relative_to(ROOT))] = {
            "sha256": file_sha256(p),
            "size_bytes": p.stat().st_size,
            "last_modified": p.stat().st_mtime,
        }

    env_info = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "platform": sys.platform,
    }

    source_lock = {
        "tag": "q0_action_identifiability_20261019",
        "protocol": "CANDIDATE_ACTION_IDENTIFIABILITY_DESIGN_20261019",
        "created_at": "2026-10-19T12:09:00+09:00",
        "env": env_info,
        "input_count": len(manifest),
    }

    with (EXEC_DIR / "INPUT_MANIFEST.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with (EXEC_DIR / "SOURCE_LOCK.json").open("w", encoding="utf-8") as f:
        json.dump(source_lock, f, indent=2)

    print(f"Locked {len(manifest)} inputs. Manifest written.")


if __name__ == "__main__":
    main()
