"""F0 data audit (numpy only). Extends EXP-156A; never trains, never reads labels into a model.

Checks per fold
  A1 shape [N,9,20,6], num in [1,9], lengths match                         (as EXP-156A)
  A2 no exact train/test sample overlap (valid agents, all 20 frames)       (as EXP-156A)
  A3 leave-one-out: train scene ids and test scene ids are disjoint          (new)
  A4 focal agent finite in all 20 frames; padded agents (index >= num) report their values (new)
  A5 coordinate scale sanity: focal median step speed in m/s (expected ~0.5-1.5)        (new)
  A6 scene_tail validation split sizes (identical rule to EXP-156)          (new)
  A7 SHA256 of the four files, recorded into the study contract              (new)
Writes AUDIT.json (refuses to overwrite) and exits non-zero on any failure.

usage: python -I audit_folds.py --data-dir <...>/processed_data_diverse --out F0/AUDIT.json
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fivefold_common import FOLDS, OBS, DT, fold_files, sha256, write_json_once, now  # noqa: E402
from fivefold_data import scene_tail_split  # noqa: E402


def digest(sample, valid):
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(sample[:valid]).view(np.uint8)); h.update(int(valid).to_bytes(2, "little"))
    return h.digest()


def audit_fold(data_dir, fold):
    files = fold_files(data_dir, fold)
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        return dict(fold=fold, passed=False, failures=[f"missing {m}" for m in missing])
    tr = np.load(files["data_train"], mmap_mode="r"); tn = np.load(files["num_train"], mmap_mode="r")
    te = np.load(files["data_test"], mmap_mode="r"); vn = np.load(files["num_test"], mmap_mode="r")
    fail = []
    if not (tr.ndim == te.ndim == 4 and tr.shape[1:] == te.shape[1:] == (9, 20, 6)):
        fail.append("A1_shape")
    if not (len(tr) == len(tn) and len(te) == len(vn) and tn.min() >= 1 and tn.max() <= 9
            and vn.min() >= 1 and vn.max() <= 9):
        fail.append("A1_counts")
    train_hash = {digest(tr[i], int(tn[i])) for i in range(len(tr))}
    overlap = int(sum(digest(te[i], int(vn[i])) in train_hash for i in range(len(te))))
    if overlap:
        fail.append("A2_overlap")
    s_tr = np.unique(np.asarray(tr[:, 0, 0, 0], np.int64)); s_te = np.unique(np.asarray(te[:, 0, 0, 0], np.int64))
    if np.intersect1d(s_tr, s_te).size:
        fail.append("A3_scene_overlap")
    rows = {}
    for name, d, num in (("train", tr, tn), ("test", te, vn)):
        xy = np.asarray(d[..., -2:], np.float64)
        if not np.isfinite(xy[:, 0]).all():
            fail.append(f"A4_focal_nonfinite_{name}")
        num = np.asarray(num, np.int64)
        pad = np.arange(9)[None] >= num[:, None]
        pad_vals = xy[pad] if pad.any() else np.zeros((0, 20, 2))
        speed = np.linalg.norm(np.diff(xy[:, 0, :OBS], axis=1), axis=-1).mean(1) / DT
        rows[name] = dict(samples=int(len(d)), scenes=np.unique(np.asarray(d[:, 0, 0, 0], np.int64)).tolist(),
                          valid_agents_min_max=[int(num.min()), int(num.max())],
                          padded_agent_slots=int(pad.sum()),
                          padded_values_all_zero=bool(np.all(pad_vals == 0)) if pad.any() else None,
                          padded_values_finite=bool(np.isfinite(pad_vals).all()) if pad.any() else None,
                          focal_obs_speed_mps_median=float(np.median(speed)),
                          focal_obs_speed_mps_p95=float(np.quantile(speed, .95)))
        med = rows[name]["focal_obs_speed_mps_median"]
        if not 0.05 < med < 3.0:
            fail.append(f"A5_speed_scale_{name}_{med:.3f}")
    tr_idx, va_idx = scene_tail_split(np.asarray(tr[:, 0, 0, 0], np.int64))
    return dict(fold=fold, passed=not fail, failures=fail, exact_train_test_overlap=overlap,
                train_scene_ids=s_tr.tolist(), test_scene_ids=s_te.tolist(), splits=rows,
                scene_tail=dict(train=int(len(tr_idx)), validation=int(len(va_idx))),
                sha256={k: sha256(p) for k, p in files.items()})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    folds = [audit_fold(a.data_dir, f) for f in FOLDS]
    for r in folds:
        print(r["fold"], "PASS" if r["passed"] else f"FAIL {r['failures']}")
    passed = all(r["passed"] for r in folds)
    write_json_once(a.out, dict(study="FIVEFOLD_B0_20261115", stage="F0_audit", at=now(),
                                data_dir=str(Path(a.data_dir).resolve()), passed=passed, folds=folds))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
