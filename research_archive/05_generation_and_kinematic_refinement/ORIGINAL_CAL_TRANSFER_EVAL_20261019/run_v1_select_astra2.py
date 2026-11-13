# -*- coding: utf-8 -*-
"""V1-only select on original_CAL using astra2 (sklearn 1.4). Saves indices for venv-k0 eval."""
from __future__ import annotations
import os, sys, json, time
from datetime import datetime
from pathlib import Path
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

ROOT = Path(r"<WORKSPACE_ROOT>")
EXEC = ROOT / "CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018"
CPV = ROOT / "COMPLEMENTARY_POOL_VALIDATION_20261018"
OUT = ROOT / "ORIGINAL_CAL_TRANSFER_EVAL_20261019"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "indices").mkdir(exist_ok=True)

sys.path.insert(0, str(EXEC))
sys.path.insert(0, str(CPV))

from k0.data_io import load_bundle, load_kin, array_sha, file_sha
from kdec.features import build_pool_features
from kdec.hgb_locked import load_hgb, score_hgb
from kdec.select_locked import select_cond_boost_11_9

SEEDS = (42, 43, 44)


def now():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S+09:00")


def main():
    print(f"START V1-select {now()}", flush=True)
    data = load_bundle()
    ix = data["split"]["original_CAL"]
    past = data["past"][ix]
    native = data["native"][ix]
    ids = data["sample_id"][ix]
    groups = data["groups"][ix]
    tracks = data["track_key"][ix].astype(str)
    summary = []
    for seed in SEEDS:
        dest = OUT / "indices" / f"v1_seed{seed}_indices.npy"
        meta = OUT / "indices" / f"v1_seed{seed}_meta.json"
        if dest.exists() and meta.exists():
            print(f"seed{seed} EXISTS", flush=True)
            summary.append(json.loads(meta.read_text(encoding="utf-8")))
            continue
        kin, src = load_kin("original_CAL", seed, data)
        assert kin is not None and kin.shape[0] == 2879
        hgb_path = CPV / "artifacts" / "v1" / f"hgb_seed{seed}.joblib"
        t0 = time.time()
        model = load_hgb(hgb_path)
        fn, fk = build_pool_features(past, native, kin.astype(np.float64))
        F = fn.shape[-1]
        B = native.shape[0]
        sn = score_hgb(fn.reshape(-1, F), model).reshape(B, -1)
        sk = score_hgb(fk.reshape(-1, F), model).reshape(B, -1)
        paths_b, pool_idx_b, idx_n_b, idx_k_b, sk_boost = select_cond_boost_11_9(
            native, kin.astype(np.float64), sn, sk
        )
        np.save(dest, pool_idx_b.astype(np.int16))
        np.savez_compressed(
            OUT / "indices" / f"v1_seed{seed}_bundle.npz",
            sample_id=ids, track_key=tracks, groups=groups,
            selected_pool_indices=pool_idx_b.astype(np.int16),
            scores_sn=sn.astype(np.float32), scores_sk=sk.astype(np.float32),
            sk_boost=sk_boost.astype(np.float32),
        )
        rec = dict(
            seed=seed, status="OK", wall_s=time.time() - t0,
            hgb=str(hgb_path), hgb_sha=file_sha(hgb_path),
            kin_src=src, indices_sha=array_sha(pool_idx_b),
            sklearn_note="astra2 sklearn for V1 HGB load only",
            finished_at=now(),
        )
        meta.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        summary.append(rec)
        print(f"seed{seed} OK wall={rec['wall_s']:.1f}s", flush=True)
    (OUT / "indices" / "v1_select_summary.json").write_text(
        json.dumps(dict(finished_at=now(), seeds=summary), indent=2), encoding="utf-8"
    )
    print("DONE V1-select", flush=True)


if __name__ == "__main__":
    main()
