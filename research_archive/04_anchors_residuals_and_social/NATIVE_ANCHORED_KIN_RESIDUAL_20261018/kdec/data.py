"""只读加载既有 ETH-UCY 缓存。"""
from __future__ import annotations
from pathlib import Path
import numpy as np

ROOT = Path(r"<WORKSPACE_ROOT>")
CACHE = ROOT / "REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz"
LABELS = ROOT / "F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz"
SPLITS = ROOT / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz"
ENCODER = ROOT / "GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz"
SIZES = dict(FIT=18909, outer=1035, inner=1856, original_CAL=2879)


def load_bundle():
    with np.load(CACHE, allow_pickle=False) as z:
        sample_id = z["sample_id"].copy()
        native = z["candidate"][..., :24].reshape(-1, 20, 12, 2).astype(np.float64)
        groups = z["change_group"].copy()
    with np.load(LABELS, allow_pickle=False) as z:
        assert np.array_equal(z["sample_id"], sample_id)
        gt = z["future"].astype(np.float64)
    with np.load(ENCODER, allow_pickle=False) as z:
        assert np.array_equal(z["sample_id"], sample_id)
        past = z["past"].astype(np.float64)
    assert np.allclose(past[:, -1], 0.0)
    sp = np.load(SPLITS, allow_pickle=False)
    assert np.array_equal(sp["sample_id"], sample_id[:25912])
    split = dict(
        FIT=sp["grouped_train_index"].copy(),
        outer=sp["outer_eval_index"].copy(),
        inner=sp["inner_cal_index"].copy(),
        original_CAL=np.arange(25912, 28791),
    )
    for k, n in SIZES.items():
        assert len(split[k]) == n
    return dict(sample_id=sample_id, past=past, gt=gt, native=native, groups=groups, split=split)
